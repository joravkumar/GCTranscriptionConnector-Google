import asyncio
import audioop
import os
import json
import time
import threading
import queue
import tempfile
from datetime import timedelta
import aiohttp
import logging
import math
import re
import wave

from config import (
    OPENAI_API_KEY,
    OPENAI_SPEECH_MODEL
)
from language_mapping import (
    normalize_language_code,
    get_openai_language_code,
    is_openai_unsupported_language,
    get_language_name_for_prompt,
    get_language_specific_prompt
)

KNOWN_ARTIFACTS = [
    "context:",
    "ring",
    "context",
    "begin",
    "beep",
    "[beep]",
    "[ring]"
]

ARTIFACT_PATTERNS = [
    r"^\s*context:?\s*",
    r"^\s*ring\s*",
    r"^\s*\[?beep\]?\s*",
    r"^\s*\[?ring\]?\s*"
]

class MockResult:
    def __init__(self):
        self.results = []

class Result:
    def __init__(self, alternatives=None, is_final=True):
        self.alternatives = alternatives or []
        self.is_final = is_final

class Alternative:
    def __init__(self, transcript="", confidence=0.9, words=None):
        self.transcript = transcript
        self.confidence = confidence
        self.words = words or [] # Note: OpenAI API used here doesn't provide reliable word timings

class Word:
    def __init__(self, word="", start_offset=None, end_offset=None, confidence=0.9):
        self.word = word
        # OpenAI API used here doesn't provide reliable word timings. Offsets will be approximate.
        self.start_offset = start_offset or timedelta(seconds=0)
        self.end_offset = end_offset or timedelta(seconds=1)
        self.confidence = confidence

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.logger = logger.getChild(f"OpenAIStream_{language}")
        self.language = normalize_language_code(language)
        self.openai_language = get_openai_language_code(self.language)
        self.is_unsupported_language = is_openai_unsupported_language(self.language)

        if self.is_unsupported_language:
            self.language_prompt = get_language_name_for_prompt(self.language)
            self.logger.info(f"Initialized StreamingTranscription with language={self.language}, unsupported by OpenAI API; using language name '{self.language_prompt}' in prompt")
        else:
            self.language_prompt = None
            self.logger.info(f"Initialized StreamingTranscription with language={self.language}, openai_language={self.openai_language}")

        self.channels = channels
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = threading.Event()
        self.running.set()

        self.vad_threshold = 200
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames_count = [0 for _ in range(channels)]
        self.speech_frames_count = [0 for _ in range(channels)]

        self.silence_threshold_frames = 8 # 8 * 100ms = 800ms of silence to trigger end of utterance
        self.min_speech_frames = 2 # Need at least 200ms of speech to start utterance

        self.accumulated_audio = [bytearray() for _ in range(channels)]
        self.accumulated_start_sample = [0 for _ in range(channels)] # Track start sample of accumulated audio

        self.last_transcripts = ["" for _ in range(channels)]
        self.last_process_time = [time.time() for _ in range(channels)]

        self.initial_frames_processed = [0 for _ in range(channels)]
        self.skip_initial_frames = 5 # Skip first 500ms

        self.token_confidence_threshold = 0.2 # Threshold for logging low confidence tokens

        self.sample_rate = 8000
        self.bytes_per_sample = 2 # For PCM16

        # Track total valid samples fed per channel (for offset calculation)
        self.valid_samples_fed = [0] * channels

        # Max buffer duration before forcing processing (e.g., 30 seconds)
        self.max_buffer_duration_sec = 30.0
        self.max_buffer_bytes = int(self.max_buffer_duration_sec * self.sample_rate * self.bytes_per_sample)

        # Timeout for processing if no silence detected (e.g., 5 seconds)
        self.processing_timeout_sec = 5.0


    def start_streaming(self):
        if not OPENAI_API_KEY:
            self.logger.error("OPENAI_API_KEY environment variable not set. Cannot start streaming.")
            raise RuntimeError("OPENAI_API_KEY not set.")

        self.running.set()
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self._streaming_recognize_thread, args=(channel,), daemon=True
            )
            self.streaming_threads[channel].start()
            self.logger.info(f"Started streaming thread for channel {channel}")

    def stop_streaming(self):
        self.logger.info("Stopping streaming threads.")
        self.running.clear()
        for q in self.audio_queues:
            q.put(None) # Send sentinel
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                 self.logger.debug(f"Joining streaming thread for channel {channel}")
                 self.streaming_threads[channel].join(timeout=2.0)
                 if self.streaming_threads[channel].is_alive():
                      self.logger.warning(f"Streaming thread for channel {channel} did not exit cleanly.")
            self.streaming_threads[channel] = None
        self.logger.info("Streaming threads stopped.")

    def _streaming_recognize_thread(self, channel):
        self.logger.info(f"Streaming thread started for channel {channel}")
        loop = None
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            while self.running.is_set():
                try:
                    audio_chunk_data = self.audio_queues[channel].get(timeout=0.1)
                    if audio_chunk_data is None: # Sentinel
                        if len(self.accumulated_audio[channel]) > 0:
                            self.logger.info(f"Channel {channel}: Processing remaining audio before exit.")
                            self._process_accumulated_audio(channel, loop, force_process=True)
                        break

                    pcm16_chunk, chunk_start_sample_offset = audio_chunk_data
                    samples_in_chunk = len(pcm16_chunk) // self.bytes_per_sample

                    # Skip initial frames
                    if self.initial_frames_processed[channel] < self.skip_initial_frames:
                        self.initial_frames_processed[channel] += 1
                        self.logger.debug(f"Channel {channel}: Skipping initial frame {self.initial_frames_processed[channel]}/{self.skip_initial_frames}")
                        continue

                    # Basic VAD
                    rms = audioop.rms(pcm16_chunk, self.bytes_per_sample)
                    is_current_speech = rms > self.vad_threshold

                    # Append audio and update start sample if buffer was empty
                    if not self.accumulated_audio[channel]:
                        self.accumulated_start_sample[channel] = chunk_start_sample_offset
                    self.accumulated_audio[channel].extend(pcm16_chunk)

                    force_process = False
                    process_now = False

                    if is_current_speech:
                        self.silence_frames_count[channel] = 0
                        self.speech_frames_count[channel] += 1

                        if not self.is_speech[channel] and self.speech_frames_count[channel] >= self.min_speech_frames:
                            self.is_speech[channel] = True
                            self.logger.debug(f"Channel {channel}: Speech detected (RMS: {rms})")
                            self.last_process_time[channel] = time.time() # Reset timer on speech start

                    else: # Silence frame
                        self.speech_frames_count[channel] = 0
                        if self.is_speech[channel]: # If we were previously speaking
                            self.silence_frames_count[channel] += 1
                            if self.silence_frames_count[channel] >= self.silence_threshold_frames:
                                self.logger.debug(f"Channel {channel}: End of speech detected after {self.silence_frames_count[channel]} silence frames.")
                                self.is_speech[channel] = False
                                process_now = True # Process the completed utterance
                        # else: We are in silence, do nothing special unless timeout/buffer full

                    # Check for processing due to timeout or buffer size
                    current_time = time.time()
                    accumulated_duration = len(self.accumulated_audio[channel]) / (self.sample_rate * self.bytes_per_sample)

                    if self.is_speech[channel] and current_time - self.last_process_time[channel] > self.processing_timeout_sec:
                         self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout ({self.processing_timeout_sec}s) during speech.")
                         process_now = True
                         force_process = True # Indicate it's not a natural end of utterance

                    if len(self.accumulated_audio[channel]) >= self.max_buffer_bytes:
                        self.logger.warning(f"Channel {channel}: Accumulated audio buffer full ({len(self.accumulated_audio[channel])} bytes >= {self.max_buffer_bytes} bytes), forcing processing.")
                        process_now = True
                        force_process = True # Indicate it's not a natural end of utterance

                    if process_now:
                        self._process_accumulated_audio(channel, loop, force_process)

                except queue.Empty:
                    # Check for timeout even if queue is empty, in case of long silence after speech
                    if self.is_speech[channel] and time.time() - self.last_process_time[channel] > self.processing_timeout_sec:
                         self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout ({self.processing_timeout_sec}s) during silence after speech.")
                         self._process_accumulated_audio(channel, loop, force_process=True)
                    continue
                except Exception as e:
                    self.logger.error(f"Channel {channel}: Error in streaming thread loop: {e}", exc_info=True)
                    # Maybe put exception in response queue?
                    self.response_queues[channel].put((e, 0))
                    break # Exit thread on unexpected error

        except Exception as e:
            self.logger.error(f"Channel {channel}: Fatal error setting up streaming thread: {e}", exc_info=True)
            self.response_queues[channel].put((e, 0)) # Signal error
        finally:
            if loop:
                loop.close()
            self.logger.info(f"Streaming thread finished for channel {channel}")

    def _process_accumulated_audio(self, channel, loop, force_process=False):
        if len(self.accumulated_audio[channel]) < self.sample_rate * self.bytes_per_sample * 0.2: # Need at least 200ms
            self.logger.debug(f"Channel {channel}: Accumulated audio too short ({len(self.accumulated_audio[channel])} bytes), skipping processing.")
            # Don't clear buffer if forced, might accumulate more later
            if not force_process:
                 self.accumulated_audio[channel] = bytearray()
                 self.is_speech[channel] = False # Reset VAD state if skipping
            return

        audio_data_to_process = bytes(self.accumulated_audio[channel])
        start_sample_offset = self.accumulated_start_sample[channel]

        # Clear the buffer *before* async processing call
        self.accumulated_audio[channel] = bytearray()
        # Reset VAD state only if it wasn't a forced process (mid-speech)
        if not force_process:
             self.is_speech[channel] = False
             self.silence_frames_count[channel] = 0
             self.speech_frames_count[channel] = 0

        self.last_process_time[channel] = time.time()

        # Check RMS again on the whole chunk
        rms = audioop.rms(audio_data_to_process, self.bytes_per_sample)
        if rms < self.vad_threshold * 0.5: # Lower threshold for accumulated audio
            self.logger.debug(f"Channel {channel}: Accumulated audio RMS ({rms}) below threshold, skipping API call.")
            return

        self.logger.info(f"Channel {channel}: Processing {len(audio_data_to_process)} bytes of audio starting at sample {start_sample_offset} (RMS: {rms})")

        temp_wav_path = None
        try:
            # Use a context manager for the temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                self._write_wav_file(temp_wav, audio_data_to_process)

            # Run the async transcription in the loop
            result_future = asyncio.run_coroutine_threadsafe(
                self._stream_transcribe_audio(temp_wav_path, channel, start_sample_offset),
                loop
            )

            # Wait for the result with a timeout
            try:
                result = result_future.result(timeout=45.0) # Generous timeout for API call + processing
                if result and not isinstance(result, Exception):
                    transcript_text = ""
                    if result.results and result.results[0].alternatives:
                        alt = result.results[0].alternatives[0]
                        transcript_text = alt.transcript

                    filtered_transcript = self._filter_spurious_artifacts(transcript_text)

                    if filtered_transcript:
                         # Update the transcript in the result object
                         if result.results and result.results[0].alternatives:
                              result.results[0].alternatives[0].transcript = filtered_transcript

                         # Send non-duplicate, non-empty transcripts
                         if filtered_transcript != self.last_transcripts[channel]:
                              # Put tuple: (result, start_sample_offset)
                              self.response_queues[channel].put((result, start_sample_offset))
                              self.last_transcripts[channel] = filtered_transcript
                         else:
                              self.logger.debug(f"Channel {channel}: Skipping duplicate transcript: '{filtered_transcript}'")
                    else:
                         self.logger.debug(f"Channel {channel}: Transcript empty after filtering.")

                elif isinstance(result, Exception):
                     self.logger.error(f"Channel {channel}: Transcription task returned an exception: {result}")
                     self.response_queues[channel].put((result, start_sample_offset)) # Propagate exception

            except asyncio.TimeoutError:
                 self.logger.error(f"Channel {channel}: Timeout waiting for transcription result.")
                 # Put an error or handle timeout? For now, just log.
            except Exception as e:
                 self.logger.error(f"Channel {channel}: Error getting transcription result: {e}")
                 self.response_queues[channel].put((e, start_sample_offset)) # Propagate exception

        except Exception as e:
            self.logger.error(f"Channel {channel}: Error processing accumulated audio: {e}", exc_info=True)
            # Ensure buffer is cleared even on error
            self.accumulated_audio[channel] = bytearray()
            self.is_speech[channel] = False
        finally:
             # Clean up temporary file if it exists
             if temp_wav_path and os.path.exists(temp_wav_path):
                 try:
                     os.unlink(temp_wav_path)
                 except OSError as e:
                     self.logger.warning(f"Channel {channel}: Could not delete temp file {temp_wav_path}: {e}")


    def _filter_spurious_artifacts(self, transcript):
        if not transcript:
            return ""

        original_transcript = transcript
        for pattern in ARTIFACT_PATTERNS:
            transcript = re.sub(pattern, "", transcript, flags=re.IGNORECASE)

        words = transcript.split()
        filtered_words = [word for word in words if word.lower() not in KNOWN_ARTIFACTS]
        transcript = " ".join(filtered_words)

        transcript = re.sub(r'\s+', ' ', transcript).strip()

        if transcript != original_transcript:
             self.logger.debug(f"Filtered artifacts: '{original_transcript}' -> '{transcript}'")

        return transcript

    def _write_wav_file(self, temp_wav_file, audio_data):
        # Using wave module for robust header writing
        try:
            with wave.open(temp_wav_file, 'wb') as wf:
                wf.setnchannels(1) # Mono
                wf.setsampwidth(self.bytes_per_sample) # 16-bit PCM
                wf.setframerate(self.sample_rate) # 8000 Hz
                wf.writeframes(audio_data)
            self.logger.debug(f"Wrote {len(audio_data)} bytes to temporary WAV file: {temp_wav_file.name}")
        except Exception as e:
            self.logger.error(f"Error writing WAV file {temp_wav_file.name}: {e}")
            raise # Re-raise the exception

    async def _stream_transcribe_audio(self, file_path, channel, start_sample_offset):
        # This runs in the separate thread's event loop
        try:
            openai_lang = self.openai_language
            prompt = "This is a customer service call. Ignore initial beeps, rings, and system sounds."

            if self.is_unsupported_language:
                self.logger.info(f"Channel {channel}: Using special handling for unsupported language {self.language}: adding '{self.language_prompt}' to prompt instead of language code")
                specific_prompt = get_language_specific_prompt(self.language)
                prompt = f"{specific_prompt} {prompt}" # Combine prompts
            else:
                self.logger.info(f"Channel {channel}: Transcribing audio starting at sample {start_sample_offset} with OpenAI model {OPENAI_SPEECH_MODEL}, language {openai_lang}")

            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

            # Use aiohttp session within the async function
            async with aiohttp.ClientSession(headers=headers) as session:
                with open(file_path, 'rb') as audio_file:
                    form_data = aiohttp.FormData()
                    form_data.add_field('file',
                                       audio_file,
                                       filename=os.path.basename(file_path),
                                       content_type='audio/wav')
                    form_data.add_field('model', OPENAI_SPEECH_MODEL)
                    # Use 'json' for structured response including logprobs if needed
                    form_data.add_field('response_format', 'json')
                    # Request logprobs for confidence calculation
                    form_data.add_field('include[]', 'logprobs')
                    # Temperature 0 for deterministic output
                    form_data.add_field('temperature', '0')
                    form_data.add_field('prompt', prompt)

                    if openai_lang and not self.is_unsupported_language:
                        form_data.add_field('language', openai_lang)

                    # NOTE: OpenAI REST API transcription does NOT support 'stream=True'
                    # for real-time low-latency streaming like Google's gRPC.
                    # It processes the whole file then returns. We simulate streaming VAD externally.
                    # Also, word timestamps require whisper-1 and verbose_json.
                    # We calculate confidence from logprobs but cannot get accurate word timestamps here.

                    full_transcript = ""
                    tokens_with_confidence = []
                    avg_confidence = 0.9 # Default
                    final_logprobs = None

                    try:
                        async with session.post(url, data=form_data, timeout=30) as response:
                            if response.status == 200:
                                response_json = await response.json()
                                full_transcript = response_json.get("text", "")
                                final_logprobs = response_json.get("logprobs") # Get logprobs if available

                                if final_logprobs:
                                    confidence_sum = 0
                                    confidence_count = 0
                                    low_confidence_tokens = []
                                    for logprob_info in final_logprobs:
                                        token = logprob_info.get('token', '')
                                        token_logprob = logprob_info.get('logprob', 0)
                                        # Clamp confidence slightly below 1.0
                                        token_confidence = min(0.999, math.exp(token_logprob))

                                        tokens_with_confidence.append((token, token_confidence))

                                        if token_confidence < self.token_confidence_threshold:
                                            low_confidence_tokens.append(token)

                                        confidence_sum += token_confidence
                                        confidence_count += 1

                                    if confidence_count > 0:
                                        avg_confidence = confidence_sum / confidence_count

                                    if low_confidence_tokens:
                                         self.logger.debug(f"Channel {channel}: Low confidence tokens (<{self.token_confidence_threshold}): {', '.join(low_confidence_tokens)}")

                                selfself.logger.debug(f"Channel {channel}: OpenAI response received. Transcript: '{full_transcript}', Avg Confidence: {avg_confidence:.4f}")

                                audio_duration_samples = (os.path.getsize(file_path) - 44) // self.bytes_per_sample # Estimate from file size minus header
                                return self._create_response_object(
                                    full_transcript,
                                    avg_confidence,
                                    start_sample_offset,
                                    audio_duration_samples,
                                    channel,
                                    tokens_with_confidence
                                )
                            else:
                                error_text = await response.text()
                                self.logger.error(f"Channel {channel}: OpenAI API error: {response.status} - {error_text}")
                                if "language" in error_text.lower():
                                    self.logger.error(f"Language-related error? Used language: {self.language}, OpenAI language: {openai_lang}, Is unsupported: {self.is_unsupported_language}")
                                # Return an exception to be handled upstream
                                return Exception(f"OpenAI API Error {response.status}: {error_text}")

                    except asyncio.TimeoutError:
                        self.logger.warning(f"Channel {channel}: OpenAI API request timeout.")
                        return Exception("OpenAI API Timeout")
                    except aiohttp.ClientError as e:
                         self.logger.error(f"Channel {channel}: aiohttp client error: {e}")
                         return Exception(f"HTTP Client Error: {e}")
                    except Exception as e:
                        self.logger.error(f"Channel {channel}: Error during OpenAI API call: {e}", exc_info=True)
                        return Exception(f"API Call Error: {e}")

        except Exception as e:
            self.logger.error(f"Channel {channel}: Error in _stream_transcribe_audio: {e}", exc_info=True)
            return Exception(f"Transcription Error: {e}")

    def _create_response_object(self, transcript, confidence, start_sample_offset, audio_duration_samples, channel=0, tokens_with_confidence=None):
        if not transcript:
            self.logger.debug(f"Channel {channel}: Empty transcript provided to create_response_object.")
            return None

        mock_result = MockResult()

        # Calculate overall utterance offset and duration based on the processed chunk
        utterance_start_sec = start_sample_offset / self.sample_rate
        utterance_duration_sec = audio_duration_samples / self.sample_rate

        # Ensure non-negative values
        utterance_start_sec = max(0, utterance_start_sec)
        utterance_duration_sec = max(0, utterance_duration_sec)

        offset_str = f"PT{utterance_start_sec:.3f}S"
        duration_str = f"PT{utterance_duration_sec:.3f}S"

        words_list = transcript.split()
        words_for_genesys = []

        if words_list:
            # OpenAI API used here doesn't provide word timestamps.
            # We will assign approximate timings and confidence based on logprobs.
            self.logger.warning(f"Channel {channel}: OpenAI model '{OPENAI_SPEECH_MODEL}' with 'response_format=json' does not provide word timestamps. Using approximate values based on overall duration and token confidence.")

            has_token_confidence = tokens_with_confidence and len(tokens_with_confidence) > 0
            word_confidences = []

            if has_token_confidence:
                # Simple heuristic: Distribute token confidences over words
                num_tokens = len(tokens_with_confidence)
                num_words = len(words_list)
                current_token_idx = 0
                for i in range(num_words):
                    # Estimate tokens per word
                    tokens_for_this_word = max(1, round((num_tokens - current_token_idx) / (num_words - i)))
                    end_idx = min(num_tokens, current_token_idx + tokens_for_this_word)

                    word_token_confs = [conf for _, conf in tokens_with_confidence[current_token_idx:end_idx]]

                    if word_token_confs:
                        # Use average confidence of mapped tokens for the word
                        word_confidences.append(sum(word_token_confs) / len(word_token_confs))
                    else:
                        word_confidences.append(confidence) # Fallback to overall confidence

                    current_token_idx = end_idx
            else:
                # No token confidence, use overall confidence for all words
                word_confidences = [confidence] * len(words_list)

            # Ensure length matches
            if len(word_confidences) != len(words_list):
                 self.logger.warning(f"Channel {channel}: Word confidence mapping resulted in mismatch ({len(word_confidences)} vs {len(words_list)} words). Using overall confidence.")
                 word_confidences = [confidence] * len(words_list)

            # Approximate word timings by dividing the total duration
            avg_word_duration_sec = utterance_duration_sec / len(words_list) if len(words_list) > 0 else utterance_duration_sec

            for i, word_text in enumerate(words_list):
                # Approximate start and end times
                word_start_time_sec = utterance_start_sec + (i * avg_word_duration_sec)
                word_end_time_sec = utterance_start_sec + ((i + 1) * avg_word_duration_sec)

                # Clamp duration to be non-negative
                word_duration_sec = max(0, word_end_time_sec - word_start_time_sec)

                # Use the mapped word confidence
                word_confidence = word_confidences[i]

                # Create Genesys-compatible Word object (using timedelta for internal consistency)
                genesys_word = Word(
                    word=word_text,
                    start_offset=timedelta(seconds=word_start_time_sec),
                    end_offset=timedelta(seconds=word_end_time_sec), # End offset relative to start of audio
                    confidence=word_confidence
                )
                words_for_genesys.append(genesys_word)

            # Log a sample of word confidences if available
            if has_token_confidence and len(words_list) > 0:
                sample_size = min(5, len(words_list))
                sample_words = [(words_list[i], f"{word_confidences[i]:.3f}") for i in range(sample_size)]
                self.logger.debug(f"Channel {channel}: Sample approx word confidences: {sample_words}")

        alternative = Alternative(
            transcript=transcript,
            confidence=confidence, # Overall utterance confidence
            words=words_for_genesys # List of Word objects with approximate timings
        )

        result = Result(
            alternatives=[alternative],
            is_final=True # OpenAI REST API only gives final results per chunk
        )

        mock_result.results = [result]
        return mock_result


    def feed_audio(self, audio_stream: bytes, channel: int):
        if not self.running.is_set():
            self.logger.warning(f"Channel {channel}: Received audio while not running, discarding.")
            return
        if not audio_stream or channel >= self.channels:
            self.logger.debug(f"Channel {channel}: Empty audio stream or invalid channel index.")
            return

        try:
            pcm16_data = audioop.ulaw2lin(audio_stream, 2)
            num_samples = len(pcm16_data) // self.bytes_per_sample

            # Calculate the starting sample offset *before* adding these samples
            chunk_start_sample_offset = self.valid_samples_fed[channel]

            # Update the total valid samples fed *after* getting the start offset
            self.valid_samples_fed[channel] += num_samples

            # Put the audio data and its starting offset into the queue
            self.audio_queues[channel].put((pcm16_data, chunk_start_sample_offset))
            self.logger.debug(f"Channel {channel}: Fed {len(pcm16_data)} bytes (starting at sample {chunk_start_sample_offset}) to audio queue.")

        except audioop.error as e:
             self.logger.error(f"Channel {channel}: audioop error converting PCMU to PCM16: {e}")
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error feeding audio: {e}")

    def get_response(self, channel: int):
        if channel >= self.channels:
            self.logger.warning(f"Invalid channel index {channel} requested in get_response.")
            return None
        try:
            # Returns tuple: (response_object_or_exception, start_sample_offset)
            response_data = self.response_queues[channel].get_nowait()
            self.logger.debug(f"Channel {channel}: Retrieved response from queue (started at sample {response_data[1]})")
            return response_data
        except queue.Empty:
            return None
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error getting response from queue: {e}")
             return None

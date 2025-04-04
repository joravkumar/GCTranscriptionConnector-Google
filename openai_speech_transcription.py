import asyncio
import audioop
import os
import json
import time
import threading
import queue
import tempfile
from datetime import timedelta, timezone
import aiohttp
import logging
from collections import deque
import math
import re
import wave # Added for WAV writing

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

# Known spurious transcription artifacts that should be filtered
KNOWN_ARTIFACTS = [
    "context:",
    "ring",
    "context",
    "begin",
    "beep",
    "[beep]",
    "[ring]"
]

# Regular expressions for more complex pattern matching
ARTIFACT_PATTERNS = [
    r"^\s*context:?\s*",  # Matches "context:" or "context" at the start
    r"^\s*ring\s*",       # Matches "ring" at the start
    r"^\s*\[?beep\]?\s*", # Matches "beep" or "[beep]" at the start
    r"^\s*\[?ring\]?\s*"  # Matches "ring" or "[ring]" at the start
]


# --- Mock Google API Response Classes ---
# These classes mimic the structure expected by audio_hook_server.py

class MockResult:
    def __init__(self):
        self.results = []

class Result:
    def __init__(self, alternatives=None, is_final=True):
        self.alternatives = alternatives or []
        self.is_final = is_final # OpenAI results are always final in this implementation

class Alternative:
    def __init__(self, transcript="", confidence=0.9, words=None, is_synthetic_timing=False):
        self.transcript = transcript
        self.confidence = confidence # Overall confidence for the alternative
        self.words = words or [] # List of Word objects
        self.is_synthetic_timing = is_synthetic_timing # Flag for OpenAI

class Word:
    def __init__(self, word="", start_offset=None, end_offset=None, confidence=0.9):
        self.word = word
        # Timestamps relative to the start of the audio stream
        self.start_offset = start_offset if start_offset is not None else timedelta(seconds=0)
        self.end_offset = end_offset if end_offset is not None else timedelta(seconds=0)
        self.confidence = confidence # Confidence for this specific word


# --- Main Streaming Transcription Class ---

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.logger = logger
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
        # Queue now stores tuples: (pcm16_data, current_total_samples)
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True

        # Audio buffering for utterance detection
        # Stores tuples: (pcm16_data, sample_time)
        self.accumulated_audio_info = [[] for _ in range(channels)]
        self.last_process_time = [time.time() for _ in range(channels)]

        # VAD parameters
        self.vad_threshold = 200
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]
        self.silence_threshold_frames = 8 # 800ms of silence

        # Track last utterance to prevent duplicates
        self.last_transcripts = ["" for _ in range(channels)]

        # Initial audio discarding
        self.initial_frames_processed = [0 for _ in range(channels)]
        self.skip_initial_frames = 5 # Skip first 5 frames (adjust as needed)

        # Minimum confidence threshold for tokens
        self.token_confidence_threshold = 0.2

    def start_streaming(self):
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self.streaming_recognize_thread, args=(channel,)
            )
            self.streaming_threads[channel].daemon = True
            self.streaming_threads[channel].start()

    def stop_streaming(self):
        self.running = False
        for channel in range(self.channels):
            try:
                self.audio_queues[channel].put(None, block=False)
            except queue.Full:
                pass # Ignore if queue is full during shutdown
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                self.streaming_threads[channel].join(timeout=1.0)

    def streaming_recognize_thread(self, channel):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            frame_duration_ms = 100 # Process audio in 100ms chunks for VAD

            while self.running:
                try:
                    # Get audio data and its corresponding sample time
                    audio_info = self.audio_queues[channel].get(timeout=0.1)
                    if audio_info is None:
                        # Process any remaining audio before shutting down
                        if self.accumulated_audio_info[channel]:
                            self._process_accumulated_audio(channel, loop)
                        break

                    pcm16_chunk, sample_time = audio_info

                    # Skip initial frames
                    if self.initial_frames_processed[channel] < self.skip_initial_frames:
                        self.initial_frames_processed[channel] += 1
                        self.logger.debug(f"Channel {channel}: Skipping initial frame {self.initial_frames_processed[channel]}")
                        continue

                    # Add chunk to buffer with its sample time
                    self.accumulated_audio_info[channel].append((pcm16_chunk, sample_time))

                    # --- VAD Logic ---
                    rms = audioop.rms(pcm16_chunk, 2) # 2 = sample width in bytes for PCM16
                    is_current_speech = rms > self.vad_threshold

                    if is_current_speech:
                        self.silence_frames[channel] = 0
                        self.speech_frames[channel] += 1
                        if not self.is_speech[channel] and self.speech_frames[channel] >= 2: # Need 2 consecutive speech frames
                            self.is_speech[channel] = True
                            self.logger.debug(f"Channel {channel}: Speech detected (RMS: {rms})")
                    else: # Silence
                        self.speech_frames[channel] = 0
                        if self.is_speech[channel]: # If silence follows speech
                            self.silence_frames[channel] += 1
                            if self.silence_frames[channel] >= self.silence_threshold_frames:
                                self.is_speech[channel] = False
                                self.logger.debug(f"Channel {channel}: End of speech detected after {self.silence_threshold_frames * frame_duration_ms}ms silence")
                                self._process_accumulated_audio(channel, loop)

                    # --- Timeout/Buffer Logic ---
                    current_time = time.time()
                    accumulated_duration_ms = sum(len(d[0]) // 16 for d in self.accumulated_audio_info[channel]) # 8000Hz * 2 bytes/sample = 16 bytes/ms

                    # Process if buffer gets too long (e.g., > 15 seconds)
                    if accumulated_duration_ms > 15000:
                         self.logger.warning(f"Channel {channel}: Audio buffer exceeded 15s ({accumulated_duration_ms}ms), forcing processing.")
                         self._process_accumulated_audio(channel, loop)
                    # Process periodically even during speech (e.g., every 5 seconds)
                    elif self.is_speech[channel] and current_time - self.last_process_time[channel] > 5.0:
                         self.logger.debug(f"Channel {channel}: Processing accumulated audio during speech (timeout)")
                         self._process_accumulated_audio(channel, loop)


                except queue.Empty:
                    # Check if there's lingering audio to process on inactivity
                    if not self.is_speech[channel] and self.accumulated_audio_info[channel] and time.time() - self.last_process_time[channel] > 1.0:
                         self.logger.debug(f"Channel {channel}: Processing remaining audio due to inactivity")
                         self._process_accumulated_audio(channel, loop)
                    continue # Normal case, just wait longer
                except Exception as e:
                    self.logger.error(f"Error in streaming thread loop for channel {channel}: {type(e).__name__} - {e}", exc_info=False)

            loop.close()
            self.logger.info(f"Streaming thread for channel {channel} finished.")

        except Exception as e:
            self.logger.error(f"Fatal error setting up streaming thread for channel {channel}: {type(e).__name__} - {str(e)}")
            self.response_queues[channel].put(e) # Signal error to main thread

    def _process_accumulated_audio(self, channel, loop):
        """Process accumulated audio using OpenAI streaming transcription"""
        if not self.accumulated_audio_info[channel]:
            return

        # Combine audio chunks and get start sample time
        audio_data = b"".join([chunk[0] for chunk in self.accumulated_audio_info[channel]])
        start_sample_time = self.accumulated_audio_info[channel][0][1] # Sample time of the first chunk
        actual_duration_samples = len(audio_data) // 2 # 2 bytes per sample for PCM16

        # Clear buffer immediately
        current_accumulated_info = self.accumulated_audio_info[channel][:]
        self.accumulated_audio_info[channel] = []
        self.last_process_time[channel] = time.time() # Update last process time

        if actual_duration_samples < 800: # Less than 100ms
            self.logger.debug(f"Channel {channel}: Accumulated audio too short ({actual_duration_samples*1000/8000:.1f} ms), skipping")
            return

        try:
            rms = audioop.rms(audio_data, 2)
            if rms < self.vad_threshold * 0.5: # Check if mostly silence
                self.logger.debug(f"Channel {channel}: Accumulated audio mostly silence (RMS: {rms}), skipping")
                return

            # Use a temporary file for the OpenAI API call
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                try:
                    self._write_wav_file(temp_wav, audio_data)
                    temp_wav.close() # Close file before passing path to async function

                    # Run the transcription in the event loop
                    result = loop.run_until_complete(
                        self.stream_transcribe_audio(temp_wav_path, channel, start_sample_time, actual_duration_samples)
                    )

                    if result and not isinstance(result, Exception):
                        transcript_text = ""
                        if result.results and result.results[0].alternatives:
                            transcript_text = result.results[0].alternatives[0].transcript

                        # Apply filtering AFTER getting the full transcript
                        filtered_transcript = self._filter_spurious_artifacts(transcript_text)

                        # Update the transcript in the result object if filtering changed it
                        if filtered_transcript != transcript_text and result.results and result.results[0].alternatives:
                             result.results[0].alternatives[0].transcript = filtered_transcript
                             # Re-generate words if transcript changed significantly? Maybe not needed if just filtering artifacts.

                        # Only send if non-empty and different from last transcript
                        if filtered_transcript and filtered_transcript != self.last_transcripts[channel]:
                            self.response_queues[channel].put(result)
                            self.last_transcripts[channel] = filtered_transcript
                        elif not filtered_transcript:
                             self.logger.debug(f"Channel {channel}: Transcript filtered down to empty string.")
                        else:
                             self.logger.debug(f"Channel {channel}: Skipping duplicate transcript: '{filtered_transcript}'")

                except Exception as api_err:
                    self.logger.error(f"Channel {channel}: Error during OpenAI API call or processing: {type(api_err).__name__} - {api_err}", exc_info=False)
                finally:
                    # Ensure temporary file is deleted
                    if os.path.exists(temp_wav_path):
                        try:
                            os.unlink(temp_wav_path)
                        except OSError as del_err:
                            self.logger.warning(f"Channel {channel}: Could not delete temp WAV file {temp_wav_path}: {del_err}")

        except Exception as e:
            self.logger.error(f"Channel {channel}: Error processing accumulated audio: {type(e).__name__} - {e}", exc_info=False)
            # Ensure buffer is cleared even on error
            self.accumulated_audio_info[channel] = []


    def _filter_spurious_artifacts(self, transcript):
        """Filter out known spurious artifacts from transcripts"""
        if not transcript:
            return ""

        original_transcript = transcript
        # Apply regex patterns first
        for pattern in ARTIFACT_PATTERNS:
            transcript = re.sub(pattern, "", transcript).strip()

        # Check simple known artifacts (case-insensitive check)
        lower_transcript = transcript.lower()
        for artifact in KNOWN_ARTIFACTS:
            if lower_transcript == artifact:
                transcript = ""
                break
            if lower_transcript.startswith(artifact + " "):
                 # Find the index to remove based on case-insensitive match
                 match_len = len(artifact)
                 if original_transcript[:match_len].lower() == artifact:
                      transcript = original_transcript[match_len:].lstrip()
                      original_transcript = transcript # Update for next iteration
                      lower_transcript = transcript.lower()


        # Remove duplicated spaces and trim
        filtered = re.sub(r'\s+', ' ', transcript).strip()
        if filtered != original_transcript:
             self.logger.debug(f"Filtered artifacts: '{original_transcript}' -> '{filtered}'")

        return filtered

    def _write_wav_file(self, temp_wav_file_obj, audio_data):
        """Write PCM16 data to a WAV file object"""
        sample_rate = 8000
        channels = 1
        sample_width = 2 # PCM16 has 2 bytes per sample

        with wave.open(temp_wav_file_obj, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)


    async def stream_transcribe_audio(self, file_path, channel, start_sample_time, actual_duration_samples):
        """Stream transcribe audio using OpenAI's transcription API"""
        try:
            openai_lang = self.openai_language
            api_start_time = time.monotonic()

            if self.is_unsupported_language:
                self.logger.info(f"Channel {channel}: Using special handling for unsupported language {self.language}: adding '{self.language_prompt}' to prompt")
            else:
                self.logger.info(f"Channel {channel}: Transcribing audio with OpenAI model {OPENAI_SPEECH_MODEL}, language={openai_lang or 'auto'}")

            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

            with open(file_path, 'rb') as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file',
                                    audio_file,
                                    filename=os.path.basename(file_path),
                                    content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'json') # Use JSON to get logprobs if needed
                form_data.add_field('stream', 'true') # Enable streaming
                form_data.add_field('include[]', 'logprobs') # Request logprobs for confidence
                form_data.add_field('temperature', '0') # Deterministic output

                # --- Prompting ---
                base_prompt = "This is a customer service call. The customer may be discussing problems with services or products."
                # Add instruction to ignore common noises
                noise_instruction = "Ignore initial beeps, rings, and system sounds."

                if self.is_unsupported_language:
                    lang_prompt = get_language_specific_prompt(self.language) # Get native prompt
                    full_prompt = f"{lang_prompt} {noise_instruction}"
                    self.logger.info(f"Channel {channel}: Using language-specific prompt: '{full_prompt}'")
                    form_data.add_field('prompt', full_prompt)
                    # Do NOT send 'language' parameter for unsupported languages
                else:
                    full_prompt = f"{base_prompt} {noise_instruction}"
                    form_data.add_field('prompt', full_prompt)
                    if openai_lang: # Only add language if supported and detected
                        form_data.add_field('language', openai_lang)

                # --- API Call ---
                full_transcript = ""
                confidence_sum = 0
                confidence_count = 0
                low_confidence_tokens = []

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data, timeout=aiohttp.ClientTimeout(total=30)) as response:
                            if response.status == 200:
                                buffer = ""
                                async for line in response.content:
                                    line = line.decode('utf-8').strip()
                                    if line.startswith('data: '):
                                        event_data = line[6:]
                                        if event_data == '[DONE]':
                                            break
                                        try:
                                            event_json = json.loads(event_data)
                                            event_type = event_json.get('type')

                                            if event_type == 'transcript.text.delta':
                                                delta = event_json.get('delta', '')
                                                # Note: We don't accumulate delta here, wait for 'done' event for full text
                                                if 'logprobs' in event_json:
                                                    for logprob in event_json['logprobs']:
                                                        token = logprob.get('token', '')
                                                        token_logprob = logprob.get('logprob', -float('inf')) # Use -inf if missing
                                                        token_confidence = min(0.999, math.exp(token_logprob)) # Cap confidence slightly below 1.0

                                                        if token_confidence < self.token_confidence_threshold:
                                                            low_confidence_tokens.append(f"'{token}'({token_confidence:.3f})")
                                                        confidence_sum += token_confidence
                                                        confidence_count += 1

                                            elif event_type == 'transcript.text.done':
                                                full_transcript = event_json.get('text', '')
                                                # Process final logprobs if present
                                                if 'logprobs' in event_json and event_json['logprobs']:
                                                     # Reset confidence calculation based on final logprobs
                                                     confidence_sum = 0
                                                     confidence_count = 0
                                                     low_confidence_tokens = []
                                                     for logprob in event_json['logprobs']:
                                                        token = logprob.get('token', '')
                                                        token_logprob = logprob.get('logprob', -float('inf'))
                                                        token_confidence = min(0.999, math.exp(token_logprob))
                                                        if token_confidence < self.token_confidence_threshold:
                                                            low_confidence_tokens.append(f"'{token}'({token_confidence:.3f})")
                                                        confidence_sum += token_confidence
                                                        confidence_count += 1
                                                break # Exit loop once done event is received

                                        except json.JSONDecodeError:
                                            self.logger.warning(f"Channel {channel}: Failed to parse streaming event: {event_data}")
                                            continue
                            else:
                                error_text = await response.text()
                                self.logger.error(f"Channel {channel}: OpenAI API error: {response.status} - {error_text}")
                                if "language" in error_text.lower():
                                    self.logger.error(f"Language-related error detail: Input lang='{self.language}', OpenAI lang='{openai_lang}', Unsupported={self.is_unsupported_language}")
                                return None # Indicate API error

                except asyncio.TimeoutError:
                    self.logger.warning(f"Channel {channel}: OpenAI API request timed out after 30s")
                    return None
                except aiohttp.ClientError as http_err:
                     self.logger.error(f"Channel {channel}: OpenAI API connection error: {http_err}")
                     return None
                except Exception as stream_err:
                    self.logger.error(f"Channel {channel}: Error processing OpenAI stream: {type(stream_err).__name__} - {stream_err}")
                    return None

                # --- Process Final Result ---
                api_duration = time.monotonic() - api_start_time
                avg_confidence = (confidence_sum / confidence_count) if confidence_count > 0 else 0.5 # Default to 0.5 if no tokens

                # Log details about the result
                self.logger.info(f"Channel {channel}: OpenAI transcription complete ({api_duration:.2f}s). Confidence: {avg_confidence:.3f}. Length: {len(full_transcript)} chars.")
                if low_confidence_tokens:
                     self.logger.debug(f"Channel {channel}: Low confidence tokens detected: {', '.join(low_confidence_tokens)}")

                # Create the response object mimicking Google's structure
                return self.create_response_object(full_transcript, avg_confidence, start_sample_time, actual_duration_samples)

        except Exception as e:
            self.logger.error(f"Channel {channel}: Error in stream_transcribe_audio: {type(e).__name__} - {str(e)}", exc_info=False)
            return None # Indicate failure

    def create_response_object(self, transcript, confidence, start_sample_time, actual_duration_samples):
        """Create a response object compatible with Google API format, using synthetic timing"""
        if not transcript:
            self.logger.debug("create_response_object: Received empty transcript, returning None.")
            return None

        mock_result = MockResult()
        words = []
        text_words = transcript.split()

        if text_words:
            actual_duration_seconds = actual_duration_samples / 8000.0
            start_time_seconds = start_sample_time / 8000.0
            avg_word_duration = actual_duration_seconds / len(text_words) if len(text_words) > 0 else 0.1

            for i, word_text in enumerate(text_words):
                # Calculate synthetic start and end times relative to the stream beginning
                word_start_offset_seconds = start_time_seconds + (i * avg_word_duration)
                word_end_offset_seconds = word_start_offset_seconds + avg_word_duration

                word = Word(
                    word=word_text,
                    # Store offsets as timedelta relative to stream start
                    start_offset=timedelta(seconds=word_start_offset_seconds),
                    end_offset=timedelta(seconds=word_end_offset_seconds),
                    confidence=confidence # Use overall confidence for each synthetic word
                )
                words.append(word)
        else:
             # Handle case where transcript is non-empty but has no spaces (e.g., single word)
             actual_duration_seconds = actual_duration_samples / 8000.0
             start_time_seconds = start_sample_time / 8000.0
             word = Word(
                 word=transcript,
                 start_offset=timedelta(seconds=start_time_seconds),
                 end_offset=timedelta(seconds=start_time_seconds + actual_duration_seconds),
                 confidence=confidence
             )
             words.append(word)


        alternative = Alternative(
            transcript=transcript,
            confidence=confidence, # Overall alternative confidence
            words=words,
            is_synthetic_timing=True # Mark timing as synthetic
        )

        result = Result(
            alternatives=[alternative],
            is_final=True # OpenAI results are treated as final here
        )

        mock_result.results = [result]
        # self.logger.debug(f"Created response object: {result.alternatives[0].transcript}, Start: {words[0].start_offset if words else 'N/A'}, End: {words[-1].end_offset if words else 'N/A'}")
        return mock_result

    def feed_audio(self, audio_stream: bytes, channel: int, total_samples: int):
        """Feed audio into the pipeline for the given channel with its sample time"""
        if not audio_stream or channel >= self.channels:
            return

        try:
            # Convert PCMU (u-law) to PCM16 for processing
            pcm16_data = audioop.ulaw2lin(audio_stream, 2)
            # Put the data and the current total_samples (as sample_time) into the queue
            self.audio_queues[channel].put((pcm16_data, total_samples), block=False)
        except queue.Full:
             self.logger.warning(f"Channel {channel}: Audio queue full, dropping frame.")
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error in feed_audio: {type(e).__name__} - {e}")


    def get_response(self, channel: int):
        """Get the next transcription response for the given channel"""
        if channel >= self.channels:
            return None
        try:
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error getting response from queue: {type(e).__name__} - {e}")
             return None

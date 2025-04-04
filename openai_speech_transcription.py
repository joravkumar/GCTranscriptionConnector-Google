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
from collections import deque
import math
import re
import wave # Use wave module for proper WAV writing

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

# --- Data Structures Mimicking Google API Response ---
class MockResult:
    """ Mimics the structure containing transcription results. """
    def __init__(self):
        self.results = []
        # Add attributes to store timing info for the utterance
        self.start_sample_count = 0
        self.duration_samples = 0

class Result:
    """ Mimics a single transcription result (usually one per utterance). """
    def __init__(self, alternatives=None, is_final=True):
        self.alternatives = alternatives if alternatives is not None else []
        self.is_final = is_final # OpenAI results are always final per chunk

class Alternative:
    """ Mimics a single transcription hypothesis. """
    def __init__(self, transcript="", confidence=0.9, words=None):
        self.transcript = transcript
        self.confidence = confidence # Overall confidence for the alternative
        self.words = words if words is not None else [] # List of Word objects

class Word:
    """ Mimics word-level information (timing is synthetic for OpenAI). """
    def __init__(self, word="", start_offset=None, end_offset=None, confidence=0.9):
        self.word = word
        # Store offsets as timedelta objects for consistency
        self.start_offset = start_offset if start_offset is not None else timedelta(seconds=0)
        self.end_offset = end_offset if end_offset is not None else timedelta(seconds=0)
        self.confidence = confidence

# --- Main Streaming Transcription Class ---
class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.logger = logger
        self.language = normalize_language_code(language)
        self.openai_language = get_openai_language_code(self.language)
        self.is_unsupported_language = is_openai_unsupported_language(self.language)

        if self.is_unsupported_language:
            self.language_prompt = get_language_name_for_prompt(self.language)
            self.logger.info(f"Unsupported OpenAI language '{self.language}'. Using prompt: '{self.language_prompt}'")
        else:
            self.language_prompt = None
            self.logger.info(f"Initialized with language={self.language}, OpenAI code={self.openai_language}")

        self.channels = channels # Number of concurrent streams to manage (usually 1 per instance)
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True

        # Audio buffering and VAD state per channel
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        self.accumulated_start_sample = [0 for _ in range(channels)] # Track start sample for accumulated audio
        self.is_first_chunk = [True for _ in range(channels)] # Track if it's the first chunk for a channel

        self.vad_threshold = 150 # Adjusted VAD threshold
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]
        self.silence_threshold_frames = 8 # Frames of silence (~800ms) to detect end of utterance
        self.last_process_time = [time.time() for _ in range(channels)]

        # Track last transcript to prevent duplicates
        self.last_transcripts = ["" for _ in range(channels)]

        # Initial audio frame skipping
        self.initial_frames_processed = [0 for _ in range(channels)]
        self.skip_initial_frames = 5 # Skip first ~500ms

        # Confidence filtering
        self.token_confidence_threshold = 0.2 # Filter tokens below this confidence

        # Ensure API key is available
        if not OPENAI_API_KEY:
            self.logger.error("OPENAI_API_KEY environment variable not set!")
            raise ValueError("OpenAI API key not configured.")

    def start_streaming(self):
        self.logger.info(f"Starting {self.channels} streaming thread(s)...")
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self.streaming_recognize_thread, args=(channel,)
            )
            self.streaming_threads[channel].daemon = True # Allow program exit even if threads are running
            self.streaming_threads[channel].start()
            self.logger.info(f"Streaming thread started for channel {channel}")

    def stop_streaming(self):
        self.logger.info("Stopping streaming threads...")
        self.running = False
        for channel in range(self.channels):
            # Signal threads to stop by putting None in queue
            self.audio_queues[channel].put(None)
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                self.logger.debug(f"Joining streaming thread for channel {channel}...")
                self.streaming_threads[channel].join(timeout=2.0) # Wait max 2 seconds
                if self.streaming_threads[channel].is_alive():
                    self.logger.warning(f"Streaming thread for channel {channel} did not exit cleanly.")
        self.logger.info("Streaming threads stopped.")

    def streaming_recognize_thread(self, channel):
        """ Background thread to process audio queue and perform VAD/transcription. """
        self.logger.info(f"Recognition thread started for channel {channel}")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            frame_duration_ms = 100 # Process audio in 100ms chunks for VAD
            frame_size_bytes = 1600 # 8000 Hz * 1 channel * 2 bytes/sample * 0.1s

            while self.running:
                try:
                    # Get audio data and associated start sample count
                    queue_item = self.audio_queues[channel].get(timeout=0.1)
                    if queue_item is None:
                        self.logger.info(f"Received None signal, processing remaining audio for channel {channel} and exiting thread.")
                        if len(self.accumulated_audio[channel]) > 0:
                             self._process_accumulated_audio(channel, loop)
                        break # Exit loop

                    pcm16_data, start_sample_for_chunk = queue_item

                    # --- Initial Frame Skipping ---
                    if self.initial_frames_processed[channel] < self.skip_initial_frames:
                        self.initial_frames_processed[channel] += 1
                        self.logger.debug(f"Channel {channel}: Skipping initial frame {self.initial_frames_processed[channel]}/{self.skip_initial_frames}")
                        continue # Skip this chunk

                    # --- VAD and Buffering ---
                    # Calculate RMS energy for VAD
                    try:
                        rms = audioop.rms(pcm16_data, 2) # 2 = sample width in bytes for PCM16
                    except audioop.error:
                        self.logger.warning(f"Channel {channel}: Could not calculate RMS for chunk (length {len(pcm16_data)}). Skipping VAD for this chunk.")
                        rms = self.vad_threshold # Assume speech if RMS fails

                    is_current_speech = rms > self.vad_threshold
                    self.logger.debug(f"Channel {channel}: Chunk RMS={rms}, VAD Threshold={self.vad_threshold}, IsSpeech={is_current_speech}")

                    # Append audio chunk to buffer regardless of VAD state initially
                    if self.is_first_chunk[channel]:
                         self.accumulated_start_sample[channel] = start_sample_for_chunk
                         self.is_first_chunk[channel] = False

                    self.accumulated_audio[channel].extend(pcm16_data)

                    # VAD State Machine
                    if is_current_speech:
                        self.silence_frames[channel] = 0 # Reset silence counter
                        if not self.is_speech[channel]:
                            # Transition to speech state
                            self.speech_frames[channel] += 1
                            if self.speech_frames[channel] >= 2: # Require 2 consecutive speech frames
                                self.is_speech[channel] = True
                                self.logger.debug(f"Channel {channel}: Speech started.")
                        else:
                             # Continue speech state
                             self.speech_frames[channel] += 1
                    else: # Silence frame
                        self.speech_frames[channel] = 0 # Reset speech counter
                        if self.is_speech[channel]:
                            # We were speaking, now silence - increment silence counter
                            self.silence_frames[channel] += 1
                            if self.silence_frames[channel] >= self.silence_threshold_frames:
                                # End of utterance detected
                                self.logger.debug(f"Channel {channel}: End of speech detected ({self.silence_frames[channel]} silence frames). Processing utterance.")
                                self._process_accumulated_audio(channel, loop)
                                self.is_speech[channel] = False # Reset state
                                self.silence_frames[channel] = 0
                        # else: # Silence continues or starts - do nothing until speech detected

                    # --- Timeout Processing ---
                    # Process buffer if it gets too long or if too much time passed since last processing
                    current_time = time.time()
                    buffer_duration_sec = len(self.accumulated_audio[channel]) / 16000.0
                    time_since_last_process = current_time - self.last_process_time[channel]

                    # Process if buffer > 15s OR (buffer > 3s AND > 5s passed since last processing)
                    if buffer_duration_sec > 15.0 or (buffer_duration_sec > 3.0 and time_since_last_process > 5.0):
                         if self.is_speech[channel]: # Only process if VAD indicates speech
                            self.logger.warning(f"Channel {channel}: Processing accumulated audio due to timeout/length (Duration: {buffer_duration_sec:.1f}s, SinceLast: {time_since_last_process:.1f}s)")
                            self._process_accumulated_audio(channel, loop)
                         else:
                            # If timeout happens during silence, just clear the buffer
                            self.logger.debug(f"Channel {channel}: Timeout/length reached during silence. Clearing buffer.")
                            self.accumulated_audio[channel] = bytearray()
                            self.is_first_chunk[channel] = True # Reset for next chunk


                except queue.Empty:
                    # Queue is empty, check if we need to process buffer due to inactivity timeout
                    if self.is_speech[channel] and time.time() - self.last_process_time[channel] > 2.0:
                         self.logger.debug(f"Channel {channel}: Processing buffer due to inactivity timeout while in speech state.")
                         self._process_accumulated_audio(channel, loop)
                         self.is_speech[channel] = False # Assume end of utterance on timeout
                         self.silence_frames[channel] = 0
                    continue # Continue waiting for audio

                except Exception as e:
                    self.logger.error(f"Error in recognition thread loop for channel {channel}: {e}", exc_info=True)
                    # Avoid tight error loops
                    time.sleep(0.1)

            loop.close()
            self.logger.info(f"Recognition thread finished for channel {channel}")

        except Exception as e:
            self.logger.error(f"Fatal error initializing recognition thread for channel {channel}: {e}", exc_info=True)
            self.response_queues[channel].put(e) # Signal error to main thread


    def _process_accumulated_audio(self, channel, loop):
        """ Processes the buffered audio chunk for transcription. """
        if len(self.accumulated_audio[channel]) < 800: # Ignore very short chunks (< 50ms)
            self.logger.debug(f"Channel {channel}: Accumulated audio too short ({len(self.accumulated_audio[channel])} bytes). Skipping.")
            # Don't clear buffer here, wait for more audio or silence timeout
            return

        audio_data_to_process = bytes(self.accumulated_audio[channel])
        start_sample_for_utterance = self.accumulated_start_sample[channel]
        duration_samples_for_utterance = len(audio_data_to_process) // 2 # 2 bytes per sample for PCM16

        # Clear buffer *before* async call
        self.accumulated_audio[channel] = bytearray()
        self.is_first_chunk[channel] = True # Reset for next accumulation
        self.last_process_time[channel] = time.time() # Update last process time


        self.logger.info(f"Channel {channel}: Processing utterance. StartSample={start_sample_for_utterance}, DurationSamples={duration_samples_for_utterance} ({duration_samples_for_utterance/8000.0:.3f}s)")

        temp_wav_path = None
        try:
            # Check RMS again on the whole chunk to avoid sending pure silence
            rms = audioop.rms(audio_data_to_process, 2)
            if rms < self.vad_threshold * 0.5: # Stricter check for the whole utterance
                self.logger.info(f"Channel {channel}: Utterance below RMS threshold ({rms:.1f}). Skipping transcription.")
                return

            # Create a temporary WAV file for the API call
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                self._write_wav_file(temp_wav, audio_data_to_process)

            self.logger.debug(f"Channel {channel}: Wrote utterance to temporary WAV file: {temp_wav_path}")

            # Schedule the async transcription task in the provided event loop
            future = asyncio.run_coroutine_threadsafe(
                self.stream_transcribe_audio(temp_wav_path, channel, start_sample_for_utterance, duration_samples_for_utterance),
                loop
            )

            # Optionally wait for result if synchronous processing is needed,
            # but typically we let the main thread handle responses via get_response()
            # result = future.result(timeout=30) # Example of waiting, usually not done here

        except audioop.error as rms_err:
             self.logger.error(f"Channel {channel}: RMS calculation failed for utterance: {rms_err}")
        except Exception as e:
            self.logger.error(f"Error processing accumulated audio for channel {channel}: {e}", exc_info=True)
        finally:
            # Clean up temp file if path was assigned
            if temp_wav_path:
                try:
                    os.unlink(temp_wav_path)
                    self.logger.debug(f"Channel {channel}: Deleted temporary WAV file: {temp_wav_path}")
                except OSError as unlink_err:
                     # Log error but don't crash the thread
                     self.logger.error(f"Channel {channel}: Error deleting temporary WAV file {temp_wav_path}: {unlink_err}")


    def _filter_spurious_artifacts(self, transcript):
        """ Filters known artifacts from the beginning or whole transcript. """
        if not transcript:
            return ""

        original_transcript = transcript
        filtered = transcript.strip()

        # Apply regex patterns first for more complex matches at the start
        for pattern in ARTIFACT_PATTERNS:
            filtered = re.sub(pattern, "", filtered, count=1).strip() # Apply only once at the start

        # Check for exact match of the whole (stripped) transcript
        if filtered in KNOWN_ARTIFACTS:
             self.logger.debug(f"Filtered artifact (exact match): '{original_transcript}' -> ''")
             return ""

        # Check if the beginning matches known artifacts (case-insensitive)
        lower_filtered = filtered.lower()
        for artifact in KNOWN_ARTIFACTS:
            if lower_filtered.startswith(artifact + " "):
                 # Find the index to remove up to
                 end_index = len(artifact) + 1
                 filtered = filtered[end_index:].strip()
                 self.logger.debug(f"Filtered artifact (prefix): '{original_transcript}' -> '{filtered}'")
                 break # Stop after first prefix match

        # Remove extra whitespace potentially introduced
        final_transcript = re.sub(r'\s+', ' ', filtered).strip()

        if final_transcript != original_transcript.strip():
             self.logger.info(f"Artifact filtering applied: '{original_transcript}' -> '{final_transcript}'")

        return final_transcript

    def _write_wav_file(self, temp_wav_file_handle, pcm16_data):
        """ Writes PCM16 data into a proper WAV format file handle. """
        sample_rate = 8000
        channels = 1 # OpenAI requires mono
        sample_width_bytes = 2 # PCM16

        try:
            with wave.open(temp_wav_file_handle, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width_bytes)
                wf.setframerate(sample_rate)
                wf.writeframes(pcm16_data)
            self.logger.debug(f"WAV file written successfully (Data length: {len(pcm16_data)} bytes)")
        except Exception as e:
            self.logger.error(f"Error writing WAV file: {e}", exc_info=True)
            raise # Re-raise the exception


    async def stream_transcribe_audio(self, file_path, channel, start_sample_count, duration_samples):
        """ Calls OpenAI API for transcription and puts result/error in response queue. """
        self.logger.info(f"Channel {channel}: Calling OpenAI API for file {file_path} (StartSample: {start_sample_count}, DurationSamples: {duration_samples})")
        result_obj = None # Initialize result_obj
        try:
            openai_lang_code = self.openai_language if not self.is_unsupported_language else None
            prompt_text = None
            if self.is_unsupported_language:
                base_prompt = get_language_specific_prompt(self.language)
                prompt_text = f"{base_prompt} Ignore initial beeps, rings, and system sounds."
                self.logger.info(f"Channel {channel}: Using language-specific prompt for {self.language}: '{prompt_text}'")
            else:
                prompt_text = "This is a customer service call. The customer may be discussing problems with services or products. Ignore initial beeps, rings, and system sounds."
                self.logger.info(f"Channel {channel}: Using standard prompt and language code '{openai_lang_code}'")


            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

            with open(file_path, 'rb') as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file', audio_file, filename=os.path.basename(file_path), content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'verbose_json') # Use verbose_json for potential segments/words later
                form_data.add_field('temperature', '0.1') # Slightly lower temperature
                if prompt_text:
                    form_data.add_field('prompt', prompt_text)
                if openai_lang_code:
                    form_data.add_field('language', openai_lang_code)

                full_transcript = ""
                confidence_sum = 0
                confidence_count = 0
                low_confidence_tokens_count = 0

                async with aiohttp.ClientSession() as session:
                     async with session.post(url, headers=headers, data=form_data, timeout=30) as response:
                        if response.status == 200:
                            response_json = await response.json()
                            self.logger.debug(f"Channel {channel}: OpenAI API Response JSON: {response_json}")

                            full_transcript = response_json.get('text', '').strip()

                            # --- Confidence Calculation (Placeholder) ---
                            # verbose_json doesn't reliably give logprobs like streaming.
                            # We'll use a default confidence for now, but this could be improved
                            # if OpenAI adds word confidences to verbose_json in the future.
                            avg_confidence = 0.90 # Default confidence for OpenAI non-streaming

                            # --- Artifact Filtering ---
                            filtered_transcript = self._filter_spurious_artifacts(full_transcript)

                            if not filtered_transcript:
                                 self.logger.info(f"Channel {channel}: Transcript was empty after filtering. Original: '{full_transcript}'")
                                 # Don't put empty results on the queue
                            elif filtered_transcript == self.last_transcripts[channel]:
                                 self.logger.info(f"Channel {channel}: Filtered transcript is same as last. Skipping duplicate. Transcript: '{filtered_transcript}'")
                                 # Don't put duplicates on the queue
                            else:
                                # --- Create Response Object ---
                                result_obj = self.create_response_object(
                                    filtered_transcript,
                                    avg_confidence,
                                    start_sample_count,
                                    duration_samples
                                )
                                if result_obj:
                                     self.logger.info(f"Channel {channel}: Transcription successful. Transcript: '{filtered_transcript}', Confidence: {avg_confidence:.3f}")
                                     self.response_queues[channel].put(result_obj)
                                     self.last_transcripts[channel] = filtered_transcript # Update last transcript
                                else:
                                     self.logger.warning(f"Channel {channel}: Failed to create response object for transcript: '{filtered_transcript}'")

                        else: # API Error
                            error_text = await response.text()
                            self.logger.error(f"Channel {channel}: OpenAI API error: {response.status} - {error_text}")
                            # Put exception on queue to notify main thread
                            self.response_queues[channel].put(Exception(f"OpenAI API Error {response.status}: {error_text}"))

        except asyncio.TimeoutError:
            self.logger.warning(f"Channel {channel}: OpenAI API request timed out.")
            self.response_queues[channel].put(TimeoutError("OpenAI API request timed out"))
        except aiohttp.ClientError as client_err:
             self.logger.error(f"Channel {channel}: Network error during OpenAI API call: {client_err}")
             self.response_queues[channel].put(client_err)
        except Exception as e:
            self.logger.error(f"Error during OpenAI transcription call for channel {channel}: {e}", exc_info=True)
            self.response_queues[channel].put(e) # Put exception on queue


    def create_response_object(self, transcript, confidence, start_sample_count, duration_samples):
        """ Creates a MockResult object compatible with the structure expected by the server. """
        if not transcript:
            self.logger.debug("Skipping response object creation for empty transcript.")
            return None

        mock_result = MockResult()
        mock_result.start_sample_count = start_sample_count
        mock_result.duration_samples = duration_samples

        # --- Generate Synthetic Word Timings ---
        words = []
        text_words = transcript.split()
        total_duration_sec = duration_samples / 8000.0

        if text_words and total_duration_sec > 0:
            avg_word_duration_sec = total_duration_sec / len(text_words)
            current_offset_sec = 0.0
            for word_text in text_words:
                start_offset = timedelta(seconds=current_offset_sec)
                end_offset = timedelta(seconds=current_offset_sec + avg_word_duration_sec)
                words.append(Word(
                    word=word_text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    confidence=confidence # Apply overall confidence to each synthetic word
                ))
                current_offset_sec += avg_word_duration_sec
        elif text_words: # Handle zero duration case
             start_offset = timedelta(seconds=0)
             end_offset = timedelta(seconds=0)
             for word_text in text_words:
                  words.append(Word(
                    word=word_text,
                    start_offset=start_offset,
                    end_offset=end_offset,
                    confidence=confidence
                ))

        alternative = Alternative(
            transcript=transcript,
            confidence=confidence,
            words=words
        )

        result = Result(
            alternatives=[alternative],
            is_final=True # OpenAI results are final per chunk
        )

        mock_result.results = [result]
        self.logger.debug(f"Created response object with {len(words)} synthetic words.")
        return mock_result


    def feed_audio(self, audio_stream: bytes, channel: int, current_total_samples: int):
        """ Feeds raw PCMU audio data and the current sample count into the processing queue. """
        if not self.running:
             self.logger.warning(f"Channel {channel}: feed_audio called but not running. Ignoring.")
             return
        if not audio_stream or channel >= self.channels:
            self.logger.warning(f"Channel {channel}: Invalid audio stream or channel index in feed_audio.")
            return

        try:
            # Convert PCMU (u-law) to PCM16 before putting in queue
            pcm16_data = audioop.ulaw2lin(audio_stream, 2)
            # Put both data and the starting sample count for this chunk onto the queue
            self.audio_queues[channel].put((pcm16_data, current_total_samples))
            self.logger.debug(f"Channel {channel}: Fed {len(pcm16_data)} bytes (PCM16) starting at sample {current_total_samples} to queue.")
        except audioop.error as conversion_err:
             self.logger.error(f"Channel {channel}: Audio conversion error (ulaw2lin): {conversion_err}. Discarding chunk.")
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error in feed_audio: {e}", exc_info=True)


    def get_response(self, channel: int):
        """ Retrieves the next transcription result or error from the response queue. """
        if channel >= self.channels:
            self.logger.error(f"Invalid channel index {channel} requested in get_response.")
            return None
        try:
            # Non-blocking get from the response queue
            response = self.response_queues[channel].get_nowait()
            self.logger.debug(f"Retrieved response from queue for channel {channel}.")
            return response
        except queue.Empty:
            # It's normal for the queue to be empty between results
            return None
        except Exception as e:
            self.logger.error(f"Error retrieving response from queue for channel {channel}: {e}", exc_info=True)
            return None

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
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True

        # Audio buffering for utterance detection
        self.audio_buffers = [[] for _ in range(channels)]
        self.buffer_durations = [0.0 for _ in range(channels)]
        self.last_process_time = [time.time() for _ in range(channels)]

        # VAD parameters
        self.vad_threshold = 200
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]

        # Silence duration to consider end of utterance (800ms)
        self.silence_threshold_frames = 8 # 8 * 100ms frame = 800ms

        # Accumulated audio buffer
        self.accumulated_audio = [bytearray() for _ in range(channels)]

        # Track last utterance to prevent duplicates
        self.last_transcripts = ["" for _ in range(channels)]

        # Initial audio discarding
        self.initial_frames_processed = [0 for _ in range(channels)]
        self.skip_initial_frames = 5

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
            # Send sentinel value to signal thread to exit
            self.audio_queues[channel].put(None)
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                self.streaming_threads[channel].join(timeout=1.0)

    def streaming_recognize_thread(self, channel):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            frame_duration_ms = 100 # Process audio in 100ms chunks for VAD
            frame_size_bytes = 1600 # 8000 Hz * 2 bytes/sample * 0.1 seconds

            current_chunk = bytearray()

            while self.running:
                try:
                    # Get data, potentially blocking with timeout
                    audio_data = self.audio_queues[channel].get(timeout=0.1)

                    if audio_data is None: # Sentinel received
                         # Process any remaining audio in current_chunk + accumulated_audio
                        current_chunk.extend(self.accumulated_audio[channel])
                        if len(current_chunk) > 0:
                             self.logger.debug(f"Channel {channel}: Processing remaining audio before shutdown ({len(current_chunk)} bytes)")
                             self._process_accumulated_audio(channel, bytes(current_chunk), loop)
                        break # Exit the loop

                    # Append new data to the current processing chunk
                    current_chunk.extend(audio_data)

                    # Process in fixed-size frames for VAD
                    while len(current_chunk) >= frame_size_bytes:
                        frame_to_process = current_chunk[:frame_size_bytes]
                        current_chunk = current_chunk[frame_size_bytes:]

                        # Skip initial frames
                        if self.initial_frames_processed[channel] < self.skip_initial_frames:
                            self.initial_frames_processed[channel] += 1
                            self.logger.debug(f"Channel {channel}: Skipping initial frame {self.initial_frames_processed[channel]}")
                            continue # Skip VAD and accumulation for initial frames

                        # VAD logic
                        rms = audioop.rms(frame_to_process, 2) # PCM16 => width=2
                        is_current_speech = rms > self.vad_threshold

                        # Always accumulate audio *after* initial skipping phase
                        self.accumulated_audio[channel].extend(frame_to_process)

                        if is_current_speech:
                            self.silence_frames[channel] = 0
                            self.speech_frames[channel] += 1
                            if not self.is_speech[channel] and self.speech_frames[channel] >= 2: # Need 2 consecutive speech frames
                                self.is_speech[channel] = True
                                self.logger.debug(f"Channel {channel}: Speech detected (RMS: {rms})")
                        else: # Silence
                            self.speech_frames[channel] = 0
                            if self.is_speech[channel]:
                                self.silence_frames[channel] += 1
                                if self.silence_frames[channel] >= self.silence_threshold_frames:
                                    self.is_speech[channel] = False
                                    self.logger.debug(f"Channel {channel}: End of speech detected after {self.silence_frames[channel]} silence frames")
                                    # Process the accumulated audio buffer
                                    self._process_accumulated_audio(channel, bytes(self.accumulated_audio[channel]), loop)
                                    self.accumulated_audio[channel] = bytearray() # Clear buffer after processing

                    # Timeout processing: If too much time passed or buffer is large
                    current_time = time.time()
                    buffer_len = len(self.accumulated_audio[channel])
                    time_since_last_process = current_time - self.last_process_time[channel]

                    # Process if speech ongoing for > 5s or buffer > 15s or silence timeout occurred implicitly
                    if self.is_speech[channel] and time_since_last_process > 5.0 and buffer_len > frame_size_bytes:
                         self.logger.debug(f"Channel {channel}: Processing accumulated audio due to ongoing speech timeout ({buffer_len} bytes)")
                         self._process_accumulated_audio(channel, bytes(self.accumulated_audio[channel]), loop)
                         self.accumulated_audio[channel] = bytearray()
                    elif buffer_len > frame_size_bytes * 150: # Approx 15 seconds buffer limit
                         self.logger.warning(f"Channel {channel}: Buffer overflow prevention, forcing processing ({buffer_len} bytes)")
                         self._process_accumulated_audio(channel, bytes(self.accumulated_audio[channel]), loop)
                         self.accumulated_audio[channel] = bytearray()


                except queue.Empty:
                    # No new audio, check if silence timeout occurred for ongoing speech
                    if self.is_speech[channel] and (time.time() - self.last_process_time[channel]) > (self.silence_threshold_frames * frame_duration_ms / 1000.0 + 0.5): # Add buffer
                         self.is_speech[channel] = False
                         self.logger.debug(f"Channel {channel}: End of speech detected due to queue timeout")
                         self._process_accumulated_audio(channel, bytes(self.accumulated_audio[channel]), loop)
                         self.accumulated_audio[channel] = bytearray()
                    continue # Continue waiting
                except Exception as e:
                    self.logger.error(f"Error in streaming thread loop for channel {channel}: {e}", exc_info=True)
                    # Maybe sleep briefly before retrying
                    time.sleep(0.1)

            loop.close()
            self.logger.info(f"Streaming thread for channel {channel} finished.")
        except Exception as e:
            self.logger.error(f"Fatal error initializing streaming thread for channel {channel}: {str(e)}", exc_info=True)
            self.response_queues[channel].put(e) # Signal error downstream

    def _process_accumulated_audio(self, channel, audio_data, loop):
        """Process accumulated audio using OpenAI streaming transcription"""
        audio_len_bytes = len(audio_data)
        if audio_len_bytes < 1600: # Less than ~0.1s, likely noise or silence
            self.logger.debug(f"Channel {channel}: Accumulated audio too short ({audio_len_bytes} bytes), skipping processing.")
            self.last_process_time[channel] = time.time()
            return

        try:
            # Check RMS again on the whole chunk to avoid pure silence
            rms = audioop.rms(audio_data, 2)
            if rms < self.vad_threshold * 0.8: # Slightly lower threshold for the whole chunk
                self.logger.debug(f"Channel {channel}: Accumulated audio chunk mostly silence (RMS: {rms}), skipping API call.")
                self.last_process_time[channel] = time.time()
                return

            self.logger.debug(f"Channel {channel}: Processing accumulated audio chunk ({audio_len_bytes} bytes, RMS: {rms})")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                try:
                     self._write_wav_file(temp_wav_path, audio_data) # Pass path, not file object
                     # Use OpenAI streaming transcription
                     result_tuple = loop.run_until_complete(
                         self.stream_transcribe_audio(temp_wav_path, channel)
                     )

                     if result_tuple and not isinstance(result_tuple, Exception):
                        transcript_text, avg_confidence = result_tuple

                        # Apply filtering AFTER getting result
                        filtered_transcript = self._filter_spurious_artifacts(transcript_text)

                        # Only send if non-empty and different from last transcript
                        if filtered_transcript and filtered_transcript != self.last_transcripts[channel]:
                            self.logger.info(f"Channel {channel}: OpenAI transcription result: '{filtered_transcript}' (Conf: {avg_confidence:.3f})")
                            # Put the filtered result and confidence into the queue
                            self.response_queues[channel].put((filtered_transcript, avg_confidence))
                            self.last_transcripts[channel] = filtered_transcript
                        elif filtered_transcript and filtered_transcript == self.last_transcripts[channel]:
                             self.logger.debug(f"Channel {channel}: Duplicate transcript detected, skipping: '{filtered_transcript}'")
                        elif not filtered_transcript:
                             self.logger.debug(f"Channel {channel}: Transcript became empty after filtering, skipping.")

                     elif isinstance(result_tuple, Exception):
                          self.logger.error(f"Channel {channel}: OpenAI transcription failed: {result_tuple}")

                except Exception as e:
                     self.logger.error(f"Channel {channel}: Error during OpenAI API call or WAV writing: {e}", exc_info=True)
                finally:
                     # Clean up temp file
                    try:
                        os.unlink(temp_wav_path)
                    except OSError as e:
                         self.logger.warning(f"Channel {channel}: Could not delete temp WAV file {temp_wav_path}: {e}")

        except Exception as e:
            self.logger.error(f"Channel {channel}: Error in _process_accumulated_audio: {e}", exc_info=True)
        finally:
             # Always update last process time regardless of success/failure
             self.last_process_time[channel] = time.time()


    def _filter_spurious_artifacts(self, transcript):
        if not transcript:
            return transcript

        original_transcript = transcript
        # Apply regex patterns first
        for pattern in ARTIFACT_PATTERNS:
            transcript = re.sub(pattern, "", transcript, flags=re.IGNORECASE)

        # Trim whitespace after regex
        transcript = transcript.strip()

        # Check simple artifacts (case-insensitive)
        for artifact in KNOWN_ARTIFACTS:
            if transcript.lower() == artifact.lower():
                self.logger.debug(f"Filtered full transcript artifact: '{original_transcript}' -> ''")
                return ""
            if transcript.lower().startswith(artifact.lower() + " "):
                # Find the actual case-insensitive match length to remove correctly
                match_len = len(artifact) + 1
                if original_transcript[:match_len].lower() == (artifact.lower() + " "):
                     transcript = original_transcript[match_len:] # Slice original to preserve case
                     self.logger.debug(f"Filtered start artifact: '{original_transcript}' -> '{transcript}'")
                     # Re-strip after removal
                     transcript = transcript.strip()


        # Remove duplicated spaces just in case
        transcript = re.sub(r'\s+', ' ', transcript).strip()

        # If filtering changed the transcript, log it
        if transcript != original_transcript:
             self.logger.debug(f"Artifact filtering applied: '{original_transcript}' -> '{transcript}'")

        return transcript


    def _write_wav_file(self, file_path, audio_data):
        """Write PCM16 data to a WAV file at the specified path"""
        sample_rate = 8000
        channels = 1
        sample_width = 2 # PCM16

        try:
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(sample_width)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data)
            self.logger.debug(f"Successfully wrote {len(audio_data)} bytes to WAV file: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to write WAV file {file_path}: {e}")
            raise # Re-raise the exception


    async def stream_transcribe_audio(self, file_path, channel):
        """Stream transcribe audio using OpenAI's API, returns (transcript, avg_confidence) or None"""
        try:
            openai_lang = self.openai_language
            prompt_to_use = "This is a customer service call. Ignore initial beeps, rings, and system sounds." # Default prompt

            if self.is_unsupported_language:
                native_prompt = get_language_specific_prompt(self.language)
                prompt_to_use = f"{native_prompt} {prompt_to_use}" # Prepend native prompt
                self.logger.info(f"Channel {channel}: Using special handling for unsupported language {self.language}. Prompt: '{prompt_to_use}'")
                openai_lang = None # Don't send language code for unsupported languages
            else:
                self.logger.info(f"Channel {channel}: Transcribing with OpenAI model {OPENAI_SPEECH_MODEL}. Language: {openai_lang}. Prompt: '{prompt_to_use}'")

            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

            with open(file_path, 'rb') as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file',
                                    audio_file,
                                    filename=os.path.basename(file_path),
                                    content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'verbose_json') # Use verbose_json to get segments/tokens if available
                form_data.add_field('timestamp_granularities[]', 'word') # Request word timestamps
                form_data.add_field('temperature', '0')
                form_data.add_field('prompt', prompt_to_use)

                if openai_lang:
                    form_data.add_field('language', openai_lang)


                full_transcript = ""
                confidence_sum = 0
                confidence_count = 0
                words_data = []

                # Note: OpenAI Transcription API doesn't truly stream results per word like Google.
                # 'verbose_json' with 'word' granularity gives the full result with timings at the end.
                # We are essentially using it as a non-streaming call here within the async structure.
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data, timeout=30) as response:
                            if response.status == 200:
                                response_json = await response.json()
                                full_transcript = response_json.get('text', '')

                                # Try to get word-level confidence if available (might not be present)
                                words_data = response_json.get('words', [])
                                if words_data:
                                     for word_info in words_data:
                                          # OpenAI doesn't directly provide confidence per word via API.
                                          # We'll rely on the server's heuristic for now.
                                          confidence_count += 1
                                     # Use a fixed high confidence if word timestamps exist but no confidence score.
                                     avg_confidence = 0.95 if words_data else 0.90
                                     self.logger.debug(f"Channel {channel}: Received {len(words_data)} words with timestamps from OpenAI.")

                                else:
                                     # Fallback: estimate confidence based on overall structure if no words array
                                     avg_confidence = 0.90 # Default estimate
                                     self.logger.debug(f"Channel {channel}: No word timestamps received from OpenAI. Using default confidence.")


                                # Basic check for low confidence based on the *presence* of words (heuristic)
                                if not full_transcript and not words_data:
                                     avg_confidence = 0.5 # Lower confidence if completely empty

                                return (full_transcript, avg_confidence)

                            else:
                                error_text = await response.text()
                                self.logger.error(f"Channel {channel}: OpenAI API error: {response.status} - {error_text}")
                                if "language" in error_text.lower() and self.is_unsupported_language:
                                    self.logger.error(f"Language-related error details: Input Lang='{self.language}', OpenAI Lang='{openai_lang}', Is Unsupported={self.is_unsupported_language}, Prompt='{prompt_to_use}'")
                                return None # Indicate failure
                except asyncio.TimeoutError:
                    self.logger.warning(f"Channel {channel}: OpenAI API request timed out.")
                    return None
                except aiohttp.ClientError as e:
                     self.logger.error(f"Channel {channel}: Network error during OpenAI API call: {e}")
                     return None
                except Exception as e:
                    self.logger.error(f"Channel {channel}: Error processing OpenAI response: {e}", exc_info=True)
                    return None
        except Exception as e:
            self.logger.error(f"Channel {channel}: Error in stream_transcribe_audio setup: {str(e)}", exc_info=True)
            return None


    def feed_audio(self, audio_stream: bytes, channel: int):
        if not audio_stream or channel >= self.channels:
            return

        try:
            # Convert PCMU (u-law) to PCM16 for internal processing
            pcm16_data = audioop.ulaw2lin(audio_stream, 2)
            self.audio_queues[channel].put(pcm16_data)
        except audioop.error as e:
             self.logger.error(f"Channel {channel}: audioop error converting frame: {e}")
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error in feed_audio: {e}", exc_info=True)


    def get_response(self, channel: int):
        """Gets the next (transcript, confidence) tuple, Exception, or None"""
        if channel >= self.channels:
            return None
        try:
            # Non-blocking get from the queue
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error retrieving response from queue: {e}")
             return e # Propagate exceptions

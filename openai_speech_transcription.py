# openai_speech_transcription.py

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
import wave # Import wave module

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


class Word:
    def __init__(self, word="", start_offset=None, end_offset=None, confidence=0.9):
        self.word = word
        self.start_offset = start_offset or timedelta(seconds=0)
        self.end_offset = end_offset or timedelta(seconds=0) # Default to 0 duration
        self.confidence = confidence

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
        self.audio_queues = [queue.Queue() for _ in range(channels)] # Store tuples (pcm16_data, adjusted_start_sample)
        self.response_queues = [queue.Queue() for _ in range(channels)] # Store dicts from create_response_object
        self.streaming_threads = [None] * channels
        self.running = True

        self.accumulated_audio = [bytearray() for _ in range(channels)]
        self.accumulated_samples = [0 for _ in range(channels)] # Track samples in current buffer
        self.utterance_start_sample = [None for _ in range(channels)] # Track absolute start sample of utterance
        self.last_sample_count = [0 for _ in range(channels)] # Track absolute sample count of last fed frame

        self.last_process_time = [time.time() for _ in range(channels)]

        self.vad_threshold = 200
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]

        self.silence_threshold_frames = 8 # ~160ms assuming ~20ms VAD frames internally

        self.last_transcripts = ["" for _ in range(channels)]

        self.initial_frames_processed = [0 for _ in range(channels)]
        self.skip_initial_frames = 5

        self.token_confidence_threshold = 0.2
        self.min_utterance_bytes = 1600 # Minimum bytes (~0.1s) to process

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
            self.audio_queues[channel].put(None)
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                self.streaming_threads[channel].join(timeout=1.0)

    def streaming_recognize_thread(self, channel):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            frame_size_ms = 20 # VAD frame size in ms
            frame_size_bytes = int(8000 * 2 * (frame_size_ms / 1000)) # Bytes per VAD frame (PCM16)

            local_buffer = bytearray()

            while self.running:
                try:
                    queue_item = self.audio_queues[channel].get(timeout=0.1)
                    if queue_item is None:
                        if len(self.accumulated_audio[channel]) >= self.min_utterance_bytes and self.utterance_start_sample[channel] is not None:
                            self.logger.debug(f"Channel {channel}: Processing remaining audio before shutdown.")
                            self._process_accumulated_audio(channel, loop, self.utterance_start_sample[channel], self.last_sample_count[channel])
                        break

                    audio_chunk, adjusted_start_sample = queue_item
                    chunk_samples = len(audio_chunk) // 2 # PCM16 -> 2 bytes/sample
                    self.last_sample_count[channel] = adjusted_start_sample + chunk_samples


                    if self.initial_frames_processed[channel] < self.skip_initial_frames:
                        self.initial_frames_processed[channel] += 1
                        self.logger.debug(f"Channel {channel}: Skipping initial frame {self.initial_frames_processed[channel]}")
                        continue

                    local_buffer.extend(audio_chunk)

                    # Process in VAD frame chunks
                    while len(local_buffer) >= frame_size_bytes:
                        vad_frame = local_buffer[:frame_size_bytes]
                        local_buffer = local_buffer[frame_size_bytes:]

                        rms = audioop.rms(vad_frame, 2)
                        is_current_speech = rms > self.vad_threshold

                        current_frame_start_sample = adjusted_start_sample + ((len(audio_chunk) - len(local_buffer) - frame_size_bytes) // 2)

                        if is_current_speech:
                            if not self.is_speech[channel]: # Start of speech detected
                                self.is_speech[channel] = True
                                self.speech_frames[channel] = 1
                                self.silence_frames[channel] = 0
                                if self.utterance_start_sample[channel] is None:
                                     self.utterance_start_sample[channel] = current_frame_start_sample
                                     self.logger.debug(f"Channel {channel}: Speech started at sample {self.utterance_start_sample[channel]}")
                                self.accumulated_audio[channel].extend(vad_frame)
                                self.accumulated_samples[channel] += frame_size_bytes // 2
                            else: # Continuing speech
                                self.speech_frames[channel] += 1
                                self.silence_frames[channel] = 0
                                self.accumulated_audio[channel].extend(vad_frame)
                                self.accumulated_samples[channel] += frame_size_bytes // 2

                        else: # Silence
                             if self.is_speech[channel]: # Silence after speech
                                self.silence_frames[channel] += 1
                                self.speech_frames[channel] = 0
                                # Still accumulate some silence for context, but maybe less?
                                # For now, keep accumulating to match original logic loosely
                                self.accumulated_audio[channel].extend(vad_frame)
                                self.accumulated_samples[channel] += frame_size_bytes // 2

                                if self.silence_frames[channel] >= self.silence_threshold_frames:
                                    self.logger.debug(f"Channel {channel}: End of speech detected due to silence.")
                                    if len(self.accumulated_audio[channel]) >= self.min_utterance_bytes and self.utterance_start_sample[channel] is not None:
                                        utterance_end_sample = current_frame_start_sample + (frame_size_bytes // 2)
                                        self._process_accumulated_audio(channel, loop, self.utterance_start_sample[channel], utterance_end_sample)
                                    else:
                                        self.logger.debug(f"Channel {channel}: Accumulated audio too short or no start sample after silence, clearing buffer.")
                                        self._reset_utterance_state(channel)
                                    self.is_speech[channel] = False
                             # else: # Silence before speech, do nothing, discard frame implicitly


                    current_time = time.time()
                    if self.is_speech[channel] and current_time - self.last_process_time[channel] > 5.0 and len(self.accumulated_audio[channel]) >= self.min_utterance_bytes * 2 and self.utterance_start_sample[channel] is not None:
                         self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout during speech.")
                         # Process up to the start of the current local_buffer
                         utterance_end_sample = self.last_sample_count[channel] - (len(local_buffer) // 2)
                         self._process_accumulated_audio(channel, loop, self.utterance_start_sample[channel], utterance_end_sample)
                         # Keep the remaining local_buffer for the next segment

                    # Prevent excessive buffer growth
                    if len(self.accumulated_audio[channel]) > 8000 * 2 * 30: # ~30 seconds
                        self.logger.warning(f"Channel {channel}: Buffer overflow risk, forcing processing.")
                        if self.utterance_start_sample[channel] is not None:
                             utterance_end_sample = self.last_sample_count[channel] - (len(local_buffer) // 2)
                             self._process_accumulated_audio(channel, loop, self.utterance_start_sample[channel], utterance_end_sample)
                        else:
                             self.logger.warning(f"Channel {channel}: Cannot process overflow, no start sample recorded. Clearing buffer.")
                             self._reset_utterance_state(channel)


                except queue.Empty:
                    # Check for timeout processing if no new audio arrived but we are in speech
                    current_time = time.time()
                    if self.is_speech[channel] and current_time - self.last_process_time[channel] > 5.0 and len(self.accumulated_audio[channel]) >= self.min_utterance_bytes and self.utterance_start_sample[channel] is not None:
                         self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout (no new audio).")
                         # End sample is the last recorded sample count
                         self._process_accumulated_audio(channel, loop, self.utterance_start_sample[channel], self.last_sample_count[channel])

                    pass
                except Exception as e:
                    self.logger.error(f"Error in streaming thread loop for channel {channel}: {e}", exc_info=True)
                    # Attempt to recover state if possible
                    self._reset_utterance_state(channel)


            loop.close()
            self.logger.info(f"Streaming thread for channel {channel} finished.")
        except Exception as e:
            self.logger.error(f"Fatal error setting up streaming thread for channel {channel}: {str(e)}", exc_info=True)
            self.response_queues[channel].put(e)


    def _reset_utterance_state(self, channel):
        self.accumulated_audio[channel] = bytearray()
        self.accumulated_samples[channel] = 0
        self.utterance_start_sample[channel] = None
        self.is_speech[channel] = False
        self.speech_frames[channel] = 0
        self.silence_frames[channel] = 0
        # Do not reset last_sample_count here, it reflects the stream position

    def _process_accumulated_audio(self, channel, loop, start_sample, end_sample):
        if len(self.accumulated_audio[channel]) < self.min_utterance_bytes:
            self.logger.debug(f"Channel {channel}: Accumulated audio {len(self.accumulated_audio[channel])} bytes too short for processing, skipping.")
            self._reset_utterance_state(channel)
            self.last_process_time[channel] = time.time()
            return

        audio_data_to_process = bytes(self.accumulated_audio[channel])
        current_start_sample = start_sample
        current_end_sample = end_sample

        # Reset state *before* async call
        self._reset_utterance_state(channel)
        self.last_process_time[channel] = time.time()

        try:
            rms = audioop.rms(audio_data_to_process, 2)
            if rms < self.vad_threshold * 0.5:
                self.logger.debug(f"Channel {channel}: Audio to process mostly silence (RMS: {rms}), skipping.")
                return

            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
                try:
                    self._write_wav_file(temp_wav, audio_data_to_process)
                    temp_wav.close() # Ensure file is written and closed before reading

                    result_dict = loop.run_until_complete(
                        self.stream_transcribe_audio(temp_wav_path, channel, current_start_sample, current_end_sample)
                    )

                    if result_dict and not isinstance(result_dict, Exception) and result_dict.get('transcript'):
                        transcript_text = result_dict['transcript']
                        if transcript_text != self.last_transcripts[channel]:
                            self.response_queues[channel].put(result_dict)
                            self.last_transcripts[channel] = transcript_text
                            self.logger.debug(f"Channel {channel}: Queued transcription result.")
                        else:
                            self.logger.debug(f"Channel {channel}: Skipping duplicate transcript.")
                    elif isinstance(result_dict, Exception):
                         self.response_queues[channel].put(result_dict) # Propagate errors

                except Exception as e:
                    self.logger.error(f"Channel {channel}: Error during transcription processing: {e}", exc_info=True)
                finally:
                    if os.path.exists(temp_wav_path):
                        try:
                            os.unlink(temp_wav_path)
                        except OSError as e:
                            self.logger.error(f"Channel {channel}: Error deleting temp WAV file {temp_wav_path}: {e}")

        except Exception as e:
            self.logger.error(f"Channel {channel}: Error in _process_accumulated_audio: {e}", exc_info=True)


    def _filter_spurious_artifacts(self, transcript):
        if not transcript:
            return transcript

        original_transcript = transcript
        for pattern in ARTIFACT_PATTERNS:
            transcript = re.sub(pattern, "", transcript)

        for artifact in KNOWN_ARTIFACTS:
            if transcript.strip().lower() == artifact:
                transcript = ""
                break
            if transcript.strip().lower().startswith(artifact + " "):
                # Case-insensitive replace of the first occurrence
                transcript = re.sub(r'(?i)^' + re.escape(artifact) + r'\s+', '', transcript.strip(), count=1)


        transcript = re.sub(r'\s+', ' ', transcript).strip()

        if transcript != original_transcript:
             self.logger.debug(f"Filtered artifacts: '{original_transcript}' -> '{transcript}'")

        return transcript

    def _write_wav_file(self, wav_file_handle, audio_data):
        sample_rate = 8000
        channels = 1
        sample_width = 2

        with wave.open(wav_file_handle, 'wb') as wf:
             wf.setnchannels(channels)
             wf.setsampwidth(sample_width)
             wf.setframerate(sample_rate)
             wf.writeframes(audio_data)


    async def stream_transcribe_audio(self, file_path, channel, start_sample, end_sample):
        try:
            openai_lang = self.openai_language
            prompt_to_use = ""

            if self.is_unsupported_language:
                base_prompt = get_language_specific_prompt(self.language)
                prompt_to_use = f"{base_prompt} Ignore initial beeps, rings, and system sounds."
                self.logger.info(f"Channel {channel}: Using language-specific prompt for {self.language_prompt}: '{prompt_to_use}'")
                openai_lang = None # Don't send language code for unsupported
            else:
                prompt_to_use = "This is a customer service call. The customer may be discussing problems with services or products. Ignore initial beeps, rings, and system sounds."
                self.logger.info(f"Channel {channel}: Using language code '{openai_lang}' and generic prompt.")


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
                form_data.add_field('response_format', 'json')
                form_data.add_field('stream', 'true')
                form_data.add_field('include[]', 'logprobs')
                form_data.add_field('temperature', '0')

                if prompt_to_use:
                    form_data.add_field('prompt', prompt_to_use)
                if openai_lang:
                    form_data.add_field('language', openai_lang)


                full_transcript = ""
                confidence_sum = 0
                confidence_count = 0
                low_confidence_tokens = []

                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data, timeout=30) as response:
                            if response.status == 200:
                                async for line in response.content:
                                    line = line.decode('utf-8').strip()
                                    if line.startswith('data: '):
                                        event_data = line[6:]
                                        if event_data == '[DONE]':
                                            break

                                        try:
                                            event_json = json.loads(event_data)
                                            event_type = event_json.get('type')

                                            if event_type == 'transcript.text.delta' or event_type == 'transcript.text.done':
                                                delta = event_json.get('delta', '') if event_type == 'transcript.text.delta' else ""
                                                if event_type == 'transcript.text.done':
                                                    full_transcript = event_json.get('text', full_transcript) # Use final text if available

                                                if 'logprobs' in event_json and event_json['logprobs']:
                                                    for logprob in event_json['logprobs']:
                                                        token = logprob.get('token', '')
                                                        token_logprob = logprob.get('logprob', -float('inf')) # Use -inf if missing
                                                        token_confidence = min(0.99, math.exp(token_logprob))

                                                        if token_confidence < self.token_confidence_threshold:
                                                            low_confidence_tokens.append(token)
                                                            self.logger.debug(f"Channel {channel}: Low confidence token: '{token}' conf={token_confidence:.4f}")

                                                        confidence_sum += token_confidence
                                                        confidence_count += 1

                                        except json.JSONDecodeError:
                                            self.logger.warning(f"Channel {channel}: Failed to parse streaming event: {event_data}")
                                            continue

                                avg_confidence = (confidence_sum / confidence_count) if confidence_count > 0 else 0.8 # Default confidence if no logprobs

                                filtered_transcript = self._filter_spurious_artifacts(full_transcript)

                                self.logger.debug(f"Channel {channel}: Final transcript='{filtered_transcript}', AvgConf={avg_confidence:.4f}")
                                if low_confidence_tokens:
                                    self.logger.debug(f"Channel {channel}: Low confidence tokens: {', '.join(low_confidence_tokens)}")

                                return self.create_response_object(filtered_transcript, avg_confidence, start_sample, end_sample)

                            else:
                                error_text = await response.text()
                                self.logger.error(f"Channel {channel}: OpenAI API error: {response.status} - {error_text}")
                                if "language" in error_text.lower():
                                    self.logger.error(f"Channel {channel}: Language-related error. Input lang: {self.language}, OpenAI lang: {openai_lang}, Unsupported: {self.is_unsupported_language}")
                                return None # Indicate error
                except asyncio.TimeoutError:
                    self.logger.warning(f"Channel {channel}: OpenAI API timeout")
                    return None
                except aiohttp.ClientError as e:
                     self.logger.error(f"Channel {channel}: Network error during OpenAI API call: {e}")
                     return None
                except Exception as e:
                    self.logger.error(f"Channel {channel}: Error processing OpenAI stream: {e}", exc_info=True)
                    return None

        except Exception as e:
            self.logger.error(f"Channel {channel}: Error in stream_transcribe_audio setup: {str(e)}", exc_info=True)
            return e # Return exception to be handled upstream

    def create_response_object(self, transcript, confidence, start_sample, end_sample):
        if not transcript:
            return None

        start_seconds = start_sample / 8000.0
        end_seconds = end_sample / 8000.0
        duration_seconds = max(0.0, end_seconds - start_seconds) # Ensure non-negative

        words_list = transcript.split()
        words = []

        if words_list and duration_seconds > 0.001: # Avoid division by zero if duration is tiny
            avg_word_duration = duration_seconds / len(words_list)
            for i, word_text in enumerate(words_list):
                word_start_offset = start_seconds + (i * avg_word_duration)
                word_end_offset = word_start_offset + avg_word_duration

                word = Word(
                    word=word_text,
                    start_offset=timedelta(seconds=word_start_offset),
                    end_offset=timedelta(seconds=word_end_offset),
                    confidence=confidence # Use overall confidence for synthetic words
                )
                words.append(word)
        elif words_list: # Handle zero duration case - assign all words to start time with zero duration
             word_start_offset = start_seconds
             word_end_offset = start_seconds
             for i, word_text in enumerate(words_list):
                 word = Word(
                    word=word_text,
                    start_offset=timedelta(seconds=word_start_offset),
                    end_offset=timedelta(seconds=word_end_offset),
                    confidence=confidence
                 )
                 words.append(word)


        # Return a dictionary instead of MockResult
        return {
            'transcript': transcript,
            'confidence': confidence,
            'words': words, # List of Word objects with timedelta offsets
            'start_seconds': start_seconds,
            'duration_seconds': duration_seconds,
            'is_final': True # OpenAI integration provides final results per utterance
        }

    def feed_audio(self, audio_stream: bytes, channel: int, adjusted_start_sample: int):
        if not audio_stream or channel >= self.channels:
            return

        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        # Put data and its starting sample position into the queue
        self.audio_queues[channel].put((pcm16_data, adjusted_start_sample))

    def get_response(self, channel: int):
        if channel >= self.channels:
            return None
        try:
            # Returns the dictionary or Exception
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None

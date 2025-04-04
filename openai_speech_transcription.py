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
import io

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
        self.words = words or []

class Word:
    def __init__(self, word="", start_offset=None, end_offset=None, confidence=0.9):
        self.word = word
        self.start_offset = start_offset or timedelta(seconds=0)
        self.end_offset = end_offset or timedelta(seconds=1)
        self.confidence = confidence

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.logger = logger
        self.language = normalize_language_code(language)
        self.openai_language = get_openai_language_code(self.language)
        self.is_unsupported_language = is_unsupported_language(self.language)
        
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
        
        self.audio_buffers = [[] for _ in range(channels)]
        self.buffer_durations = [0.0 for _ in range(channels)]
        self.last_process_time = [time.time() for _ in range(channels)]
        
        self.vad_threshold = 200
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]
        
        self.silence_threshold_frames = 8
        
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        
        self.last_transcripts = ["" for _ in range(channels)]
        
        self.initial_frames_processed = [0 for _ in range(channels)]
        self.skip_initial_frames = 5

        self.token_confidence_threshold = 0.2
        
        self.total_samples = [0 for _ in range(channels)]
        
        self.speech_start_samples = [0 for _ in range(channels)]
        
        self.sample_rate = 8000
        
        self.audio_position_samples = [0 for _ in range(channels)]
        self.last_utterance_end_samples = [0 for _ in range(channels)]

        self.rate_limit_backoff = [0.5 for _ in range(channels)]
        self.max_retry_count = 3

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
            
            frame_size = 800  # 100ms of audio at 8kHz
            
            while self.running:
                try:
                    audio_chunk = self.audio_queues[channel].get(timeout=0.1)
                    if audio_chunk is None:
                        if len(self.accumulated_audio[channel]) > 0:
                            self._process_accumulated_audio(channel, loop)
                        break
                    
                    samples_in_chunk = len(audio_chunk) // 2
                    self.total_samples[channel] += samples_in_chunk
                    
                    if self.initial_frames_processed[channel] < self.skip_initial_frames:
                        self.initial_frames_processed[channel] += 1
                        self.logger.debug(f"Channel {channel}: Skipping initial frame {self.initial_frames_processed[channel]} to avoid startup artifacts")
                        continue
                    
                    rms = audioop.rms(audio_chunk, 2)
                    is_current_speech = rms > self.vad_threshold
                    
                    self.accumulated_audio[channel].extend(audio_chunk)
                    
                    if is_current_speech:
                        self.silence_frames[channel] = 0
                        self.speech_frames[channel] += 1
                        
                        if not self.is_speech[channel] and self.speech_frames[channel] >= 2:
                            self.is_speech[channel] = True
                            self.logger.debug(f"Channel {channel}: Speech detected")
                            self.speech_start_samples[channel] = self.total_samples[channel] - samples_in_chunk
                            
                    else:
                        self.speech_frames[channel] = 0
                        
                        if self.is_speech[channel]:
                            self.silence_frames[channel] += 1
                            
                            if self.silence_frames[channel] >= self.silence_threshold_frames:
                                self.is_speech[channel] = False
                                self.logger.debug(f"Channel {channel}: End of speech detected")
                                
                                self.audio_position_samples[channel] = self.speech_start_samples[channel]
                                
                                self._process_accumulated_audio(channel, loop)
                                
                                self.last_utterance_end_samples[channel] = self.total_samples[channel]
                    
                    current_time = time.time()
                    if current_time - self.last_process_time[channel] > 3.0 and len(self.accumulated_audio[channel]) > frame_size * 30:
                        self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout")
                        self.audio_position_samples[channel] = self.speech_start_samples[channel] if self.is_speech[channel] else self.last_utterance_end_samples[channel]
                        self._process_accumulated_audio(channel, loop)
                    
                    if len(self.accumulated_audio[channel]) > frame_size * 300:
                        self.logger.warning(f"Channel {channel}: Buffer overflow, forcing processing")
                        self.audio_position_samples[channel] = self.speech_start_samples[channel] if self.is_speech[channel] else self.last_utterance_end_samples[channel]
                        self._process_accumulated_audio(channel, loop)
                
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Error in streaming thread for channel {channel}: {e}")
                    
            loop.close()
        except Exception as e:
            self.logger.error(f"Fatal error in streaming thread for channel {channel}: {str(e)}")
            self.response_queues[channel].put(e)

    def _process_accumulated_audio(self, channel, loop):
        if len(self.accumulated_audio[channel]) < 1600:
            self.logger.debug(f"Accumulated audio too short ({len(self.accumulated_audio[channel])} bytes), skipping")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
            return
            
        try:
            audio_data = bytes(self.accumulated_audio[channel])
            rms = audioop.rms(audio_data, 2)
            
            if rms < self.vad_threshold * 0.7:
                self.logger.debug(f"Audio mostly silence (RMS: {rms}), skipping")
                self.accumulated_audio[channel] = bytearray()
                self.last_process_time[channel] = time.time()
                return
            
            current_position = self.audio_position_samples[channel]
            audio_duration = len(audio_data) // 2
            
            result = loop.run_until_complete(
                self.stream_transcribe_audio(audio_data, channel, current_position, audio_duration)
            )
            
            if result and not isinstance(result, Exception):
                transcript_text = ""
                if result.results and result.results[0].alternatives:
                    transcript_text = result.results[0].alternatives[0].transcript
                
                filtered_transcript = self._filter_spurious_artifacts(transcript_text)
                
                if result.results and result.results[0].alternatives and filtered_transcript:
                    result.results[0].alternatives[0].transcript = filtered_transcript
                
                if filtered_transcript and filtered_transcript != self.last_transcripts[channel]:
                    self.response_queues[channel].put(result)
                    self.last_transcripts[channel] = filtered_transcript
                    
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
        except Exception as e:
            self.logger.error(f"Error processing accumulated audio: {e}")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()

    def _filter_spurious_artifacts(self, transcript):
        if not transcript:
            return transcript
            
        for pattern in ARTIFACT_PATTERNS:
            transcript = re.sub(pattern, "", transcript)
            
        for artifact in KNOWN_ARTIFACTS:
            if transcript.strip() == artifact:
                return ""
            
            if transcript.strip().startswith(artifact + " "):
                transcript = transcript.replace(artifact + " ", "", 1)
            
        transcript = re.sub(r'\s+', ' ', transcript).strip()
        
        return transcript

    def _write_wav_file_to_buffer(self, buffer, audio_data):
        sample_rate = 8000
        channels = 1
        sample_width = 2
        
        buffer.write(b'RIFF')
        buffer.write((36 + len(audio_data)).to_bytes(4, 'little'))
        buffer.write(b'WAVE')
        
        buffer.write(b'fmt ')
        buffer.write((16).to_bytes(4, 'little'))
        buffer.write((1).to_bytes(2, 'little'))
        buffer.write((channels).to_bytes(2, 'little'))
        buffer.write((sample_rate).to_bytes(4, 'little'))
        buffer.write((sample_rate * channels * sample_width).to_bytes(4, 'little'))
        buffer.write((channels * sample_width).to_bytes(2, 'little'))
        buffer.write((sample_width * 8).to_bytes(2, 'little'))
        
        buffer.write(b'data')
        buffer.write(len(audio_data).to_bytes(4, 'little'))
        buffer.write(audio_data)

    async def stream_transcribe_audio(self, audio_data, channel, current_position, audio_duration):
        try:
            openai_lang = self.openai_language
            
            if self.is_unsupported_language:
                self.logger.info(f"Using special handling for unsupported language {self.language}: adding '{self.language_prompt}' to prompt instead of language code")
            else:
                self.logger.info(f"Transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
                self.logger.info(f"Using language code for OpenAI: '{openai_lang}' (converted from '{self.language}')")
            
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            wav_buffer = io.BytesIO()
            self._write_wav_file_to_buffer(wav_buffer, audio_data)
            wav_buffer.seek(0)
            
            form_data = aiohttp.FormData()
            form_data.add_field('file', 
                               wav_buffer, 
                               filename='audio.wav',
                               content_type='audio/wav')
            form_data.add_field('model', OPENAI_SPEECH_MODEL)
            form_data.add_field('response_format', 'json')
            form_data.add_field('stream', 'true')
            form_data.add_field('include[]', 'logprobs')
            form_data.add_field('temperature', '0')
            
            if self.is_unsupported_language:
                prompt = get_language_specific_prompt(self.language)
                base_prompt = prompt
                prompt = f"{base_prompt} Ignore initial beeps, rings, and system sounds."
                self.logger.info(f"Using language-specific prompt for {self.language_prompt}: '{prompt}'")
                form_data.add_field('prompt', prompt)
            else:
                if openai_lang:
                    form_data.add_field('language', openai_lang)
                
                prompt = "This is a customer service call. The customer may be discussing problems with services or products. Ignore initial beeps, rings, and system sounds."
                form_data.add_field('prompt', prompt)
            
            full_transcript = ""
            tokens_with_confidence = []
            avg_confidence = 0.9
            confidence_sum = 0
            confidence_count = 0
            low_confidence_tokens = []
            
            retry_count = 0
            max_retries = self.max_retry_count
            retry_delay = self.rate_limit_backoff[channel]
            
            while retry_count <= max_retries and self.running:
                try:
                    timeout = aiohttp.ClientTimeout(total=30)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, headers=headers, data=form_data) as response:
                            if response.status == 200:
                                async for line in response.content:
                                    line_text = line.decode('utf-8').strip()
                                    if not line_text.startswith('data: '):
                                        continue
                                    
                                    event_data = line_text[6:]
                                    if event_data == '[DONE]':
                                        break
                                    
                                    try:
                                        event_json = json.loads(event_data)
                                        event_type = event_json.get('type')
                                        
                                        if event_type == 'transcript.text.delta':
                                            delta = event_json.get('delta', '')
                                            full_transcript += delta
                                            
                                            if 'logprobs' in event_json:
                                                for logprob in event_json['logprobs']:
                                                    token = logprob.get('token', '')
                                                    token_logprob = logprob.get('logprob', 0)
                                                    token_confidence = min(0.99, math.exp(token_logprob))
                                                    
                                                    tokens_with_confidence.append((token, token_confidence))
                                                    
                                                    if token_confidence < self.token_confidence_threshold:
                                                        low_confidence_tokens.append(token)
                                                        self.logger.debug(f"Low confidence token: '{token}' with confidence {token_confidence:.4f}")
                                                    
                                                    confidence_sum += token_confidence
                                                    confidence_count += 1
                                            
										elif event_type == 'transcript.text.done':
                                            full_transcript = event_json.get('text', '')
                                            
                                            if 'logprobs' in event_json and event_json['logprobs']:
                                                tokens_with_confidence = []
                                                confidence_sum = 0
                                                confidence_count = 0
                                                
                                                for logprob in event_json['logprobs']:
                                                    token = logprob.get('token', '')
                                                    token_logprob = logprob.get('logprob', 0)
                                                    token_confidence = min(0.99, math.exp(token_logprob))
                                                    
                                                    tokens_with_confidence.append((token, token_confidence))
                                                    
                                                    if token_confidence < self.token_confidence_threshold:
                                                        low_confidence_tokens.append(token)
                                                        self.logger.debug(f"Low confidence token (final): '{token}' with confidence {token_confidence:.4f}")
                                                    
                                                    confidence_sum += token_confidence
                                                    confidence_count += 1
                                        
                                    except json.JSONDecodeError:
                                        self.logger.warning(f"Failed to parse streaming event: {event_data}")
                                        continue
                                
                                if confidence_count > 0:
                                    avg_confidence = confidence_sum / confidence_count
                                
                                filtered_transcript = self._filter_spurious_artifacts(full_transcript)
                                
                                self.logger.debug(f"Original transcript: {full_transcript}")
                                self.logger.debug(f"Filtered transcript: {filtered_transcript}")
                                self.logger.debug(f"Average confidence: {avg_confidence}")
                                self.logger.debug(f"Number of tokens with confidence: {len(tokens_with_confidence)}")
                                if low_confidence_tokens:
                                    self.logger.debug(f"Low confidence tokens: {', '.join(low_confidence_tokens)}")
                                
                                self.rate_limit_backoff[channel] = max(0.5, self.rate_limit_backoff[channel] * 0.9)
                                
                                return self.create_response_object(
                                    filtered_transcript, 
                                    avg_confidence, 
                                    current_position, 
                                    audio_duration, 
                                    channel, 
                                    tokens_with_confidence
                                )
                            elif response.status == 429:
                                error_text = await response.text()
                                self.logger.warning(f"OpenAI API rate limit exceeded: {response.status} - {error_text}")
                                retry_delay = min(30, self.rate_limit_backoff[channel] * 2)
                                self.rate_limit_backoff[channel] = retry_delay
                                retry_count += 1
                                self.logger.info(f"Retrying in {retry_delay}s (attempt {retry_count}/{max_retries})")
                                await asyncio.sleep(retry_delay)
                            else:
                                error_text = await response.text()
                                self.logger.error(f"OpenAI API error: {response.status} - {error_text}")
                                if "language" in error_text.lower():
                                    self.logger.error(f"Language-related error. Used language: {self.language}, OpenAI language: {openai_lang}, Is unsupported: {self.is_unsupported_language}")
                                break
                except asyncio.TimeoutError:
                    self.logger.warning(f"OpenAI API timeout for channel {channel}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        await asyncio.sleep(retry_delay)
                    else:
                        break
                except aiohttp.ClientError as e:
                    self.logger.error(f"HTTP error in OpenAI streaming API: {str(e)}")
                    retry_count += 1
                    if retry_count <= max_retries:
                        await asyncio.sleep(retry_delay)
                    else:
                        break
                except Exception as e:
                    self.logger.error(f"Error in OpenAI streaming API: {str(e)}")
                    break
            
            return None
        except Exception as e:
            self.logger.error(f"Error in stream_transcribe_audio: {str(e)}")
            return None

    def create_response_object(self, transcript, confidence, current_position, audio_duration, channel=0, tokens_with_confidence=None):
        if not transcript:
            return None
            
        mock_result = MockResult()
        
        text_words = transcript.split()
        words = []
        
        if text_words:
            total_duration = audio_duration / self.sample_rate
            
            avg_duration = total_duration / len(text_words)
            
            abs_start_time = current_position / self.sample_rate
            
            has_token_confidence = tokens_with_confidence and len(tokens_with_confidence) > 0
            
            if has_token_confidence:
                word_confidences = []
                
                if len(tokens_with_confidence) >= len(text_words):
                    tokens_per_word = len(tokens_with_confidence) / len(text_words)
                    
                    for i in range(len(text_words)):
                        start_idx = int(i * tokens_per_word)
                        end_idx = int((i + 1) * tokens_per_word)
                        
                        token_confs = [conf for _, conf in tokens_with_confidence[start_idx:end_idx]]
                        
                        if token_confs:
                            word_confidences.append(sum(token_confs) / len(token_confs))
                        else:
                            word_confidences.append(confidence)
                else:
                    self.logger.debug(f"Fewer tokens than words: {len(tokens_with_confidence)} tokens, {len(text_words)} words")
                    word_confidences = [confidence] * len(text_words)
            else:
                word_confidences = [confidence] * len(text_words)
            
            if len(word_confidences) != len(text_words):
                self.logger.warning(f"Word confidence mismatch: {len(word_confidences)} confidences, {len(text_words)} words")
                word_confidences = [confidence] * len(text_words)
            
            for i, word_text in enumerate(text_words):
                word_start_time = abs_start_time + (i * avg_duration)
                word_end_time = abs_start_time + ((i + 1) * avg_duration)
                
                word_confidence = word_confidences[i]
                
                word = Word(
                    word=word_text,
                    start_offset=timedelta(seconds=word_start_time),
                    end_offset=timedelta(seconds=word_end_time),
                    confidence=word_confidence
                )
                words.append(word)
            
            if has_token_confidence and len(text_words) > 0:
                sample_size = min(5, len(text_words))
                sample_words = [(text_words[i], word_confidences[i]) for i in range(sample_size)]
                self.logger.debug(f"Sample word confidences: {sample_words}")
        
        alternative = Alternative(
            transcript=transcript,
            confidence=confidence,
            words=words
        )
        
        result = Result(
            alternatives=[alternative],
            is_final=True
        )
        
        mock_result.results = [result]
        return mock_result

    def feed_audio(self, audio_stream: bytes, channel: int):
        if not audio_stream or channel >= self.channels:
            return
        
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        self.audio_queues[channel].put(pcm16_data)

    def get_response(self, channel: int):
        if channel >= self.channels:
            return None
        try:
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None

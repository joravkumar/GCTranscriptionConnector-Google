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

from config import (
    OPENAI_API_KEY,
    OPENAI_SPEECH_MODEL
)
from language_mapping import normalize_language_code, get_openai_language_code

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
        
        # VAD parameters for detecting speech segments
        self.vad_threshold = 200
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]
        
        # Silence duration to consider end of utterance (800ms)
        self.silence_threshold_frames = 8
        
        # Keep track of accumulated audio for streaming
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        
        # Track last utterance to prevent duplicates
        self.last_transcripts = ["" for _ in range(channels)]

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
            
            # Frame size for processing (100ms of audio at 8kHz)
            frame_size = 800  # 8000 Hz / 10 frames per second
            
            while self.running:
                try:
                    audio_chunk = self.audio_queues[channel].get(timeout=0.1)
                    if audio_chunk is None:
                        # Process any remaining audio before shutting down
                        if len(self.accumulated_audio[channel]) > 0:
                            self._process_accumulated_audio(channel, loop)
                        break
                    
                    # Approximate frame duration
                    frame_duration = len(audio_chunk) / 16000.0  # 8000 Hz * 2 bytes per sample
                    
                    # Check audio energy for VAD
                    rms = audioop.rms(audio_chunk, 2)
                    is_current_speech = rms > self.vad_threshold
                    
                    # Always add to accumulated buffer for continuous streaming
                    self.accumulated_audio[channel].extend(audio_chunk)
                    
                    # VAD state machine for utterance detection
                    if is_current_speech:
                        self.silence_frames[channel] = 0
                        self.speech_frames[channel] += 1
                        
                        if not self.is_speech[channel] and self.speech_frames[channel] >= 2:
                            self.is_speech[channel] = True
                            self.logger.debug(f"Channel {channel}: Speech detected")
                            
                    else:
                        # Not speech
                        self.speech_frames[channel] = 0
                        
                        if self.is_speech[channel]:
                            self.silence_frames[channel] += 1
                            
                            # If we've detected enough silence after speech, process the utterance
                            if self.silence_frames[channel] >= self.silence_threshold_frames:
                                self.is_speech[channel] = False
                                self.logger.debug(f"Channel {channel}: End of speech detected")
                                self._process_accumulated_audio(channel, loop)
                    
                    # Periodically process accumulated audio even during continuous speech
                    current_time = time.time()
                    if current_time - self.last_process_time[channel] > 3.0 and len(self.accumulated_audio[channel]) > frame_size * 30:
                        self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout")
                        self._process_accumulated_audio(channel, loop)
                    
                    # Prevent buffer overflow
                    if len(self.accumulated_audio[channel]) > frame_size * 300:  # ~30 seconds
                        self.logger.warning(f"Channel {channel}: Buffer overflow, forcing processing")
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
        """Process accumulated audio using OpenAI streaming transcription"""
        if len(self.accumulated_audio[channel]) < 1600:  # Less than 0.1s at 8kHz
            self.logger.debug(f"Accumulated audio too short ({len(self.accumulated_audio[channel])} bytes), skipping")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
            return
            
        try:
            # Check RMS to avoid processing silence
            audio_data = bytes(self.accumulated_audio[channel])
            rms = audioop.rms(audio_data, 2)
            
            if rms < self.vad_threshold * 0.7:
                self.logger.debug(f"Audio mostly silence (RMS: {rms}), skipping")
                self.accumulated_audio[channel] = bytearray()
                self.last_process_time[channel] = time.time()
                return
                
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                self._write_wav_file(temp_wav, audio_data)
                
                try:
                    # Use OpenAI streaming transcription
                    result = loop.run_until_complete(
                        self.stream_transcribe_audio(temp_wav.name, channel)
                    )
                    
                    if result and not isinstance(result, Exception):
                        # Check if transcript is different from last one
                        transcript_text = ""
                        if result.results and result.results[0].alternatives:
                            transcript_text = result.results[0].alternatives[0].transcript
                        
                        # Only send if non-empty and different from last transcript
                        if transcript_text and transcript_text != self.last_transcripts[channel]:
                            self.response_queues[channel].put(result)
                            self.last_transcripts[channel] = transcript_text
                except Exception as e:
                    self.logger.error(f"Error in OpenAI streaming transcription: {e}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_wav.name)
                    except:
                        pass
                        
            # Reset accumulated audio and update last process time
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
        except Exception as e:
            self.logger.error(f"Error processing accumulated audio: {e}")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()

    def _write_wav_file(self, temp_wav, audio_data):
        """Write PCM16 data to a WAV file"""
        sample_rate = 8000
        channels = 1
        sample_width = 2
        
        # Write WAV header
        temp_wav.write(b'RIFF')
        temp_wav.write((36 + len(audio_data)).to_bytes(4, 'little'))
        temp_wav.write(b'WAVE')
        
        temp_wav.write(b'fmt ')
        temp_wav.write((16).to_bytes(4, 'little'))
        temp_wav.write((1).to_bytes(2, 'little'))
        temp_wav.write((channels).to_bytes(2, 'little'))
        temp_wav.write((sample_rate).to_bytes(4, 'little'))
        temp_wav.write((sample_rate * channels * sample_width).to_bytes(4, 'little'))
        temp_wav.write((channels * sample_width).to_bytes(2, 'little'))
        temp_wav.write((sample_width * 8).to_bytes(2, 'little'))
        
        temp_wav.write(b'data')
        temp_wav.write(len(audio_data).to_bytes(4, 'little'))
        temp_wav.write(audio_data)
        temp_wav.flush()

    async def stream_transcribe_audio(self, file_path, channel):
        """Stream transcribe audio using OpenAI's streaming API"""
        try:
            openai_lang = self.openai_language
            self.logger.info(f"Simple transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
            self.logger.info(f"Using language code for OpenAI: '{openai_lang}' (converted from '{self.language}')")
            
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
                
                # Enable streaming for real-time results
                form_data.add_field('stream', 'true')
                
                # Include logprobs for token confidence
                form_data.add_field('include[]', 'logprobs')
                
                # Set temperature to 0 for most deterministic results
                form_data.add_field('temperature', '0')
                
                # Only specify language if it's not empty
                if openai_lang:
                    form_data.add_field('language', openai_lang)
                
                # Add contextual prompt for service call transcriptions
                prompt = "This is a customer service call. The customer may be discussing problems with services or products."
                form_data.add_field('prompt', prompt)
                
                full_transcript = ""
                words = []
                avg_confidence = 0.9  # Default
                confidence_sum = 0
                confidence_count = 0
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data, timeout=30) as response:
                            if response.status == 200:
                                # Process streaming response
                                buffer = ""
                                async for line in response.content:
                                    line = line.decode('utf-8').strip()
                                    if line.startswith('data: '):
                                        event_data = line[6:]  # Remove 'data: ' prefix
                                        if event_data == '[DONE]':
                                            break
                                            
                                        try:
                                            event_json = json.loads(event_data)
                                            event_type = event_json.get('type')
                                            
                                            # Handle different event types
                                            if event_type == 'transcript.text.delta':
                                                delta = event_json.get('delta', '')
                                                full_transcript += delta
                                                
                                                # Extract logprobs for confidence
                                                if 'logprobs' in event_json:
                                                    for logprob in event_json['logprobs']:
                                                        token_logprob = logprob.get('logprob', 0)
                                                        # Convert logprob to linear probability and cap at 0.99
                                                        token_confidence = min(0.99, math.exp(token_logprob))
                                                        confidence_sum += token_confidence
                                                        confidence_count += 1
                                                
                                            elif event_type == 'transcript.text.done':
                                                # Final transcript with all tokens
                                                full_transcript = event_json.get('text', '')
                                                
                                                # Calculate final confidence from logprobs
                                                if 'logprobs' in event_json and event_json['logprobs']:
                                                    for logprob in event_json['logprobs']:
                                                        token_logprob = logprob.get('logprob', 0)
                                                        token_confidence = min(0.99, math.exp(token_logprob))
                                                        confidence_sum += token_confidence
                                                        confidence_count += 1
                                                
                                        except json.JSONDecodeError:
                                            self.logger.warning(f"Failed to parse streaming event: {event_data}")
                                            continue
                                
                                # Calculate average confidence
                                if confidence_count > 0:
                                    avg_confidence = confidence_sum / confidence_count
                                
                                self.logger.debug(f"Full transcript: {full_transcript}")
                                self.logger.debug(f"Average confidence: {avg_confidence}")
                                
                                # Create response object similar to Google API response
                                return self.create_response_object(full_transcript, avg_confidence)
                            else:
                                error_text = await response.text()
                                self.logger.error(f"OpenAI API error: {response.status} - {error_text}")
                                return None
                except asyncio.TimeoutError:
                    self.logger.warning(f"OpenAI API timeout for channel {channel}")
                    return None
                except Exception as e:
                    self.logger.error(f"Error in OpenAI streaming API: {str(e)}")
                    return None
        except Exception as e:
            self.logger.error(f"Error in stream_transcribe_audio: {str(e)}")
            return None

    def create_response_object(self, transcript, confidence):
        """Create a response object compatible with Google API format"""
        if not transcript:
            return None
            
        mock_result = MockResult()
        
        # Split transcript into words for token-level information
        text_words = transcript.split()
        words = []
        
        if text_words:
            # Create synthetic timing for words
            total_duration = min(max(len(text_words) * 0.3, 1.0), 10.0)
            avg_duration = total_duration / len(text_words)
            
            for i, word_text in enumerate(text_words):
                start_time = i * avg_duration
                end_time = (i + 1) * avg_duration
                
                word = Word(
                    word=word_text,
                    start_offset=timedelta(seconds=start_time),
                    end_offset=timedelta(seconds=end_time),
                    confidence=confidence  # Use calculated confidence
                )
                words.append(word)
        
        alternative = Alternative(
            transcript=transcript,
            confidence=confidence,  # Use calculated confidence
            words=words
        )
        
        result = Result(
            alternatives=[alternative],
            is_final=True
        )
        
        mock_result.results = [result]
        return mock_result

    def feed_audio(self, audio_stream: bytes, channel: int):
        """Feed audio into the pipeline for the given channel"""
        if not audio_stream or channel >= self.channels:
            return
        
        # Convert PCMU (u-law) to PCM16 for processing
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        self.audio_queues[channel].put(pcm16_data)

    def get_response(self, channel: int):
        """Get the next transcription response for the given channel"""
        if channel >= self.channels:
            return None
        try:
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None

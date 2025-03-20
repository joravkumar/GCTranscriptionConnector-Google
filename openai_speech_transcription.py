import asyncio
import audioop
import os
import json
import hashlib
import time
import threading
import queue
import tempfile
import base64
from datetime import timedelta
import aiohttp
import logging

from config import (
    OPENAI_API_KEY,
    OPENAI_SPEECH_MODEL
)
from language_mapping import normalize_language_code, get_openai_language_code

class MockResult:
    def __init__(self):
        self.results = []

class StreamingTranscription:
    model_format_support = {}
    
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
        self.temp_files = [None] * channels
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        self.chunk_size = 24000
        self.last_chunk_time = [time.time() for _ in range(channels)]
        self.max_delay = 2.0
        self.silence_threshold = 500  # RMS threshold for silence detection

    def start_streaming(self):
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self.streaming_recognize_thread, args=(channel,)
            )
            self.streaming_threads[channel].start()

    def stop_streaming(self):
        self.running = False
        for channel in range(self.channels):
            self.audio_queues[channel].put(None)
        for channel in range(self.channels):
            if self.streaming_threads[channel]:
                self.streaming_threads[channel].join()
        
        for temp_file in self.temp_files:
            if temp_file and not temp_file.closed:
                temp_file.close()

    def streaming_recognize_thread(self, channel):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            while self.running:
                try:
                    audio_chunk = self.audio_queues[channel].get(timeout=0.1)
                    if audio_chunk is None:
                        break
                    
                    self.accumulated_audio[channel].extend(audio_chunk)
                    current_time = time.time()
                    
                    if (len(self.accumulated_audio[channel]) >= self.chunk_size or 
                            current_time - self.last_chunk_time[channel] >= self.max_delay):
                        
                        if len(self.accumulated_audio[channel]) > 0:
                            audio_data = bytes(self.accumulated_audio[channel])
                            
                            # Check if the audio is silent
                            rms = audioop.rms(audio_data, 2)
                            if rms < self.silence_threshold:
                                # Skip processing silent audio
                                self.accumulated_audio[channel] = bytearray()
                                self.last_chunk_time[channel] = current_time
                                continue
                            
                            self.accumulated_audio[channel] = bytearray()
                            self.last_chunk_time[channel] = current_time
                            
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                                sample_rate = 8000
                                channels = 1
                                sample_width = 2
                                
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
                            
                            try:
                                result = loop.run_until_complete(
                                    self.transcribe_audio(temp_wav.name, channel)
                                )
                                self.response_queues[channel].put(result)
                            except Exception as e:
                                self.logger.error(f"Error in OpenAI transcription for channel {channel}: {e}")
                                self.response_queues[channel].put(e)
                            finally:
                                if os.path.exists(temp_wav.name):
                                    os.unlink(temp_wav.name)

                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Error in streaming recognize thread for channel {channel}: {e}")
                    self.response_queues[channel].put(e)
                    
            loop.close()
        except Exception as e:
            self.logger.error(f"Fatal error in streaming recognize thread for channel {channel}: {e}")
            self.response_queues[channel].put(e)

    async def transcribe_audio(self, file_path, channel):
        try:
            openai_lang = self.openai_language
            self.logger.info(f"Transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
            self.logger.info(f"Using language code for OpenAI: '{openai_lang}' (converted from '{self.language}')")
            
            # Check audio for silence
            with open(file_path, 'rb') as f:
                audio_data = f.read()
                
            # Get the audio data part (after WAV header)
            audio_content = audio_data[44:]  # Skip WAV header
            
            # Check RMS value
            rms = audioop.rms(audio_content, 2)
            if rms < self.silence_threshold:
                self.logger.info(f"Audio is silent (RMS: {rms}), skipping transcription")
                return MockResult()
            
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            # Check if we already know this model's capabilities
            if OPENAI_SPEECH_MODEL in self.model_format_support and 'verbose_json' not in self.model_format_support[OPENAI_SPEECH_MODEL]:
                self.logger.debug(f"Using json format for model {OPENAI_SPEECH_MODEL} (cached capability)")
                return await self.transcribe_audio_simple(file_path, channel)
            
            with open(file_path, 'rb') as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file', 
                                   audio_file, 
                                   filename=os.path.basename(file_path),
                                   content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'verbose_json')
                form_data.add_field('language', openai_lang)
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data) as response:
                            if response.status == 200:
                                # Cache successful format support
                                if OPENAI_SPEECH_MODEL not in self.model_format_support:
                                    self.model_format_support[OPENAI_SPEECH_MODEL] = ['verbose_json']
                                
                                result_json = await response.json()
                                self.logger.debug(f"OpenAI transcription result: {result_json}")
                                
                                # Check if the response is empty or contains only whitespace
                                if not result_json.get('text') or result_json.get('text').strip() == "":
                                    self.logger.info("OpenAI returned empty transcript, returning empty result")
                                    return MockResult()
                                
                                return self.convert_to_google_format(result_json, channel)
                            else:
                                error_text = await response.text()
                                error_msg = f"OpenAI API error: {response.status} - {error_text}"
                                self.logger.error(error_msg)
                                return MockResult()
                except Exception as e:
                    error_text = str(e)
                    if ("verbose_json" in error_text and "not compatible" in error_text) or \
                       ("verbose_json" in error_text and "unsupported_value" in error_text):
                        # Update model format support cache
                        if OPENAI_SPEECH_MODEL not in self.model_format_support:
                            self.model_format_support[OPENAI_SPEECH_MODEL] = []
                        if 'verbose_json' in self.model_format_support.get(OPENAI_SPEECH_MODEL, []):
                            self.model_format_support[OPENAI_SPEECH_MODEL].remove('verbose_json')
                        
                        self.logger.warning(f"Model {OPENAI_SPEECH_MODEL} doesn't support verbose_json, falling back to json format")
                        return await self.transcribe_audio_simple(file_path, channel)
                    else:
                        self.logger.error(f"Error in OpenAI transcription: {e}")
                        return MockResult()
        except Exception as e:
            self.logger.error(f"Error in OpenAI transcription: {e}")
            return MockResult()

    async def transcribe_audio_simple(self, file_path, channel):
        try:
            openai_lang = self.openai_language
            self.logger.info(f"Simple transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
            self.logger.info(f"Using language code for OpenAI: '{openai_lang}' (converted from '{self.language}')")
            
            # Check audio for silence
            with open(file_path, 'rb') as f:
                audio_data = f.read()
                
            # Get the audio data part (after WAV header)
            audio_content = audio_data[44:]  # Skip WAV header
            
            # Check RMS value
            rms = audioop.rms(audio_content, 2)
            if rms < self.silence_threshold:
                self.logger.info(f"Audio is silent (RMS: {rms}), skipping transcription")
                return MockResult()
            
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
                form_data.add_field('language', openai_lang)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=form_data) as response:
                        if response.status == 200:
                            # Update model format support cache
                            if OPENAI_SPEECH_MODEL not in self.model_format_support:
                                self.model_format_support[OPENAI_SPEECH_MODEL] = []
                            if 'json' not in self.model_format_support.get(OPENAI_SPEECH_MODEL, []):
                                self.model_format_support[OPENAI_SPEECH_MODEL].append('json')
                            
                            result_json = await response.json()
                            self.logger.debug(f"OpenAI simple transcription result: {result_json}")
                            
                            # Check if the response is empty or contains only whitespace
                            if not result_json.get('text') or result_json.get('text').strip() == "":
                                self.logger.info("OpenAI returned empty transcript, returning empty result")
                                return MockResult()
                            
                            return self.convert_simple_to_google_format(result_json, channel)
                        else:
                            error_text = await response.text()
                            error_msg = f"OpenAI API error: {response.status} - {error_text}"
                            self.logger.error(error_msg)
                            return MockResult()
        except Exception as e:
            self.logger.error(f"Error in simple OpenAI transcription: {e}")
            return MockResult()

    def convert_to_google_format(self, openai_response, channel):
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        
        mock_result = type('MockResult', (), {})
        mock_result.results = []
        
        if 'text' in openai_response and openai_response['text'].strip():
            result = type('Result', (), {})
            result.alternatives = []
            result.is_final = True
            
            alternative = type('Alternative', (), {})
            alternative.transcript = openai_response['text']
            alternative.confidence = 0.9
            
            if 'words' in openai_response and openai_response['words']:
                alternative.words = []
                for word_info in openai_response['words']:
                    word = type('Word', (), {})
                    word.word = word_info.get('word', '')
                    
                    start_sec = word_info.get('start', 0)
                    end_sec = word_info.get('end', 0)
                    word.start_offset = timedelta(seconds=start_sec)
                    word.end_offset = timedelta(seconds=end_sec)
                    
                    word.confidence = word_info.get('confidence', 0.9)
                    
                    alternative.words.append(word)
            else:
                alternative.words = []
                word = type('Word', (), {})
                word.word = openai_response['text']
                word.start_offset = timedelta(seconds=0)
                word.end_offset = timedelta(seconds=2)
                word.confidence = 0.9
                alternative.words.append(word)
            
            result.alternatives.append(alternative)
            mock_result.results.append(result)
        
        return mock_result

    def convert_simple_to_google_format(self, openai_response, channel):
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        
        mock_result = type('MockResult', (), {})
        mock_result.results = []
        
        if 'text' in openai_response and openai_response['text'].strip():
            result = type('Result', (), {})
            result.alternatives = []
            result.is_final = True
            
            alternative = type('Alternative', (), {})
            alternative.transcript = openai_response['text']
            alternative.confidence = 0.9
            
            words = openai_response['text'].split()
            if words:
                alternative.words = []
                avg_duration = 2.0 / len(words)
                
                for i, word_text in enumerate(words):
                    word = type('Word', (), {})
                    word.word = word_text
                    word.start_offset = timedelta(seconds=i * avg_duration)
                    word.end_offset = timedelta(seconds=(i + 1) * avg_duration)
                    word.confidence = 0.9
                    alternative.words.append(word)
            else:
                alternative.words = []
            
            result.alternatives.append(alternative)
            mock_result.results.append(result)
        
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

async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> dict:
    if not audio_stream:
        logger.warning("openai_speech_transcription - No audio data received for transcription.")
        return {"transcript": "", "words": []}
    
    try:
        logger.info("Starting OpenAI transcription process")
        
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1

        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        logger.info(f"Converted PCMU to PCM16: {len(pcm16_data)} bytes, channels={channels}")
        
        rms_value = audioop.rms(pcm16_data, 2)
        logger.debug(f"PCM16 RMS value: {rms_value}")
        
        if rms_value < 500:  # Higher threshold for silence detection
            logger.info("PCM16 RMS value below threshold, skipping transcription")
            return {"transcript": "", "words": []}

        source_language_raw = "en-US"
        if negotiated_media and "language" in negotiated_media:
            source_language_raw = negotiated_media["language"]
        
        source_language = normalize_language_code(source_language_raw)
        openai_language = get_openai_language_code(source_language)
        logger.info(f"Source language determined as: {source_language} (OpenAI format: {openai_language})")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            sample_rate = 8000
            sample_width = 2
            
            temp_wav.write(b'RIFF')
            temp_wav.write((36 + len(pcm16_data)).to_bytes(4, 'little'))
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
            temp_wav.write(len(pcm16_data).to_bytes(4, 'little'))
            temp_wav.write(pcm16_data)
            temp_wav.flush()
            
            file_path = temp_wav.name
        
        try:
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            response_format = 'json' # More reliable to use json format directly
            
            try:
                form_data = aiohttp.FormData()
                with open(file_path, 'rb') as audio_file:
                    form_data.add_field('file', 
                                      audio_file, 
                                      filename=os.path.basename(file_path),
                                      content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', response_format)
                form_data.add_field('language', openai_language)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=form_data) as response:
                        if response.status == 200:
                            result_json = await response.json()
                            logger.debug(f"OpenAI transcription result: {result_json}")
                            
                            transcript = result_json.get('text', '').strip()
                            
                            # If transcript is empty, return empty result
                            if not transcript:
                                return {
                                    "transcript": "",
                                    "confidence": 0.0,
                                    "words": []
                                }
                            
                            words_list = []
                            words = transcript.split()
                            
                            if words:
                                avg_duration = 2.0 / len(words)
                                for i, word_text in enumerate(words):
                                    words_list.append({
                                        "word": word_text,
                                        "start_time": i * avg_duration,
                                        "end_time": (i + 1) * avg_duration,
                                        "confidence": 0.9
                                    })
                            
                            return {
                                "transcript": transcript,
                                "confidence": 0.9,
                                "words": words_list
                            }
                        else:
                            error_text = await response.text()
                            logger.error(f"OpenAI API error: {response.status} - {error_text}")
                            return {"transcript": "", "words": []}
            
            except Exception as e:
                logger.error(f"Error in OpenAI API call: {str(e)}")
                return {"transcript": "", "words": []}
                
        finally:
            if os.path.exists(file_path):
                os.unlink(file_path)
                
    except Exception as e:
        logger.error(f"openai_speech_transcription - Error during transcription: {e}", exc_info=True)
        return {"transcript": "", "words": []}

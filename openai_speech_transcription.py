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
        self.temp_files = [None] * channels
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        
        # Increase chunk size to capture more complete phrases (6 seconds of audio)
        self.chunk_size = 48000
        
        self.last_chunk_time = [time.time() for _ in range(channels)]
        
        # Use a longer max delay to allow more complete utterances
        self.max_delay = 3.0
        
        # Lower silence threshold to capture more speech
        self.silence_threshold = 50
        
        # Track silence for VAD
        self.silence_duration = [0.0 for _ in range(channels)]
        self.is_speaking = [False for _ in range(channels)]
        self.vad_silence_threshold = 200
        self.min_silence_duration = 0.5
        self.min_speech_duration = 0.3
        self.speech_buffers = [[] for _ in range(channels)]
        self.speech_start_time = [None for _ in range(channels)]
        self.last_speech_end_time = [0.0 for _ in range(channels)]

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
                    
                    # Process VAD for better utterance detection
                    self._process_vad(audio_chunk, channel, loop)
                    
                    # Also add to the accumulated buffer for timeout-based processing
                    self.accumulated_audio[channel].extend(audio_chunk)
                    current_time = time.time()
                    
                    # Process accumulated audio if max delay is reached
                    if current_time - self.last_chunk_time[channel] >= self.max_delay and len(self.accumulated_audio[channel]) > 0:
                        audio_data = bytes(self.accumulated_audio[channel])
                        
                        # Skip processing very short or silent audio
                        rms = audioop.rms(audio_data, 2)
                        if rms < self.silence_threshold or len(audio_data) < 8000:
                            self.accumulated_audio[channel] = bytearray()
                            self.last_chunk_time[channel] = current_time
                            continue
                            
                        self.accumulated_audio[channel] = bytearray()
                        self.last_chunk_time[channel] = current_time
                        
                        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                            self._write_wav_file(temp_wav, audio_data)
                            
                            try:
                                result = loop.run_until_complete(
                                    self.transcribe_audio(temp_wav.name, channel)
                                )
                                
                                if result and not isinstance(result, Exception):
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

    def _process_vad(self, audio_chunk, channel, loop):
        try:
            rms = audioop.rms(audio_chunk, 2)
            is_silence = rms < self.vad_silence_threshold
            dt = 0.1  # Approximate time step for VAD
            
            if is_silence:
                self.silence_duration[channel] += dt
                
                # If we detected the end of speech
                if self.is_speaking[channel] and self.silence_duration[channel] >= self.min_silence_duration:
                    self.is_speaking[channel] = False
                    
                    # If we have accumulated enough speech
                    if self.speech_start_time[channel] is not None:
                        speech_duration = time.time() - self.speech_start_time[channel]
                        
                        if speech_duration >= self.min_speech_duration and len(self.speech_buffers[channel]) > 0:
                            # Concatenate all buffers into one audio segment
                            complete_audio = b''.join(self.speech_buffers[channel])
                            
                            if len(complete_audio) > 8000:  # At least 1 second of audio
                                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                                    self._write_wav_file(temp_wav, complete_audio)
                                    
                                    try:
                                        result = loop.run_until_complete(
                                            self.transcribe_audio(temp_wav.name, channel)
                                        )
                                        
                                        if result and not isinstance(result, Exception):
                                            self.response_queues[channel].put(result)
                                    except Exception as e:
                                        self.logger.error(f"VAD transcription error for channel {channel}: {e}")
                                    finally:
                                        if os.path.exists(temp_wav.name):
                                            os.unlink(temp_wav.name)
                            
                        # Reset speech buffer
                        self.speech_buffers[channel] = []
                        self.speech_start_time[channel] = None
                        self.last_speech_end_time[channel] = time.time()
            else:
                # Reset silence counter when we detect speech
                self.silence_duration[channel] = 0.0
                
                # If this is the start of a new speech segment
                if not self.is_speaking[channel]:
                    self.is_speaking[channel] = True
                    
                    # Only start a new utterance if there's been enough silence
                    elapsed_since_last = time.time() - self.last_speech_end_time[channel]
                    if self.speech_start_time[channel] is None and elapsed_since_last >= self.min_silence_duration:
                        self.speech_start_time[channel] = time.time()
                
                # Add chunk to speech buffer if we're tracking speech
                if self.speech_start_time[channel] is not None:
                    self.speech_buffers[channel].append(audio_chunk)
                    
        except Exception as e:
            self.logger.error(f"Error in VAD processing for channel {channel}: {e}")

    def _write_wav_file(self, temp_wav, audio_data):
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

    async def transcribe_audio(self, file_path, channel):
        try:
            openai_lang = self.openai_language
            self.logger.info(f"Simple transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
            self.logger.info(f"Using language code for OpenAI: '{openai_lang}' (converted from '{self.language}')")
            
            # Skip very small audio files
            file_size = os.path.getsize(file_path)
            if file_size < 8000:  # Less than 1 second of audio (8000 bytes at 8kHz)
                self.logger.info(f"Audio file too small ({file_size} bytes), skipping transcription")
                return None
            
            # Check audio for silence
            with open(file_path, 'rb') as f:
                f.seek(44)  # Skip WAV header
                audio_content = f.read()
            
            rms = audioop.rms(audio_content, 2)
            if rms < self.silence_threshold:
                self.logger.info(f"Audio is silent (RMS: {rms}), skipping transcription")
                return None
            
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
                
                # Add temperature=0 for more deterministic results
                form_data.add_field('temperature', '0')
                
                # Add prompt for context (improves accuracy)
                prompt = "This is a customer service call. The customer may be discussing problems with services or products."
                form_data.add_field('prompt', prompt)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=form_data) as response:
                        if response.status == 200:
                            result_json = await response.json()
                            self.logger.debug(f"OpenAI simple transcription result: {result_json}")
                            
                            # If the response is empty or contains only whitespace, return None
                            if not result_json.get('text') or result_json.get('text').strip() == "":
                                self.logger.info("OpenAI returned empty transcript, skipping")
                                return None
                            
                            return self.convert_to_google_format(result_json, channel)
                        else:
                            error_text = await response.text()
                            error_msg = f"OpenAI API error: {response.status} - {error_text}"
                            self.logger.error(error_msg)
                            
                            # Retry with exponential backoff for 429 (rate limit) errors
                            if response.status == 429:
                                retry_after = int(response.headers.get('Retry-After', 2))
                                self.logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                                await asyncio.sleep(retry_after)
                                return await self.transcribe_audio(file_path, channel)
                                
                            return None
        except Exception as e:
            self.logger.error(f"Error in OpenAI transcription: {e}")
            return None

    def convert_to_google_format(self, openai_response, channel):
        mock_result = MockResult()
        
        if 'text' in openai_response and openai_response['text'].strip():
            transcript = openai_response['text'].strip()
            
            # Create word objects with more realistic timing
            words = []
            text_words = transcript.split()
            
            if text_words:
                # More realistic duration based on word count
                total_duration = min(max(len(text_words) * 0.3, 1.0), 5.0)
                avg_duration = total_duration / len(text_words)
                
                for i, word_text in enumerate(text_words):
                    word = Word(
                        word=word_text,
                        start_offset=timedelta(seconds=i * avg_duration),
                        end_offset=timedelta(seconds=(i + 1) * avg_duration),
                        confidence=0.9  # OpenAI doesn't provide word-level confidence
                    )
                    words.append(word)
            
            # Create alternative and result objects
            alternative = Alternative(
                transcript=transcript,
                confidence=0.9,  # OpenAI doesn't provide utterance-level confidence
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
        
        # Lower threshold for better sensitivity
        if rms_value < 50:
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
            
            try:
                form_data = aiohttp.FormData()
                with open(file_path, 'rb') as audio_file:
                    form_data.add_field('file', 
                                      audio_file, 
                                      filename=os.path.basename(file_path),
                                      content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'json')
                form_data.add_field('language', openai_language)
                form_data.add_field('temperature', '0')
                form_data.add_field('prompt', "This is a customer service call. The customer may be discussing problems with services or products.")
                
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
                                # More realistic duration based on word count
                                total_duration = min(max(len(words) * 0.3, 1.0), 5.0)
                                avg_duration = total_duration / len(words)
                                
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
                            
                            # Handle rate limiting
                            if response.status == 429:
                                retry_after = int(response.headers.get('Retry-After', 2))
                                logger.warning(f"Rate limited, waiting {retry_after}s before retry")
                                await asyncio.sleep(retry_after)
                                return await translate_audio(audio_stream, negotiated_media, logger)
                                
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

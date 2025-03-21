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
        
        # Audio buffering
        self.audio_buffers = [[] for _ in range(channels)]
        self.buffer_durations = [0.0 for _ in range(channels)]

        # VAD parameters - calibrated for better utterance detection
        self.is_speech = [False for _ in range(channels)]
        self.silence_counters = [0 for _ in range(channels)]
        self.speech_counters = [0 for _ in range(channels)]
        
        # More conservative silence threshold to avoid cutting off speech
        self.vad_threshold = 200
        
        # Longer silence required to consider end of utterance (800ms)
        self.silence_threshold_frames = 8
        
        # Shorter speech required to start an utterance (100ms)
        self.speech_threshold_frames = 1
        
        # Utterance building
        self.utterance_buffers = [[] for _ in range(channels)]
        self.max_utterance_duration = 15.0  # Max seconds to force processing even during continuous speech
        self.utterance_durations = [0.0 for _ in range(channels)]
        self.last_process_time = [time.time() for _ in range(channels)]
        
        # Prevent sending duplicate utterances
        self.last_transcripts = ["" for _ in range(channels)]
        
        # Context buffers to maintain conversation context
        self.context_buffers = [deque(maxlen=3) for _ in range(channels)]
        
        # API backoff
        self.backoff_time = 1.0
        self.max_backoff = 16.0

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
            
            while self.running:
                try:
                    audio_chunk = self.audio_queues[channel].get(timeout=0.1)
                    if audio_chunk is None:
                        # Process any remaining audio before shutting down
                        if self.utterance_buffers[channel]:
                            self._process_utterance(channel, loop)
                        break
                    
                    # Each frame is approximately 100ms (800 bytes at 8kHz)
                    frame_duration = len(audio_chunk) / 16000.0  # 8000 Hz * 2 bytes per sample
                    
                    # Analyze audio energy for VAD
                    rms = audioop.rms(audio_chunk, 2)
                    is_current_speech = rms > self.vad_threshold
                    
                    # Add to audio buffer regardless of speech detection
                    self.audio_buffers[channel].append(audio_chunk)
                    self.buffer_durations[channel] += frame_duration
                    
                    # Too much data in buffer without processing, force processing
                    if self.buffer_durations[channel] > 30.0:
                        self.logger.warning(f"Buffer overflow in channel {channel}, forcing processing")
                        self._process_buffer(channel, loop)
                        continue
                    
                    # VAD state machine - improved to better detect utterance boundaries
                    if is_current_speech:
                        # Reset silence counter when we detect speech
                        self.silence_counters[channel] = 0
                        self.speech_counters[channel] += 1
                        
                        # If we've detected enough speech frames, mark as speech
                        if not self.is_speech[channel] and self.speech_counters[channel] >= self.speech_threshold_frames:
                            self.is_speech[channel] = True
                            self.logger.debug(f"Channel {channel}: Speech started")
                            
                        # Add to utterance buffer when in speech mode
                        if self.is_speech[channel]:
                            self.utterance_buffers[channel].append(audio_chunk)
                            self.utterance_durations[channel] += frame_duration
                            
                            # If utterance is getting too long, process it to avoid delay
                            if self.utterance_durations[channel] >= self.max_utterance_duration:
                                self.logger.debug(f"Channel {channel}: Max utterance duration reached, processing")
                                self._process_utterance(channel, loop)
                    else:
                        # Not speech
                        self.speech_counters[channel] = 0
                        
                        if self.is_speech[channel]:
                            self.silence_counters[channel] += 1
                            
                            # If we're in speech mode but detected enough silence, end the utterance
                            if self.silence_counters[channel] >= self.silence_threshold_frames:
                                self.is_speech[channel] = False
                                self.logger.debug(f"Channel {channel}: Speech ended, processing utterance")
                                
                                # Only process if we have enough audio data
                                if self.utterance_durations[channel] > 0.3:
                                    self._process_utterance(channel, loop)
                    
                    # Still process the buffer periodically even without VAD triggers
                    current_time = time.time()
                    if current_time - self.last_process_time[channel] > 5.0 and self.buffer_durations[channel] > 1.0:
                        self._process_buffer(channel, loop)
                
                except queue.Empty:
                    pass
                except Exception as e:
                    self.logger.error(f"Error in streaming recognize thread for channel {channel}: {e}")
                    
            loop.close()
        except Exception as e:
            self.logger.error(f"Fatal error in streaming recognize thread for channel {channel}: {str(e)}")
            self.response_queues[channel].put(e)

    def _process_utterance(self, channel, loop):
        """Process an utterance detected by VAD"""
        if not self.utterance_buffers[channel]:
            self.utterance_durations[channel] = 0.0
            return
        
        try:
            # Combine all audio chunks in the utterance
            combined_audio = b''.join(self.utterance_buffers[channel])
            
            # Skip processing if utterance is too short or likely not speech
            if len(combined_audio) < 3200:  # Less than 0.2 seconds at 8kHz, 16-bit
                self.logger.debug(f"Utterance too short ({len(combined_audio)} bytes), skipping")
                self.utterance_buffers[channel] = []
                self.utterance_durations[channel] = 0.0
                return
            
            # Process the utterance
            self._transcribe_audio_data(combined_audio, channel, loop)
            
            # Reset utterance buffer and duration
            self.utterance_buffers[channel] = []
            self.utterance_durations[channel] = 0.0
        except Exception as e:
            self.logger.error(f"Error processing utterance for channel {channel}: {e}")
            self.utterance_buffers[channel] = []
            self.utterance_durations[channel] = 0.0

    def _process_buffer(self, channel, loop):
        """Process the entire buffer regardless of VAD state"""
        if not self.audio_buffers[channel]:
            self.buffer_durations[channel] = 0.0
            return
            
        try:
            # Combine all audio chunks in the buffer
            combined_audio = b''.join(self.audio_buffers[channel])
            
            # Skip processing if buffer is too short or mostly silence
            if len(combined_audio) < 6400:  # Less than 0.4 seconds at 8kHz, 16-bit
                self.logger.debug(f"Buffer too short ({len(combined_audio)} bytes), skipping")
                self.audio_buffers[channel] = []
                self.buffer_durations[channel] = 0.0
                return
                
            # Check if mostly silence
            rms = audioop.rms(combined_audio, 2)
            if rms < self.vad_threshold * 0.7:
                self.logger.debug(f"Buffer mostly silence (RMS: {rms}), skipping")
                self.audio_buffers[channel] = []
                self.buffer_durations[channel] = 0.0
                return
                
            # Process the buffer
            self._transcribe_audio_data(combined_audio, channel, loop)
            
            # Reset buffer and duration
            self.audio_buffers[channel] = []
            self.buffer_durations[channel] = 0.0
            self.last_process_time[channel] = time.time()
        except Exception as e:
            self.logger.error(f"Error processing buffer for channel {channel}: {e}")
            self.audio_buffers[channel] = []
            self.buffer_durations[channel] = 0.0

    def _transcribe_audio_data(self, audio_data, channel, loop):
        """Send audio data to OpenAI for transcription"""
        if not audio_data:
            return
            
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            self._write_wav_file(temp_wav, audio_data)
            
            try:
                # Build context from previous utterances
                context = " ".join(self.context_buffers[channel])
                
                result = loop.run_until_complete(
                    self.transcribe_audio(temp_wav.name, channel, context)
                )
                
                if result and not isinstance(result, Exception):
                    # Only send if we got a non-empty transcript that's different from the last one
                    transcript_text = ""
                    if result.results and result.results[0].alternatives:
                        transcript_text = result.results[0].alternatives[0].transcript
                    
                    if transcript_text and transcript_text != self.last_transcripts[channel]:
                        self.response_queues[channel].put(result)
                        self.last_transcripts[channel] = transcript_text
                        
                        # Add to context buffer for future transcriptions
                        self.context_buffers[channel].append(transcript_text)
            except Exception as e:
                self.logger.error(f"Error in OpenAI transcription for channel {channel}: {e}")
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_wav.name)
                except:
                    pass

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

    async def transcribe_audio(self, file_path, channel, context=""):
        """Send audio file to OpenAI for transcription"""
        try:
            openai_lang = self.openai_language
            self.logger.info(f"Simple transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
            self.logger.info(f"Using language code for OpenAI: '{openai_lang}' (converted from '{self.language}')")
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size < 4000:  # Less than 0.25 seconds at 8kHz, 16-bit
                self.logger.info(f"Audio file too small ({file_size} bytes), skipping transcription")
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
                
                # Only specify language if it's not empty
                if openai_lang:
                    form_data.add_field('language', openai_lang)
                
                # Set temperature to 0 for most deterministic results
                form_data.add_field('temperature', '0')
                
                # Add contextual prompt for better accuracy
                prompt = "This is a customer service call. "
                
                # Add context from previous utterances if available
                if context:
                    prompt += f"Previous conversation: {context} "
                
                form_data.add_field('prompt', prompt)
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data, timeout=10) as response:
                            if response.status == 200:
                                result_json = await response.json()
                                transcript = result_json.get('text', '').strip()
                                
                                # Check for OpenAI simple transcription
                                self.logger.debug(f"OpenAI simple transcription result: {result_json}")
                                
                                # If the transcript is empty or only whitespace, return None
                                if not transcript:
                                    return None
                                
                                # Convert to our internal format for compatibility
                                return self.convert_to_google_format(result_json, channel)
                            else:
                                error_text = await response.text()
                                error_msg = f"OpenAI API error: {response.status} - {error_text}"
                                self.logger.error(error_msg)
                                
                                # Handle rate limiting with exponential backoff
                                if response.status == 429:
                                    retry_after = int(response.headers.get('Retry-After', self.backoff_time))
                                    self.logger.warning(f"Rate limited, backing off for {retry_after}s")
                                    
                                    # Apply exponential backoff but cap at max_backoff
                                    self.backoff_time = min(self.backoff_time * 2, self.max_backoff)
                                    
                                    await asyncio.sleep(retry_after)
                                    return await self.transcribe_audio(file_path, channel, context)
                                
                                # For other errors, reset backoff
                                self.backoff_time = 1.0
                                return None
                except asyncio.TimeoutError:
                    self.logger.warning(f"OpenAI API timeout for channel {channel}")
                    return None
                except Exception as e:
                    self.logger.error(f"Error in OpenAI API request: {str(e)}")
                    return None
        except Exception as e:
            self.logger.error(f"Error in transcribe_audio: {str(e)}")
            return None

    def convert_to_google_format(self, openai_response, channel):
        """Convert OpenAI response to Google format for compatibility"""
        mock_result = MockResult()
        
        if 'text' in openai_response and openai_response['text'].strip():
            transcript = openai_response['text'].strip()
            
            words = []
            text_words = transcript.split()
            
            if text_words:
                # Generate realistic word timings
                total_duration = min(max(len(text_words) * 0.3, 1.0), 10.0)
                avg_duration = total_duration / len(text_words)
                
                for i, word_text in enumerate(text_words):
                    start_time = i * avg_duration
                    end_time = (i + 1) * avg_duration
                    
                    word = Word(
                        word=word_text,
                        start_offset=timedelta(seconds=start_time),
                        end_offset=timedelta(seconds=end_time),
                        confidence=0.9  # OpenAI doesn't provide word-level confidence
                    )
                    words.append(word)
            
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

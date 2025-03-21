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
        self.vad_threshold = 300  # Increased for better speech detection
        self.vad_min_duration = 0.5  # Minimum speech duration in seconds
        self.is_speech = [False for _ in range(channels)]
        self.silence_frames = [0 for _ in range(channels)]
        self.speech_frames = [0 for _ in range(channels)]
        
        # Silence duration to consider end of utterance (700ms)
        self.silence_threshold_frames = 7
        
        # Keep track of accumulated audio for streaming
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        self.audio_start_times = [time.time() for _ in range(channels)]
        
        # Track last utterances to prevent duplicates
        self.last_transcripts = ["" for _ in range(channels)]
        self.transcript_history = [deque(maxlen=5) for _ in range(channels)]

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
                    
                    # Check audio energy for VAD
                    rms = audioop.rms(audio_chunk, 2)
                    
                    # Add to accumulated buffer
                    self.accumulated_audio[channel].extend(audio_chunk)
                    
                    # More robust speech detection
                    is_current_speech = rms > self.vad_threshold
                    
                    # VAD state machine for utterance detection
                    if is_current_speech:
                        self.silence_frames[channel] = 0
                        self.speech_frames[channel] += 1
                        
                        # Make sure we have enough continuous speech frames before considering this speech
                        if not self.is_speech[channel] and self.speech_frames[channel] >= 3:
                            self.is_speech[channel] = True
                            self.logger.debug(f"Channel {channel}: Speech detected")
                            # Reset audio start time to better track utterance duration
                            if not self.audio_start_times[channel]:
                                self.audio_start_times[channel] = time.time()
                    else:
                        # Not speech
                        if self.is_speech[channel]:
                            self.silence_frames[channel] += 1
                            
                            # If we've detected enough silence after speech, process the utterance
                            if self.silence_frames[channel] >= self.silence_threshold_frames:
                                # Only process if we had enough speech
                                speech_duration = time.time() - self.audio_start_times[channel]
                                if speech_duration >= self.vad_min_duration:
                                    self.logger.debug(f"Channel {channel}: End of speech detected")
                                    self._process_accumulated_audio(channel, loop)
                                else:
                                    # Too short to be meaningful speech, discard it
                                    self.logger.debug(f"Channel {channel}: Speech too short ({speech_duration:.2f}s), discarding")
                                    self.accumulated_audio[channel] = bytearray()
                                
                                # Reset speech detection state
                                self.is_speech[channel] = False
                                self.speech_frames[channel] = 0
                                self.audio_start_times[channel] = None
                        else:
                            self.speech_frames[channel] = 0
                    
                    # Periodically process accumulated audio (every 2.5s) to avoid buffering too much
                    current_time = time.time()
                    if current_time - self.last_process_time[channel] > 2.5 and len(self.accumulated_audio[channel]) > frame_size * 40:
                        self.logger.debug(f"Channel {channel}: Processing accumulated audio due to timeout")
                        self._process_accumulated_audio(channel, loop)
                    
                    # Prevent buffer overflow (hard limit at 20 seconds)
                    if len(self.accumulated_audio[channel]) > frame_size * 160:
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
        # Need at least 0.2s of audio to be meaningful
        min_audio_size = 3200
        if len(self.accumulated_audio[channel]) < min_audio_size:
            self.logger.debug(f"Accumulated audio too short ({len(self.accumulated_audio[channel])} bytes), skipping")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
            self.audio_start_times[channel] = None
            return
            
        try:
            # Check RMS to avoid processing silence
            audio_data = bytes(self.accumulated_audio[channel])
            rms = audioop.rms(audio_data, 2)
            
            # Skip processing if audio is mostly silence
            if rms < 50:
                self.logger.debug(f"Audio mostly silence (RMS: {rms}), skipping")
                self.accumulated_audio[channel] = bytearray()
                self.last_process_time[channel] = time.time()
                self.audio_start_times[channel] = None
                return
                
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                self._write_wav_file(temp_wav, audio_data)
                
                try:
                    # Use OpenAI transcription
                    result = loop.run_until_complete(
                        self.transcribe_audio(temp_wav.name, channel)
                    )
                    
                    if result and not isinstance(result, Exception):
                        # Check if transcript is valid and non-empty
                        transcript_text = ""
                        
                        if result.results and result.results[0].alternatives:
                            alt = result.results[0].alternatives[0]
                            transcript_text = alt.transcript
                            
                            # Only process if transcript is non-empty
                            if transcript_text:
                                # Check for duplicates or significant overlap with recent transcripts
                                is_duplicate = False
                                for prev_transcript in self.transcript_history[channel]:
                                    # Check if new transcript is contained within a previous one
                                    if transcript_text in prev_transcript or prev_transcript in transcript_text:
                                        is_duplicate = True
                                        break
                                        
                                if not is_duplicate:
                                    # Add to history and send response
                                    self.transcript_history[channel].append(transcript_text)
                                    self.response_queues[channel].put(result)
                                    self.logger.debug(f"Full transcript: {transcript_text}")
                                    self.logger.debug(f"Average confidence: {alt.confidence}")
                                else:
                                    self.logger.debug(f"Skipping duplicate transcript: {transcript_text}")
                            else:
                                self.logger.debug("Empty transcript, skipping.")
                except Exception as e:
                    self.logger.error(f"Error in OpenAI transcription: {e}")
                finally:
                    # Clean up temp file
                    try:
                        os.unlink(temp_wav.name)
                    except:
                        pass
                        
            # Reset accumulated audio and update last process time
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
            self.audio_start_times[channel] = None
        except Exception as e:
            self.logger.error(f"Error processing accumulated audio: {e}")
            self.accumulated_audio[channel] = bytearray()
            self.last_process_time[channel] = time.time()
            self.audio_start_times[channel] = None

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

    async def transcribe_audio(self, file_path, channel):
        """Transcribe audio using OpenAI's API"""
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
                
                # Set temperature to 0 for most deterministic results
                form_data.add_field('temperature', '0')
                
                # Only specify language if it's not empty
                if openai_lang:
                    form_data.add_field('language', openai_lang)
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data, timeout=10) as response:
                            if response.status == 200:
                                response_json = await response.json()
                                transcript = response_json.get('text', '')
                                
                                # Handle empty transcripts
                                if not transcript.strip():
                                    return None
                                
                                # Calculate a reasonable confidence score
                                # For OpenAI, since we don't have direct token-level confidence,
                                # we use a fixed base confidence that's high for deterministic outputs
                                confidence = 0.8
                                
                                # Create response object compatible with Google API format
                                return self.create_response_object(transcript, confidence)
                            else:
                                error_text = await response.text()
                                self.logger.error(f"OpenAI API error: {response.status} - {error_text}")
                                return None
                except asyncio.TimeoutError:
                    self.logger.warning(f"OpenAI API timeout for channel {channel}")
                    return None
                except Exception as e:
                    self.logger.error(f"Error in OpenAI API: {str(e)}")
                    return None
        except Exception as e:
            self.logger.error(f"Error in transcribe_audio: {str(e)}")
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

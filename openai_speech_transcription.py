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

def normalize_language_code(lang: str) -> str:
    """
    Normalize language codes to the proper BCP-47 format (e.g. "es-es" -> "es-ES", "en-us" -> "en-US").
    If the language code does not contain a hyphen, return it as is.
    """
    if '-' in lang:
        parts = lang.split('-')
        if len(parts) == 2:
            return f"{parts[0].lower()}-{parts[1].upper()}"
    return lang

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.language = normalize_language_code(language)
        self.channels = channels
        self.logger = logger
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True
        self.temp_files = [None] * channels
        self.accumulated_audio = [bytearray() for _ in range(channels)]
        self.chunk_size = 24000  # Approximate chunk size (3 seconds of audio at 8kHz)
        self.last_chunk_time = [time.time() for _ in range(channels)]
        self.max_delay = 2.0  # Maximum time to wait before processing audio

    def start_streaming(self):
        """Start the streaming recognition threads for each channel."""
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self.streaming_recognize_thread, args=(channel,)
            )
            self.streaming_threads[channel].start()

    def stop_streaming(self):
        """Stop the streaming recognition threads and clean up."""
        self.running = False
        for channel in range(self.channels):
            self.audio_queues[channel].put(None)  # Signal thread to exit
        for channel in range(self.channels):
            if self.streaming_threads[channel]:
                self.streaming_threads[channel].join()
        
        # Clean up any temp files
        for temp_file in self.temp_files:
            if temp_file and not temp_file.closed:
                temp_file.close()

    def streaming_recognize_thread(self, channel):
        """Run the streaming transcription using OpenAI API in a separate thread for a specific channel."""
        try:
            # Create a session for HTTP requests
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Main processing loop
            while self.running:
                try:
                    # Get the next audio chunk or wait
                    audio_chunk = self.audio_queues[channel].get(timeout=0.1)
                    if audio_chunk is None:  # Exit signal
                        break
                    
                    # Process the chunk
                    self.accumulated_audio[channel].extend(audio_chunk)
                    current_time = time.time()
                    
                    # Process accumulated audio if we have enough or if enough time has passed
                    if (len(self.accumulated_audio[channel]) >= self.chunk_size or 
                            current_time - self.last_chunk_time[channel] >= self.max_delay):
                        
                        if len(self.accumulated_audio[channel]) > 0:
                            # Process the accumulated audio
                            audio_data = bytes(self.accumulated_audio[channel])
                            
                            # Reset for next chunk
                            self.accumulated_audio[channel] = bytearray()
                            self.last_chunk_time[channel] = current_time
                            
                            # Create temp file for the audio chunk
                            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                                # Write WAV header
                                sample_rate = 8000
                                channels = 1
                                sample_width = 2  # 16-bit PCM
                                
                                # WAV header
                                temp_wav.write(b'RIFF')
                                temp_wav.write((36 + len(audio_data)).to_bytes(4, 'little'))  # File size
                                temp_wav.write(b'WAVE')
                                
                                # Format chunk
                                temp_wav.write(b'fmt ')
                                temp_wav.write((16).to_bytes(4, 'little'))  # Format chunk size
                                temp_wav.write((1).to_bytes(2, 'little'))  # Audio format (PCM)
                                temp_wav.write((channels).to_bytes(2, 'little'))  # Channels
                                temp_wav.write((sample_rate).to_bytes(4, 'little'))  # Sample rate
                                temp_wav.write((sample_rate * channels * sample_width).to_bytes(4, 'little'))  # Byte rate
                                temp_wav.write((channels * sample_width).to_bytes(2, 'little'))  # Block align
                                temp_wav.write((sample_width * 8).to_bytes(2, 'little'))  # Bits per sample
                                
                                # Data chunk
                                temp_wav.write(b'data')
                                temp_wav.write(len(audio_data).to_bytes(4, 'little'))  # Data size
                                temp_wav.write(audio_data)
                                temp_wav.flush()
                            
                            # Call OpenAI API to transcribe the chunk
                            try:
                                result = loop.run_until_complete(
                                    self.transcribe_audio(temp_wav.name, channel)
                                )
                                # Put the result in the response queue
                                self.response_queues[channel].put(result)
                            except Exception as e:
                                self.logger.error(f"Error in OpenAI transcription for channel {channel}: {e}")
                                self.response_queues[channel].put(e)
                            finally:
                                # Clean up the temp file
                                if os.path.exists(temp_wav.name):
                                    os.unlink(temp_wav.name)

                except queue.Empty:
                    # No audio data available yet
                    pass
                except Exception as e:
                    self.logger.error(f"Error in streaming recognize thread for channel {channel}: {e}")
                    self.response_queues[channel].put(e)
                    
            loop.close()
        except Exception as e:
            self.logger.error(f"Fatal error in streaming recognize thread for channel {channel}: {e}")
            self.response_queues[channel].put(e)

    async def transcribe_audio(self, file_path, channel):
        """
        Send audio file to OpenAI for transcription.
        Returns a response object that mimics the structure expected by the existing code.
        """
        try:
            self.logger.info(f"Transcribing audio from channel {channel} with OpenAI model {OPENAI_SPEECH_MODEL}")
            
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            # Prepare form data with the file
            with open(file_path, 'rb') as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file', 
                                   audio_file, 
                                   filename=os.path.basename(file_path),
                                   content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'verbose_json')
                form_data.add_field('language', self.language)
                
                # For newer models that don't support verbose_json, we'll handle the fallback
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data) as response:
                            if response.status == 200:
                                result_json = await response.json()
                                self.logger.debug(f"OpenAI transcription result: {result_json}")
                                
                                # Convert OpenAI response to match Google's format
                                return self.convert_to_google_format(result_json, channel)
                            else:
                                error_text = await response.text()
                                error_msg = f"OpenAI API error: {response.status} - {error_text}"
                                self.logger.error(error_msg)
                                raise Exception(error_msg)
                except Exception as e:
                    # If we get an error with verbose_json, try again with text format
                    if "verbose_json" in str(e) and (OPENAI_SPEECH_MODEL == "gpt-4o-mini-transcribe" or 
                                                    OPENAI_SPEECH_MODEL == "gpt-4o-transcribe"):
                        self.logger.warning(f"Model {OPENAI_SPEECH_MODEL} doesn't support verbose_json, falling back to json format")
                        return await self.transcribe_audio_simple(file_path, channel)
                    else:
                        raise e
        except Exception as e:
            self.logger.error(f"Error in OpenAI transcription: {e}")
            raise e

    async def transcribe_audio_simple(self, file_path, channel):
        """Fallback method for models that don't support verbose_json."""
        try:
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            # Prepare form data with the file
            with open(file_path, 'rb') as audio_file:
                form_data = aiohttp.FormData()
                form_data.add_field('file', 
                                   audio_file, 
                                   filename=os.path.basename(file_path),
                                   content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'json')
                form_data.add_field('language', self.language)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=form_data) as response:
                        if response.status == 200:
                            result_json = await response.json()
                            self.logger.debug(f"OpenAI simple transcription result: {result_json}")
                            
                            # Create a simple response with just the text
                            return self.convert_simple_to_google_format(result_json, channel)
                        else:
                            error_text = await response.text()
                            error_msg = f"OpenAI API error: {response.status} - {error_text}"
                            self.logger.error(error_msg)
                            raise Exception(error_msg)
        except Exception as e:
            self.logger.error(f"Error in simple OpenAI transcription: {e}")
            raise e

    def convert_to_google_format(self, openai_response, channel):
        """
        Convert OpenAI response to match the Google Speech-to-Text response format
        for compatibility with existing code.
        """
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        
        # Create a mock Google Speech result
        mock_result = type('MockResult', (), {})
        mock_result.results = []
        
        # If the response has text, create a result
        if 'text' in openai_response:
            result = type('Result', (), {})
            result.alternatives = []
            result.is_final = True
            
            alternative = type('Alternative', (), {})
            alternative.transcript = openai_response['text']
            alternative.confidence = 0.9  # Default confidence
            
            # Handle word-level timestamps if available
            if 'words' in openai_response and openai_response['words']:
                alternative.words = []
                for word_info in openai_response['words']:
                    word = type('Word', (), {})
                    word.word = word_info.get('word', '')
                    
                    # Convert timestamps to timedelta objects
                    start_sec = word_info.get('start', 0)
                    end_sec = word_info.get('end', 0)
                    word.start_offset = timedelta(seconds=start_sec)
                    word.end_offset = timedelta(seconds=end_sec)
                    
                    # Set confidence for each word
                    word.confidence = word_info.get('confidence', 0.9)
                    
                    alternative.words.append(word)
            else:
                # For models that don't provide word-level timestamps
                # Create a single "word" with the full transcript
                alternative.words = []
                word = type('Word', (), {})
                word.word = openai_response['text']
                word.start_offset = timedelta(seconds=0)
                word.end_offset = timedelta(seconds=2)  # Default 2-second duration
                word.confidence = 0.9
                alternative.words.append(word)
            
            result.alternatives.append(alternative)
            mock_result.results.append(result)
        
        return mock_result

    def convert_simple_to_google_format(self, openai_response, channel):
        """
        Convert simple OpenAI json response (without timestamps) to match the Google format.
        This is used for models that don't support verbose_json.
        """
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        
        # Create a mock Google Speech result
        mock_result = type('MockResult', (), {})
        mock_result.results = []
        
        # If the response has text, create a result
        if 'text' in openai_response:
            result = type('Result', (), {})
            result.alternatives = []
            result.is_final = True
            
            alternative = type('Alternative', (), {})
            alternative.transcript = openai_response['text']
            alternative.confidence = 0.9  # Default confidence
            
            # Since we don't have word-level timestamps, create them artificially
            words = openai_response['text'].split()
            if words:
                alternative.words = []
                avg_duration = 2.0 / len(words)  # Distribute over 2 seconds
                
                for i, word_text in enumerate(words):
                    word = type('Word', (), {})
                    word.word = word_text
                    word.start_offset = timedelta(seconds=i * avg_duration)
                    word.end_offset = timedelta(seconds=(i + 1) * avg_duration)
                    word.confidence = 0.9
                    alternative.words.append(word)
            else:
                # Empty transcript
                alternative.words = []
            
            result.alternatives.append(alternative)
            mock_result.results.append(result)
        
        return mock_result

    def feed_audio(self, audio_stream: bytes, channel: int):
        """Feed audio data (PCMU) into the streaming queue for a specific channel after converting to PCM16."""
        if not audio_stream or channel >= self.channels:
            return
        # Convert from PCMU (u-law) to PCM16
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        self.audio_queues[channel].put(pcm16_data)

    def get_response(self, channel: int):
        """Retrieve the next transcription response from the queue for a specific channel."""
        if channel >= self.channels:
            return None
        try:
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None

async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> dict:
    """
    Process audio with OpenAI's API for non-streaming transcription.
    This matches the interface of the Google function but uses OpenAI.
    """
    if not audio_stream:
        logger.warning("openai_speech_transcription - No audio data received for transcription.")
        return {"transcript": "", "words": []}
    
    try:
        logger.info("Starting OpenAI transcription process")
        
        # Determine number of channels from negotiated media; default to 1.
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1

        # Convert the incoming PCMU (u-law) data to PCM16.
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        logger.info(f"Converted PCMU to PCM16: {len(pcm16_data)} bytes, channels={channels}")
        
        # Compute RMS value for debugging to assess audio energy.
        rms_value = audioop.rms(pcm16_data, 2)
        logger.debug(f"PCM16 RMS value: {rms_value}")
        
        # If energy is too low, skip transcription to avoid arbitrary results.
        if rms_value < 50:
            logger.info("PCM16 RMS value below threshold, skipping transcription")
            return {"transcript": "", "words": []}

        # Extract the source language from negotiated_media if provided; default to "en-US".
        source_language_raw = "en-US"
        if negotiated_media and "language" in negotiated_media:
            source_language_raw = negotiated_media["language"]
        source_language = normalize_language_code(source_language_raw)
        logger.info(f"Source language determined as: {source_language}")
        
        # Create temp file for the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            # Write WAV header
            sample_rate = 8000
            sample_width = 2  # 16-bit PCM
            
            # WAV header
            temp_wav.write(b'RIFF')
            temp_wav.write((36 + len(pcm16_data)).to_bytes(4, 'little'))  # File size
            temp_wav.write(b'WAVE')
            
            # Format chunk
            temp_wav.write(b'fmt ')
            temp_wav.write((16).to_bytes(4, 'little'))  # Format chunk size
            temp_wav.write((1).to_bytes(2, 'little'))  # Audio format (PCM)
            temp_wav.write((channels).to_bytes(2, 'little'))  # Channels
            temp_wav.write((sample_rate).to_bytes(4, 'little'))  # Sample rate
            temp_wav.write((sample_rate * channels * sample_width).to_bytes(4, 'little'))  # Byte rate
            temp_wav.write((channels * sample_width).to_bytes(2, 'little'))  # Block align
            temp_wav.write((sample_width * 8).to_bytes(2, 'little'))  # Bits per sample
            
            # Data chunk
            temp_wav.write(b'data')
            temp_wav.write(len(pcm16_data).to_bytes(4, 'little'))  # Data size
            temp_wav.write(pcm16_data)
            temp_wav.flush()
            
            file_path = temp_wav.name
        
        try:
            # Call OpenAI API
            url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            # First try with verbose_json if possible
            try:
                form_data = aiohttp.FormData()
                with open(file_path, 'rb') as audio_file:
                    form_data.add_field('file', 
                                      audio_file, 
                                      filename=os.path.basename(file_path),
                                      content_type='audio/wav')
                form_data.add_field('model', OPENAI_SPEECH_MODEL)
                form_data.add_field('response_format', 'verbose_json')
                form_data.add_field('language', source_language)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, headers=headers, data=form_data) as response:
                        if response.status == 200:
                            result_json = await response.json()
                            logger.debug(f"OpenAI transcription result: {result_json}")
                            
                            # Extract transcript and words
                            transcript = result_json.get('text', '')
                            words_list = []
                            
                            if 'words' in result_json:
                                for word_info in result_json['words']:
                                    words_list.append({
                                        "word": word_info.get('word', ''),
                                        "start_time": word_info.get('start', 0),
                                        "end_time": word_info.get('end', 0),
                                        "confidence": word_info.get('confidence', 0.9)
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
            
            # For models that don't support verbose_json
            except Exception as e:
                if "verbose_json" in str(e) and (OPENAI_SPEECH_MODEL == "gpt-4o-mini-transcribe" or 
                                               OPENAI_SPEECH_MODEL == "gpt-4o-transcribe"):
                    logger.warning(f"Model {OPENAI_SPEECH_MODEL} doesn't support verbose_json, falling back to json format")
                    
                    form_data = aiohttp.FormData()
                    with open(file_path, 'rb') as audio_file:
                        form_data.add_field('file', 
                                          audio_file, 
                                          filename=os.path.basename(file_path),
                                          content_type='audio/wav')
                    form_data.add_field('model', OPENAI_SPEECH_MODEL)
                    form_data.add_field('response_format', 'json')
                    form_data.add_field('language', source_language)
                    
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, headers=headers, data=form_data) as response:
                            if response.status == 200:
                                result_json = await response.json()
                                logger.debug(f"OpenAI simple transcription result: {result_json}")
                                
                                # Extract just the transcript
                                transcript = result_json.get('text', '')
                                
                                # Create artificial word timings
                                words = transcript.split()
                                words_list = []
                                
                                if words:
                                    avg_duration = 2.0 / len(words)  # Distribute words over 2 seconds
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
                else:
                    raise e
                
        finally:
            # Clean up temp file
            if os.path.exists(file_path):
                os.unlink(file_path)
                
    except Exception as e:
        logger.error(f"openai_speech_transcription - Error during transcription: {e}", exc_info=True)
        return {"transcript": "", "words": []}

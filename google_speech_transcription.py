import asyncio
import queue
import threading
import time
import audioop
import logging
from datetime import timedelta
import uuid
import tempfile
import os

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from language_mapping import normalize_language_code
from config import GOOGLE_SPEECH_MODEL

# Create the Speech client once and reuse it
try:
    speech_client = SpeechClient()
except Exception as e:
    logging.error(f"Failed to create Google Speech client: {e}")
    speech_client = None

class StreamingTranscription:
    def __init__(self, language, channels, logger):
        self.language = normalize_language_code(language)
        self.channels = channels
        self.logger = logger
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True
        self.last_frame_time = 0
        
        # For Chirp 2, we need to provide a recognition config
        self.config = cloud_speech.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=[self.language],
            model=GOOGLE_SPEECH_MODEL,
            adaptation=None,
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_word_confidence=True,
                profanity_filter=False
            )
        )

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

    def streaming_recognize_thread(self, channel):
        """Run the streaming recognition in a separate thread for a specific channel."""
        try:
            # Create a session for stream operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create the streaming recognize generator
            generator = self.streaming_requests_generator(channel)
            
            # Start the streaming recognition
            responses = speech_client.streaming_recognize(
                requests=generator(),
                timeout=300.0
            )
            
            # Process the responses
            for response in responses:
                if not self.running:
                    break
                self.response_queues[channel].put(response)
            
            loop.close()
        except Exception as e:
            self.logger.error(f"Error in streaming recognize thread for channel {channel}: {e}")
            self.response_queues[channel].put(e)

    def streaming_requests_generator(self, channel):
        """Generate streaming requests for audio data."""
        def gen_requests():
            try:
                # First request is the config
                yield cloud_speech.StreamingRecognizeRequest(
                    recognizer=f"projects/{speech_client.project}/locations/global/recognizers/_",
                    streaming_config=cloud_speech.StreamingRecognitionConfig(
                        config=self.config,
                        streaming_features=cloud_speech.StreamingRecognitionFeatures(
                            interim_results=True,
                            voice_activity_timeout=cloud_speech.VoiceActivityTimeout(
                                speech_start_timeout=timedelta(seconds=30.0),
                                speech_end_timeout=timedelta(seconds=3.0)
                            )
                        )
                    )
                )
                
                # Subsequent requests are the audio data
                while self.running:
                    # Get audio data from the queue
                    audio_data = self.audio_queues[channel].get()
                    if audio_data is None:  # End signal
                        break
                    
                    # Send the audio data
                    yield cloud_speech.StreamingRecognizeRequest(
                        audio=audio_data
                    )
                    
                self.logger.debug(f"Streaming request generator for channel {channel} completed")
            except Exception as e:
                self.logger.error(f"Error in streaming request generator for channel {channel}: {e}")
                raise
        return gen_requests

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
    Process audio with Google Cloud Speech-to-Text API for non-streaming transcription.
    
    Args:
        audio_stream: Raw PCMU audio data
        negotiated_media: Media format information
        logger: Logger instance
        
    Returns:
        dict: A dictionary containing the transcript and word-level details
    """
    if not audio_stream:
        logger.warning("google_speech_transcription - No audio data received for transcription.")
        return {"transcript": "", "words": []}
    
    if speech_client is None:
        logger.error("google_speech_transcription - Google Speech client not initialized.")
        return {"transcript": "", "words": []}
    
    try:
        logger.info("Starting Google Speech-to-Text transcription process")
        
        # Determine number of channels from negotiated media
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1

        # Convert audio from PCMU to PCM16
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        logger.info(f"Converted PCMU to PCM16: {len(pcm16_data)} bytes, channels={channels}")
        
        # Calculate audio energy
        rms_value = audioop.rms(pcm16_data, 2)
        logger.debug(f"PCM16 RMS value: {rms_value}")
        
        # Skip processing if audio is too quiet
        if rms_value < 50:
            logger.info("PCM16 RMS value below threshold, skipping transcription")
            return {"transcript": "", "words": []}

        # Extract the source language
        source_language = "en-US"
        if negotiated_media and "language" in negotiated_media:
            source_language = normalize_language_code(negotiated_media["language"])
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
            # Read the file
            with open(file_path, "rb") as audio_file:
                content = audio_file.read()
                
            # Set up recognition config
            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=[source_language],
                model=GOOGLE_SPEECH_MODEL,
                features=cloud_speech.RecognitionFeatures(
                    enable_word_time_offsets=True,
                    enable_word_confidence=True,
                )
            )
            
            # Perform recognition
            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{speech_client.project}/locations/global/recognizers/_",
                config=config,
                content=content
            )
            
            response = speech_client.recognize(request=request)
            
            # Process results
            transcript = ""
            words_list = []
            
            for result in response.results:
                if not result.alternatives:
                    continue
                    
                best_alternative = result.alternatives[0]
                transcript += best_alternative.transcript + " "
                
                # Extract word-level information if available
                for word in getattr(best_alternative, "words", []):
                    word_info = {
                        "word": word.word,
                        "start_time": word.start_offset.total_seconds(),
                        "end_time": word.end_offset.total_seconds(),
                        "confidence": getattr(word, "confidence", 1.0)
                    }
                    words_list.append(word_info)
            
            transcript = transcript.strip()
            
            # Create the result
            result = {
                "transcript": transcript,
                "confidence": getattr(best_alternative, "confidence", 1.0) if transcript else 0.0,
                "words": words_list
            }
            
            return result
        
        finally:
            # Clean up
            if os.path.exists(file_path):
                os.unlink(file_path)
                
    except Exception as e:
        logger.error(f"google_speech_transcription - Error during transcription: {e}", exc_info=True)
        return {"transcript": "", "words": []}

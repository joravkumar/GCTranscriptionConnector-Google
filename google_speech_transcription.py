import asyncio
import audioop
import os
import json
import hashlib
import time
import threading
import queue
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions
from config import (
    GOOGLE_APPLICATION_CREDENTIALS,
    GOOGLE_CLOUD_PROJECT,
    GOOGLE_SPEECH_MODEL
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

try:
    credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
    _credentials = service_account.Credentials.from_service_account_info(credentials_info)
except Exception as e:
    _credentials = None

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.language = normalize_language_code(language)
        self.channels = channels  # Number of channels to manage
        self.logger = logger
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True

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
        """Run the StreamingRecognize API call in a separate thread for a specific channel."""
        try:
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )
            recognition_config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=8000,
                    audio_channel_count=1  # Each stream is mono
                ),
                language_codes=[self.language],
                model=GOOGLE_SPEECH_MODEL,
                features=cloud_speech.RecognitionFeatures(
                    enable_word_time_offsets=True,
                    enable_word_confidence=True
                )
            )

            streaming_config = cloud_speech.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=cloud_speech.StreamingRecognitionFeatures(
                    interim_results=False,
                    enable_voice_activity_events=True
                )
            )
            config_request = cloud_speech.StreamingRecognizeRequest(
                recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
                streaming_config=streaming_config,
            )

            def audio_generator():
                yield config_request
                while self.running:
                    try:
                        pcm16_data = self.audio_queues[channel].get(timeout=0.1)
                        if pcm16_data is None:
                            break
                        yield cloud_speech.StreamingRecognizeRequest(audio=pcm16_data)
                    except queue.Empty:
                        continue

            responses_iterator = client.streaming_recognize(requests=audio_generator())
            for response in responses_iterator:
                self.logger.info(f"Streaming recognition response for channel {channel}: {response}")
                self.response_queues[channel].put(response)
        except Exception as e:
            self.logger.error(f"Streaming recognition error for channel {channel}: {e}")
            self.response_queues[channel].put(e)

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
    if not audio_stream:
        logger.warning("google_speech_transcription - No audio data received for transcription.")
        return {"transcript": "", "words": []}
    try:
        logger.info("Starting transcription process")
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

        def transcribe():
            if not GOOGLE_CLOUD_PROJECT:
                raise ValueError("GOOGLE_CLOUD_PROJECT not configured.")
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )
            # Explicit decoding config to match the raw PCM16 data we are passing.
            explicit_config = cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000,
                audio_channel_count=channels
            )

            # Build the recognition config
            config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=explicit_config,
                language_codes=[source_language],
                model=GOOGLE_SPEECH_MODEL,
                features=cloud_speech.RecognitionFeatures(
                    enable_word_time_offsets=True,
                    enable_word_confidence=True
                )
            )
            # If the source language is not English, add translation_config so that the transcript is translated to en-US.
            if source_language.lower() != "en-us":
                config.translation_config = cloud_speech.TranslationConfig(
                    target_language="en-US"
                )

            logger.debug(
                "google_speech_transcription - Sending recognition request with "
                f"LINEAR16 decoding, sample_rate=8000, channels={channels}, "
                f"language={source_language}, model={GOOGLE_SPEECH_MODEL}"
            )

            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
                config=config,
                content=pcm16_data,
            )
            start_time = time.time()
            response = client.recognize(request=request)
            duration = time.time() - start_time
            logger.info(f"Recognition API call took {duration:.3f} seconds")

            result_data = {}
            # Process the first result available.
            for result in response.results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    words_list = []
                    if hasattr(alt, "words") and alt.words:
                        for word in alt.words:
                            start = 0.0
                            end = 0.0
                            if word.start_offset is not None:
                                start = word.start_offset.total_seconds()
                            if word.end_offset is not None:
                                end = word.end_offset.total_seconds()
                            words_list.append({
                                "word": word.word,
                                "start_time": start,
                                "end_time": end,
                                "confidence": word.confidence if word.confidence is not None else 1.0
                            })
                    result_data = {
                        "transcript": alt.transcript,
                        "confidence": alt.confidence if alt.confidence is not None else 1.0,
                        "words": words_list
                    }
                    break
            if not result_data:
                logger.warning("google_speech_transcription - No transcript alternatives returned from recognition API")
                result_data = {"transcript": "", "words": []}
            return result_data

        result = await asyncio.to_thread(transcribe)
        logger.info(f"Transcription result: {result}")
        return result
    except Exception as e:
        logger.error(f"google_speech_transcription - Error during transcription: {e}", exc_info=True)
        return {"transcript": "", "words": []}

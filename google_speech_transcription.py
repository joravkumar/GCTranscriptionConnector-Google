import asyncio
import audioop
import os
import json
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

async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> str:
    if not audio_stream:
        logger.warning(f"google_speech_transcription - No audio data received for transcription.")
        return ""
    try:
        logger.debug(f"google_speech_transcription - Translating audio chunk of length {len(audio_stream)} bytes")
        # Determine number of channels from negotiated media; default to 1.
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1

        # Convert the incoming PCMU (u-law) data to PCM16.
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        logger.debug(
            f"google_speech_transcription - Converted PCMU to PCM16: {len(pcm16_data)} bytes, sample_width=2, "
            f"frame_rate=8000, channels={channels}"
        )

        # Extract the source language from negotiated_media if provided; default to "en-US".
        source_language_raw = "en-US"
        if negotiated_media and "language" in negotiated_media:
            source_language_raw = negotiated_media["language"]
        source_language = normalize_language_code(source_language_raw)
        logger.debug(f"google_speech_transcription - Source language determined as: {source_language}")

        def transcribe():
            if not GOOGLE_CLOUD_PROJECT:
                raise ValueError("GOOGLE_CLOUD_PROJECT not configured.")
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )
            # Build the explicit decoding configuration.
            explicit_config = cloud_speech.ExplicitDecodingConfig(
                encoding=1,  # LINEAR16
                sample_rate_hertz=8000,
                audio_channel_count=channels,
            )
            # Build the recognition configuration
            config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=explicit_config,
                language_codes=[source_language],
                model=GOOGLE_SPEECH_MODEL,
                features=cloud_speech.RecognitionFeatures(
                    enable_word_time_offsets=True
                )
            )
            # If the source language is not English, add translation_config so that the transcript is translated to en-US.
            if source_language.lower() != "en-us":
                config.translation_config = cloud_speech.TranslationConfig(target_language="en-US")
            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
                config=config,
                content=pcm16_data,
            )
            response = client.recognize(request=request)
            transcripts = []
            for result in response.results:
                if result.alternatives:
                    transcripts.append(result.alternatives[0].transcript)
            return " ".join(transcripts)

        transcript = await asyncio.to_thread(transcribe)
        logger.debug(f"google_speech_transcription - Received transcript: {transcript}")
        return transcript
    except Exception as e:
        logger.error(f"google_speech_transcription - Error during transcription: {e}", exc_info=True)
        return ""

import asyncio
import audioop
import os
import json
import hashlib
import time
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

async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> dict:
    if not audio_stream:
        logger.warning("google_speech_transcription - No audio data received for transcription.")
        return {"transcript": "", "words": []}
    try:
        logger.debug(f"google_speech_transcription - Translating audio chunk of length: {len(audio_stream)} bytes")
        # Determine number of channels from negotiated media; default to 1.
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1

        # Convert the incoming PCMU (u-law) data to PCM16.
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        logger.debug(f"google_speech_transcription - Converted PCMU to PCM16: {len(pcm16_data)} bytes, sample_width=2, frame_rate=8000, channels={channels}")
        # Compute RMS value for debugging to assess audio energy.
        rms_value = audioop.rms(pcm16_data, 2)
        logger.debug(f"google_speech_transcription - PCM16 RMS value: {rms_value}")

        # Compute MD5 hash of PCM16 data for correlation.
        hash_digest = hashlib.md5(pcm16_data).hexdigest()
        logger.debug(f"google_speech_transcription - PCM16 data MD5: {hash_digest}")

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
            # Use auto_decoding_config for auto-detection of audio parameters.
            config = cloud_speech.RecognitionConfig(
                auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                language_codes=[source_language],
                model=GOOGLE_SPEECH_MODEL,
                features=cloud_speech.RecognitionFeatures(
                    enable_word_time_offsets=True,
                    enable_word_confidence=True
                )
            )
            # If the source language is not English, add translation_config so that the transcript is translated to en-US.
            if source_language.lower() != "en-us":
                config.translation_config = cloud_speech.TranslationConfig(target_language="en-US")
            logger.debug(f"google_speech_transcription - Sending recognition request with auto_decoding_config, channels={channels}, language={source_language}, model={GOOGLE_SPEECH_MODEL}")
            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
                config=config,
                content=pcm16_data,
            )
            start_time = time.time()
            response = client.recognize(request=request)
            duration = time.time() - start_time
            logger.debug(f"google_speech_transcription - Recognition API call took {duration:.3f} seconds")
            result_data = {}
            # Process the first result available.
            for result in response.results:
                if result.alternatives:
                    alt = result.alternatives[0]
                    words_list = []
                    if hasattr(alt, "words") and alt.words:
                        for word in alt.words:
                            start = (word.start_time.seconds + word.start_time.nanos/1e9) if word.start_time else 0.0
                            end = (word.end_time.seconds + word.end_time.nanos/1e9) if word.end_time else 0.0
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
        logger.debug(f"google_speech_transcription - Received result: {result}")
        return result
    except Exception as e:
        logger.error(f"google_speech_transcription - Error during transcription: {e}", exc_info=True)
        return {"transcript": "", "words": []}

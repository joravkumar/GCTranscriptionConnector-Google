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
    GOOGLE_SPEECH_MODEL,
    GEMINI_API_KEY,
    GOOGLE_TRANSLATION_MODEL,
    GOOGLE_TRANSLATION_DEST_LANGUAGE
)
from google import genai

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
        self.channels = channels
        self.logger = logger
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.streaming_thread = None
        self.running = True

    def start_streaming(self):
        """Start the streaming recognition thread."""
        self.streaming_thread = threading.Thread(target=self.streaming_recognize_thread)
        self.streaming_thread.start()

    def stop_streaming(self):
        """Stop the streaming recognition thread and clean up."""
        self.running = False
        self.audio_queue.put(None)  # Signal the thread to exit
        if self.streaming_thread:
            self.streaming_thread.join()

    def streaming_recognize_thread(self):
        """Run the StreamingRecognize API call in a separate thread."""
        try:
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )
            # Log the language and channels for debugging
            self.logger.info(f"Setting up streaming recognition - Language: {self.language}, Channels: {self.channels}")
            recognition_config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=8000,
                    audio_channel_count=self.channels  # Set to 2 for stereo
                ),
                language_codes=[self.language],
                model=GOOGLE_SPEECH_MODEL,
                features=cloud_speech.RecognitionFeatures(
                    enable_word_time_offsets=True,
                    enable_word_confidence=True,
                    enable_automatic_punctuation=True,
                    enable_spoken_punctuation=True,
                    enable_spoken_emojis=True,
                    max_alternatives=1,
                    profanity_filter=False,
                    enable_separate_recognition_per_channel=True if self.channels > 1 else False
                )
            )

            # Streaming config with interim_results disabled so we only see final output
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
                        pcm16_data = self.audio_queue.get(timeout=0.1)
                        if pcm16_data is None:
                            break
                        yield cloud_speech.StreamingRecognizeRequest(audio=pcm16_data)
                    except queue.Empty:
                        continue

            # Transcribes the audio into text
            responses_iterator = client.streaming_recognize(requests=audio_generator())
            for response in responses_iterator:
                self.logger.debug(f"Streaming recognition response: {response}")
                self.response_queue.put(response)
        except Exception as e:
            self.logger.error(f"Streaming recognition error: {e}")
            self.response_queue.put(e)

    def feed_audio(self, audio_stream: bytes):
        """Feed audio data (PCMU) into the streaming queue after converting to PCM16."""
        if not audio_stream:
            return
        # Convert from PCMU (u-law) to PCM16
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        # Keep stereo audio as is, no mono conversion
        self.logger.debug(f"Converted PCMU to PCM16: {len(pcm16_data)} bytes")
        self.audio_queue.put(pcm16_data)

    def get_response(self):
        """Retrieve the next transcription response from the queue, if available."""
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None

async def translate_with_gemini(text: str, source_language: str, dest_language: str, logger) -> str:
    """Translate text using Gemini API with structured output."""
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        system_prompt = f"""
        You are a professional translator. Translate the following text from {source_language} to {dest_language}.
        Provide a professional and accurate translation. Respond with only the translated text, nothing else.

        Schema for the response:
        - Type: string
        - Description: The translated text in {dest_language}
        """
        response = client.models.generate_content(
            model=GOOGLE_TRANSLATION_MODEL,
            contents=[system_prompt, text],
            config={
                'response_mime_type': 'text/plain',
            }
        )
        translated_text = response.text.strip()
        logger.debug(f"Translated '{text}' to '{translated_text}'")
        return translated_text
    except Exception as e:
        logger.error(f"Error translating text with Gemini API: {e}")
        return text  # Fallback to original text if translation fails

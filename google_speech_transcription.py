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
    if '-' in lang:
        parts = lang.split('-')
        if len(parts) == 2:
            return f"{parts[0].lower()}-{parts[1].upper()}"
    return lang

try:
    credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
    _credentials = service_account.Credentials.from_service_account_info(credentials_info)
except Exception:
    _credentials = None

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.language = normalize_language_code(language)
        self.channels = channels
        self.logger = logger
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True
        self.lock = threading.Lock()

    def start_streaming(self):
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self.streaming_recognize_thread, args=(channel,)
            )
            self.streaming_threads[channel].start()

    def stop_streaming(self):
        self.running = False
        for channel in range(self.channels):
            with self.lock:
                self.audio_queues[channel].put(None)
        for channel in range(self.channels):
            if self.streaming_threads[channel]:
                self.streaming_threads[channel].join()

    def streaming_recognize_thread(self, channel):
        try:
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )
            features = cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True
            )
            if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2':
                features.enable_word_confidence = True
            recognition_config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=8000,
                    audio_channel_count=1
                ),
                language_codes=[self.language],
                model=GOOGLE_SPEECH_MODEL,
                features=features
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
                    with self.lock:
                        try:
                            pcm16_data = self.audio_queues[channel].get(timeout=0.1)
                            if pcm16_data is None:
                                break
                            yield cloud_speech.StreamingRecognizeRequest(audio=pcm16_data)
                        except queue.Empty:
                            continue
            responses_iterator = client.streaming_recognize(requests=audio_generator())
            for response in responses_iterator:
                with self.lock:
                    self.response_queues[channel].put(response)
        except Exception as e:
            with self.lock:
                self.response_queues[channel].put(e)

    def feed_audio(self, audio_stream: bytes, channel: int):
        if not audio_stream or channel >= self.channels or len(audio_stream) % 2 != 0:
            return
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        with self.lock:
            self.audio_queues[channel].put(pcm16_data)

    def get_response(self, channel: int):
        if channel >= self.channels:
            return None
        with self.lock:
            try:
                return self.response_queues[channel].get_nowait()
            except queue.Empty:
                return None

async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> dict:
    if not audio_stream:
        return {"transcript": "", "words": []}
    try:
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        rms_value = audioop.rms(pcm16_data, 2)
        if rms_value < 50:
            return {"transcript": "", "words": []}
        source_language_raw = "en-US"
        if negotiated_media and "language" in negotiated_media:
            source_language_raw = negotiated_media["language"]
        source_language = normalize_language_code(source_language_raw)
        def transcribe():
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )
            explicit_config = cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000,
                audio_channel_count=channels
            )
            features = cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True
            )
            if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2':
                features.enable_word_confidence = True
            config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=explicit_config,
                language_codes=[source_language],
                model=GOOGLE_SPEECH_MODEL,
                features=features
            )
            if source_language.lower() != "en-us":
                config.translation_config = cloud_speech.TranslationConfig(
                    target_language="en-US"
                )
            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
                config=config,
                content=pcm16_data,
            )
            response = client.recognize(request=request)
            result_data = {}
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
                            confidence = 1.0
                            if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2':
                                confidence = word.confidence if word.confidence is not None else 1.0
                            words_list.append({
                                "word": word.word,
                                "start_time": start,
                                "end_time": end,
                                "confidence": confidence
                            })
                    overall_confidence = 1.0
                    if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2':
                        overall_confidence = alt.confidence if alt.confidence is not None else 1.0
                    result_data = {
                        "transcript": alt.transcript,
                        "confidence": overall_confidence,
                        "words": words_list
                    }
                    break
            if not result_data:
                result_data = {"transcript": "", "words": []}
            return result_data
        result = await asyncio.to_thread(transcribe)
        return result
    except Exception:
        return {"transcript": "", "words": []}

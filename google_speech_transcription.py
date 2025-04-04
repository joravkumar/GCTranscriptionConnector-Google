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
    # Attempt to load credentials from JSON string first (e.g., env var)
    credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
    _credentials = service_account.Credentials.from_service_account_info(credentials_info)
except json.JSONDecodeError:
    # If not JSON, assume it's a file path
    _credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
except Exception as e:
     # Log error and potentially raise or handle gracefully
    print(f"Error loading Google credentials: {e}") # Use print as logger might not be setup yet
    _credentials = None # Ensure _credentials is None if loading fails


class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.language = normalize_language_code(language)
        self.channels = channels
        self.logger = logger
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = True
        self.recognizer_path = f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_"
        self.is_chirp_2 = GOOGLE_SPEECH_MODEL.lower() == 'chirp_2'

    def start_streaming(self):
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self.streaming_recognize_thread, args=(channel,)
            )
            self.streaming_threads[channel].daemon = True # Allow main thread to exit even if these error
            self.streaming_threads[channel].start()

    def stop_streaming(self):
        self.running = False
        for channel in range(self.channels):
            self.audio_queues[channel].put(None)
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                try:
                    self.streaming_threads[channel].join(timeout=1.0)
                    if self.streaming_threads[channel].is_alive():
                         self.logger.warning(f"Streaming thread for channel {channel} did not terminate gracefully.")
                except Exception as e:
                     self.logger.error(f"Error joining streaming thread for channel {channel}: {e}")

    def streaming_recognize_thread(self, channel):
        try:
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )

            features = cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True
            )

            if self.is_chirp_2:
                features.enable_word_confidence = True
                self.logger.info(f"Using Chirp 2 model with word-level confidence enabled for channel {channel}")
            else:
                self.logger.info(f"Using {GOOGLE_SPEECH_MODEL} model without word-level confidence for channel {channel} - will use default confidence")

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
                    enable_voice_activity_events=True # Get VAD events
                )
            )
            config_request = cloud_speech.StreamingRecognizeRequest(
                recognizer=self.recognizer_path,
                streaming_config=streaming_config,
            )

            def audio_generator():
                yield config_request
                while self.running:
                    try:
                        pcm16_data = self.audio_queues[channel].get(timeout=0.1)
                        if pcm16_data is None:
                            self.logger.debug(f"Audio queue for channel {channel} received None, ending generator.")
                            break
                        yield cloud_speech.StreamingRecognizeRequest(audio=pcm16_data)
                    except queue.Empty:
                        continue
                    except Exception as gen_e:
                         self.logger.error(f"Error in audio generator for channel {channel}: {gen_e}")
                         break # Exit generator on error
                self.logger.debug(f"Audio generator for channel {channel} finished.")


            self.logger.info(f"Starting Google streaming recognize for channel {channel}")
            responses_iterator = client.streaming_recognize(requests=audio_generator())

            for response in responses_iterator:
                 # Only put responses with actual results or VAD events
                if response.results or response.speech_event_type:
                    self.response_queues[channel].put(response)
                else:
                     self.logger.debug(f"Received empty response from Google on channel {channel}, discarding.")

            self.logger.info(f"Google streaming recognize loop finished for channel {channel}")

        except Exception as e:
            self.logger.error(f"Fatal streaming recognition error for channel {channel}: {e}", exc_info=True)
            self.response_queues[channel].put(e) # Signal error to the main loop
        finally:
             self.logger.info(f"Exiting streaming_recognize_thread for channel {channel}")


    def feed_audio(self, audio_stream: bytes, channel: int):
        if not self.running or not audio_stream or channel >= self.channels:
            return
        try:
            pcm16_data = audioop.ulaw2lin(audio_stream, 2)
            self.audio_queues[channel].put(pcm16_data)
        except Exception as e:
             self.logger.error(f"Error converting or queueing audio for channel {channel}: {e}")


    def get_response(self, channel: int):
        if channel >= self.channels:
            return None
        try:
            return self.response_queues[channel].get_nowait()
        except queue.Empty:
            return None
        except Exception as e:
            self.logger.error(f"Error getting response from queue for channel {channel}: {e}")
            return None


async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> dict:

    # This function appears to be for non-streaming, batch-like recognition.
    # It's kept for potential compatibility but the main focus is StreamingTranscription.
    # Confidence default updated here as well.

    if not audio_stream:
        logger.warning("google_speech_transcription (translate_audio) - No audio data received.")
        return {"transcript": "", "confidence": 0.99, "words": []}
    try:
        logger.info("Starting non-streaming transcription process (translate_audio)")
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0: channels = 1

        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        logger.info(f"(translate_audio) Converted PCMU to PCM16: {len(pcm16_data)} bytes, channels={channels}")
        rms_value = audioop.rms(pcm16_data, 2)
        logger.debug(f"(translate_audio) PCM16 RMS value: {rms_value}")

        if rms_value < 50:
            logger.info("(translate_audio) PCM16 RMS value below threshold, skipping transcription")
            return {"transcript": "", "confidence": 0.99, "words": []}

        source_language_raw = "en-US"
        if negotiated_media and "language" in negotiated_media:
            source_language_raw = negotiated_media["language"]
        source_language = normalize_language_code(source_language_raw)
        logger.info(f"(translate_audio) Source language determined as: {source_language}")

        is_chirp_2 = GOOGLE_SPEECH_MODEL.lower() == 'chirp_2'

        def transcribe():
            if not GOOGLE_CLOUD_PROJECT:
                raise ValueError("GOOGLE_CLOUD_PROJECT not configured.")

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

            if is_chirp_2:
                features.enable_word_confidence = True
                logger.info(f"(translate_audio) Using Chirp 2 model with word-level confidence enabled")
            else:
                logger.info(f"(translate_audio) Using {GOOGLE_SPEECH_MODEL} model without word-level confidence - will use default confidence")

            config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=explicit_config,
                language_codes=[source_language],
                model=GOOGLE_SPEECH_MODEL,
                features=features
            )
            # Note: translation_config is not used in the streaming part, only here.
            if source_language.lower() != "en-us":
                config.translation_config = cloud_speech.TranslationConfig(
                    target_language_codes=["en-US"] # API expects list
                )
                logger.info("(translate_audio) Added translation_config to en-US")


            logger.debug(
                "google_speech_transcription (translate_audio) - Sending recognition request with "
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
            logger.info(f"(translate_audio) Recognition API call took {duration:.3f} seconds")

            result_data = {}
            default_confidence = 0.99 # *** FIX: Default confidence changed to 0.99 ***

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

                            # Use word confidence if available (Chirp 2), otherwise default
                            confidence = default_confidence
                            if is_chirp_2 and hasattr(word, 'confidence') and word.confidence is not None and word.confidence > 0.0:
                                confidence = word.confidence

                            words_list.append({
                                "word": word.word,
                                "start_time": start,
                                "end_time": end,
                                "confidence": confidence
                            })

                    # Use overall confidence if available, otherwise default
                    overall_confidence = default_confidence
                    if hasattr(alt, 'confidence') and alt.confidence is not None and alt.confidence > 0.0:
                       # Per docs, overall confidence might not be reliable for Chirp models
                       # If we have word confidences, maybe averaging them is better?
                       if words_list and all('confidence' in w for w in words_list):
                           avg_word_conf = sum(w['confidence'] for w in words_list) / len(words_list)
                           overall_confidence = avg_word_conf
                           logger.debug(f"(translate_audio) Using average word confidence as overall: {overall_confidence:.4f}")
                       else:
                           overall_confidence = alt.confidence # Use API value if no word confidences
                    logger.debug(f"(translate_audio) Using final overall confidence: {overall_confidence:.4f}")


                    result_data = {
                        "transcript": alt.transcript,
                        "confidence": overall_confidence,
                        "words": words_list
                    }
                    break # Process only the first result

            if not result_data:
                logger.warning("google_speech_transcription (translate_audio) - No transcript alternatives returned")
                result_data = {"transcript": "", "confidence": default_confidence, "words": []}

            return result_data

        result = await asyncio.to_thread(transcribe)
        logger.info(f"(translate_audio) Transcription result: {result}")
        return result
    except Exception as e:
        logger.error(f"google_speech_transcription (translate_audio) - Error during transcription: {e}", exc_info=True)
        return {"transcript": "", "confidence": 0.99, "words": []} # Return default confidence on error

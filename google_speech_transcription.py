import asyncio
import audioop
import os
import json
import time
import threading
import queue
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.oauth2 import service_account
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPIError, Cancelled

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

_credentials = None
try:
    if GOOGLE_APPLICATION_CREDENTIALS:
        if os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
             _credentials = service_account.Credentials.from_service_account_file(GOOGLE_APPLICATION_CREDENTIALS)
        else:
             credentials_info = json.loads(GOOGLE_APPLICATION_CREDENTIALS)
             _credentials = service_account.Credentials.from_service_account_info(credentials_info)
    else:
         # Fallback to default credentials if GOOGLE_APPLICATION_CREDENTIALS is not set
         from google.auth import default
         _credentials, _ = default()
         if not _credentials:
              raise ValueError("Could not automatically determine Google Cloud credentials.")
except Exception as e:
    print(f"Error loading Google Cloud credentials: {e}")
    _credentials = None # Ensure it's None if loading fails

class StreamingTranscription:
    def __init__(self, language: str, channels: int, logger):
        self.language = normalize_language_code(language)
        self.channels = channels
        self.logger = logger.getChild(f"GoogleStream_{self.language}")
        self.audio_queues = [queue.Queue() for _ in range(channels)]
        self.response_queues = [queue.Queue() for _ in range(channels)]
        self.streaming_threads = [None] * channels
        self.running = threading.Event() # Use Event for thread-safe running flag
        self.running.set() # Start in running state
        self.sample_rate = 8000
        self.valid_samples_fed = [0] * channels # Track valid samples fed per channel

    def start_streaming(self):
        if not _credentials:
             self.logger.error("Google Cloud credentials not loaded. Cannot start streaming.")
             raise RuntimeError("Google Cloud credentials failed to load.")
        if not GOOGLE_CLOUD_PROJECT:
             self.logger.error("GOOGLE_CLOUD_PROJECT environment variable not set. Cannot start streaming.")
             raise RuntimeError("GOOGLE_CLOUD_PROJECT not set.")

        self.running.set()
        for channel in range(self.channels):
            self.streaming_threads[channel] = threading.Thread(
                target=self._streaming_recognize_thread, args=(channel,), daemon=True
            )
            self.streaming_threads[channel].start()
            self.logger.info(f"Started streaming thread for channel {channel}")

    def stop_streaming(self):
        self.logger.info("Stopping streaming threads.")
        self.running.clear() # Signal threads to stop
        for q in self.audio_queues:
            q.put(None) # Send sentinel to unblock threads waiting on queue
        for channel in range(self.channels):
            if self.streaming_threads[channel] and self.streaming_threads[channel].is_alive():
                self.logger.debug(f"Joining streaming thread for channel {channel}")
                self.streaming_threads[channel].join(timeout=2.0) # Add timeout
                if self.streaming_threads[channel].is_alive():
                    self.logger.warning(f"Streaming thread for channel {channel} did not exit cleanly.")
            self.streaming_threads[channel] = None
        self.logger.info("Streaming threads stopped.")

    def _streaming_recognize_thread(self, channel):
        self.logger.info(f"Streaming thread started for channel {channel}")
        client = None
        responses_iterator = None
        try:
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )

            features = cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_automatic_punctuation=True # Enable automatic punctuation
            )

            if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2':
                features.enable_word_confidence = True
                self.logger.info(f"Channel {channel}: Using Chirp 2 model with word-level confidence enabled.")
            else:
                self.logger.info(f"Channel {channel}: Using {GOOGLE_SPEECH_MODEL} model without word-level confidence - will use default confidence of 1.0.")

            recognition_config = cloud_speech.RecognitionConfig(
                # Use explicit_decoding_config for LINEAR16
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=self.sample_rate,
                    audio_channel_count=1 # Each stream processes one mono channel
                ),
                language_codes=[self.language],
                model=GOOGLE_SPEECH_MODEL,
                features=features
            )

            streaming_config = cloud_speech.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=cloud_speech.StreamingRecognitionFeatures(
                    interim_results=False, # Genesys expects final results
                    enable_voice_activity_events=False # VAD not needed here, handled externally if required
                )
            )
            config_request = cloud_speech.StreamingRecognizeRequest(
                recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
                streaming_config=streaming_config,
            )

            def audio_generator():
                yield config_request
                while self.running.is_set():
                    try:
                        audio_chunk_data = self.audio_queues[channel].get(timeout=0.1)
                        if audio_chunk_data is None: # Sentinel value received
                            self.logger.debug(f"Channel {channel}: Sentinel received, ending audio generator.")
                            break
                        pcm16_data, chunk_start_sample_offset = audio_chunk_data
                        yield cloud_speech.StreamingRecognizeRequest(audio=pcm16_data)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        self.logger.error(f"Channel {channel}: Error in audio generator: {e}")
                        break
                self.logger.debug(f"Channel {channel}: Audio generator finished.")


            self.logger.info(f"Channel {channel}: Starting Google streaming recognition request.")
            responses_iterator = client.streaming_recognize(requests=audio_generator())

            for response in responses_iterator:
                if not self.running.is_set():
                    self.logger.info(f"Channel {channel}: Stop signal received, exiting response loop.")
                    break

                # Retrieve the starting sample offset associated with this response
                # This is tricky as the API doesn't directly link responses to input chunks.
                # We rely on the order and the fact that get_response() retrieves the start offset.
                # For now, we pass the start offset along with the response in the queue.
                # This assumes the start offset retrieved by get_response corresponds to this response.
                # A more robust method might involve request IDs if the API supported it.
                # We will pass the start_sample_offset when putting into the response queue.
                # The caller (AudioHookServer) will retrieve it.

                self.logger.debug(f"Channel {channel}: Received response from Google API.")
                # Find the start offset associated with the *first* word in this response, if available
                current_chunk_start_offset = self.valid_samples_fed[channel] # Default assumption
                if response.results and response.results[0].alternatives:
                     alt = response.results[0].alternatives[0]
                     if hasattr(alt, 'words') and alt.words:
                          # This offset needs to be determined more accurately.
                          # How to link this response back to the specific audio chunk's start offset?
                          # The current implementation in feed_audio calculates it *before* queueing.
                          # Let's assume the get_response will fetch the correct offset.
                          # We will put (response, start_offset) tuple in the queue.
                          pass # Offset is handled when putting into queue

                self.response_queues[channel].put((response, current_chunk_start_offset)) # Put response and assumed offset

        except Cancelled:
             self.logger.info(f"Channel {channel}: Google API request cancelled.")
        except GoogleAPIError as e:
            self.logger.error(f"Channel {channel}: Google API error during streaming: {e}")
            self.response_queues[channel].put((e, 0)) # Put exception in queue with dummy offset
        except Exception as e:
            self.logger.error(f"Channel {channel}: Unexpected error in streaming thread: {e}", exc_info=True)
            self.response_queues[channel].put((e, 0)) # Put exception in queue
        finally:
            if responses_iterator:
                # Attempt to close the iterator gracefully if needed, though exiting the loop should handle it.
                pass
            self.logger.info(f"Streaming thread finished for channel {channel}")


    def feed_audio(self, audio_stream: bytes, channel: int):
        if not self.running.is_set():
             self.logger.warning(f"Channel {channel}: Received audio while not running, discarding.")
             return
        if not audio_stream or channel >= self.channels:
            self.logger.debug(f"Channel {channel}: Empty audio stream or invalid channel index.")
            return

        try:
            pcm16_data = audioop.ulaw2lin(audio_stream, 2)
            num_samples = len(pcm16_data) // 2

            # Calculate the starting sample offset *before* adding these samples
            chunk_start_sample_offset = self.valid_samples_fed[channel]

            # Update the total valid samples fed *after* getting the start offset
            self.valid_samples_fed[channel] += num_samples

            # Put the audio data and its starting offset into the queue
            self.audio_queues[channel].put((pcm16_data, chunk_start_sample_offset))
            self.logger.debug(f"Channel {channel}: Fed {len(pcm16_data)} bytes (starting at sample {chunk_start_sample_offset}) to audio queue.")

        except audioop.error as e:
             self.logger.error(f"Channel {channel}: audioop error converting PCMU to PCM16: {e}")
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error feeding audio: {e}")

    def get_response(self, channel: int):
        if channel >= self.channels:
            self.logger.warning(f"Invalid channel index {channel} requested in get_response.")
            return None
        try:
            # Returns tuple: (response_object_or_exception, start_sample_offset)
            response_data = self.response_queues[channel].get_nowait()
            self.logger.debug(f"Channel {channel}: Retrieved response from queue (started at sample {response_data[1]})")
            return response_data
        except queue.Empty:
            return None
        except Exception as e:
             self.logger.error(f"Channel {channel}: Error getting response from queue: {e}")
             return None

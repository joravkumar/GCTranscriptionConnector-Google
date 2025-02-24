import asyncio
import uuid
import json
import time
import websockets
import tempfile
import audioop
import logging
from pydub import AudioSegment
from websockets.exceptions import ConnectionClosed
import queue

from config import (
    RATE_LIMIT_MAX_RETRIES,
    GENESYS_MSG_RATE_LIMIT,
    GENESYS_BINARY_RATE_LIMIT,
    GENESYS_MSG_BURST_LIMIT,
    GENESYS_BINARY_BURST_LIMIT,
    MAX_AUDIO_BUFFER_SIZE,
    SUPPORTED_LANGUAGES
)
from rate_limiter import RateLimiter
from utils import format_json, parse_iso8601_duration
from google_speech_transcription import normalize_language_code

from collections import deque
logger = logging.getLogger("AudioHookServer")

class AudioHookServer:
    def __init__(self, websocket):
        self.session_id = str(uuid.uuid4())
        self.ws = websocket
        self.client_seq = 0
        self.server_seq = 0
        self.running = True
        self.negotiated_media = None
        self.start_time = time.time()
        self.logger = logger.getChild(f"AudioHookServer_{self.session_id}")
        self.audio_frames_sent = 0
        self.audio_frames_received = 0
        self.rate_limit_state = {
            "retry_count": 0,
            "last_retry_time": 0,
            "in_backoff": False
        }

        self.message_limiter = RateLimiter(GENESYS_MSG_RATE_LIMIT, GENESYS_MSG_BURST_LIMIT)
        self.binary_limiter = RateLimiter(GENESYS_BINARY_RATE_LIMIT, GENESYS_BINARY_BURST_LIMIT)

        self.audio_buffer = deque(maxlen=MAX_AUDIO_BUFFER_SIZE)
        self.last_frame_time = 0

        # New: total samples processed for offset calculation (kept for compatibility)
        self.total_samples = 0

        # New: store conversation language from open message
        self.conversation_language = "en-US"

        # New: queue for streaming audio frames and event loop reference
        self.audio_queue = queue.Queue()
        self.loop = asyncio.get_event_loop()

        self.logger.info(f"New session started: {self.session_id}")

    async def handle_error(self, msg: dict):
        error_code = msg["parameters"].get("code")
        error_params = msg["parameters"]

        if error_code == 429:
            retry_after = None

            if "retryAfter" in error_params:
                retry_after_duration = error_params["retryAfter"]
                try:
                    retry_after = parse_iso8601_duration(retry_after_duration)
                    self.logger.info(
                        f"[Rate Limit] Using Genesys-provided retryAfter duration: {retry_after}s "
                        f"(parsed from {retry_after_duration})"
                    )
                except ValueError as e:
                    self.logger.warning(
                        f"[Rate Limit] Failed to parse Genesys retryAfter format: {retry_after_duration}. "
                        f"Error: {str(e)}"
                    )

            if retry_after is None and hasattr(self.ws, 'response_headers'):
                http_retry_after = (
                    self.ws.response_headers.get('Retry-After') or 
                    self.ws.response_headers.get('retry-after')
                )
                if http_retry_after:
                    try:
                        retry_after = float(http_retry_after)
                        self.logger.info(
                            f"[Rate Limit] Using HTTP header Retry-After duration: {retry_after}s"
                        )
                    except ValueError:
                        try:
                            retry_after = parse_iso8601_duration(http_retry_after)
                            self.logger.info(
                                f"[Rate Limit] Using HTTP header Retry-After duration: {retry_after}s "
                                f"(parsed from ISO8601)"
                            )
                        except ValueError:
                            self.logger.warning(
                                f"[Rate Limit] Failed to parse HTTP Retry-After format: {http_retry_after}"
                            )

            self.logger.warning(
                f"[Rate Limit] Received 429 error. "
                f"Session: {self.session_id}, "
                f"Current duration: {time.time() - self.start_time:.2f}s, "
                f"Retry count: {self.rate_limit_state['retry_count']}, "
                f"RetryAfter: {retry_after}s"
            )

            self.rate_limit_state["in_backoff"] = True
            self.rate_limit_state["retry_count"] += 1

            if self.rate_limit_state["retry_count"] > RATE_LIMIT_MAX_RETRIES:
                self.logger.error(
                    f"[Rate Limit] Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded. "
                    f"Session: {self.session_id}, "
                    f"Total retries: {self.rate_limit_state['retry_count']}, "
                    f"Duration: {time.time() - self.start_time:.2f}s"
                )
                await self.disconnect_session(reason="error", info="Rate limit max retries exceeded")
                return False

            self.logger.warning(
                f"[Rate Limit] Rate limited, attempt {self.rate_limit_state['retry_count']}/{RATE_LIMIT_MAX_RETRIES}. "
                f"Backing off for {retry_after if retry_after is not None else 'default delay'}s. "
                f"Session: {self.session_id}, "
                f"Duration: {time.time() - self.start_time:.2f}s"
            )

            await asyncio.sleep(retry_after if retry_after is not None else 3)
            self.rate_limit_state["in_backoff"] = False
            self.logger.info(
                f"[Rate Limit] Backoff complete, resuming operations. "
                f"Session: {self.session_id}"
            )

            return True
        return False

    async def handle_message(self, msg: dict):
        msg_type = msg.get("type")
        seq = msg.get("seq", 0)
        self.client_seq = seq

        if self.rate_limit_state.get("in_backoff") and msg_type != "error":
            self.logger.debug(f"Skipping message type {msg_type} during rate limit backoff")
            return

        if msg_type == "error":
            handled = await self.handle_error(msg)
            if handled:
                return

        if msg_type == "open":
            await self.handle_open(msg)
        elif msg_type == "ping":
            await self.handle_ping(msg)
        elif msg_type == "close":
            await self.handle_close(msg)
        elif msg_type in ["update", "resume", "pause"]:
            self.logger.debug(f"Ignoring message of type {msg_type}")
        else:
            self.logger.debug(f"Ignoring unknown message type: {msg_type}")

    async def handle_open(self, msg: dict):
        self.session_id = msg["id"]

        # Capture conversation language from the open message parameters
        if "language" in msg["parameters"]:
            self.conversation_language = normalize_language_code(msg["parameters"]["language"])
        else:
            self.conversation_language = "en-US"

        is_probe = (
            msg["parameters"].get("conversationId") == "00000000-0000-0000-0000-000000000000" and
            msg["parameters"].get("participant", {}).get("id") == "00000000-0000-0000-0000-000000000000"
        )

        if is_probe:
            self.logger.info("Detected probe connection")
            # Prepare supported languages list from the comma-separated config value
            supported_langs = [lang.strip() for lang in SUPPORTED_LANGUAGES.split(",")]
            opened_msg = {
                "version": "2",
                "type": "opened",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {
                    "startPaused": False,
                    "media": [],
                    "supportedLanguages": supported_langs
                }
            }
            self.server_seq += 1
            await self._send_json(opened_msg)
            return

        offered_media = msg["parameters"].get("media", [])
        chosen = None
        for m in offered_media:
            if (m.get("format") == "PCMU" and m.get("rate") == 8000):
                chosen = m
                break

        if not chosen:
            resp = {
                "version": "2",
                "type": "disconnect",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {
                    "reason": "error",
                    "info": "No supported format found"
                }
            }
            self.server_seq += 1
            await self._send_json(resp)
            self.running = False
            return

        self.negotiated_media = chosen

        opened_msg = {
            "version": "2",
            "type": "opened",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "startPaused": False,
                "media": [chosen]
            }
        }
        self.server_seq += 1
        await self._send_json(opened_msg)
        self.logger.info(f"Session opened. Negotiated media format: {chosen}")
        # Start streaming recognition after session is opened
        self.streaming_task = asyncio.create_task(self.start_streaming_recognition())

    async def handle_ping(self, msg: dict):
        pong_msg = {
            "version": "2",
            "type": "pong",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {}
        }
        self.server_seq += 1
        try:
            await asyncio.wait_for(self._send_json(pong_msg), timeout=1.0)
        except asyncio.TimeoutError:
            self.logger.error("Failed to send pong response within timeout")

    async def handle_close(self, msg: dict):
        self.logger.info(f"Received 'close' from Genesys. Reason: {msg['parameters'].get('reason')}")
        
        closed_msg = {
            "version": "2",
            "type": "closed",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "summary": ""
            }
        }
        self.server_seq += 1
        try:
            await asyncio.wait_for(self._send_json(closed_msg), timeout=4.0)
        except asyncio.TimeoutError:
            self.logger.error("Failed to send closed response within timeout")

        duration = time.time() - self.start_time
        self.logger.info(
            f"Session stats - Duration: {duration:.2f}s, "
            f"Frames sent: {self.audio_frames_sent}, "
            f"Frames received: {self.audio_frames_received}"
        )

        # Stop streaming recognition by sending sentinel to queue
        self.audio_queue.put(None)
        self.running = False

    async def disconnect_session(self, reason="completed", info=""):
        try:
            if not self.session_id:
                return

            disconnect_msg = {
                "version": "2", 
                "type": "disconnect",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {
                    "reason": reason,
                    "info": info,
                    "outputVariables": {}
                }
            }
            self.server_seq += 1
            await asyncio.wait_for(self._send_json(disconnect_msg), timeout=5.0)
            try:
                await asyncio.wait_for(self.ws.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Client did not acknowledge disconnect for session {self.session_id}")
        except Exception as e:
            self.logger.error(f"Error in disconnect_session: {e}")
        finally:
            self.running = False

    async def handle_audio_frame(self, frame_bytes: bytes):
        self.audio_frames_received += 1
        self.logger.debug(f"Received audio frame from Genesys: {len(frame_bytes)} bytes (frame #{self.audio_frames_received})")
        # Convert to PCM16 and put into queue for streaming recognition
        pcm16_data = audioop.ulaw2lin(frame_bytes, 2)
        samples = len(pcm16_data) // 2
        self.total_samples += samples  # Keep track of total samples for compatibility
        self.audio_queue.put(pcm16_data)
        # Optionally keep audio_buffer for debugging
        self.audio_buffer.append(frame_bytes)

    async def start_streaming_recognition(self):
        await asyncio.to_thread(self.run_streaming_recognition)

    def run_streaming_recognition(self):
        from google.cloud.speech_v2 import SpeechClient
        from google.cloud.speech_v2.types import cloud_speech
        from google.api_core.client_options import ClientOptions
        from config import GOOGLE_CLOUD_PROJECT, GOOGLE_SPEECH_MODEL, _credentials

        try:
            client = SpeechClient(
                credentials=_credentials,
                client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com")
            )
            requests = self.audio_stream()
            self.logger.info("Starting streaming recognition with Google Speech-to-Text API")
            responses = client.streaming_recognize(requests=requests)
            for response in responses:
                for result in response.results:
                    transcript = result.alternatives[0].transcript.strip()
                    confidence = result.alternatives[0].confidence
                    is_final = result.is_final
                    words = [
                        {
                            "word": word.word,
                            "start_time": word.start_time.total_seconds(),
                            "end_time": word.end_time.total_seconds(),
                            "confidence": word.confidence
                        }
                        for word in result.alternatives[0].words
                    ]
                    # Calculate offset and duration from word timestamps
                    if words:
                        start_time = words[0]["start_time"]
                        end_time = words[-1]["end_time"]
                        offset = f"PT{start_time:.2f}S"
                        duration = f"PT{(end_time - start_time):.2f}S"
                    else:
                        offset = f"PT0.00S"
                        duration = f"PT0.00S"

                    # Filter out low-confidence or arbitrary transcriptions
                    arbitrary_words = {"sí", "mhm"}
                    if transcript.lower() in arbitrary_words and confidence < 0.7:
                        self.logger.debug(f"Skipping low-confidence arbitrary transcript: {transcript} (confidence: {confidence})")
                        continue

                    self.logger.info(f"Streaming transcript obtained: {transcript} (confidence: {confidence}, isFinal: {is_final})")
                    future = self.loop.run_coroutine_threadsafe(
                        self.send_transcript_event(transcript, confidence, is_final, offset, duration, words)
                    )
        except Exception as e:
            self.logger.error(f"Error in streaming recognition: {e}", exc_info=True)
            future = self.loop.run_coroutine_threadsafe(
                self.disconnect_session(reason="error", info="Streaming recognition error")
            )
        finally:
            self.logger.info("Streaming recognition stopped")

    def audio_stream(self):
        from google.cloud.speech_v2.types import cloud_speech
        from config import GOOGLE_CLOUD_PROJECT, GOOGLE_SPEECH_MODEL

        source_language = self.conversation_language
        channels = 1
        if self.negotiated_media and "channels" in self.negotiated_media:
            channels = len(self.negotiated_media["channels"])
            if channels == 0:
                channels = 1

        explicit_config = cloud_speech.ExplicitDecodingConfig(
            encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=8000,
            audio_channel_count=channels
        )
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=explicit_config,
            language_codes=[source_language],
            model=GOOGLE_SPEECH_MODEL,
            features=cloud_speech.RecognitionFeatures(
                enable_word_time_offsets=True,
                enable_word_confidence=True
            )
        )
        if source_language.lower() != "en-us":
            recognition_config.translation_config = cloud_speech.TranslationConfig(target_language="en-US")
        streaming_config = cloud_speech.StreamingRecognitionConfig(
            config=recognition_config,
            streaming_features=cloud_speech.StreamingRecognitionFeatures(
                interim_results=True
            )
        )
        config_request = cloud_speech.StreamingRecognizeRequest(
            recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
            streaming_config=streaming_config,
        )
        yield config_request

        self.logger.debug("Starting to yield audio frames from queue")
        while True:
            pcm16_data = self.audio_queue.get()
            if pcm16_data is None:
                self.logger.debug("Received sentinel, stopping audio stream")
                break
            yield cloud_speech.StreamingRecognizeRequest(audio=pcm16_data)
            self.audio_queue.task_done()

    async def send_transcript_event(self, transcript, confidence, is_final, offset, duration, words):
        tokens = []
        for word_info in words:
            token_duration = word_info["end_time"] - word_info["start_time"]
            tokens.append({
                "type": "word",
                "value": word_info["word"],
                "confidence": word_info["confidence"],
                "offset": f"PT{word_info['start_time']:.2f}S",
                "duration": f"PT{token_duration:.2f}S",
                "language": self.conversation_language
            })
        transcript_event = {
            "version": "2",
            "type": "event",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "entities": [
                    {
                        "type": "transcript",
                        "data": {
                            "id": str(uuid.uuid4()),
                            "channelId": 0,
                            "isFinal": is_final,
                            "offset": offset,
                            "duration": duration,
                            "alternatives": [
                                {
                                    "confidence": confidence,
                                    "languages": [self.conversation_language],
                                    "interpretations": [
                                        {
                                            "type": "display",
                                            "transcript": transcript,
                                            "tokens": tokens
                                        }
                                    ]
                                }
                            ]
                        }
                    }
                ]
            }
        }
        self.server_seq += 1
        self.logger.debug(f"Sending transcript event to Genesys: {transcript} (isFinal: {is_final})")
        await self._send_json(transcript_event)

    async def _send_json(self, msg: dict):
        try:
            if not await self.message_limiter.acquire():
                current_rate = self.message_limiter.get_current_rate()
                self.logger.warning(
                    f"Message rate limit exceeded (current rate: {current_rate:.2f}/s). "
                    f"Message type: {msg.get('type')}. Dropping to maintain compliance."
                )
                return

            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
        except ConnectionClosed:
            self.logger.warning("Genesys WebSocket closed while sending JSON message.")
            self.running = False

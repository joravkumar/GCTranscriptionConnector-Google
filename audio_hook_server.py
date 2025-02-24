import asyncio
import uuid
import json
import time
import websockets
import tempfile
import audioop
import logging
from websockets.exceptions import ConnectionClosed

from config import (
    RATE_LIMIT_MAX_RETRIES,
    GENESYS_MSG_RATE_LIMIT,
    GENESYS_BINARY_RATE_LIMIT,
    GENESYS_MSG_BURST_LIMIT,
    GENESYS_BINARY_BURST_LIMIT,
    MAX_AUDIO_BUFFER_SIZE,
    SUPPORTED_LANGUAGES,
    GOOGLE_TRANSLATION_DEST_LANGUAGE
)
from rate_limiter import RateLimiter
from utils import format_json, parse_iso8601_duration
from google_speech_transcription import StreamingTranscription, normalize_language_code
from google_gemini_translation import translate_with_gemini

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

        # Total samples processed for offset calculation
        self.total_samples = 0

        # Store conversation language from open message
        self.conversation_language = "en-US"

        # Streaming transcription
        self.streaming_transcription = None
        self.process_responses_task = None

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
        self.logger.info(f"Conversation language set to: {self.conversation_language}")

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
            if await self._send_json(opened_msg):
                self.server_seq += 1
            else:
                await self.disconnect_session(reason="error", info="Failed to send opened message")
                return
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
            if await self._send_json(resp):
                self.server_seq += 1
            else:
                await self.disconnect_session(reason="error", info="Failed to send disconnect message")
                return
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
        if await self._send_json(opened_msg):
            self.server_seq += 1
        else:
            await self.disconnect_session(reason="error", info="Failed to send opened message")
            return
        self.logger.info(f"Session opened. Negotiated media format: {chosen}")

        # Set channels to 2 for stereo audio (two participants)
        channels = 2

        # Initialize and start streaming transcription
        self.streaming_transcription = StreamingTranscription(self.conversation_language, channels, self.logger)
        self.streaming_transcription.start_streaming()
        self.process_responses_task = asyncio.create_task(self.process_transcription_responses())

    async def handle_ping(self, msg: dict):
        pong_msg = {
            "version": "2",
            "type": "pong",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {}
        }
        if await self._send_json(pong_msg):
            self.server_seq += 1
        else:
            self.logger.error("Failed to send pong response")
            await self.disconnect_session(reason="error", info="Failed to send pong message")

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
        if await self._send_json(closed_msg):
            self.server_seq += 1
        else:
            self.logger.error("Failed to send closed response")
            await self.disconnect_session(reason="error", info="Failed to send closed message")

        duration = time.time() - self.start_time
        self.logger.info(
            f"Session stats - Duration: {duration:.2f}s, "
            f"Frames sent: {self.audio_frames_sent}, "
            f"Frames received: {self.audio_frames_received}"
        )

        self.running = False
        if self.streaming_transcription:
            self.streaming_transcription.stop_streaming()
        if self.process_responses_task:
            self.process_responses_task.cancel()

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
            if await self._send_json(disconnect_msg):
                self.server_seq += 1
            else:
                self.logger.error("Failed to send disconnect message")
            try:
                await asyncio.wait_for(self.ws.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Client did not acknowledge disconnect for session {self.session_id}")
        except Exception as e:
            self.logger.error(f"Error in disconnect_session: {e}")
        finally:
            self.running = False
            if self.streaming_transcription:
                self.streaming_transcription.stop_streaming()
            if self.process_responses_task:
                self.process_responses_task.cancel()

    async def handle_audio_frame(self, frame_bytes: bytes):
        self.audio_frames_received += 1
        self.logger.debug(f"Received audio frame from Genesys: {len(frame_bytes)} bytes (frame #{self.audio_frames_received})")

        # Assume 2 channels for stereo audio
        channels = 2
        sample_times = len(frame_bytes) // channels
        self.total_samples += sample_times

        # Pass raw PCMU to the transcription logic
        if self.streaming_transcription:
            self.streaming_transcription.feed_audio(frame_bytes)

        self.audio_buffer.append(frame_bytes)

    async def process_transcription_responses(self):
        while self.running:
            self.logger.debug("Checking for transcription responses")
            response = self.streaming_transcription.get_response()
            if response:
                self.logger.debug(f"Processing transcription response: {response}")
                if isinstance(response, Exception):
                    self.logger.error(f"Streaming recognition error: {response}")
                    await self.disconnect_session(reason="error", info="Streaming recognition failed")
                    break
                for result in response.results:
                    if not result.alternatives:
                        continue
                    alt = result.alternatives[0]
                    transcript_text = alt.transcript
                    is_final = result.is_final
                    channel_tag = result.channel_tag if hasattr(result, 'channel_tag') else 0

                    # Translate the transcript using Gemini API
                    translated_text = await translate_with_gemini(
                        transcript_text,
                        self.conversation_language,
                        GOOGLE_TRANSLATION_DEST_LANGUAGE,
                        self.logger
                    )

                    self.logger.info(f"Transcript text: {translated_text} (is_final={is_final}, channel={channel_tag})")
                    if is_final and alt.words:
                        words_joined = " ".join([w.word for w in alt.words])
                        self.logger.info(f"Transcribed words: {words_joined}")

                    if is_final and alt.words:
                        words = []
                        for word_info in alt.words:
                            start_time = word_info.start_offset.total_seconds()
                            end_time = word_info.end_offset.total_seconds()
                            words.append({
                                "type": "word",
                                "value": word_info.word,
                                "confidence": word_info.confidence,
                                "offset": f"PT{start_time:.2f}S",
                                "duration": f"PT{(end_time - start_time):.2f}S",
                                "language": GOOGLE_TRANSLATION_DEST_LANGUAGE
                            })
                        offset = words[0]["offset"] if words else "PT0S"
                        duration = f"PT{(end_time - start_time):.2f}S" if words else "PT0S"
                    else:
                        current_offset = self.total_samples / 8000.0
                        offset = f"PT{current_offset:.2f}S"
                        duration = "PT0S"
                        words = None

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
                                        "channelId": channel_tag,  # Assign to correct participant
                                        "isFinal": is_final,
                                        "offset": offset,
                                        "duration": duration,
                                        "alternatives": [
                                            {
                                                "confidence": alt.confidence if is_final else 1.0,
                                                "languages": [GOOGLE_TRANSLATION_DEST_LANGUAGE],
                                                "interpretations": [
                                                    {
                                                        "type": "display",
                                                        "transcript": translated_text,
                                                        "tokens": words if words else []
                                                    }
                                                ]
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                    if await self._send_json(transcript_event):
                        self.server_seq += 1
                    else:
                        self.logger.debug("Transcript event dropped due to rate limiting")
            else:
                await asyncio.sleep(0.01)

    async def _send_json(self, msg: dict):
        try:
            if not await self.message_limiter.acquire():
                current_rate = self.message_limiter.get_current_rate()
                self.logger.warning(
                    f"Message rate limit exceeded (current rate: {current_rate:.2f}/s). "
                    f"Message type: {msg.get('type')}. Dropping to maintain compliance."
                )
                return False  # Message not sent

            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
            return True  # Message sent
        except ConnectionClosed:
            self.logger.warning("Genesys WebSocket closed while sending JSON message.")
            self.running = False
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

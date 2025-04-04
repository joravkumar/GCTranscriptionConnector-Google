import asyncio
import uuid
import json
import time
import websockets
import tempfile
import audioop
import logging
import importlib
from websockets.exceptions import ConnectionClosed, ConnectionClosedOK

from config import (
    RATE_LIMIT_MAX_RETRIES,
    GENESYS_MSG_RATE_LIMIT,
    GENESYS_BINARY_RATE_LIMIT,
    GENESYS_MSG_BURST_LIMIT,
    GENESYS_BINARY_BURST_LIMIT,
    MAX_AUDIO_BUFFER_SIZE,
    SUPPORTED_LANGUAGES,
    DEFAULT_SPEECH_PROVIDER
)
from rate_limiter import RateLimiter
from utils import format_json, parse_iso8601_duration

from language_mapping import normalize_language_code
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

        self.total_samples_received = 0
        self.is_paused_by_server = False
        self.is_paused_by_client = False

        self.input_language = "en-US"
        self.destination_language = "en-US"

        self.enable_translation = False

        self.speech_provider = DEFAULT_SPEECH_PROVIDER

        self.StreamingTranscription = None

        self.streaming_transcriptions = []
        self.process_responses_tasks = []

        self.logger.info(f"New session started: {self.session_id}")

    def _load_transcription_provider(self, provider_name=None):
        provider = provider_name or self.speech_provider
        provider = provider.lower()

        try:
            if provider == 'openai':
                module = importlib.import_module('openai_speech_transcription')
            else:
                module = importlib.import_module('google_speech_transcription')

            self.StreamingTranscription = module.StreamingTranscription
            self.logger.info(f"Loaded transcription provider: {provider}")
        except ImportError as e:
            self.logger.error(f"Failed to load transcription provider '{provider}': {e}")
            if provider != 'google':
                self.logger.warning(f"Falling back to Google transcription provider")
                self._load_transcription_provider('google')
            else:
                raise

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
                f"Backing off for {retry_after if retry_after is not None else 3}s. "
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
        else:
            self.logger.error(f"Received error message: {json.dumps(msg)}")
            # Potentially disconnect based on other error codes if needed
            return False # Indicate not handled specifically as rate limit

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
            else:
                # If error wasn't a handled 429, maybe disconnect?
                # await self.disconnect_session(reason="error", info=f"Received unhandled error: {msg.get('parameters', {}).get('info', 'Unknown error')}")
                self.logger.warning(f"Unhandled error message received: {msg}")
                return

        if msg_type == "open":
            await self.handle_open(msg)
        elif msg_type == "ping":
            await self.handle_ping(msg)
        elif msg_type == "close":
            await self.handle_close(msg)
        elif msg_type == "discarded":
            await self.handle_discarded(msg)
        elif msg_type == "paused":
            await self.handle_paused(msg)
        elif msg_type == "resumed":
            await self.handle_resumed(msg)
        elif msg_type == "pause": # Server requests pause
            await self.handle_server_pause_request(msg)
        elif msg_type == "resume": # Server requests resume
            await self.handle_server_resume_request(msg)
        elif msg_type in ["update"]:
            await self.handle_update(msg)
        else:
            self.logger.debug(f"Ignoring unknown message type: {msg_type}")

    async def handle_discarded(self, msg: dict):
        discarded_duration_str = msg["parameters"].get("discarded")
        if discarded_duration_str:
            try:
                gap = parse_iso8601_duration(discarded_duration_str)
                self.logger.info(f"Received 'discarded' message: Client indicates {gap:.2f}s of audio was lost before this point. Server timeline is unaffected.")
                # No offset adjustment needed on the server side. The client's 'position' header tracks its timeline.
            except ValueError as e:
                self.logger.warning(f"Failed to parse discarded duration '{discarded_duration_str}': {e}")
        else:
            self.logger.warning("Received 'discarded' message without 'discarded' parameter.")

    async def handle_paused(self, msg: dict):
        # This message indicates the client has paused sending audio.
        self.is_paused_by_client = True
        self.logger.info("Received 'paused' message from client: Client has paused sending audio.")
        # No offset adjustment needed. We just stop receiving audio.

    async def handle_resumed(self, msg: dict):
        # This message indicates the client has resumed sending audio.
        self.is_paused_by_client = False
        self.logger.info("Received 'resumed' message from client: Client has resumed sending audio.")
        # No offset adjustment needed. We just start receiving audio again.

    async def handle_server_pause_request(self, msg: dict):
        # Server requests to pause the stream
        self.is_paused_by_server = True
        self.logger.info("Received 'pause' request from server. Acknowledging and pausing.")
        paused_ack = {
            "version": "2",
            "type": "paused",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {}
        }
        if await self._send_json(paused_ack):
            self.server_seq += 1
        else:
            self.logger.error("Failed to send paused acknowledgement")
            await self.disconnect_session(reason="error", info="Failed to send paused acknowledgement")

    async def handle_server_resume_request(self, msg: dict):
        # Server requests to resume the stream
        self.is_paused_by_server = False
        self.logger.info("Received 'resume' request from server.")
        if self.is_paused_by_client:
            # If client is also paused, acknowledge server resume but indicate still paused
            self.logger.info("Client is still paused, acknowledging server resume but remaining paused.")
            paused_ack = {
                "version": "2",
                "type": "paused",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {}
            }
            if await self._send_json(paused_ack):
                self.server_seq += 1
            else:
                self.logger.error("Failed to send paused acknowledgement (during server resume)")
                await self.disconnect_session(reason="error", info="Failed to send paused acknowledgement")
        else:
            # If client is not paused, send resumed message
            self.logger.info("Client is not paused, sending resumed acknowledgement.")
            resumed_ack = {
                "version": "2",
                "type": "resumed",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {}
            }
            if await self._send_json(resumed_ack):
                self.server_seq += 1
            else:
                self.logger.error("Failed to send resumed acknowledgement")
                await self.disconnect_session(reason="error", info="Failed to send resumed acknowledgement")

    async def handle_update(self, msg: dict):
        self.logger.info(f"Received 'update' message: {json.dumps(msg['parameters'])}")
        if "language" in msg["parameters"]:
            new_lang = normalize_language_code(msg["parameters"]["language"])
            if new_lang != self.input_language:
                self.logger.warning(f"Language update requested from {self.input_language} to {new_lang}. Restarting transcription streams.")
                self.input_language = new_lang
                # Stop existing transcription tasks/threads
                for task in self.process_responses_tasks:
                    task.cancel()
                for transcription in self.streaming_transcriptions:
                    transcription.stop_streaming()

                # Re-initialize and start new transcription instances/tasks
                channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
                if channels == 0: channels = 1
                self.streaming_transcriptions = [self.StreamingTranscription(self.input_language, 1, self.logger) for _ in range(channels)]
                for transcription in self.streaming_transcriptions:
                    transcription.start_streaming()
                self.process_responses_tasks = [asyncio.create_task(self.process_transcription_responses(channel)) for channel in range(channels)]
                self.logger.info(f"Transcription streams restarted for new language: {self.input_language}")
            else:
                self.logger.info(f"Language update received but language ({new_lang}) is unchanged.")
        # Handle other updatable parameters here if needed in the future

    async def handle_open(self, msg: dict):
        self.session_id = msg["id"]

        custom_config = msg["parameters"].get("customConfig", {})

        self.input_language = normalize_language_code(custom_config.get("inputLanguage", "en-US"))

        self.enable_translation = custom_config.get("enableTranslation", False)

        self.destination_language = normalize_language_code(msg["parameters"].get("language", "en-US"))

        self.speech_provider = custom_config.get("transcriptionVendor", DEFAULT_SPEECH_PROVIDER)

        self._load_transcription_provider(self.speech_provider)

        is_probe = (
            msg["parameters"].get("conversationId") == "00000000-0000-0000-0000-000000000000" and
            msg["parameters"].get("participant", {}).get("id") == "00000000-0000-0000-0000-000000000000"
        )

        supported_langs = [lang.strip() for lang in SUPPORTED_LANGUAGES.split(",")]

        if is_probe:
            self.logger.info("Detected probe connection")
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
                await self.disconnect_session(reason="error", info="Failed to send opened message for probe")
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
                    "info": "No supported format found (PCMU @ 8000 Hz required)"
                }
            }
            if await self._send_json(resp):
                self.server_seq += 1
            else:
                self.logger.error("Failed to send disconnect message (no supported format)")
            self.running = False
            return

        self.negotiated_media = chosen
        start_paused = custom_config.get("startPaused", False)
        self.is_paused_by_server = start_paused # Initialize server pause state

        opened_params = {
            "startPaused": start_paused,
            "media": [chosen]
        }
        if msg["parameters"].get("supportedLanguages") is True:
             opened_params["supportedLanguages"] = supported_langs

        opened_msg = {
            "version": "2",
            "type": "opened",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": opened_params
        }

        if await self._send_json(opened_msg):
            self.server_seq += 1
        else:
            await self.disconnect_session(reason="error", info="Failed to send opened message")
            return
        self.logger.info(f"Session opened. Negotiated media format: {chosen}, Start Paused: {start_paused}")

        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1

        self.streaming_transcriptions = [self.StreamingTranscription(self.input_language, 1, self.logger) for _ in range(channels)]
        for transcription in self.streaming_transcriptions:
            transcription.start_streaming()
        self.process_responses_tasks = [asyncio.create_task(self.process_transcription_responses(channel)) for channel in range(channels)]

        # Handle case where session starts paused by client (e.g., sensitive data)
        # Genesys sends 'paused' immediately after 'opened' in this case.
        # We don't need special handling here; handle_paused will set the flag.

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

        duration = time.time() - self.start_time
        self.logger.info(
            f"Session stats - Duration: {duration:.2f}s, "
            f"Frames sent (events): {self.audio_frames_sent}, " # Renamed for clarity
            f"Frames received (audio): {self.audio_frames_received}"
        )

        self.running = False
        for transcription in self.streaming_transcriptions:
            transcription.stop_streaming()
        for task in self.process_responses_tasks:
            if task and not task.done():
                task.cancel()
        # Ensure tasks are awaited/cancelled properly
        await asyncio.gather(*self.process_responses_tasks, return_exceptions=True)


    async def disconnect_session(self, reason="completed", info=""):
        if not self.running: # Avoid duplicate disconnect attempts
             return
        self.running = False # Stop processing loops immediately

        try:
            if not self.session_id:
                self.logger.warning("Attempted to disconnect session without a session ID.")
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
            # Use a timeout for sending the disconnect message
            try:
                await asyncio.wait_for(self._send_json(disconnect_msg), timeout=2.0)
                self.server_seq += 1
            except asyncio.TimeoutError:
                 self.logger.warning(f"Timeout sending disconnect message for session {self.session_id}")
            except Exception as send_err:
                 self.logger.error(f"Error sending disconnect message for session {self.session_id}: {send_err}")

            # Attempt graceful WebSocket closure
            if self.ws and self.ws.open:
                try:
                    self.logger.info(f"Closing WebSocket connection for session {self.session_id}")
                    await asyncio.wait_for(self.ws.close(code=1000, reason=reason), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning(f"Timeout waiting for WebSocket close confirmation for session {self.session_id}")
                except ConnectionClosedOK:
                    self.logger.info(f"WebSocket connection closed gracefully for session {self.session_id}")
                except ConnectionClosed as close_err:
                     self.logger.warning(f"WebSocket connection closed abruptly for session {self.session_id}: Code {close_err.code}, Reason: {close_err.reason}")
                except Exception as close_err:
                    self.logger.error(f"Error closing WebSocket for session {self.session_id}: {close_err}")

        except Exception as e:
            self.logger.error(f"Error during disconnect_session for {self.session_id}: {e}")
        finally:
            # Ensure transcription resources are stopped regardless of errors
            for transcription in self.streaming_transcriptions:
                try:
                    transcription.stop_streaming()
                except Exception as stop_err:
                    self.logger.error(f"Error stopping transcription stream for session {self.session_id}: {stop_err}")
            for task in self.process_responses_tasks:
                if task and not task.done():
                    task.cancel()
            # Wait briefly for tasks to cancel
            await asyncio.sleep(0.1)
            self.logger.info(f"Session {self.session_id} disconnected.")

    async def handle_audio_frame(self, frame_bytes: bytes):
        self.audio_frames_received += 1
        self.logger.debug(f"Received audio frame from Genesys: {len(frame_bytes)} bytes (frame #{self.audio_frames_received})")

        # Check pause state *before* processing
        if self.is_paused_by_client or self.is_paused_by_server:
            self.logger.debug("Audio frame received while paused, discarding.")
            return # Discard frame if paused by either client or server

        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1

        # Calculate samples *in this frame*
        # Assuming PCMU (1 byte/sample), sample rate 8000 Hz
        samples_in_frame = len(frame_bytes) // channels
        self.total_samples_received += samples_in_frame

        if not self.streaming_transcriptions:
             self.logger.warning("Received audio frame but no transcription instances available.")
             return

        try:
            if channels == 2:
                left_channel_bytes = frame_bytes[0::2]  # External channel
                right_channel_bytes = frame_bytes[1::2] # Internal channel
                if len(self.streaming_transcriptions) > 0:
                     self.streaming_transcriptions[0].feed_audio(left_channel_bytes, 0)
                if len(self.streaming_transcriptions) > 1:
                     self.streaming_transcriptions[1].feed_audio(right_channel_bytes, 1)
            elif channels == 1:
                 if len(self.streaming_transcriptions) > 0:
                      self.streaming_transcriptions[0].feed_audio(frame_bytes, 0)
            else:
                 self.logger.error(f"Unsupported channel count: {channels}")

            self.audio_buffer.append(frame_bytes)
            self.last_frame_time = time.time()

        except Exception as e:
            self.logger.error(f"Error feeding audio to transcription engine: {e}", exc_info=True)
            await self.disconnect_session(reason="error", info="Error processing audio frame")


    async def process_transcription_responses(self, channel):
        try:
            while self.running:
                if not self.streaming_transcriptions or channel >= len(self.streaming_transcriptions):
                    self.logger.warning(f"Transcription instance for channel {channel} not available or index out of bounds.")
                    await asyncio.sleep(0.1)
                    continue

                transcription_instance = self.streaming_transcriptions[channel]
                response_data = transcription_instance.get_response(channel)

                if response_data:
                    response, start_sample_offset = response_data # Unpack response and its start offset

                    self.logger.info(f"Processing transcription response on channel {channel} (started at sample {start_sample_offset}): {response}")
                    if isinstance(response, Exception):
                        self.logger.error(f"Streaming recognition error received from channel {channel}: {response}")
                        await self.disconnect_session(reason="error", info="Streaming recognition failed")
                        break # Exit loop for this channel on error

                    if not hasattr(response, 'results') or not response.results:
                        self.logger.debug(f"Empty results in response for channel {channel}, skipping.")
                        continue

                    for result in response.results:
                        if not result.alternatives:
                            continue

                        alt = result.alternatives[0]
                        transcript_text = alt.transcript
                        if not transcript_text or transcript_text.isspace():
                             self.logger.debug(f"Empty transcript in alternative for channel {channel}, skipping.")
                             continue

                        source_lang = self.input_language
                        translated_text = transcript_text
                        dest_lang = source_lang

                        if self.enable_translation and source_lang.split('-')[0] != self.destination_language.split('-')[0]:
                            translated_text = await translate_with_gemini(transcript_text, source_lang, dest_lang, self.logger)
                            if translated_text is None:
                                self.logger.warning(f"Translation failed for text: '{transcript_text}'. Skipping transcription event.")
                                continue
                            dest_lang = self.destination_language
                        else:
                            translated_text = transcript_text # Use original if no translation needed/failed
                            dest_lang = source_lang

                        default_confidence = 1.0

                        # Determine if accurate word timings are available
                        has_word_timings = hasattr(alt, "words") and alt.words and all(
                            hasattr(w, "start_offset") and w.start_offset is not None and
                            hasattr(w, "end_offset") and w.end_offset is not None
                            for w in alt.words
                        )

                        # Calculate overall utterance offset and duration based on available timings
                        if has_word_timings:
                            utterance_start_offset_sec = alt.words[0].start_offset.total_seconds()
                            utterance_end_offset_sec = alt.words[-1].end_offset.total_seconds()
                            # Adjust relative offsets from transcription engine by the absolute start sample of the chunk
                            utterance_abs_start_sec = (start_sample_offset / 8000.0) + utterance_start_offset_sec
                            utterance_abs_end_sec = (start_sample_offset / 8000.0) + utterance_end_offset_sec
                            utterance_duration_sec = max(0, utterance_abs_end_sec - utterance_abs_start_sec) # Ensure non-negative
                        elif hasattr(alt, "total_billed_time"): # Fallback for some APIs, might not be accurate utterance duration
                             utterance_abs_start_sec = start_sample_offset / 8000.0
                             utterance_duration_sec = alt.total_billed_time.total_seconds()
                             self.logger.warning("Using total_billed_time as fallback for duration.")
                        else:
                             # Fallback if no timings available (e.g., maybe OpenAI without verbose_json)
                             utterance_abs_start_sec = start_sample_offset / 8000.0
                             # Estimate duration based on received audio chunk length if possible, otherwise default
                             # This requires knowing the chunk size associated with the response, which is tricky. Defaulting.
                             utterance_duration_sec = 1.0 # Default duration if no other info
                             self.logger.warning("No word timings or duration info found, using estimated start and default duration.")


                        # Ensure offsets are non-negative
                        utterance_abs_start_sec = max(0, utterance_abs_start_sec)

                        offset_str = f"PT{utterance_abs_start_sec:.3f}S" # Use 3 decimal places for precision
                        duration_str = f"PT{utterance_duration_sec:.3f}S"

                        overall_confidence = default_confidence
                        if hasattr(alt, "confidence") and alt.confidence is not None and alt.confidence > 0.0:
                            overall_confidence = alt.confidence

                        tokens = []
                        if self.enable_translation:
                            # Approximate token timings for translated text
                            words_list = translated_text.split()
                            if words_list and utterance_duration_sec > 0:
                                per_word_duration = utterance_duration_sec / len(words_list)
                                for i, word in enumerate(words_list):
                                    token_offset_sec = utterance_abs_start_sec + i * per_word_duration
                                    tokens.append({
                                        "type": "word",
                                        "value": word,
                                        "confidence": overall_confidence, # Use overall confidence for translated words
                                        "offset": f"PT{token_offset_sec:.3f}S",
                                        "duration": f"PT{per_word_duration:.3f}S",
                                        "language": dest_lang
                                    })
                            else: # Handle empty translation or zero duration
                                tokens.append({
                                    "type": "word",
                                    "value": translated_text,
                                    "confidence": overall_confidence,
                                    "offset": offset_str,
                                    "duration": duration_str, # Use overall duration
                                    "language": dest_lang
                                })
                        else:
                            # Use original transcript and timings if available
                            if has_word_timings:
                                for w in alt.words:
                                    token_relative_start_sec = w.start_offset.total_seconds()
                                    token_relative_end_sec = w.end_offset.total_seconds()
                                    token_abs_start_sec = max(0, (start_sample_offset / 8000.0) + token_relative_start_sec)
                                    token_duration_sec = max(0, token_relative_end_sec - token_relative_start_sec)

                                    word_confidence = default_confidence
                                    if hasattr(w, "confidence") and w.confidence is not None and w.confidence > 0.0:
                                        word_confidence = w.confidence

                                    tokens.append({
                                        "type": "word",
                                        "value": w.word,
                                        "confidence": word_confidence,
                                        "offset": f"PT{token_abs_start_sec:.3f}S",
                                        "duration": f"PT{token_duration_sec:.3f}S",
                                        "language": dest_lang # Should be input_language here
                                    })
                            else:
                                # Fallback: Create a single token for the whole utterance if no word timings
                                self.logger.warning("Creating single token for utterance as word timings are unavailable.")
                                tokens.append({
                                    "type": "word",
                                    "value": transcript_text,
                                    "confidence": overall_confidence,
                                    "offset": offset_str,
                                    "duration": duration_str,
                                    "language": dest_lang # Should be input_language here
                                })

                        alternative_payload = {
                            "confidence": overall_confidence,
                            "interpretations": [
                                {
                                    "type": "display",
                                    "transcript": translated_text, # Send translated or original text
                                    "tokens": tokens
                                }
                            ]
                        }
                        # Add languages field only if language detection occurred (or translation happened)
                        detected_languages = []
                        if hasattr(alt, "language_code") and alt.language_code:
                             detected_languages.append(alt.language_code)
                        if self.enable_translation and dest_lang not in detected_languages:
                             detected_languages.append(dest_lang)
                        if detected_languages:
                             alternative_payload["languages"] = detected_languages


                        transcript_event = {
                            "version": "2",
                            "type": "event",
                            "seq": self.server_seq + 1, # Increment sequence number here
                            "clientseq": self.client_seq,
                            "id": self.session_id,
                            "parameters": {
                                "entities": [
                                    {
                                        "type": "transcript",
                                        "data": {
                                            "id": str(uuid.uuid4()),
                                            "channelId": channel,
                                            "isFinal": result.is_final,
                                            "offset": offset_str,
                                            "duration": duration_str,
                                            "alternatives": [alternative_payload]
                                        }
                                    }
                                ]
                            }
                        }
                        self.logger.info(f"Sending transcription event to Genesys (seq={self.server_seq + 1}): {json.dumps(transcript_event)}")
                        if await self._send_json(transcript_event):
                            self.server_seq += 1 # Increment only if send was successful
                            self.audio_frames_sent += 1 # Count events sent
                        else:
                            self.logger.warning(f"Transcript event (seq={self.server_seq + 1}) dropped due to rate limiting or send error.")
                            # Do not increment server_seq if send failed

                else:
                    # No response available, yield control briefly
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
             self.logger.info(f"Transcription response processing task for channel {channel} cancelled.")
        except Exception as e:
             self.logger.error(f"Error in process_transcription_responses for channel {channel}: {e}", exc_info=True)
             # Consider disconnecting if the processing task fails critically
             # await self.disconnect_session(reason="error", info=f"Transcription processing error on channel {channel}")

    async def _send_json(self, msg: dict):
        if not self.ws or not self.ws.open:
             self.logger.warning(f"WebSocket closed or unavailable, cannot send message type {msg.get('type')}")
             self.running = False # Ensure loops stop if WS is closed
             return False

        try:
            if not await self.message_limiter.acquire():
                current_rate = self.message_limiter.get_current_rate()
                self.logger.warning(
                    f"Message rate limit exceeded (current rate: {current_rate:.2f}/s). "
                    f"Message type: {msg.get('type')}. Dropping to maintain compliance."
                )
                return False

            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
            return True
        except ConnectionClosedOK:
            self.logger.info("Genesys WebSocket closed gracefully while sending JSON message.")
            self.running = False
            return False
        except ConnectionClosed as e:
            self.logger.warning(f"Genesys WebSocket closed unexpectedly while sending JSON message. Code: {e.code}, Reason: {e.reason}")
            self.running = False
            return False
        except Exception as e:
            self.logger.error(f"Error sending message type {msg.get('type')}: {e}", exc_info=True)
            # Consider disconnecting on persistent send errors
            # await self.disconnect_session(reason="error", info=f"Failed to send message: {e}")
            return False

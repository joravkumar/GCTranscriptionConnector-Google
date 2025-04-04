import asyncio
import uuid
import json
import time
import websockets
import tempfile
import audioop
import logging
from websockets.exceptions import ConnectionClosed
from datetime import timedelta


from config import (
    RATE_LIMIT_MAX_RETRIES,
    GENESYS_MSG_RATE_LIMIT,
    GENESYS_BINARY_RATE_LIMIT,
    GENESYS_MSG_BURST_LIMIT,
    GENESYS_BINARY_BURST_LIMIT,
    MAX_AUDIO_BUFFER_SIZE,
    SUPPORTED_LANGUAGES,
    SPEECH_PROVIDER,
    GOOGLE_SPEECH_MODEL
)
from rate_limiter import RateLimiter
from utils import format_json, parse_iso8601_duration

from language_mapping import normalize_language_code

if SPEECH_PROVIDER == 'openai':
    from openai_speech_transcription import StreamingTranscription
else:
    from google_speech_transcription import StreamingTranscription

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

        self.total_samples = 0
        self.offset_adjustment = 0
        self.pause_start_time = None

        self.input_language = "en-US"
        self.destination_language = "en-US"
        self.enable_translation = False

        self.streaming_transcriptions = []
        self.process_responses_tasks = []

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
        elif msg_type == "discarded":
            await self.handle_discarded(msg)
        elif msg_type == "paused":
            await self.handle_paused(msg)
        elif msg_type == "resumed":
            await self.handle_resumed(msg)
        elif msg_type in ["update"]:
            self.logger.debug(f"Ignoring message type {msg_type}")
        else:
            self.logger.debug(f"Ignoring unknown message type: {msg_type}")

    async def handle_discarded(self, msg: dict):
        discarded_duration_str = msg["parameters"].get("discarded")
        if discarded_duration_str:
            try:
                gap = parse_iso8601_duration(discarded_duration_str)
                gap_samples = int(gap * 8000)
                self.offset_adjustment += gap_samples
                self.logger.info(f"Handled 'discarded' message: gap duration {gap}s, adding {gap_samples} samples to offset adjustment.")
            except ValueError as e:
                self.logger.warning(f"Failed to parse discarded duration '{discarded_duration_str}': {e}")
        else:
            self.logger.warning("Received 'discarded' message without 'discarded' parameter.")

    async def handle_paused(self, msg: dict):
        if self.pause_start_time is None:
            self.pause_start_time = time.time()
            self.logger.info("Handled 'paused' message: pause started.")
        else:
            self.logger.warning("Received 'paused' message while already paused.")

    async def handle_resumed(self, msg: dict):
        if self.pause_start_time is not None:
            pause_duration = time.time() - self.pause_start_time
            gap_samples = int(pause_duration * 8000)
            self.offset_adjustment += gap_samples
            self.logger.info(f"Handled 'resumed' message: pause duration {pause_duration:.2f}s, adding {gap_samples} samples to offset adjustment.")
            self.pause_start_time = None
        else:
            self.logger.warning("Received 'resumed' message without a preceding 'paused' event.")

    async def handle_open(self, msg: dict):
        self.session_id = msg["id"]

        custom_config = msg["parameters"].get("customConfig", {})
        self.input_language = normalize_language_code(custom_config.get("inputLanguage", "en-US"))
        self.enable_translation = custom_config.get("enableTranslation", False)
        self.destination_language = normalize_language_code(msg["parameters"].get("language", "en-US"))

        is_probe = (
            msg["parameters"].get("conversationId") == "00000000-0000-0000-0000-000000000000" and
            msg["parameters"].get("participant", {}).get("id") == "00000000-0000-0000-0000-000000000000"
        )

        if is_probe:
            self.logger.info("Detected probe connection")
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

        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1

        self.streaming_transcriptions = [StreamingTranscription(self.input_language, 1, self.logger) for _ in range(channels)]
        for transcription in self.streaming_transcriptions:
            transcription.start_streaming()
        self.process_responses_tasks = [asyncio.create_task(self.process_transcription_responses(channel)) for channel in range(channels)]

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
        for transcription in self.streaming_transcriptions:
            transcription.stop_streaming()
        for task in self.process_responses_tasks:
            task.cancel()

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
            for transcription in self.streaming_transcriptions:
                transcription.stop_streaming()
            for task in self.process_responses_tasks:
                task.cancel()

    async def handle_audio_frame(self, frame_bytes: bytes):
        self.audio_frames_received += 1
        self.logger.debug(f"Received audio frame from Genesys: {len(frame_bytes)} bytes (frame #{self.audio_frames_received})")

        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1

        sample_times = len(frame_bytes) // channels
        current_adjusted_samples = self.total_samples - self.offset_adjustment
        self.total_samples += sample_times


        if channels == 2:
            left_channel = frame_bytes[0::2]
            right_channel = frame_bytes[1::2]
            # Pass adjusted sample count for *this* frame's start
            left_adjusted_start = current_adjusted_samples
            right_adjusted_start = current_adjusted_samples
            self.streaming_transcriptions[0].feed_audio(left_channel, 0, left_adjusted_start)
            self.streaming_transcriptions[1].feed_audio(right_channel, 0, right_adjusted_start)
        else:
            # Pass adjusted sample count for *this* frame's start
            adjusted_start = current_adjusted_samples
            self.streaming_transcriptions[0].feed_audio(frame_bytes, 0, adjusted_start)

        self.audio_buffer.append(frame_bytes)

    async def process_transcription_responses(self, channel):
        while self.running:
            response = self.streaming_transcriptions[channel].get_response(0)
            if response:
                self.logger.info(f"Processing transcription response on channel {channel}")
                if isinstance(response, Exception):
                    self.logger.error(f"Streaming recognition error on channel {channel}: {response}")
                    await self.disconnect_session(reason="error", info="Streaming recognition failed")
                    break

                if SPEECH_PROVIDER == 'openai':
                    if not response or not response.get('transcript'):
                        self.logger.debug(f"OpenAI response empty or missing transcript for channel {channel}")
                        await asyncio.sleep(0.01)
                        continue

                    transcript_text = response['transcript']
                    source_lang = self.input_language
                    dest_lang = self.destination_language

                    if self.enable_translation:
                        translated_text = await translate_with_gemini(transcript_text, source_lang, dest_lang, self.logger)
                        if translated_text is None:
                            self.logger.warning(f"Translation failed for text: '{transcript_text}'. Skipping transcription event.")
                            await asyncio.sleep(0.01)
                            continue
                    else:
                        translated_text = transcript_text
                        dest_lang = source_lang

                    overall_start = response['start_seconds']
                    overall_duration = response['duration_seconds']
                    overall_confidence = response['confidence']

                    offset_str = f"PT{overall_start:.3f}S" # Use 3 decimals for precision
                    duration_str = f"PT{overall_duration:.3f}S" # Use 3 decimals for precision

                    tokens = []
                    response_words = response.get('words', [])

                    if self.enable_translation:
                         # Generate synthetic tokens for translated text
                        words_list = translated_text.split()
                        if words_list and overall_duration > 0.001: # Avoid division by zero
                            per_word_duration = overall_duration / len(words_list)
                            for i, word in enumerate(words_list):
                                token_offset = overall_start + i * per_word_duration
                                token_duration = per_word_duration
                                tokens.append({
                                    "type": "word",
                                    "value": word,
                                    "confidence": overall_confidence,
                                    "offset": f"PT{token_offset:.3f}S",
                                    "duration": f"PT{token_duration:.3f}S",
                                    "language": dest_lang
                                })
                        elif translated_text: # Handle case with duration 0 or no words
                             tokens.append({
                                "type": "word",
                                "value": translated_text,
                                "confidence": overall_confidence,
                                "offset": offset_str,
                                "duration": duration_str,
                                "language": dest_lang
                            })

                    elif response_words: # Use OpenAI's generated word timings
                        for w in response_words:
                            token_offset = w.start_offset.total_seconds()
                            token_duration = w.end_offset.total_seconds() - w.start_offset.total_seconds()
                            tokens.append({
                                "type": "word",
                                "value": w.word,
                                "confidence": w.confidence,
                                "offset": f"PT{token_offset:.3f}S",
                                "duration": f"PT{token_duration:.3f}S",
                                "language": dest_lang
                            })
                    elif translated_text: # Fallback if no words but have transcript
                         tokens.append({
                            "type": "word",
                            "value": translated_text,
                            "confidence": overall_confidence,
                            "offset": offset_str,
                            "duration": duration_str,
                            "language": dest_lang
                        })


                elif SPEECH_PROVIDER == 'google':
                    if not response.results or not response.results[0].alternatives:
                         self.logger.debug(f"Google response empty or missing alternatives for channel {channel}")
                         await asyncio.sleep(0.01)
                         continue

                    result = response.results[0]
                    alt = result.alternatives[0]
                    transcript_text = alt.transcript
                    source_lang = self.input_language

                    if self.enable_translation:
                        dest_lang = self.destination_language
                        translated_text = await translate_with_gemini(transcript_text, source_lang, dest_lang, self.logger)
                        if translated_text is None:
                            self.logger.warning(f"Translation failed for text: '{transcript_text}'. Skipping transcription event.")
                            await asyncio.sleep(0.01)
                            continue
                    else:
                        dest_lang = source_lang
                        translated_text = transcript_text

                    adjustment_seconds = self.offset_adjustment / 8000.0
                    default_confidence = 1.0

                    use_word_timings = hasattr(alt, "words") and alt.words and len(alt.words) > 0 and all(
                        hasattr(w, "start_offset") and w.start_offset is not None and hasattr(w, "end_offset") and w.end_offset is not None for w in alt.words
                    )

                    if use_word_timings:
                        overall_start = alt.words[0].start_offset.total_seconds() - adjustment_seconds
                        overall_end = alt.words[-1].end_offset.total_seconds() - adjustment_seconds
                        overall_duration = max(0.0, overall_end - overall_start) # Ensure non-negative duration
                    else:
                        # Fallback - this might be inaccurate, especially duration. Consider improving if possible.
                        # Use the start time of the *first* frame associated with this result if available.
                        # For now, use the previous potentially inaccurate logic as a placeholder.
                        # A better fallback might require passing timing info from the transcription class.
                        first_sample_time = (self.total_samples - (len(self.audio_buffer[-1] if self.audio_buffer else b'') // (1 if self.negotiated_media.get("channels", []) != 2 else 2))) / 8000.0
                        overall_start = max(0.0, first_sample_time - adjustment_seconds) # Use a more reasonable start estimate
                        overall_duration = 0.5 # Assign a default duration if none is available
                        self.logger.warning(f"Using fallback timing for Google STT result on channel {channel}. Accuracy may be reduced.")


                    offset_str = f"PT{overall_start:.3f}S"
                    duration_str = f"PT{overall_duration:.3f}S"


                    overall_confidence = default_confidence
                    if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2':
                         if hasattr(alt, "confidence") and alt.confidence is not None:
                             overall_confidence = alt.confidence
                    elif hasattr(alt, "confidence") and alt.confidence is not None: # Use if available for other models too
                         overall_confidence = alt.confidence


                    tokens = []
                    if self.enable_translation:
                        words_list = translated_text.split()
                        if words_list and overall_duration > 0.001:
                            per_word_duration = overall_duration / len(words_list)
                            for i, word in enumerate(words_list):
                                token_offset = overall_start + i * per_word_duration
                                confidence = overall_confidence # Use overall confidence for translated words
                                tokens.append({
                                    "type": "word",
                                    "value": word,
                                    "confidence": confidence,
                                    "offset": f"PT{token_offset:.3f}S",
                                    "duration": f"PT{per_word_duration:.3f}S",
                                    "language": dest_lang
                                })
                        elif translated_text:
                             tokens.append({
                                "type": "word",
                                "value": translated_text,
                                "confidence": overall_confidence,
                                "offset": offset_str,
                                "duration": duration_str,
                                "language": dest_lang
                            })
                    elif use_word_timings:
                        for w in alt.words:
                            token_offset = w.start_offset.total_seconds() - adjustment_seconds
                            token_end_offset = w.end_offset.total_seconds() - adjustment_seconds
                            token_duration = max(0.0, token_end_offset - token_offset) # Ensure non-negative

                            word_confidence = default_confidence
                            if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2':
                                if hasattr(w, "confidence") and w.confidence is not None:
                                     word_confidence = w.confidence
                            elif hasattr(w, "confidence") and w.confidence is not None: # Use if available
                                word_confidence = w.confidence

                            tokens.append({
                                "type": "word",
                                "value": w.word,
                                "confidence": word_confidence,
                                "offset": f"PT{token_offset:.3f}S",
                                "duration": f"PT{token_duration:.3f}S",
                                "language": dest_lang
                            })
                    elif translated_text: # Fallback if no word timings but have original transcript
                         tokens.append({
                            "type": "word",
                            "value": translated_text, # Use original transcript here
                            "confidence": overall_confidence,
                            "offset": offset_str,
                            "duration": duration_str,
                            "language": dest_lang
                        })
                else:
                    self.logger.error(f"Unsupported SPEECH_PROVIDER: {SPEECH_PROVIDER}")
                    await asyncio.sleep(0.01)
                    continue

                # Common part: Construct and send the event if tokens exist
                if tokens:
                    alternative = {
                        "confidence": overall_confidence,
                        **({"languages": [dest_lang]} if self.enable_translation else {}),
                        "interpretations": [
                            {
                                "type": "display",
                                "transcript": translated_text,
                                "tokens": tokens
                            }
                        ]
                    }

                    channel_id = channel

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
                                        "channelId": channel_id,
                                        "isFinal": True, # Assuming final results for now
                                        "offset": offset_str,
                                        "duration": duration_str,
                                        "alternatives": [
                                            alternative
                                        ]
                                    }
                                }
                            ]
                        }
                    }
                    self.logger.info(f"Sending transcription event to Genesys: {json.dumps(transcript_event)}")
                    if await self._send_json(transcript_event):
                        self.server_seq += 1
                    else:
                        self.logger.debug("Transcript event dropped due to rate limiting")
                else:
                     self.logger.debug(f"No tokens generated for transcript on channel {channel}. Skipping event.")

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
                return False

            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
            return True
        except ConnectionClosed:
            self.logger.warning("Genesys WebSocket closed while sending JSON message.")
            self.running = False
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

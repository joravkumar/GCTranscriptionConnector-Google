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

# Import language normalization function
from language_mapping import normalize_language_code

# Import the appropriate speech transcription module based on the configured provider
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

        # Total samples processed for offset calculation
        self.total_samples = 0
        # Offset adjustment for control messages (discarded/paused)
        self.offset_adjustment = 0
        # Timestamp when pause started, if any
        self.pause_start_time = None

        # New language attributes:
        # input_language comes from customConfig.inputLanguage (for transcription)
        # destination_language comes from the "language" field (for translation)
        self.input_language = "en-US"
        self.destination_language = "en-US"

        # Translation enable flag from customConfig.enableTranslation
        self.enable_translation = False

        # Streaming transcription for each channel
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
                gap_samples = int(gap * 8000) # assuming 8kHz sample rate
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
            gap_samples = int(pause_duration * 8000) # assuming 8kHz sample rate
            self.offset_adjustment += gap_samples
            self.logger.info(f"Handled 'resumed' message: pause duration {pause_duration:.2f}s, adding {gap_samples} samples to offset adjustment.")
            self.pause_start_time = None
        else:
            self.logger.warning("Received 'resumed' message without a preceding 'paused' event.")

    async def handle_open(self, msg: dict):
        self.session_id = msg["id"]

        # Extract customConfig from the open message
        custom_config = msg["parameters"].get("customConfig", {})

        # Set input_language from customConfig.inputLanguage, default to "en-US"
        self.input_language = normalize_language_code(custom_config.get("inputLanguage", "en-US"))

        # Set enable_translation from customConfig.enableTranslation, default to False
        self.enable_translation = custom_config.get("enableTranslation", False)

        # Set destination_language from the "language" field
        self.destination_language = normalize_language_code(msg["parameters"].get("language", "en-US"))

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

        # Determine number of channels
        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1

        # Initialize streaming transcription for each channel as mono using the input language
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

        # Compute how many "sample times" based on negotiated channels
        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1

        # For PCMU, each channel is 1 byte per sample time
        sample_times = len(frame_bytes) // channels
        self.total_samples += sample_times

        # Deinterleave stereo audio and feed to respective transcription instances
        if channels == 2:
            left_channel = frame_bytes[0::2] # External channel
            right_channel = frame_bytes[1::2] # Internal channel
            self.streaming_transcriptions[0].feed_audio(left_channel, 0)
            self.streaming_transcriptions[1].feed_audio(right_channel, 1) # Use correct channel index
        else:
            self.streaming_transcriptions[0].feed_audio(frame_bytes, 0)

        self.audio_buffer.append(frame_bytes)

    async def process_transcription_responses(self, channel):
         while self.running:
            response = self.streaming_transcriptions[channel].get_response(0) # Assuming one channel per instance
            if response:
                if isinstance(response, Exception):
                    self.logger.error(f"Streaming recognition error on channel {channel}: {response}")
                    await self.disconnect_session(reason="error", info="Streaming recognition failed")
                    break

                # --- Common variables ---
                final_text = ""
                overall_confidence = 0.9 # Default confidence
                offset_str = "PT0S"
                duration_str = "PT0S"
                tokens = []
                is_final = True # OpenAI and Google streaming mostly give final results per utterance
                source_lang = self.input_language
                dest_lang = self.destination_language if self.enable_translation else self.input_language

                # --- Provider-specific processing ---
                if SPEECH_PROVIDER == 'openai':
                    if isinstance(response, tuple) and len(response) == 2:
                        transcript_text, avg_confidence = response
                        if not transcript_text:
                            continue

                        overall_confidence = avg_confidence if avg_confidence is not None else 0.9

                        # Calculate offset and duration based on server's sample tracking
                        effective_end_samples = self.total_samples - self.offset_adjustment
                        words_list = transcript_text.split()

                        # Estimate duration based on word count (heuristic)
                        estimated_duration_seconds = max(0.1, min(15.0, len(words_list) * 0.4))
                        overall_start_seconds = max(0.0, (effective_end_samples / 8000.0) - estimated_duration_seconds)

                        offset_str = f"PT{overall_start_seconds:.3f}S"
                        duration_str = f"PT{estimated_duration_seconds:.3f}S"
                        final_text = transcript_text

                        # Generate synthetic tokens
                        if words_list:
                            per_word_duration = estimated_duration_seconds / len(words_list) if len(words_list) > 0 else 0
                            for i, word in enumerate(words_list):
                                token_offset = overall_start_seconds + i * per_word_duration
                                tokens.append({
                                    "type": "word",
                                    "value": word,
                                    "confidence": overall_confidence, # Use average confidence for all tokens
                                    "offset": f"PT{token_offset:.3f}S",
                                    "duration": f"PT{per_word_duration:.3f}S",
                                    "language": source_lang # Assume source lang for now
                                })
                        else:
                             # Handle empty transcript case if necessary, though filtered before
                            pass

                    else:
                        self.logger.warning(f"Unexpected response format from OpenAI provider: {response}")
                        continue

                else: # Google provider
                    self.logger.info(f"Processing Google transcription response on channel {channel}: {response}")
                    if not response.results or not response.results[0].alternatives:
                        continue

                    alt = response.results[0].alternatives[0]
                    transcript_text = alt.transcript
                    is_final = response.results[0].is_final

                    # Calculate offset adjustment in seconds
                    adjustment_seconds = self.offset_adjustment / 8000.0

                    # Use actual word timings if available
                    use_word_timings = hasattr(alt, "words") and alt.words and all(
                        hasattr(w, "start_offset") and w.start_offset is not None and hasattr(w, "end_offset") and w.end_offset is not None for w in alt.words
                    )

                    if use_word_timings:
                        # Use first word start and last word end for overall timing
                        overall_start_seconds = alt.words[0].start_offset.total_seconds() - adjustment_seconds
                        overall_end_seconds = alt.words[-1].end_offset.total_seconds() - adjustment_seconds
                        overall_duration_seconds = max(0.0, overall_end_seconds - overall_start_seconds)
                        overall_start_seconds = max(0.0, overall_start_seconds) # Ensure not negative

                        offset_str = f"PT{overall_start_seconds:.3f}S"
                        duration_str = f"PT{overall_duration_seconds:.3f}S"

                        # Set overall confidence
                        if hasattr(alt, "confidence") and alt.confidence is not None:
                            overall_confidence = alt.confidence
                        elif GOOGLE_SPEECH_MODEL.lower() != 'chirp_2':
                            overall_confidence = 1.0 # Default for non-Chirp2

                        # Build tokens with actual timing
                        for w in alt.words:
                             token_start_seconds = w.start_offset.total_seconds() - adjustment_seconds
                             token_end_seconds = w.end_offset.total_seconds() - adjustment_seconds
                             token_duration_seconds = max(0.0, token_end_seconds - token_start_seconds)
                             token_start_seconds = max(0.0, token_start_seconds)

                             word_confidence = 1.0 # Default
                             if hasattr(w, "confidence") and w.confidence is not None:
                                 word_confidence = w.confidence

                             tokens.append({
                                 "type": "word",
                                 "value": w.word,
                                 "confidence": word_confidence,
                                 "offset": f"PT{token_start_seconds:.3f}S",
                                 "duration": f"PT{token_duration_seconds:.3f}S",
                                 "language": source_lang
                             })
                    else:
                         # No word-level timings (should be rare with Chirp/Chirp2 but handle anyway)
                        effective_end_samples = self.total_samples - self.offset_adjustment
                        overall_start_seconds = max(0.0, (effective_end_samples / 8000.0) - 1.0) # Estimate 1 sec duration
                        overall_duration_seconds = 1.0

                        offset_str = f"PT{overall_start_seconds:.3f}S"
                        duration_str = f"PT{overall_duration_seconds:.3f}S"

                        if hasattr(alt, "confidence") and alt.confidence is not None:
                            overall_confidence = alt.confidence
                        elif GOOGLE_SPEECH_MODEL.lower() != 'chirp_2':
                             overall_confidence = 1.0

                        # Create a single token for the whole transcript
                        tokens.append({
                            "type": "word",
                            "value": transcript_text,
                            "confidence": overall_confidence,
                            "offset": offset_str,
                            "duration": duration_str,
                            "language": source_lang
                        })

                    final_text = transcript_text

                # --- Translation (Common) ---
                translated_text = final_text
                if self.enable_translation:
                    translated_text = await translate_with_gemini(final_text, source_lang, dest_lang, self.logger)
                    if translated_text is None:
                        self.logger.warning(f"Translation failed for text: '{final_text}'. Sending original transcript.")
                        translated_text = final_text # Fallback to original
                        dest_lang = source_lang # Revert dest_lang if translation failed
                    else:
                         # If translation succeeded, we need to regenerate synthetic tokens for the translated text
                        if SPEECH_PROVIDER == 'openai' or (SPEECH_PROVIDER == 'google' and not use_word_timings):
                            # Recalculate tokens based on translated words
                            translated_words_list = translated_text.split()
                            tokens = []
                            if translated_words_list:
                                # Use the same overall timing calculated before translation
                                current_overall_start = overall_start_seconds
                                current_overall_duration = overall_duration_seconds
                                per_word_duration = current_overall_duration / len(translated_words_list) if len(translated_words_list) > 0 else 0

                                for i, word in enumerate(translated_words_list):
                                    token_offset = current_overall_start + i * per_word_duration
                                    tokens.append({
                                        "type": "word",
                                        "value": word,
                                        "confidence": overall_confidence,
                                        "offset": f"PT{token_offset:.3f}S",
                                        "duration": f"PT{per_word_duration:.3f}S",
                                        "language": dest_lang
                                    })
                            else:
                                # Handle empty translation? Add a single token with overall timing.
                                 tokens.append({
                                    "type": "word",
                                    "value": translated_text,
                                    "confidence": overall_confidence,
                                    "offset": offset_str,
                                    "duration": duration_str,
                                    "language": dest_lang
                                 })
                        else: # Google with word timings - Update language in existing tokens
                             for token in tokens:
                                 token["language"] = dest_lang


                # --- Construct and Send Event (Common) ---
                if not translated_text or not tokens:
                    self.logger.debug(f"Skipping empty transcript event for channel {channel}")
                    continue

                alternative = {
                    "confidence": overall_confidence,
                    **({"languages": [dest_lang]} if self.enable_translation else {}), # Report language only if translated
                    "interpretations": [
                        {
                            "type": "display",
                            "transcript": translated_text,
                            "tokens": tokens
                        }
                    ]
                }

                channel_id = channel # Integer channel index

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
                                    "isFinal": is_final,
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
                self.logger.info(f"Sending transcription event to Genesys (Channel {channel}): {json.dumps(transcript_event)}")
                if await self._send_json(transcript_event):
                    self.server_seq += 1
                else:
                    self.logger.debug("Transcript event dropped due to rate limiting or connection issues")
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
                return False # Message not sent

            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
            return True # Message sent
        except ConnectionClosed:
            self.logger.warning("Genesys WebSocket closed while sending JSON message.")
            self.running = False
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {e}")
            return False

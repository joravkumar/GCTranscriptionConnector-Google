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
    SPEECH_PROVIDER,
    GOOGLE_SPEECH_MODEL # Import GOOGLE_SPEECH_MODEL
)
from rate_limiter import RateLimiter
from utils import format_json, parse_iso8601_duration

# Import language normalization function
from language_mapping import normalize_language_code

# Import the appropriate speech transcription module based on the configured provider
if SPEECH_PROVIDER == 'openai':
    from openai_speech_transcription import StreamingTranscription
else:
    # Assuming google_speech_transcription also follows the one-instance-per-channel model
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

        # Total samples processed for offset calculation (based *only* on received audio)
        self.total_samples_processed = 0
        # Timestamp when pause started, if any
        self.pause_start_time = None

        # Language attributes:
        self.input_language = "en-US"
        self.destination_language = "en-US"

        # Translation enable flag
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
        # Handle other errors if needed
        self.logger.error(f"Received error message from Genesys: {format_json(msg)}")
        # Decide if the error is fatal
        # await self.disconnect_session(reason="error", info=f"Received error from client: {error_code}")
        return False # Indicate error wasn't handled for rate limit purposes

    async def handle_message(self, msg: dict):
        msg_type = msg.get("type")
        seq = msg.get("seq", 0)
        self.client_seq = seq

        if self.rate_limit_state.get("in_backoff") and msg_type != "error":
            self.logger.debug(f"Skipping message type {msg_type} during rate limit backoff")
            return

        if msg_type == "error":
            handled = await self.handle_error(msg)
            if handled: # Only return if it was a rate limit error that was handled
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
        elif msg_type == "update":
            await self.handle_update(msg) # Add handler for update message
        else:
            self.logger.debug(f"Ignoring unknown or unhandled message type: {msg_type}")

    async def handle_discarded(self, msg: dict):
        # This message indicates audio was lost *by the client* before sending.
        # The server does NOT adjust its own sample count based on this.
        # The server's sample count only reflects audio *received*.
        # Genesys documentation: "Paused" or "discarded" durations must not be included as part of the offset
        discarded_duration_str = msg["parameters"].get("discarded")
        if discarded_duration_str:
            try:
                gap = parse_iso8601_duration(discarded_duration_str)
                self.logger.info(f"Received 'discarded' message: client lost {gap}s of audio.")
            except ValueError as e:
                self.logger.warning(f"Failed to parse discarded duration '{discarded_duration_str}': {e}")
        else:
            self.logger.warning("Received 'discarded' message without 'discarded' parameter.")

    async def handle_paused(self, msg: dict):
        # This message indicates the client has paused sending audio.
        # The server does NOT adjust its own sample count based on this.
        # The server's sample count only reflects audio *received*.
        if self.pause_start_time is None:
            self.pause_start_time = time.time() # Track pause locally if needed for server logic, but not for offset calculation
            self.logger.info("Received 'paused' message from client.")
        else:
            # This could happen if server also requested pause. Log it.
            self.logger.info("Received 'paused' message while already locally tracking a pause.")

    async def handle_resumed(self, msg: dict):
        # This message indicates the client has resumed sending audio.
        # The server does NOT adjust its own sample count based on this.
        if self.pause_start_time is not None:
            pause_duration = time.time() - self.pause_start_time
            self.logger.info(f"Received 'resumed' message. Client pause lasted {pause_duration:.2f}s (local tracking).")
            self.pause_start_time = None
        else:
            self.logger.info("Received 'resumed' message without a preceding 'paused' event (or local tracking).")

    async def handle_update(self, msg: dict):
        # Handle changes during the call, e.g., language change
        self.logger.info(f"Received 'update' message: {format_json(msg)}")
        params = msg.get("parameters", {})
        if "language" in params:
            new_lang = normalize_language_code(params["language"])
            if new_lang != self.destination_language:
                self.logger.info(f"Updating destination language from {self.destination_language} to {new_lang}")
                self.destination_language = new_lang
                # Potentially notify transcription/translation components if needed
            else:
                 self.logger.info(f"Received language update, but language {new_lang} is already set.")
        # Handle other updatable parameters in the future here

    async def handle_open(self, msg: dict):
        self.session_id = msg["id"]

        custom_config = msg["parameters"].get("customConfig", {})
        self.input_language = normalize_language_code(custom_config.get("inputLanguage", "en-US"))
        self.enable_translation = custom_config.get("enableTranslation", False)
        self.destination_language = normalize_language_code(msg["parameters"].get("language", "en-US"))

        self.logger.info(f"Handling 'open'. Input Lang: {self.input_language}, Dest Lang: {self.destination_language}, Translate Enabled: {self.enable_translation}")

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
                    "media": [], # No media for probe
                    "supportedLanguages": supported_langs
                }
            }
            if await self._send_json(opened_msg):
                self.server_seq += 1
            else:
                # Don't disconnect, just log error for probe
                 self.logger.error("Failed to send opened message for probe connection")
            return # Don't proceed further for probe

        offered_media = msg["parameters"].get("media", [])
        chosen = None
        for m in offered_media:
            # Require PCMU at 8000 Hz
            if (m.get("format") == "PCMU" and m.get("rate") == 8000):
                chosen = m
                break

        if not chosen:
            self.logger.error(f"No supported media format found in offer: {offered_media}")
            resp = {
                "version": "2",
                "type": "disconnect",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq,
                "id": self.session_id,
                "parameters": {
                    "reason": "error",
                    "info": "No supported media format found (PCMU, 8000Hz required)"
                }
            }
            if await self._send_json(resp):
                self.server_seq += 1
            # No need to call disconnect_session here, sending disconnect is enough
            self.running = False
            return

        self.negotiated_media = chosen
        self.logger.info(f"Negotiated media format: {self.negotiated_media}")

        opened_msg = {
            "version": "2",
            "type": "opened",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "startPaused": False,
                "media": [chosen] # Echo back the chosen media
            }
        }
        if await self._send_json(opened_msg):
            self.server_seq += 1
        else:
            await self.disconnect_session(reason="error", info="Failed to send opened message")
            return
        self.logger.info(f"Session opened successfully.")

        # Determine number of channels based on the *chosen* media
        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media else 1
        if channels == 0:
            channels = 1 # Default to 1 if channels array is empty or missing

        self.logger.info(f"Initializing {channels} transcription channel(s) using {SPEECH_PROVIDER} provider with input language {self.input_language}")

        # Initialize one streaming transcription instance *per channel*
        # Each instance will handle only its own audio stream (effectively mono)
        self.streaming_transcriptions = [
            StreamingTranscription(
                language=self.input_language,
                # channels=1, # This parameter might be removed from StreamingTranscription if each instance is always mono
                logger=self.logger.getChild(f"TranscriptionChannel_{i}")
            ) for i in range(channels)
        ]

        for i, transcription in enumerate(self.streaming_transcriptions):
            transcription.start_streaming()
            # Start a separate response processing task for each instance/channel
            task = asyncio.create_task(self.process_transcription_responses(i)) # Pass channel index 'i'
            self.process_responses_tasks.append(task)

        self.logger.info("Transcription response processing tasks started.")


    async def handle_ping(self, msg: dict):
        self.logger.debug("Received ping, sending pong")
        pong_msg = {
            "version": "2",
            "type": "pong",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {} # Pong parameters are typically empty
        }
        if await self._send_json(pong_msg):
            self.server_seq += 1
        else:
            self.logger.error("Failed to send pong response")
            # Consider if a failed pong should trigger disconnect
            # await self.disconnect_session(reason="error", info="Failed to send pong message")


    async def handle_close(self, msg: dict):
        self.logger.info(f"Received 'close' from Genesys. Reason: {msg['parameters'].get('reason')}")

        # Stop feeding audio to transcription instances immediately
        self.running = False # Signal loops to stop
        for transcription in self.streaming_transcriptions:
             transcription.stop_streaming() # Gracefully stop transcription threads/tasks

        # Wait briefly for any final transcription results
        await asyncio.sleep(0.5) # Adjust delay as needed

        # Cancel response processing tasks if still running
        for task in self.process_responses_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    self.logger.info("Response processing task cancelled.")

        # Send the 'closed' message
        closed_msg = {
            "version": "2",
            "type": "closed",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq,
            "id": self.session_id,
            "parameters": {
                "summary": "Session closed normally." # Optional summary
            }
        }
        if await self._send_json(closed_msg):
            self.server_seq += 1
            self.logger.info("Sent 'closed' response to Genesys.")
        else:
            self.logger.error("Failed to send closed response")
            # No need to disconnect here, client will close the connection

        duration = time.time() - self.start_time
        self.logger.info(
            f"Session stats - Duration: {duration:.2f}s, "
            f"Frames sent (to Genesys): {self.audio_frames_sent}, "
            f"Frames received (from Genesys): {self.audio_frames_received}, "
            f"Total Samples Processed: {self.total_samples_processed}"
        )

        # self.running is already False, the main connection loop will exit


    async def disconnect_session(self, reason="completed", info=""):
        # Use this for server-initiated disconnects or fatal errors
        if not self.running:
             self.logger.info("Disconnect requested, but session already stopping.")
             return # Avoid sending disconnect if already closing

        self.logger.warning(f"Disconnecting session. Reason: {reason}, Info: {info}")
        self.running = False # Stop processing loops

        for transcription in self.streaming_transcriptions:
             transcription.stop_streaming()
        for task in self.process_responses_tasks:
             if not task.done():
                task.cancel()

        try:
            if not self.session_id:
                 self.logger.warning("Cannot send disconnect, session_id not set.")
                 return

            disconnect_msg = {
                "version": "2",
                "type": "disconnect",
                "seq": self.server_seq + 1,
                "clientseq": self.client_seq, # Use last known client seq
                "id": self.session_id,
                "parameters": {
                    "reason": reason,
                    "info": info,
                    "outputVariables": {} # Optional output variables
                }
            }
            # Try sending disconnect, but don't rely on it if connection is already bad
            try:
                if self.ws.open:
                    if await self._send_json(disconnect_msg):
                        self.server_seq += 1
                        self.logger.info("Sent disconnect message to Genesys.")
                    else:
                        self.logger.error("Failed to send disconnect message (rate limit or other send error).")
                else:
                    self.logger.warning("WebSocket already closed, cannot send disconnect message.")
            except Exception as send_err:
                 self.logger.error(f"Error sending disconnect message: {send_err}")

            # Don't wait for closed, just close our end if possible
            if self.ws.open:
                await self.ws.close(code=1000, reason=f"Server initiated disconnect: {reason}")
                self.logger.info("Closed WebSocket connection from server side.")

        except Exception as e:
            self.logger.error(f"Error during disconnect_session cleanup: {e}")
        finally:
             # Ensure running is false even if errors occurred
             self.running = False


    async def handle_audio_frame(self, frame_bytes: bytes):
        if not self.negotiated_media:
            self.logger.warning("Received audio frame before media negotiation completed. Ignoring.")
            return

        self.audio_frames_received += 1
        self.logger.debug(f"Received audio frame: {len(frame_bytes)} bytes (frame #{self.audio_frames_received})")

        # Determine number of channels from the *negotiated* media
        num_negotiated_channels = len(self.negotiated_media.get("channels", []))
        if num_negotiated_channels == 0:
            num_negotiated_channels = 1 # Should not happen if negotiation succeeded, but safeguard

        # For PCMU, each channel is 1 byte per sample time
        # Calculate samples *per channel* in this frame
        samples_in_frame_per_channel = len(frame_bytes) // num_negotiated_channels
        samples_before_this_frame = self.total_samples_processed

        # Update total samples processed *by the server*
        self.total_samples_processed += samples_in_frame_per_channel
        self.logger.debug(f"Processed {samples_in_frame_per_channel} samples this frame. Total samples processed: {self.total_samples_processed}")


        # Deinterleave stereo audio and feed to respective transcription instances
        if num_negotiated_channels == 2:
            if len(self.streaming_transcriptions) < 2:
                 self.logger.error("Negotiated 2 channels but only have < 2 transcription instances. Check initialization.")
                 return
            left_channel_pcmu = frame_bytes[0::2]  # External channel
            right_channel_pcmu = frame_bytes[1::2] # Internal channel
            # Pass the sample count *before* this frame was processed for offset calculation
            # Call feed_audio on the correct instance for each channel
            self.streaming_transcriptions[0].feed_audio(left_channel_pcmu, samples_before_this_frame)
            self.streaming_transcriptions[1].feed_audio(right_channel_pcmu, samples_before_this_frame)
            self.logger.debug(f"Fed L channel ({len(left_channel_pcmu)} bytes) to instance 0, R channel ({len(right_channel_pcmu)} bytes) to instance 1.")
        elif num_negotiated_channels == 1:
             if len(self.streaming_transcriptions) < 1:
                 self.logger.error("Negotiated 1 channel but have 0 transcription instances. Check initialization.")
                 return
             # Pass the sample count *before* this frame was processed
             self.streaming_transcriptions[0].feed_audio(frame_bytes, samples_before_this_frame)
             self.logger.debug(f"Fed mono channel ({len(frame_bytes)} bytes) to instance 0.")
        else:
             self.logger.warning(f"Unsupported number of channels negotiated: {num_negotiated_channels}. Ignoring audio frame.")

        # Keep a small buffer if needed for other purposes, but not primary audio processing
        self.audio_buffer.append(frame_bytes)
        self.last_frame_time = time.time()


    async def process_transcription_responses(self, channel_index):
        # This task processes responses for a *specific* channel index (0 or 1)
        self.logger.info(f"Starting response processing loop for channel_index {channel_index}")
        if channel_index >= len(self.streaming_transcriptions):
             self.logger.error(f"Invalid channel_index {channel_index} passed to process_transcription_responses. Max index: {len(self.streaming_transcriptions)-1}")
             return

        transcription_instance = self.streaming_transcriptions[channel_index]

        while self.running:
            try:
                # Get response from the specific transcription instance for this channel_index
                # The get_response method in the instance itself doesn't need the index anymore
                response = transcription_instance.get_response()

                if response:
                    self.logger.info(f"Processing transcription response on channel_index {channel_index}")
                    self.logger.debug(f"Response data (channel_index {channel_index}): {response}") # Log raw response for debug

                    if isinstance(response, Exception):
                        self.logger.error(f"Streaming recognition error received on channel_index {channel_index}: {response}")
                        # Decide if error is fatal for the whole session
                        # await self.disconnect_session(reason="error", info=f"Streaming recognition failed on channel {channel_index}")
                        # break # Exit loop for this channel on error, or maybe continue? For now, log and continue.
                        await asyncio.sleep(0.1) # Avoid tight loop on continuous errors
                        continue

                    # --- Process Google Response ---
                    if SPEECH_PROVIDER == 'google':
                        # Google response object has 'results' attribute
                        if not hasattr(response, 'results') or not response.results:
                             self.logger.debug(f"Empty or invalid Google response on channel_index {channel_index}")
                             await asyncio.sleep(0.01)
                             continue

                        for result in response.results:
                            if not result.alternatives:
                                continue

                            alt = result.alternatives[0]
                            transcript_text = alt.transcript
                            source_lang = self.input_language
                            final_text = transcript_text
                            final_lang = source_lang

                            if self.enable_translation:
                                dest_lang = self.destination_language
                                translated_text = await translate_with_gemini(transcript_text, source_lang, dest_lang, self.logger)
                                if translated_text is None:
                                    self.logger.warning(f"Translation failed for text: '{transcript_text}'. Sending original transcript.")
                                    # Keep original text and language if translation fails
                                else:
                                     final_text = translated_text
                                     final_lang = dest_lang

                            # --- Timing Calculation (Google) ---
                            offset_str = "PT0S"
                            duration_str = "PT0S"
                            overall_confidence = 1.0 # Default for non-Chirp2 or if confidence missing
                            tokens = []

                            # Check if we have proper word-level timing
                            has_word_timings = hasattr(alt, "words") and alt.words and len(alt.words) > 0 and all(
                                hasattr(w, "start_offset") and w.start_offset is not None and hasattr(w, "end_offset") and w.end_offset is not None for w in alt.words
                            )

                            if has_word_timings:
                                # Calculate overall offset/duration from actual word timings
                                overall_start_sec = alt.words[0].start_offset.total_seconds()
                                overall_end_sec = alt.words[-1].end_offset.total_seconds()
                                # Ensure start/end are non-negative and end >= start
                                overall_start_sec = max(0.0, overall_start_sec)
                                overall_end_sec = max(overall_start_sec, overall_end_sec)
                                overall_duration_sec = overall_end_sec - overall_start_sec

                                offset_str = f"PT{overall_start_sec:.3f}S"
                                duration_str = f"PT{overall_duration_sec:.3f}S"

                                # Build tokens using actual word timings
                                for w in alt.words:
                                    token_start_sec = max(0.0, w.start_offset.total_seconds())
                                    token_end_sec = max(token_start_sec, w.end_offset.total_seconds())
                                    token_duration_sec = token_end_sec - token_start_sec

                                    word_confidence = 1.0 # Default
                                    if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2' and hasattr(w, "confidence") and w.confidence is not None:
                                        word_confidence = w.confidence

                                    tokens.append({
                                        "type": "word",
                                        "value": w.word,
                                        "confidence": word_confidence,
                                        "offset": f"PT{token_start_sec:.3f}S",
                                        "duration": f"PT{token_duration_sec:.3f}S",
                                        "language": final_lang # Use final language for all tokens
                                    })

                                # Overall confidence for the alternative (only from Chirp 2)
                                if GOOGLE_SPEECH_MODEL.lower() == 'chirp_2' and hasattr(alt, "confidence") and alt.confidence is not None:
                                    overall_confidence = alt.confidence

                            else:
                                # No word timings (e.g., Chirp non-2)
                                self.logger.warning(f"Google result on channel_index {channel_index} lacks word timings. Using approximate overall timing.")
                                # Use total samples processed *up to this point* as a rough offset estimate
                                # This is NOT ideal but better than zero. Needs refinement.
                                approx_offset_sec = self.total_samples_processed / 8000.0
                                offset_str = f"PT{approx_offset_sec:.3f}S"
                                duration_str = "PT0.1S" # Assign a minimal duration

                                # Create a single token for the whole transcript
                                tokens = [{
                                    "type": "word",
                                    "value": final_text, # Use the potentially translated text
                                    "confidence": overall_confidence, # Default 1.0 here
                                    "offset": offset_str,
                                    "duration": duration_str,
                                    "language": final_lang
                                }]

                            # Build the alternative structure
                            alternative = {
                                "confidence": overall_confidence,
                                **({"languages": [final_lang]} if final_lang != self.input_language else {}),
                                "interpretations": [
                                    {
                                        "type": "display",
                                        "transcript": final_text,
                                        "tokens": tokens
                                    }
                                ]
                            }
                            # Pass the correct external channel_index
                            await self._send_transcript_event(channel_index, result.is_final, offset_str, duration_str, alternative)


                    # --- Process OpenAI Response ---
                    elif SPEECH_PROVIDER == 'openai':
                        # OpenAI response object is our custom MockResult
                        if not hasattr(response, 'results') or not response.results:
                             self.logger.debug(f"Empty or invalid OpenAI response on channel_index {channel_index}")
                             await asyncio.sleep(0.01)
                             continue

                        # OpenAI MockResult structure
                        result = response.results[0] # Assuming one result per response
                        if not result.alternatives:
                             continue

                        alt = result.alternatives[0]
                        transcript_text = alt.transcript # Original transcript from OpenAI
                        source_lang = self.input_language
                        final_text = transcript_text
                        final_lang = source_lang

                        if self.enable_translation:
                            dest_lang = self.destination_language
                            translated_text = await translate_with_gemini(transcript_text, source_lang, dest_lang, self.logger)
                            if translated_text is None:
                                self.logger.warning(f"Translation failed for text: '{transcript_text}'. Sending original transcript.")
                                # Keep original text and language
                            else:
                                final_text = translated_text
                                final_lang = dest_lang

                        # --- Timing Calculation (OpenAI) ---
                        # Use start_sample_count and duration_samples stored in the MockResult
                        start_samples = getattr(response, 'start_sample_count', 0)
                        duration_samples = getattr(response, 'duration_samples', 0)

                        overall_start_sec = start_samples / 8000.0
                        overall_duration_sec = duration_samples / 8000.0

                        # Ensure non-negative and valid duration
                        overall_start_sec = max(0.0, overall_start_sec)
                        overall_duration_sec = max(0.0, overall_duration_sec)

                        offset_str = f"PT{overall_start_sec:.3f}S"
                        duration_str = f"PT{overall_duration_sec:.3f}S"

                        # --- Token Generation (OpenAI - Synthetic Timing) ---
                        tokens = []
                        words_list = final_text.split() # Split final (possibly translated) text
                        overall_confidence = alt.confidence # Use confidence from OpenAI Alternative

                        if words_list and overall_duration_sec > 0:
                            per_word_duration_sec = overall_duration_sec / len(words_list)
                            for i, word_text in enumerate(words_list):
                                token_start_sec = overall_start_sec + (i * per_word_duration_sec)
                                token_duration_sec = per_word_duration_sec

                                tokens.append({
                                    "type": "word",
                                    "value": word_text,
                                    "confidence": overall_confidence, # Apply overall confidence to each synthetic token
                                    "offset": f"PT{token_start_sec:.3f}S",
                                    "duration": f"PT{token_duration_sec:.3f}S",
                                    "language": final_lang # Use final language
                                })
                        elif final_text: # Handle case of non-empty transcript but zero duration or no words
                             tokens = [{
                                "type": "word",
                                "value": final_text,
                                "confidence": overall_confidence,
                                "offset": offset_str,
                                "duration": duration_str, # Could be PT0.000S
                                "language": final_lang
                            }]


                        # Build the alternative structure
                        alternative = {
                            "confidence": overall_confidence,
                            **({"languages": [final_lang]} if final_lang != self.input_language else {}),
                            "interpretations": [
                                {
                                    "type": "display",
                                    "transcript": final_text,
                                    "tokens": tokens
                                }
                            ]
                        }
                        # Pass the correct external channel_index
                        await self._send_transcript_event(channel_index, result.is_final, offset_str, duration_str, alternative)

                    else:
                         self.logger.warning(f"Unknown SPEECH_PROVIDER '{SPEECH_PROVIDER}' encountered in response processing.")

                else:
                    # No response available, wait briefly
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                self.logger.info(f"Response processing task for channel_index {channel_index} cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Error in process_transcription_responses loop for channel_index {channel_index}: {e}", exc_info=True)
                # Avoid continuous error loops, wait before retrying
                await asyncio.sleep(0.1)
        self.logger.info(f"Exiting response processing loop for channel_index {channel_index}")


    async def _send_transcript_event(self, channel_index, is_final, offset_str, duration_str, alternative_data):
         """Helper function to construct and send a transcript event."""
         # Ensure channel_index is an integer 0 or 1 for the event payload
         payload_channel_id = int(channel_index)

         transcript_event = {
            "version": "2",
            "type": "event",
            "seq": self.server_seq + 1,
            "clientseq": self.client_seq, # Use last known client seq
            "id": self.session_id,
            "parameters": {
                "entities": [
                    {
                        "type": "transcript",
                        "data": {
                            "id": str(uuid.uuid4()),
                            "channelId": payload_channel_id, # Use the integer index
                            "isFinal": is_final,
                            "offset": offset_str,
                            "duration": duration_str,
                            "alternatives": [ alternative_data ]
                        }
                    }
                ]
            }
        }
         self.logger.info(f"Sending transcript event to Genesys (channelId {payload_channel_id}, final={is_final})")
         self.logger.debug(f"Transcript Event Data: {format_json(transcript_event)}")
         if await self._send_json(transcript_event):
            self.server_seq += 1
            self.audio_frames_sent += 1 # Increment count of messages sent to Genesys
         else:
            self.logger.warning(f"Transcript event dropped for channelId {payload_channel_id} (likely due to rate limiting or connection issue)")


    async def _send_json(self, msg: dict):
        try:
            # Check rate limit before sending
            if not await self.message_limiter.acquire():
                current_rate = self.message_limiter.get_current_rate()
                self.logger.warning(
                    f"Message rate limit exceeded (current rate: {current_rate:.2f}/s). "
                    f"Message type: {msg.get('type')}. Dropping message."
                )
                return False # Message not sent

            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
            return True # Message sent
        except ConnectionClosed as cc:
            self.logger.warning(f"Genesys WebSocket closed while sending JSON message: Code={cc.code}, Reason='{cc.reason}'")
            self.running = False # Stop the session if connection closed
            return False
        except Exception as e:
            self.logger.error(f"Error sending message: {type(e).__name__} - {e}")
            # Consider if certain exceptions should stop the session
            # self.running = False
            return False

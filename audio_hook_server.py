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
        self.audio_frames_sent = 0 # Note: This seems unused, server sends events, not frames
        self.audio_frames_received = 0
        self.rate_limit_state = {
            "retry_count": 0,
            "last_retry_time": 0,
            "in_backoff": False
        }

        self.message_limiter = RateLimiter(GENESYS_MSG_RATE_LIMIT, GENESYS_MSG_BURST_LIMIT)
        self.binary_limiter = RateLimiter(GENESYS_BINARY_RATE_LIMIT, GENESYS_BINARY_BURST_LIMIT)

        self.audio_buffer = deque(maxlen=MAX_AUDIO_BUFFER_SIZE) # Note: This seems unused
        self.last_frame_time = 0 # Note: This seems unused

        # --- Offset Tracking ---
        self.total_samples_processed = 0 # Total samples *received* from client (before adjustments)
        self.offset_adjustment_samples = 0 # Total samples to *subtract* due to pauses/discards
        self.pause_start_time = None # Timestamp when pause started, if any

        # --- Language and Translation ---
        self.input_language = "en-US" # Default, updated in handle_open
        self.destination_language = "en-US" # Default, updated in handle_open
        self.enable_translation = False # Default, updated in handle_open

        # --- Transcription State ---
        self.streaming_transcriptions = [] # List of StreamingTranscription instances (one per channel)
        self.process_responses_tasks = [] # Asyncio tasks for processing responses

        self.logger.info(f"New session started: {self.session_id}")

    async def handle_error(self, msg: dict):
        error_code = msg["parameters"].get("code")
        error_params = msg["parameters"]

        if error_code == 429: # Rate limit error
            retry_after = None
            retry_after_source = "default"

            # Try parsing Genesys-provided ISO duration
            if "retryAfter" in error_params:
                retry_after_duration_str = error_params["retryAfter"]
                try:
                    retry_after = parse_iso8601_duration(retry_after_duration_str)
                    retry_after_source = f"Genesys ({retry_after_duration_str})"
                except ValueError as e:
                    self.logger.warning(f"[Rate Limit] Failed to parse Genesys retryAfter format '{retry_after_duration_str}': {e}")

            # Fallback to HTTP header if available and Genesys format failed
            if retry_after is None and hasattr(self.ws, 'response_headers'):
                http_retry_after = (
                    self.ws.response_headers.get('Retry-After') or
                    self.ws.response_headers.get('retry-after')
                )
                if http_retry_after:
                    try:
                        # Try parsing as seconds first
                        retry_after = float(http_retry_after)
                        retry_after_source = f"HTTP header ({http_retry_after}s)"
                    except ValueError:
                        # Try parsing as ISO duration
                        try:
                            retry_after = parse_iso8601_duration(http_retry_after)
                            retry_after_source = f"HTTP header (ISO: {http_retry_after})"
                        except ValueError:
                            self.logger.warning(f"[Rate Limit] Failed to parse HTTP Retry-After format: {http_retry_after}")

            # Use a default backoff if no specific duration found
            if retry_after is None:
                retry_after = 3.0 # Default 3 seconds
                retry_after_source = "default (3s)"

            self.logger.warning(
                f"[Rate Limit] Received 429 error. Session: {self.session_id}, "
                f"Retry count: {self.rate_limit_state['retry_count']}, "
                f"Using RetryAfter: {retry_after:.2f}s from {retry_after_source}"
            )

            self.rate_limit_state["in_backoff"] = True
            self.rate_limit_state["retry_count"] += 1

            if self.rate_limit_state["retry_count"] > RATE_LIMIT_MAX_RETRIES:
                self.logger.error(
                    f"[Rate Limit] Max retries ({RATE_LIMIT_MAX_RETRIES}) exceeded. Session: {self.session_id}. "
                    f"Disconnecting."
                )
                await self.disconnect_session(reason="error", info="Rate limit max retries exceeded")
                return False # Indicate disconnect

            self.logger.info(f"[Rate Limit] Backing off for {retry_after:.2f}s...")
            await asyncio.sleep(retry_after)
            self.rate_limit_state["in_backoff"] = False
            self.logger.info(f"[Rate Limit] Backoff complete, resuming operations. Session: {self.session_id}")
            return True # Indicate handled, continue session

        # Handle other error codes if needed
        self.logger.error(f"Received unhandled error message: {format_json(msg)}")
        # Optionally disconnect for critical errors
        # await self.disconnect_session(reason="error", info=f"Received error code {error_code}")
        return False # Indicate not handled (or disconnect happened)

    async def handle_message(self, msg: dict):
        msg_type = msg.get("type")
        seq = msg.get("seq", 0)
        self.client_seq = seq # Track client sequence number

        # If in rate limit backoff, only process 'error' messages
        if self.rate_limit_state.get("in_backoff") and msg_type != "error":
            self.logger.debug(f"Skipping message type '{msg_type}' during rate limit backoff")
            return

        # Route message based on type
        if msg_type == "error":
            await self.handle_error(msg)
        elif msg_type == "open":
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
            await self.handle_update(msg) # Added handler for update
        else:
            self.logger.debug(f"Ignoring unknown or unhandled message type: {msg_type}")

    async def handle_discarded(self, msg: dict):
        discarded_duration_str = msg["parameters"].get("discarded")
        if discarded_duration_str:
            try:
                gap_seconds = parse_iso8601_duration(discarded_duration_str)
                # Assuming 8kHz sample rate based on typical PCMU
                sample_rate = self.negotiated_media.get("rate", 8000) if self.negotiated_media else 8000
                gap_samples = int(gap_seconds * sample_rate)
                self.offset_adjustment_samples += gap_samples
                self.logger.info(f"Handled 'discarded' message: gap duration {gap_seconds:.3f}s, adding {gap_samples} samples to offset adjustment. New total adjustment: {self.offset_adjustment_samples} samples.")
            except ValueError as e:
                self.logger.warning(f"Failed to parse discarded duration '{discarded_duration_str}': {e}")
        else:
            self.logger.warning("Received 'discarded' message without 'discarded' parameter.")

    async def handle_paused(self, msg: dict):
        # This message can be received unsolicited (client pause) or as a response to server 'pause'
        if self.pause_start_time is None:
            self.pause_start_time = time.time() # Record when the pause *actually* started according to messages
            self.logger.info("Session is now paused (either client-initiated or server confirmation).")
        else:
            # If we receive 'paused' while already paused, it might be a confirmation or redundant client pause
            self.logger.debug("Received 'paused' message while already in paused state.")

    async def handle_resumed(self, msg: dict):
         # This message indicates the client is no longer paused
        if self.pause_start_time is not None:
            pause_duration_seconds = time.time() - self.pause_start_time
            sample_rate = self.negotiated_media.get("rate", 8000) if self.negotiated_media else 8000
            gap_samples = int(pause_duration_seconds * sample_rate)
            self.offset_adjustment_samples += gap_samples
            self.logger.info(f"Handled 'resumed' message: pause ended. Duration {pause_duration_seconds:.3f}s, adding {gap_samples} samples to offset adjustment. New total adjustment: {self.offset_adjustment_samples} samples.")
            self.pause_start_time = None # Reset pause start time
        else:
            # Receiving 'resumed' when not paused might happen if server resumed but client was never paused
            self.logger.info("Received 'resumed' message, session is now active (if it wasn't already).")


    async def handle_update(self, msg: dict):
        # Handle changes during the conversation, e.g., language change
        params = msg.get("parameters", {})
        if "language" in params:
            new_language_raw = params["language"]
            new_language_normalized = normalize_language_code(new_language_raw)
            if new_language_normalized != self.destination_language:
                self.logger.info(f"Language updated via 'update' message: '{self.destination_language}' -> '{new_language_normalized}'")
                self.destination_language = new_language_normalized
                # Note: Input language for transcription usually doesn't change mid-call.
                # If the STT engine needs reconfiguration, it would happen here.
                # For now, we just update the destination language for potential translation.
            else:
                 self.logger.debug(f"Received 'update' message with unchanged language: {new_language_normalized}")
        else:
             self.logger.debug(f"Received 'update' message with no relevant changes: {params}")
        # Server should not send a response to an 'update' message unless disconnecting.


    async def handle_open(self, msg: dict):
        self.session_id = msg["id"] # Use the session ID provided by Genesys
        params = msg["parameters"]

        # --- Language Configuration ---
        custom_config = params.get("customConfig", {})
        self.input_language = normalize_language_code(custom_config.get("inputLanguage", "en-US"))
        self.enable_translation = custom_config.get("enableTranslation", False)
        # Destination language comes from the main 'language' parameter
        self.destination_language = normalize_language_code(params.get("language", "en-US"))

        self.logger.info(f"Open received. Session: {self.session_id}. Input lang: {self.input_language}. Dest lang: {self.destination_language}. Translate enabled: {self.enable_translation}")

        # --- Probe Connection Handling ---
        # Genesys sends a probe with null IDs to check connectivity and get supported languages
        is_probe = (
            params.get("conversationId") == "00000000-0000-0000-0000-000000000000" and
            params.get("participant", {}).get("id") == "00000000-0000-0000-0000-000000000000"
        )

        if is_probe:
            self.logger.info("Detected probe connection. Responding with supported languages.")
            supported_langs = [lang.strip() for lang in SUPPORTED_LANGUAGES.split(",") if lang.strip()]
            opened_msg_params = {
                "startPaused": False, # Probe doesn't need pausing
                "media": [], # No media needed for probe
                "supportedLanguages": supported_langs
            }
            # If client requested supported languages explicitly, include them
            if params.get("supportedLanguages"):
                 opened_msg_params["supportedLanguages"] = supported_langs

            opened_msg = self._build_message("opened", parameters=opened_msg_params)

            if await self._send_json(opened_msg):
                self.server_seq += 1
                self.logger.info(f"Sent 'opened' for probe with languages: {supported_langs}")
            else:
                # If send fails, we can't really disconnect a probe cleanly
                self.logger.error("Failed to send 'opened' response for probe connection.")
            self.running = False # Close handler for probe connection
            return

        # --- Regular Connection Handling ---
        offered_media = params.get("media", [])
        chosen_media_config = None
        # Find the first supported media format (prefer stereo PCMU)
        supported_formats = [
             {"format": "PCMU", "rate": 8000, "channels": ["external", "internal"]},
             {"format": "PCMU", "rate": 8000, "channels": ["external"]},
             {"format": "PCMU", "rate": 8000, "channels": ["internal"]},
        ]

        for supported in supported_formats:
             for offered in offered_media:
                  # Compare format, rate, and channel structure
                  if (offered.get("format") == supported["format"] and
                      offered.get("rate") == supported["rate"] and
                      len(offered.get("channels", [])) == len(supported["channels"]) and
                      all(c in offered.get("channels", []) for c in supported["channels"])):
                       chosen_media_config = offered # Choose the offered config that matches
                       break
             if chosen_media_config:
                  break


        if not chosen_media_config:
            self.logger.error(f"No supported media format found in offer: {offered_media}")
            await self.disconnect_session(reason="error", info="No supported media format found")
            return

        self.negotiated_media = chosen_media_config
        self.logger.info(f"Negotiated media format: {self.negotiated_media}")

        # --- Send 'opened' Response ---
        opened_msg_params = {
            "startPaused": False, # Start unpaused by default
            "media": [self.negotiated_media] # Echo back the chosen media config
        }
        # Include supported languages if requested
        if params.get("supportedLanguages"):
            supported_langs = [lang.strip() for lang in SUPPORTED_LANGUAGES.split(",") if lang.strip()]
            opened_msg_params["supportedLanguages"] = supported_langs

        opened_msg = self._build_message("opened", parameters=opened_msg_params)

        if await self._send_json(opened_msg):
            self.server_seq += 1
            self.logger.info("Session opened successfully.")
        else:
            self.logger.error("Failed to send 'opened' message.")
            # No need to call disconnect_session here, _send_json handles closure on failure
            return

        # --- Initialize Transcription Streams ---
        num_channels = len(self.negotiated_media.get("channels", []))
        if num_channels == 0:
             num_channels = 1 # Default to 1 channel if none specified

        self.logger.info(f"Initializing {num_channels} transcription stream(s) for language '{self.input_language}' using {SPEECH_PROVIDER} provider.")
        try:
            self.streaming_transcriptions = [
                StreamingTranscription(self.input_language, 1, self.logger.getChild(f"Stream_{channel}"))
                for channel in range(num_channels)
            ]
            for transcription in self.streaming_transcriptions:
                transcription.start_streaming() # Start background threads

            # Create tasks to process responses from each stream
            self.process_responses_tasks = [
                asyncio.create_task(self.process_transcription_responses(channel))
                for channel in range(num_channels)
            ]
            self.logger.info("Transcription streams and response processors started.")
        except Exception as e:
             self.logger.error(f"Failed to initialize transcription streams: {e}", exc_info=True)
             await self.disconnect_session(reason="error", info="Failed to initialize transcription backend")


    async def handle_ping(self, msg: dict):
        # Respond immediately to ping
        pong_msg = self._build_message("pong")
        if not await self._send_json(pong_msg):
            self.logger.warning("Failed to send pong response.")
            # Don't necessarily disconnect here, connection might recover
        else:
             self.server_seq += 1

    async def handle_close(self, msg: dict):
        self.logger.info(f"Received 'close' from Genesys. Reason: {msg['parameters'].get('reason', 'N/A')}")
        self.running = False # Signal loops to stop

        # Stop transcription streams gracefully
        for transcription in self.streaming_transcriptions:
            if transcription:
                transcription.stop_streaming()
        # Cancel response processing tasks
        for task in self.process_responses_tasks:
            if task and not task.done():
                task.cancel()
        try:
             if self.process_responses_tasks:
                  await asyncio.gather(*self.process_responses_tasks, return_exceptions=True)
        except asyncio.CancelledError:
             self.logger.info("Response processing tasks cancelled.")


        # Send 'closed' response
        closed_msg = self._build_message("closed", parameters={"summary": "Session closed normally"})
        if await self._send_json(closed_msg):
            self.server_seq += 1
            self.logger.info("Sent 'closed' response to Genesys.")
        else:
            self.logger.error("Failed to send 'closed' response.")
            # Connection likely already closed by _send_json

        duration = time.time() - self.start_time
        self.logger.info(
            f"Session stats - Duration: {duration:.2f}s, "
            f"Frames received: {self.audio_frames_received}"
        )
        # No need to close websocket here, Genesys should close after receiving 'closed'


    async def disconnect_session(self, reason="completed", info=""):
        if not self.running: # Avoid duplicate disconnects
             return
        self.running = False # Stop processing loops immediately
        self.logger.warning(f"Disconnecting session {self.session_id}. Reason: {reason}, Info: {info}")

        # Stop transcription streams
        for transcription in self.streaming_transcriptions:
             if transcription:
                  transcription.stop_streaming()
        # Cancel response processing tasks
        for task in self.process_responses_tasks:
             if task and not task.done():
                  task.cancel()
        try:
             if self.process_responses_tasks:
                  await asyncio.gather(*self.process_responses_tasks, return_exceptions=True)
        except asyncio.CancelledError:
             self.logger.info("Response processing tasks cancelled during disconnect.")

        # Try sending a 'disconnect' message if websocket is still open
        if self.ws.open:
            disconnect_msg = self._build_message("disconnect", parameters={"reason": reason, "info": info, "outputVariables": {}})
            if await self._send_json(disconnect_msg):
                self.server_seq += 1
                self.logger.info("Sent 'disconnect' message to Genesys.")
                # Wait briefly for Genesys to potentially close
                try:
                    await asyncio.wait_for(self.ws.wait_closed(), timeout=1.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Timeout waiting for client to close connection after disconnect message.")
                except ConnectionClosed:
                    pass # Expected if client closes quickly
            else:
                self.logger.warning("Failed to send 'disconnect' message (connection likely already closed).")

        # Ensure websocket is closed from server side if still open
        if self.ws.open:
             await self.ws.close(code=1000, reason=f"Server initiated disconnect: {reason}")
             self.logger.info("Server closed WebSocket connection.")


    def _build_message(self, msg_type: str, parameters: dict = None) -> dict:
        """Helper to construct standard message structure"""
        return {
            "version": "2",
            "type": msg_type,
            "seq": self.server_seq + 1, # Tentative sequence number (incremented on successful send)
            "clientseq": self.client_seq, # Echo last received client sequence
            "id": self.session_id,
            "parameters": parameters if parameters is not None else {}
        }


    async def handle_audio_frame(self, frame_bytes: bytes):
        self.audio_frames_received += 1
        # self.logger.debug(f"Received audio frame: {len(frame_bytes)} bytes (frame #{self.audio_frames_received})")

        if not self.negotiated_media:
            self.logger.warning("Received audio frame before media negotiation complete. Ignoring.")
            return

        # Calculate number of sample times in this frame
        sample_rate = self.negotiated_media.get("rate", 8000)
        num_channels = len(self.negotiated_media.get("channels", []))
        if num_channels == 0: num_channels = 1
        bytes_per_sample_per_channel = 1 # For PCMU
        bytes_per_sample_time = num_channels * bytes_per_sample_per_channel
        sample_times = len(frame_bytes) // bytes_per_sample_time

        # Store the starting sample time for this frame *before* incrementing total
        current_frame_start_sample = self.total_samples_processed

        # Increment total samples processed *from the client*
        self.total_samples_processed += sample_times

        # Feed audio to appropriate transcription instances
        if num_channels == 2:
            # Deinterleave stereo PCMU audio
            left_channel_ulaw = frame_bytes[0::2]  # External (assuming standard order)
            right_channel_ulaw = frame_bytes[1::2] # Internal (assuming standard order)

            # Feed each channel with its corresponding starting sample time
            if len(self.streaming_transcriptions) > 0 and self.streaming_transcriptions[0]:
                 self.streaming_transcriptions[0].feed_audio(left_channel_ulaw, 0, current_frame_start_sample)
            if len(self.streaming_transcriptions) > 1 and self.streaming_transcriptions[1]:
                 self.streaming_transcriptions[1].feed_audio(right_channel_ulaw, 1, current_frame_start_sample)

        elif num_channels == 1:
            # Feed mono audio with its starting sample time
            if len(self.streaming_transcriptions) > 0 and self.streaming_transcriptions[0]:
                 self.streaming_transcriptions[0].feed_audio(frame_bytes, 0, current_frame_start_sample)
        else:
             self.logger.warning(f"Unsupported number of channels ({num_channels}) in negotiated media. Ignoring audio frame.")

        # Note: self.audio_buffer seems unused, removing append
        # self.audio_buffer.append(frame_bytes)


    async def process_transcription_responses(self, channel: int):
        """Coroutine to continuously process responses from a single transcription stream"""
        self.logger.info(f"Starting response processor for channel {channel}")
        sample_rate = self.negotiated_media.get("rate", 8000) if self.negotiated_media else 8000

        while self.running:
            try:
                if not self.streaming_transcriptions or channel >= len(self.streaming_transcriptions) or not self.streaming_transcriptions[channel]:
                     await asyncio.sleep(0.1) # Wait if transcription not ready
                     continue

                response = self.streaming_transcriptions[channel].get_response(channel) # Pass channel index (though OpenAI class ignores it)

                if response:
                    # self.logger.debug(f"Processing transcription response on channel {channel}: {response}")
                    if isinstance(response, Exception):
                        self.logger.error(f"Streaming recognition error on channel {channel}: {response}", exc_info=response)
                        await self.disconnect_session(reason="error", info=f"Streaming recognition failed on channel {channel}")
                        break # Stop processing for this channel

                    if not hasattr(response, 'results') or not response.results:
                         self.logger.debug(f"Channel {channel}: Received empty or invalid response object.")
                         continue

                    for result in response.results:
                        if not result.alternatives:
                            continue

                        # --- Process the top alternative ---
                        alt = result.alternatives[0]
                        original_transcript_text = alt.transcript

                        if not original_transcript_text or not original_transcript_text.strip():
                             self.logger.debug(f"Channel {channel}: Skipping empty transcript.")
                             continue

                        # --- Translation Step (if enabled) ---
                        text_to_send = original_transcript_text
                        language_to_send = self.input_language # Default to input lang
                        if self.enable_translation:
                            dest_lang = self.destination_language
                            source_lang = self.input_language # Assume transcription is in input lang
                            self.logger.info(f"Channel {channel}: Translating '{original_transcript_text}' from {source_lang} to {dest_lang}")
                            translated_text = await translate_with_gemini(original_transcript_text, source_lang, dest_lang, self.logger)
                            if translated_text:
                                text_to_send = translated_text
                                language_to_send = dest_lang # Language of the *sent* text is destination
                                self.logger.info(f"Channel {channel}: Translation successful: '{text_to_send}'")
                            else:
                                self.logger.warning(f"Channel {channel}: Translation failed for '{original_transcript_text}'. Sending original transcript.")
                                # Keep text_to_send as original_transcript_text
                                # Keep language_to_send as input_language
                        else:
                             language_to_send = self.input_language # No translation, language is input


                        # --- Determine Timing Source ---
                        # Check for the flag set by the OpenAI transcription class
                        is_synthetic = getattr(alt, 'is_synthetic_timing', False)
                        has_word_data = hasattr(alt, "words") and alt.words

                        if not has_word_data:
                             self.logger.warning(f"Channel {channel}: Transcription result missing word data. Cannot generate event.")
                             continue # Need word data for offsets/durations

                        # --- Calculate Offsets and Durations ---
                        # These calculations MUST account for self.offset_adjustment_samples
                        # Offsets/Durations are relative to the *start of the stream*, adjusted for pauses/discards.

                        # Get the stream-relative start/end times from the word data
                        # (These are already timedeltas relative to stream start, thanks to the revised OpenAI class)
                        first_word_start_seconds = alt.words[0].start_offset.total_seconds()
                        last_word_end_seconds = alt.words[-1].end_offset.total_seconds()

                        # Calculate the adjusted start offset in seconds
                        adjustment_seconds = self.offset_adjustment_samples / sample_rate
                        overall_start_adjusted_seconds = max(0.0, first_word_start_seconds - adjustment_seconds)

                        # Calculate the duration based on the word timings (unadjusted)
                        overall_duration_seconds = max(0.0, last_word_end_seconds - first_word_start_seconds)

                        # Format according to ISO 8601 Duration (PT#S)
                        offset_str = f"PT{overall_start_adjusted_seconds:.3f}S"
                        duration_str = f"PT{overall_duration_seconds:.3f}S"

                        # --- Determine Confidence ---
                        # Use overall confidence from alternative if available, else default
                        overall_confidence = alt.confidence if hasattr(alt, 'confidence') and alt.confidence is not None else 0.9

                        # Specific handling for Google Chirp model (non-Chirp 2) which doesn't provide reliable confidence
                        if SPEECH_PROVIDER == 'google' and GOOGLE_SPEECH_MODEL.lower() == 'chirp':
                             overall_confidence = 1.0 # Force 1.0 for original Chirp model


                        # --- Build Tokens ---
                        tokens = []
                        for w in alt.words:
                            word_start_seconds = w.start_offset.total_seconds()
                            word_end_seconds = w.end_offset.total_seconds()

                            # Calculate adjusted offset for the token
                            token_start_adjusted_seconds = max(0.0, word_start_seconds - adjustment_seconds)
                            # Duration is based on the word's unadjusted start/end
                            token_duration_seconds = max(0.0, word_end_seconds - word_start_seconds)

                            token_offset_str = f"PT{token_start_adjusted_seconds:.3f}S"
                            token_duration_str = f"PT{token_duration_seconds:.3f}S"

                            # Use word confidence if available and reliable, else use overall
                            token_confidence = w.confidence if hasattr(w, 'confidence') and w.confidence is not None else overall_confidence
                            # Override for Google Chirp model
                            if SPEECH_PROVIDER == 'google' and GOOGLE_SPEECH_MODEL.lower() == 'chirp':
                                 token_confidence = 1.0

                            tokens.append({
                                "type": "word", # Assuming all tokens are words for now
                                "value": w.word,
                                "confidence": round(token_confidence, 4), # Round confidence
                                "offset": token_offset_str,
                                "duration": token_duration_str,
                                "language": language_to_send # Language of the token is the language being sent
                            })

                        if not tokens:
                             self.logger.warning(f"Channel {channel}: No tokens generated for transcript '{text_to_send}'. Skipping event.")
                             continue

                        # --- Construct Transcript Event ---
                        alternative_payload = {
                            "confidence": round(overall_confidence, 4),
                            # "languages": [language_to_send], # Optional: report language if detected/translated
                            "interpretations": [
                                {
                                    "type": "display",
                                    "transcript": text_to_send, # The (potentially translated) text
                                    "tokens": tokens
                                }
                            ]
                        }
                        # Add language list if translation happened or if STT provided it
                        if self.enable_translation or (hasattr(alt, 'languages') and alt.languages):
                             alternative_payload["languages"] = [language_to_send]


                        transcript_data = {
                            "id": str(uuid.uuid4()), # Unique ID for this transcript event
                            "channelId": channel, # 0 for external, 1 for internal
                            "isFinal": result.is_final,
                            "offset": offset_str,
                            "duration": duration_str,
                            "alternatives": [alternative_payload]
                        }

                        transcript_event = self._build_message(
                            "event",
                            parameters={"entities": [{"type": "transcript", "data": transcript_data}]}
                        )

                        # --- Send Event ---
                        self.logger.info(f"Channel {channel}: Sending transcript event. Offset: {offset_str}, Duration: {duration_str}, Text: '{text_to_send[:50]}...'")
                        # self.logger.debug(f"Transcript event details: {format_json(transcript_event)}")
                        if await self._send_json(transcript_event):
                            self.server_seq += 1
                        else:
                            # Rate limiting or connection closed, message was dropped or failed
                            self.logger.warning(f"Channel {channel}: Failed to send transcript event (rate limited or connection closed).")
                            # If connection closed, self.running should be false, loop will exit

                else:
                    # No response from queue, wait briefly
                    await asyncio.sleep(0.02)

            except asyncio.CancelledError:
                 self.logger.info(f"Response processor for channel {channel} cancelled.")
                 break
            except Exception as e:
                self.logger.error(f"Error in response processor for channel {channel}: {type(e).__name__} - {e}", exc_info=True)
                # Avoid continuous error loops, wait a bit
                await asyncio.sleep(1)

        self.logger.info(f"Response processor for channel {channel} stopped.")


    async def _send_json(self, msg: dict):
        """Send JSON message with rate limiting and error handling"""
        try:
            if not self.ws.open:
                 self.logger.warning(f"Attempted to send message type '{msg.get('type')}' but WebSocket is closed.")
                 self.running = False # Ensure loops stop
                 return False

            if not await self.message_limiter.acquire():
                current_rate = self.message_limiter.get_current_rate()
                self.logger.warning(
                    f"Message rate limit exceeded (current rate: {current_rate:.2f}/s). "
                    f"Dropping message type: {msg.get('type')}"
                )
                return False # Message not sent

            # self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
            return True # Message sent successfully

        except ConnectionClosed as cc:
            self.logger.warning(f"WebSocket closed while sending message type '{msg.get('type')}'. Code: {cc.code}, Reason: {cc.reason}")
            self.running = False # Stop loops
            return False
        except Exception as e:
            self.logger.error(f"Error sending message type '{msg.get('type')}': {type(e).__name__} - {e}", exc_info=False)
            # Consider closing connection on persistent send errors
            # await self.disconnect_session(reason="error", info=f"Failed to send message: {e}")
            self.running = False # Stop loops on potentially critical send error
            return False

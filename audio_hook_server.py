import asyncio
import uuid
import json
import time
import websockets
import tempfile
import audioop
import logging
import importlib
from websockets.exceptions import ConnectionClosed

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
        self.total_samples = 0
        self.discarded_samples = 0
        self.paused_samples = 0
        self.pause_start_time = None
        self.input_language = "en-US"
        self.destination_language = "en-US"
        self.enable_translation = False
        self.speech_provider = DEFAULT_SPEECH_PROVIDER
        self.StreamingTranscription = None
        self.streaming_transcriptions = []
        self.process_responses_tasks = []
        self.lock = asyncio.Lock()

    def _load_transcription_provider(self, provider_name=None):
        provider = provider_name or self.speech_provider
        provider = provider.lower()
        try:
            if provider == 'openai':
                module = importlib.import_module('openai_speech_transcription')
            else:
                module = importlib.import_module('google_speech_transcription')
            self.StreamingTranscription = module.StreamingTranscription
        except ImportError as e:
            if provider != 'google':
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
                except ValueError:
                    pass
            if retry_after is None and hasattr(self.ws, 'response_headers'):
                http_retry_after = (
                    self.ws.response_headers.get('Retry-After') or 
                    self.ws.response_headers.get('retry-after')
                )
                if http_retry_after:
                    try:
                        retry_after = float(http_retry_after)
                    except ValueError:
                        try:
                            retry_after = parse_iso8601_duration(http_retry_after)
                        except ValueError:
                            pass
            self.rate_limit_state["in_backoff"] = True
            self.rate_limit_state["retry_count"] += 1
            if self.rate_limit_state["retry_count"] > RATE_LIMIT_MAX_RETRIES:
                await self.disconnect_session(reason="error", info="Rate limit max retries exceeded")
                return False
            await asyncio.sleep(retry_after if retry_after is not None else 3)
            self.rate_limit_state["in_backoff"] = False
            return True
        return False

    async def handle_message(self, msg: dict):
        msg_type = msg.get("type")
        seq = msg.get("seq", 0)
        self.client_seq = seq
        async with self.lock:
            if self.rate_limit_state.get("in_backoff") and msg_type != "error":
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
                pass
            else:
                pass

    async def handle_discarded(self, msg: dict):
        discarded_duration_str = msg["parameters"].get("discarded")
        if discarded_duration_str:
            try:
                gap = parse_iso8601_duration(discarded_duration_str)
                gap_samples = int(gap * 8000)
                async with self.lock:
                    self.discarded_samples += gap_samples
            except ValueError:
                pass

    async def handle_paused(self, msg: dict):
        if self.pause_start_time is None:
            self.pause_start_time = time.time()

    async def handle_resumed(self, msg: dict):
        if self.pause_start_time is not None:
            pause_duration = time.time() - self.pause_start_time
            gap_samples = int(pause_duration * 8000)
            async with self.lock:
                self.paused_samples += gap_samples
            self.pause_start_time = None

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
        if is_probe:
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
        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1
        async with self.lock:
            self.streaming_transcriptions = [self.StreamingTranscription(self.input_language, 1, self.logger) for _ in range(channels)]
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
            await self.disconnect_session(reason="error", info="Failed to send pong message")

    async def handle_close(self, msg: dict):
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
            await self.disconnect_session(reason="error", info="Failed to send closed message")
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
            await asyncio.wait_for(self.ws.wait_closed(), timeout=5.0)
        except asyncio.TimeoutError:
            pass
        except Exception:
            pass
        finally:
            self.running = False
            for transcription in self.streaming_transcriptions:
                transcription.stop_streaming()
            for task in self.process_responses_tasks:
                task.cancel()

    async def handle_audio_frame(self, frame_bytes: bytes):
        if not frame_bytes or len(frame_bytes) % 2 != 0:
            return
        self.audio_frames_received += 1
        channels = len(self.negotiated_media.get("channels", [])) if self.negotiated_media and "channels" in self.negotiated_media else 1
        if channels == 0:
            channels = 1
        sample_times = len(frame_bytes) // channels
        async with self.lock:
            self.total_samples += sample_times
            self.audio_buffer.append(frame_bytes)
        if channels == 2:
            left_channel = frame_bytes[0::2]
            right_channel = frame_bytes[1::2]
            self.streaming_transcriptions[0].feed_audio(left_channel, 0)
            self.streaming_transcriptions[1].feed_audio(right_channel, 0)
        else:
            self.streaming_transcriptions[0].feed_audio(frame_bytes, 0)

    async def process_transcription_responses(self, channel):
        while self.running:
            response = self.streaming_transcriptions[channel].get_response(0)
            if response:
                if isinstance(response, Exception):
                    await self.disconnect_session(reason="error", info="Streaming recognition failed")
                    break
                for result in response.results:
                    if not result.alternatives:
                        continue
                    alt = result.alternatives[0]
                    transcript_text = alt.transcript
                    source_lang = self.input_language
                    if self.enable_translation:
                        dest_lang = self.destination_language
                        translated_text = await translate_with_gemini(transcript_text, source_lang, dest_lang, self.logger)
                        if translated_text is None:
                            continue
                    else:
                        dest_lang = source_lang
                        translated_text = transcript_text
                    async with self.lock:
                        adjustment_seconds = self.discarded_samples / 8000.0
                    default_confidence = 1.0
                    use_word_timings = hasattr(alt, "words") and alt.words and len(alt.words) > 0 and all(
                        hasattr(w, "start_offset") and w.start_offset is not None for w in alt.words
                    )
                    if use_word_timings:
                        overall_start = alt.words[0].start_offset.total_seconds()
                        overall_end = alt.words[-1].end_offset.total_seconds()
                        overall_duration = overall_end - overall_start
                    else:
                        overall_start = (self.total_samples - self.discarded_samples) / 8000.0
                        overall_duration = 1.0
                    overall_start -= adjustment_seconds
                    if overall_start < 0:
                        overall_start = 0
                    offset_str = f"PT{overall_start:.2f}S"
                    duration_str = f"PT{overall_duration:.2f}S"
                    overall_confidence = default_confidence
                    if hasattr(alt, "confidence") and alt.confidence is not None and alt.confidence > 0.0:
                        overall_confidence = alt.confidence
                    if self.enable_translation:
                        words_list = translated_text.split()
                        if words_list and overall_duration > 0:
                            per_word_duration = overall_duration / len(words_list)
                            tokens = []
                            for i, word in enumerate(words_list):
                                token_offset = overall_start + i * per_word_duration
                                confidence = overall_confidence
                                tokens.append({
                                    "type": "word",
                                    "value": word,
                                    "confidence": confidence,
                                    "offset": f"PT{token_offset:.2f}S",
                                    "duration": f"PT{per_word_duration:.2f}S",
                                    "language": dest_lang
                                })
                        else:
                            tokens = [{
                                "type": "word",
                                "value": translated_text,
                                "confidence": overall_confidence,
                                "offset": offset_str,
                                "duration": duration_str,
                                "language": dest_lang
                            }]
                    else:
                        if use_word_timings:
                            tokens = []
                            for w in alt.words:
                                token_offset = w.start_offset.total_seconds() - adjustment_seconds
                                token_duration = w.end_offset.total_seconds() - w.start_offset.total_seconds()
                                if token_offset < 0:
                                    token_offset = 0
                                word_confidence = default_confidence
                                if hasattr(w, "confidence") and w.confidence is not None and w.confidence > 0.0:
                                    word_confidence = w.confidence
                                tokens.append({
                                    "type": "word",
                                    "value": w.word,
                                    "confidence": word_confidence,
                                    "offset": f"PT{token_offset:.2f}S",
                                    "duration": f"PT{token_duration:.2f}S",
                                    "language": dest_lang
                                })
                        else:
                            tokens = [{
                                "type": "word",
                                "value": transcript_text,
                                "confidence": overall_confidence,
                                "offset": offset_str,
                                "duration": duration_str,
                                "language": dest_lang
                            }]
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
                                        "isFinal": result.is_final,
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
                    if await self._send_json(transcript_event):
                        self.server_seq += 1
            else:
                await asyncio.sleep(0.01)

    async def _send_json(self, msg: dict):
        try:
            if not await self.message_limiter.acquire():
                return False
            max_retries = 3
            retry_delay = 1
            for attempt in range(max_retries):
                try:
                    await self.ws.send(json.dumps(msg))
                    return True
                except ConnectionClosed:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        self.running = False
                        return False
                except Exception:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        return False
        except Exception:
            return False

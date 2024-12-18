"""
Genesys Audio Connector <--> OpenAI Realtime API bridging server
"""

import asyncio
import json
import uuid
import logging
import audioop
import time
import base64
import os
import ssl
import http
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

import websockets

# Load environment variables from .env file
env_path = Path('.') / '.env'
if not env_path.exists():
    raise FileNotFoundError("Please create a .env file with OPENAI_API_KEY and OPENAI_MODEL")
load_dotenv(env_path)

DEBUG = os.getenv('DEBUG', 'false').lower()
OPENAI_VOICE = os.getenv('OPENAI_VOICE', 'sage')

# Read instructions from system_prompt.txt
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    INSTRUCTIONS = f.read()

###############################################################################
# Configuration constants
###############################################################################

# Server settings
GENESYS_LISTEN_HOST = "0.0.0.0"   # Bind to all available interfaces
GENESYS_LISTEN_PORT = 443         # Genesys will connect to wss://yourdomain:443/audiohook
GENESYS_PATH = "/audiohook"

# OpenAI Realtime API settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

OPENAI_MODEL = os.getenv('OPENAI_MODEL')
if not OPENAI_MODEL:
    OPENAI_MODEL = "gpt-4o-mini-realtime-preview"

OPENAI_REALTIME_URL = f"wss://api.openai.com/v1/realtime?model={OPENAI_MODEL}"

# SSL Configuration
SSL_CERT_PATH = "C:/tools/nginx-1.27.3/certs/gcaudiotogemini.ddns.net-chain.pem"
SSL_KEY_PATH = "C:/tools/nginx-1.27.3/certs/gcaudiotogemini.ddns.net-key.pem"

if not os.path.exists(SSL_CERT_PATH) or not os.path.exists(SSL_KEY_PATH):
    raise FileNotFoundError("SSL certificate files not found at specified paths.")

ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ssl_context.load_cert_chain(SSL_CERT_PATH, SSL_KEY_PATH)
ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
ssl_context.maximum_version = ssl.TLSVersion.TLSv1_3
ssl_context.set_ciphers('ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256')

# Logging
LOG_FILE = "logging.txt"
LOGGING_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"

if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

logging.basicConfig(
    level=logging.DEBUG,
    format=LOGGING_FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("GenesysOpenAIBridge")
websockets_logger = logging.getLogger('websockets')

# Add a filter to redact sensitive info if DEBUG=true
class SensitiveInfoFilter(logging.Filter):
    def filter(self, record):
        if DEBUG == 'true':
            msg = record.getMessage()
            # Redact Authorization header value
            if 'Authorization:' in msg:
                msg = re.sub(r"(Authorization:\s*Bearer\s+)\S+", r"\1********", msg)
            # Redact Sec-WebSocket-Key header value
            if 'Sec-WebSocket-Key:' in msg:
                msg = re.sub(r"(Sec-WebSocket-Key:\s*)\S+", r"\1********", msg)
            record.msg = msg
        return True

websockets_logger.addFilter(SensitiveInfoFilter())

if DEBUG != 'true':
    logger.setLevel(logging.INFO)
    websockets_logger.setLevel(logging.INFO)

###############################################################################
# Utility Functions
###############################################################################

def decode_pcmu_to_pcm16(ulaw_bytes: bytes) -> bytes:
    """
    Decodes PCMU (μ-law) to 16-bit linear PCM, maintaining 8kHz sample rate.
    """
    return audioop.ulaw2lin(ulaw_bytes, 2)

def encode_pcm16_to_pcmu(pcm16_bytes: bytes) -> bytes:
    """
    Encodes 16-bit linear PCM to PCMU (μ-law), maintaining 8kHz sample rate.
    Uses audioop for high-quality conversion.
    """
    return audioop.lin2ulaw(pcm16_bytes, 2)

def format_json(obj: dict) -> str:
    return json.dumps(obj, indent=2)

async def handle_http_request(path: str, request_headers):
    logger.info(f"[HTTP] Remote address: {request_headers.get('Host', 'unknown')}")

    required_headers = [
        'audiohook-organization-id',
        'audiohook-correlation-id',
        'audiohook-session-id',
        'x-api-key'
    ]

    logger.info("[HTTP] Received headers:")
    for header, value in request_headers.items():
        masked_value = '*' * 8 if header.lower() in ['x-api-key', 'authorization'] else value
        logger.info(f"[HTTP]   {header}: {masked_value}")

    header_keys = {k.lower(): v for k, v in request_headers.items()}
    missing_headers = [h for h in required_headers if h not in header_keys]

    if missing_headers:
        logger.warning(f"[HTTP] Missing required headers: {', '.join(missing_headers)}")
        return http.HTTPStatus.UNAUTHORIZED, [], b'Missing required headers\n'

    if header_keys.get('upgrade', '').lower() != 'websocket':
        logger.warning("[HTTP] Not a WebSocket upgrade request")
        return http.HTTPStatus.BAD_REQUEST, [], b'WebSocket upgrade required\n'

    if header_keys.get('sec-websocket-version', '') != '13':
        logger.warning("[HTTP] Invalid WebSocket version")
        return http.HTTPStatus.BAD_REQUEST, [], b'WebSocket version 13 required\n'

    if not header_keys.get('sec-websocket-key'):
        logger.warning("[HTTP] Missing WebSocket key")
        return http.HTTPStatus.BAD_REQUEST, [], b'WebSocket key required\n'

    logger.info("[HTTP] Upgrade request validation successful")
    return None

###############################################################################
# OpenAI Realtime Client
###############################################################################

class OpenAIRealtimeClient:
    def __init__(self, session_id: str):
        self.ws = None
        self.running = False
        self.read_task = None
        self._lock = asyncio.Lock()
        self.session_id = session_id
        self.logger = logger.getChild(f"OpenAIClient_{session_id}")
        self.start_time = time.time()

    async def connect(self, instructions=INSTRUCTIONS, voice=OPENAI_VOICE):
        from websockets.asyncio.client import connect as ws_connect
        
        ws_headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }

        try:
            self.logger.info(f"Connecting to OpenAI Realtime API WebSocket using model: {OPENAI_MODEL}...")
            self.logger.debug(f"WebSocket URL: {OPENAI_REALTIME_URL}")
            connect_start = time.time()
            
            self.ws = await asyncio.wait_for(
                ws_connect(
                    OPENAI_REALTIME_URL,
                    additional_headers=ws_headers,
                    max_size=2**23,
                    compression=None,
                    max_queue=32
                ),
                timeout=10.0
            )

            connect_time = time.time() - connect_start
            self.logger.info(f"OpenAI WebSocket connection established in {connect_time:.2f}s")
            self.running = True

            msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            server_event = json.loads(msg)
            self.logger.debug(f"Received from OpenAI on connect:\n{format_json(server_event)}")
            if server_event.get("type") != "session.created":
                self.logger.error("Did not receive session.created event.")
                await self.close()
                raise RuntimeError("OpenAI session not created")

            session_update = {
                "type": "session.update",
                "session": {
                    "modalities": ["audio", "text"],
                    "instructions": instructions,
                    "voice": voice,
                    "input_audio_format": "g711_ulaw",
                    "output_audio_format": "g711_ulaw",
                    "turn_detection": {
                        "type": "server_vad",
                        "threshold": 0.5,
                        "prefix_padding_ms": 300,
                        "silence_duration_ms": 500,
                        "create_response": True
                    }
                }
            }
            await self._safe_send(json.dumps(session_update))

            updated_ok = False
            while True:
                msg = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
                ev = json.loads(msg)
                self.logger.debug(f"Received after session.update:\n{format_json(ev)}")
                if ev.get("type") == "session.updated":
                    updated_ok = True
                    break
            if not updated_ok:
                self.logger.error("Session update not confirmed.")
                await self.close()
                raise RuntimeError("OpenAI session update not confirmed")

        except (asyncio.TimeoutError, websockets.exceptions.WebSocketException, TypeError) as e:
            self.logger.error(f"Error establishing OpenAI connection: {e}")
            self.logger.error(f"Model: {OPENAI_MODEL}")
            self.logger.error(f"URL: {OPENAI_REALTIME_URL}")
            if isinstance(e, websockets.exceptions.WebSocketException):
                self.logger.error(f"WebSocket specific error details: {str(e)}")
            await self.close()
            raise RuntimeError(f"Failed to connect to OpenAI: {str(e)}")

    async def _safe_send(self, message: str):
        async with self._lock:
            if self.ws and self.running:
                try:
                    msg_dict = json.loads(message)
                    self.logger.debug(f"Sending to OpenAI:\n{format_json(msg_dict)}")
                except json.JSONDecodeError:
                    self.logger.debug(f"Sending raw to OpenAI: {message[:200]}...")
                await self.ws.send(message)

    async def send_audio(self, pcmu_8k: bytes):
        if not self.running or self.ws is None:
            return
        self.logger.debug(f"Sending audio frame to OpenAI: {len(pcmu_8k)} bytes")
        encoded = base64.b64encode(pcmu_8k).decode("utf-8")
        msg = {
            "type": "input_audio_buffer.append",
            "audio": encoded
        }
        await self._safe_send(json.dumps(msg))

    async def start_receiving(self, on_audio_callback):
        if not self.running or not self.ws:
            return

        async def _read_loop():
            try:
                while self.running:
                    raw = await self.ws.recv()
                    try:
                        msg_dict = json.loads(raw)
                        self.logger.debug(f"Received from OpenAI:\n{format_json(msg_dict)}")
                        ev_type = msg_dict.get("type", "")
                        if ev_type == "response.audio.delta":
                            delta_b64 = msg_dict.get("delta", "")
                            if delta_b64:
                                pcmu_8k = base64.b64decode(delta_b64)
                                on_audio_callback(pcmu_8k)
                    except json.JSONDecodeError:
                        self.logger.debug(f"Received raw from OpenAI: {raw[:200]}...")
            except websockets.exceptions.ConnectionClosed:
                self.logger.info("OpenAI websocket closed.")
                self.running = False
            except Exception as e:
                self.logger.error(f"Error reading from OpenAI: {e}")
                self.running = False

        self.read_task = asyncio.create_task(_read_loop())

    async def close(self):
        duration = time.time() - self.start_time
        self.logger.info(f"Closing OpenAI connection after {duration:.2f}s")
        self.running = False
        if self.ws:
            try:
                await self.ws.close()
            except Exception as e:
                self.logger.error(f"Error closing OpenAI connection: {e}")
            self.ws = None
        if self.read_task:
            self.read_task.cancel()
            self.read_task = None

###############################################################################
# Genesys AudioHook protocol server
###############################################################################

class AudioHookServer:
    def __init__(self, websocket):
        self.session_id = str(uuid.uuid4())
        self.ws = websocket
        self.client_seq = 0
        self.server_seq = 0
        self.openai_client = OpenAIRealtimeClient(self.session_id)
        self.running = True
        self.negotiated_media = None
        self.start_time = time.time()
        self.logger = logger.getChild(f"AudioHookServer_{self.session_id}")
        self.audio_frames_sent = 0
        self.audio_frames_received = 0

        self.logger.info(f"New session started: {self.session_id}")

    async def handle_message(self, msg: dict):
        msg_type = msg.get("type")
        seq = msg.get("seq", 0)
        self.client_seq = seq

        self.logger.debug(f"Received message from Genesys:\n{format_json(msg)}")

        if msg_type == "open":
            await self.handle_open(msg)
        elif msg_type == "ping":
            await self.handle_ping(msg)
        elif msg_type == "close":
            await self.handle_close(msg)
        elif msg_type == "error":
            self.logger.warning(f"Received 'error' from Genesys:\n{format_json(msg)}")
        elif msg_type in ["update", "resume", "pause"]:
            self.logger.debug(f"Ignoring message of type {msg_type}")
        else:
            self.logger.debug(f"Ignoring unknown message type: {msg_type}")

    async def handle_open(self, msg: dict):
        self.session_id = msg["id"]

        is_probe = (
            msg["parameters"].get("conversationId") == "00000000-0000-0000-0000-000000000000" and
            msg["parameters"].get("participant", {}).get("id") == "00000000-0000-0000-0000-000000000000"
        )

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
                    "media": []
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

        # Connect to OpenAI
        try:
            await self.openai_client.connect(voice=OPENAI_VOICE, instructions=INSTRUCTIONS)
        except Exception as e:
            self.logger.error(f"OpenAI connection failed: {e}")
            await self.disconnect_session(reason="error", info=str(e))
            return

        def on_audio_callback(pcmu_8k: bytes):
            asyncio.create_task(self.handle_openai_audio(pcmu_8k))

        await self.openai_client.start_receiving(on_audio_callback)

    async def handle_openai_audio(self, pcmu_8k: bytes):
        if not self.running:
            return
        self.logger.debug(f"Processing OpenAI audio frame: {len(pcmu_8k)} bytes")
        
        await self.send_binary_to_genesys(pcmu_8k)

    async def send_binary_to_genesys(self, data: bytes):
        try:
            await self.ws.send(data)
            self.audio_frames_sent += 1
            self.logger.debug(f"Sent audio frame to Genesys: {len(data)} bytes (frame #{self.audio_frames_sent})")
        except websockets.ConnectionClosed:
            self.logger.warning("Genesys WebSocket closed while sending audio frame.")
            self.running = False

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
            "parameters": {}
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
        await self.openai_client.close()
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
                    "info": info
                }
            }

            self.server_seq += 1
            await asyncio.wait_for(self._send_json(disconnect_msg), timeout=5.0)
            try:
                await asyncio.wait_for(self.ws.wait_closed(), timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"Client did not acknowledge disconnect for session {self.session_id}")

        except Exception as e:
            logger.error(f"Error in disconnect_session: {e}")
        finally:
            self.running = False

    async def handle_audio_frame(self, frame_bytes: bytes):
        if not self.openai_client.running:
            return

        self.audio_frames_received += 1
        self.logger.debug(f"Received audio frame from Genesys: {len(frame_bytes)} bytes (frame #{self.audio_frames_received})")
        
        await self.openai_client.send_audio(frame_bytes)

    async def _send_json(self, msg: dict):
        try:
            self.logger.debug(f"Sending message to Genesys:\n{format_json(msg)}")
            await self.ws.send(json.dumps(msg))
        except websockets.ConnectionClosed:
            self.logger.warning("Genesys WebSocket closed while sending JSON message.")
            self.running = False

async def validate_request(path, request_headers):
    logger.info(f"\n{'='*50}\n[HTTP] Starting WebSocket upgrade validation")
    logger.info(f"[HTTP] Target path: {GENESYS_PATH}")
    logger.info(f"[HTTP] Remote address: {request_headers.get('Host', 'unknown')}")

    logger.info("[HTTP] Full headers received:")
    for name, value in request_headers.items():
        if name.lower() in ['x-api-key', 'authorization']:
            logger.info(f"[HTTP]   {name}: {'*' * 8}")
        else:
            logger.info(f"[HTTP]   {name}: {value}")

    normalized_path = path.rstrip('/')
    normalized_target = GENESYS_PATH.rstrip('/')

    if normalized_path != normalized_target:
        logger.error("[HTTP] Path mismatch:")
        logger.error(f"[HTTP]   Expected: {GENESYS_PATH}")
        logger.error(f"[HTTP]   Normalized received: {normalized_path}")
        logger.error(f"[HTTP]   Normalized expected: {normalized_target}")
        return http.HTTPStatus.NOT_FOUND, [], b'Invalid path\n'

    required_headers = [
        'audiohook-organization-id',
        'audiohook-correlation-id',
        'audiohook-session-id',
        'x-api-key',
        'upgrade',
        'sec-websocket-version',
        'sec-websocket-key'
    ]

    header_keys = {k.lower(): v for k, v in request_headers.items()}
    logger.info("[HTTP] Normalized headers for validation:")
    for k, v in header_keys.items():
        if k in ['x-api-key', 'authorization']:
            logger.info(f"[HTTP]   {k}: {'*' * 8}")
        else:
            logger.info(f"[HTTP]   {k}: {v}")

    missing_headers = []
    found_headers = []
    for h in required_headers:
        if h.lower() not in header_keys:
            missing_headers.append(h)
        else:
            found_headers.append(h)

    if missing_headers:
        error_msg = f"Missing required headers: {', '.join(missing_headers)}"
        logger.error(f"[HTTP] Connection rejected - {error_msg}")
        logger.error("[HTTP] Found headers: " + ", ".join(found_headers))
        return http.HTTPStatus.UNAUTHORIZED, [], error_msg.encode()

    upgrade_header = header_keys.get('upgrade', '').lower()
    logger.info(f"[HTTP] Checking upgrade header: {upgrade_header}")
    if upgrade_header != 'websocket':
        error_msg = f"Invalid upgrade header: {upgrade_header}"
        logger.error(f"[HTTP] {error_msg}")
        return http.HTTPStatus.BAD_REQUEST, [], b'WebSocket upgrade required\n'

    ws_version = header_keys.get('sec-websocket-version', '')
    logger.info(f"[HTTP] Checking WebSocket version: {ws_version}")
    if ws_version != '13':
        error_msg = f"Invalid WebSocket version: {ws_version}"
        logger.error(f"[HTTP] {error_msg}")
        return http.HTTPStatus.BAD_REQUEST, [], b'WebSocket version 13 required\n'

    ws_key = header_keys.get('sec-websocket-key')
    if not ws_key:
        logger.error("[HTTP] Missing WebSocket key")
        return http.HTTPStatus.BAD_REQUEST, [], b'WebSocket key required\n'
    logger.info("[HTTP] Found valid WebSocket key")

    ws_protocol = header_keys.get('sec-websocket-protocol', '')
    if ws_protocol:
        logger.info(f"[HTTP] WebSocket protocol requested: {ws_protocol}")
        if 'audiohook' not in ws_protocol.lower():
            logger.warning("[HTTP] Client didn't request 'audiohook' protocol")

    connection_header = header_keys.get('connection', '').lower()
    logger.info(f"[HTTP] Connection header: {connection_header}")
    if 'upgrade' not in connection_header:
        logger.warning("[HTTP] Connection header doesn't contain 'upgrade'")

    logger.info("[HTTP] All validation checks passed successfully")
    logger.info(f"[HTTP] Proceeding with WebSocket upgrade")
    logger.info("="*50)
    return None

async def handle_genesys_connection(websocket):
    connection_id = str(uuid.uuid4())[:8]
    logger.info(f"\n{'='*50}\n[WS-{connection_id}] New WebSocket connection handler started")

    session = None

    try:
        logger.info(f"Received WebSocket connection from {websocket.remote_address}")
        logger.info(f"[WS-{connection_id}] Remote address: {websocket.remote_address}")
        logger.info(f"[WS-{connection_id}] Connection state: {websocket.state}")

        ws_attributes = ['path', 'remote_address', 'local_address', 'state', 'open', 'protocol']
        logger.info(f"[WS-{connection_id}] WebSocket object attributes:")
        for attr in ws_attributes:
            value = getattr(websocket, attr, "Not available")
            logger.info(f"[WS-{connection_id}]   {attr}: {value}")

        logger.info(f"[WS-{connection_id}] WebSocket connection established; handshake was validated beforehand.")

        session = AudioHookServer(websocket)
        logger.info(f"[WS-{connection_id}] Session created with ID: {session.session_id}")

        logger.info(f"[WS-{connection_id}] Starting main message loop")
        while session.running:
            try:
                logger.debug(f"[WS-{connection_id}] Waiting for next message...")
                msg = await websocket.recv()
                if isinstance(msg, bytes):
                    logger.debug(f"[WS-{connection_id}] Received binary frame: {len(msg)} bytes")
                    await session.handle_audio_frame(msg)
                else:
                    try:
                        data = json.loads(msg)
                        logger.debug(f"[WS-{connection_id}] Received JSON message:\n{format_json(data)}")
                        await session.handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"[WS-{connection_id}] Error parsing JSON: {e}")
                        await session.disconnect_session("error", f"JSON parse error: {e}")
                    except Exception as e:
                        logger.error(f"[WS-{connection_id}] Error processing message: {e}")
                        await session.disconnect_session("error", f"Message processing error: {e}")

            except websockets.ConnectionClosed as e:
                logger.info(f"[WS-{connection_id}] Connection closed: code={e.code}, reason={e.reason}")
                break
            except Exception as e:
                logger.error(f"[WS-{connection_id}] Unexpected error: {e}", exc_info=True)
                break

        logger.info(f"[WS-{connection_id}] Session loop ended, cleaning up")
        await session.openai_client.close()
        logger.info(f"[WS-{connection_id}] Session cleanup complete")

    except Exception as e:
        logger.error(f"[WS-{connection_id}] Fatal connection error: {e}", exc_info=True)
        if session is None:
            session = AudioHookServer(websocket)
        await session.disconnect_session(reason="error", info=f"Internal error: {str(e)}")
    finally:
        logger.info(f"[WS-{connection_id}] Connection handler finished\n{'='*50}")

async def main():
    startup_msg = f"""
{'='*80}
Genesys-OpenAI Bridging Server
Starting up at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Host: {GENESYS_LISTEN_HOST}
Port: {GENESYS_LISTEN_PORT}
Path: {GENESYS_PATH}
SSL: Public CA Signed Certs
Log File: {os.path.abspath(LOG_FILE)}
{'='*80}
"""
    logger.info(startup_msg)

    websockets_logger = logging.getLogger('websockets')
    if DEBUG != 'true':
        websockets_logger.setLevel(logging.INFO)

    websockets_logger.addHandler(logging.FileHandler(LOG_FILE))

    try:
        async with websockets.serve(
            handle_genesys_connection,
            GENESYS_LISTEN_HOST,
            GENESYS_LISTEN_PORT,
            ssl=ssl_context,
            max_size=64000,
            ping_interval=None,
            ping_timeout=None
        ):
            logger.info(
                f"Server is listening for Genesys AudioHook connections on "
                f"wss://{GENESYS_LISTEN_HOST}:{GENESYS_LISTEN_PORT}{GENESYS_PATH}"
            )
            logger.info("SSL context info:")
            logger.info(f"  Cert path: {SSL_CERT_PATH}")
            logger.info(f"  Key path: {SSL_KEY_PATH}")
            logger.info(f"  Protocol: {ssl_context.protocol}")
            logger.info(f"  Verify mode: {ssl_context.verify_mode}")
            logger.info(f"  Options: {ssl_context.options}")

            try:
                await asyncio.Future()  # run forever
            except asyncio.CancelledError:
                logger.info("Server shutdown initiated")
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down via KeyboardInterrupt.")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        logger.info("Server shutdown complete.")

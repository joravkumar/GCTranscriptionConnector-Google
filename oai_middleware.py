import asyncio
import json
import uuid
import logging
import http
import os
from datetime import datetime
import hmac
import hashlib
import base64

from config import (
    GENESYS_LISTEN_HOST,
    GENESYS_LISTEN_PORT,
    GENESYS_PATH,
    DEBUG,
    GENESYS_API_KEY,
    GENESYS_ORG_ID
)
from audio_hook_server import AudioHookServer
from utils import format_json

# ---------------------------
# Simple Logging Setup
# ---------------------------
if DEBUG == 'true':
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("GenesysOpenAIBridge")

# Updated import: get the protocol from websockets.server (websockets 15.0)
from websockets.server import WebSocketServerProtocol

class CustomWebSocketServerProtocol(WebSocketServerProtocol):
    async def handshake(self, *args, **kwargs):
        try:
            return await super().handshake(*args, **kwargs)
        except Exception as exc:
            logger.error(f"Handshake failed: {exc}", exc_info=True)
            raise

async def validate_request(connection, request):
    """
    This function is called by websockets.serve() to validate the HTTP request
    before upgrading to a WebSocket.
    Signature verification is disabled; only the API key is required.
    """
    # Added health endpoint support for Digital Ocean health checks
    if request.path == "/health":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    
    path_str = request.path
    raw_headers = dict(request.headers)

    logger.info(f"\n{'='*50}\n[HTTP] Starting WebSocket upgrade validation")
    logger.info(f"[HTTP] Target path: {GENESYS_PATH}")
    logger.info(f"[HTTP] Remote address: {raw_headers.get('host', 'unknown')}")

    logger.info("[HTTP] Full headers received:")
    for name, value in raw_headers.items():
        if name.lower() in ['x-api-key', 'authorization']:
            logger.info(f"[HTTP]   {name}: {'*' * 8}")
        else:
            logger.info(f"[HTTP]   {name}: {value}")

    normalized_path = path_str.rstrip('/')
    normalized_target = GENESYS_PATH.rstrip('/')
    if normalized_path != normalized_target:
        logger.error("[HTTP] Path mismatch:")
        logger.error(f"[HTTP]   Expected: {GENESYS_PATH}")
        logger.error(f"[HTTP]   Normalized received: {normalized_path}")
        logger.error(f"[HTTP]   Normalized expected: {normalized_target}")
        return connection.respond(http.HTTPStatus.NOT_FOUND, "Invalid path\n")

    required_headers = [
        'audiohook-organization-id',
        'audiohook-correlation-id',
        'audiohook-session-id',
        'x-api-key',
        'upgrade',
        'sec-websocket-version',
        'sec-websocket-key'
    ]

    header_keys = {k.lower(): v for k, v in raw_headers.items()}
    logger.info("[HTTP] Normalized headers for validation:")
    for k, v in header_keys.items():
        if k in ['x-api-key', 'authorization']:
            logger.info(f"[HTTP]   {k}: {'*' * 8}")
        else:
            logger.info(f"[HTTP]   {k}: {v}")

    if header_keys.get('x-api-key') != GENESYS_API_KEY:
        logger.error("Invalid X-API-KEY header value.")
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Invalid API key\n")

    if header_keys.get('audiohook-organization-id') != GENESYS_ORG_ID:
        logger.error("Invalid Audiohook-Organization-Id header value.")
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Invalid Audiohook-Organization-Id\n")

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
        return connection.respond(http.HTTPStatus.UNAUTHORIZED, error_msg)

    upgrade_header = header_keys.get('upgrade', '').lower()
    logger.info(f"[HTTP] Checking upgrade header: {upgrade_header}")
    if upgrade_header != 'websocket':
        error_msg = f"Invalid upgrade header: {upgrade_header}"
        logger.error(f"[HTTP] {error_msg}")
        return connection.respond(http.HTTPStatus.BAD_REQUEST, "WebSocket upgrade required\n")

    ws_version = header_keys.get('sec-websocket-version', '')
    logger.info(f"[HTTP] Checking WebSocket version: {ws_version}")
    if ws_version != '13':
        error_msg = f"Invalid WebSocket version: {ws_version}"
        logger.error(f"[HTTP] {error_msg}")
        return connection.respond(http.HTTPStatus.BAD_REQUEST, "WebSocket version 13 required\n")

    ws_key = header_keys.get('sec-websocket-key')
    if not ws_key:
        logger.error("[HTTP] Missing WebSocket key")
        return connection.respond(http.HTTPStatus.BAD_REQUEST, "WebSocket key required\n")
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

        ws_attributes = ['path', 'remote_address', 'local_address', 'state', 'open', 'subprotocol']
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
                    except Exception as ex:
                        logger.error(f"[WS-{connection_id}] Error processing message: {ex}")
                        await session.disconnect_session("error", f"Message processing error: {ex}")
            except Exception as ex:
                from websockets.exceptions import ConnectionClosed
                if isinstance(ex, ConnectionClosed):
                    logger.info(f"[WS-{connection_id}] Connection closed: code={ex.code}, reason={ex.reason}")
                else:
                    logger.error(f"[WS-{connection_id}] Unexpected error: {ex}", exc_info=True)
                break

        logger.info(f"[WS-{connection_id}] Session loop ended, cleaning up")
        if session and hasattr(session, 'openai_client') and session.openai_client:
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
Genesys-OpenAIBridge Server
Starting up at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Host: {GENESYS_LISTEN_HOST}
Port: {GENESYS_LISTEN_PORT}
Path: {GENESYS_PATH}
{'='*80}
"""
    logger.info(startup_msg)

    websockets_logger = logging.getLogger('websockets')
    if DEBUG != 'true':
        websockets_logger.setLevel(logging.INFO)

    # Monkey-patch the default protocol in the server.
    import websockets.server
    websockets.server.WebSocketServerProtocol = CustomWebSocketServerProtocol

    try:
        async with websockets.serve(
            handle_genesys_connection,
            GENESYS_LISTEN_HOST,
            int(GENESYS_LISTEN_PORT),
            process_request=validate_request,
            max_size=64000,
            ping_interval=None,
            ping_timeout=None
        ):
            logger.info(
                f"Server is listening for Genesys AudioHook connections on "
                f"ws://{GENESYS_LISTEN_HOST}:{GENESYS_LISTEN_PORT}{GENESYS_PATH}"
            )
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

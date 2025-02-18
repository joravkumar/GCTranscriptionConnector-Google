import asyncio
import json
import uuid
import logging
import websockets
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
    GENESYS_ORG_ID,
    GENESYS_CLIENT_SECRET
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

def verify_signature(signature_header, signature_input_header, headers, method, path):
    """
    Verifies the signature using HMAC-SHA256.
    Expected signature header format: "sig1=:<base64 signature>:"
    Expected signature-input header format:
      sig1=("@request-target" "@authority" "audiohook-organization-id" "audiohook-session-id"
      "audiohook-correlation-id" "x-api-key");keyid="..."; nonce="...";alg="hmac-sha256";created=...;expires=...
    """

    logger.debug("Entering verify_signature...")

    # Log the raw signature header, but mask most of it.
    if len(signature_header) > 12:
        logger.debug(f"Raw signature_header (partially masked): {signature_header[:12]}... (len={len(signature_header)})")
    else:
        logger.debug(f"Raw signature_header: {signature_header}")

    # Log the raw signature-input header, but mask it if long.
    if len(signature_input_header) > 50:
        logger.debug(f"Raw signature_input_header (partially masked): {signature_input_header[:50]}...")
    else:
        logger.debug(f"Raw signature_input_header: {signature_input_header}")

    try:
        # Parse signature-input header to extract the list of headers.
        if not signature_input_header.startswith("sig1="):
            logger.debug("Signature input header does not start with 'sig1='.")
            return False
        sig_input = signature_input_header[len("sig1="):].strip()
        if not (sig_input.startswith("(") and ")" in sig_input):
            logger.debug("Signature input header missing parentheses.")
            return False
        end_paren_index = sig_input.find(")")
        headers_part = sig_input[1:end_paren_index]
        header_list = headers_part.split()
        header_list = [h.strip('"') for h in header_list]
    except Exception as e:
        logger.error(f"Error parsing signature input: {e}")
        return False

    logger.debug(f"Header list to be signed: {header_list}")

    # Build the signing string according to the headers list.
    signing_lines = []
    for header in header_list:
        if header == "@request-target":
            line = f"@request-target: {method.lower()} {path}"
            signing_lines.append(line)
        elif header == "@authority":
            if "host" in headers:
                line = f"@authority: {headers['host']}"
                signing_lines.append(line)
            else:
                logger.debug("Missing 'host' in headers while building signing string.")
                return False
        else:
            if header.lower() not in headers:
                logger.debug(f"Missing header '{header}' in the request.")
                return False
            line = f"{header.lower()}: {headers[header.lower()]}"
            signing_lines.append(line)

    signing_string = "\n".join(signing_lines)

    # For debugging, show the signing string but mask the x-api-key if present:
    masked_signing_string = signing_string
    if "x-api-key:" in masked_signing_string:
        # Attempt to mask only the value after 'x-api-key: '
        masked_signing_string = masked_signing_string.replace(
            f"x-api-key: {headers['x-api-key']}",
            "x-api-key: ****"
        )

    logger.debug(f"Constructed signing string:\n{masked_signing_string}")

    # Genesys docs: the client secret is base64-encoded, so decode it first.
    try:
        decoded_secret = base64.b64decode(GENESYS_CLIENT_SECRET)
        logger.debug(f"Decoded client secret length: {len(decoded_secret)} bytes")
    except Exception as e:
        logger.error(f"Failed to base64-decode GENESYS_CLIENT_SECRET: {e}")
        return False

    computed_hmac = hmac.new(
        decoded_secret,
        signing_string.encode('utf-8'),
        hashlib.sha256
    ).digest()
    computed_signature = base64.b64encode(computed_hmac).decode('utf-8')

    # Extract the provided signature.
    if not signature_header.startswith("sig1=:") or not signature_header.endswith(":"):
        logger.debug("Signature header format is invalid (missing 'sig1=:' prefix or ':' suffix).")
        return False

    provided_signature = signature_header[len("sig1=:"):-1]

    # Show partial logs for computed vs. provided signature
    logger.debug(f"Computed signature (first 10 chars): {computed_signature[:10]}...")
    logger.debug(f"Provided signature (first 10 chars): {provided_signature[:10]}...")

    is_match = hmac.compare_digest(computed_signature, provided_signature)
    logger.debug(f"Signature match result: {is_match}")
    return is_match


async def validate_request(path, request_headers):
    """
    This function is called by websockets.serve() to validate the HTTP request
    before upgrading to a WebSocket.
    In newer versions of websockets, 'path' may not be a string. We attempt to
    extract a proper path string from it if needed.
    """
    # Attempt to convert 'path' to a proper string:
    if not isinstance(path, str):
        if hasattr(path, "resource"):
            path_str = path.resource
        elif hasattr(path, "path"):
            path_str = path.path
        elif hasattr(path, "request") and hasattr(path.request, "path"):
            path_str = path.request.path
        else:
            path_str = str(path)
    else:
        path_str = path

    # Convert request_headers to a dictionary.
    if hasattr(request_headers, "headers"):
        raw_headers = dict(request_headers.headers)
    else:
        raw_headers = dict(request_headers)

    logger.info(f"\n{'='*50}\n[HTTP] Starting WebSocket upgrade validation")
    logger.info(f"[HTTP] Target path: {GENESYS_PATH}")
    logger.info(f"[HTTP] Remote address: {raw_headers.get('Host', 'unknown')}")

    logger.info("[HTTP] Full headers received:")
    for name, value in raw_headers.items():
        if name.lower() in ['x-api-key', 'authorization']:
            logger.info(f"[HTTP]   {name}: {'*' * 8}")
        else:
            logger.info(f"[HTTP]   {name}: {value}")

    # Perform the path comparison.
    normalized_path = path_str.rstrip('/')
    normalized_target = GENESYS_PATH.rstrip('/')
    if normalized_path != normalized_target:
        logger.error("[HTTP] Path mismatch:")
        logger.error(f"[HTTP]   Expected: {GENESYS_PATH}")
        logger.error(f"[HTTP]   Normalized received: {normalized_path}")
        logger.error(f"[HTTP]   Normalized expected: {normalized_target}")
        return (http.HTTPStatus.NOT_FOUND.value, [], b'Invalid path\n')

    required_headers = [
        'audiohook-organization-id',
        'audiohook-correlation-id',
        'audiohook-session-id',
        'x-api-key',
        'upgrade',
        'sec-websocket-version',
        'sec-websocket-key'
    ]

    # Lowercase all keys for consistent lookups.
    header_keys = {k.lower(): v for k, v in raw_headers.items()}
    logger.info("[HTTP] Normalized headers for validation:")
    for k, v in header_keys.items():
        if k in ['x-api-key', 'authorization']:
            logger.info(f"[HTTP]   {k}: {'*' * 8}")
        else:
            logger.info(f"[HTTP]   {k}: {v}")

    # Verify X-API-KEY header.
    if header_keys.get('x-api-key') != GENESYS_API_KEY:
        logger.error("Invalid X-API-KEY header value.")
        return (http.HTTPStatus.UNAUTHORIZED.value, [], b"Invalid API key\n")

    # Verify Audiohook-Organization-Id matches expected organization.
    if header_keys.get('audiohook-organization-id') != GENESYS_ORG_ID:
        logger.error("Invalid Audiohook-Organization-Id header value.")
        return (http.HTTPStatus.UNAUTHORIZED.value, [], b"Invalid Audiohook-Organization-Id\n")

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
        return (http.HTTPStatus.UNAUTHORIZED.value, [], error_msg.encode())

    upgrade_header = header_keys.get('upgrade', '').lower()
    logger.info(f"[HTTP] Checking upgrade header: {upgrade_header}")
    if upgrade_header != 'websocket':
        error_msg = f"Invalid upgrade header: {upgrade_header}"
        logger.error(f"[HTTP] {error_msg}")
        return (http.HTTPStatus.BAD_REQUEST.value, [], b'WebSocket upgrade required\n')

    ws_version = header_keys.get('sec-websocket-version', '')
    logger.info(f"[HTTP] Checking WebSocket version: {ws_version}")
    if ws_version != '13':
        error_msg = f"Invalid WebSocket version: {ws_version}"
        logger.error(f"[HTTP] {error_msg}")
        return (http.HTTPStatus.BAD_REQUEST.value, [], b'WebSocket version 13 required\n')

    ws_key = header_keys.get('sec-websocket-key')
    if not ws_key:
        logger.error("[HTTP] Missing WebSocket key")
        return (http.HTTPStatus.BAD_REQUEST.value, [], b'WebSocket key required\n')
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

    # Log optional signature headers if provided.
    if 'signature' in header_keys:
        logger.info("[HTTP] Received Signature header.")
    if 'signature-input' in header_keys:
        logger.info("[HTTP] Received Signature-Input header.")

    # If a client secret is configured, verify that signature headers exist and are valid.
    if GENESYS_CLIENT_SECRET:
        if 'signature' not in header_keys or 'signature-input' not in header_keys:
            logger.error("Missing signature headers despite client secret being configured.")
            return (http.HTTPStatus.UNAUTHORIZED.value, [], b"Missing signature headers\n")
        if not verify_signature(header_keys['signature'], header_keys['signature-input'], header_keys, "GET", path_str):
            logger.error("Invalid signature.")
            return (http.HTTPStatus.UNAUTHORIZED.value, [], b"Invalid signature\n")

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
                    except Exception as ex:
                        logger.error(f"[WS-{connection_id}] Error processing message: {ex}")
                        await session.disconnect_session("error", f"Message processing error: {ex}")

            except websockets.ConnectionClosed as e:
                logger.info(f"[WS-{connection_id}] Connection closed: code={e.code}, reason={e.reason}")
                break
            except Exception as ex:
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
Genesys-OpenAI Bridging Server
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

    try:
        async with websockets.serve(
            handle_genesys_connection,
            GENESYS_LISTEN_HOST,
            GENESYS_LISTEN_PORT,
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

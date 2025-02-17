import os
import ssl
import logging
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

DEBUG = os.getenv('DEBUG', 'false').lower()

# Audio buffering settings
MAX_AUDIO_BUFFER_SIZE = 50
AUDIO_FRAME_SEND_INTERVAL = 0.15

# Server settings
GENESYS_LISTEN_HOST = "0.0.0.0"
GENESYS_LISTEN_PORT = 443
GENESYS_PATH = "/audiohook"

# OpenAI Realtime API settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Rate limiting constants 
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_DELAY = 3
RATE_LIMIT_WINDOW = 300
RATE_LIMIT_PHASES = [
    {"window": 300, "delay": 3},
    {"window": 600, "delay": 9},
    {"window": float('inf'), "delay": 27}
]

# Genesys rate limiting constants (to respect Audio Hook limits)
GENESYS_MSG_RATE_LIMIT = 5
GENESYS_BINARY_RATE_LIMIT = 5
GENESYS_MSG_BURST_LIMIT = 25
GENESYS_BINARY_BURST_LIMIT = 25
GENESYS_RATE_WINDOW = 1.0 

LOGGING_FORMAT = "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s"

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
websockets_logger.setLevel(logging.INFO)

if DEBUG != 'true':
    logger.setLevel(logging.INFO)

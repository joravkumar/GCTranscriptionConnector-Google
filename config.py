import os
from dotenv import load_dotenv

load_dotenv()  # load variables from .env

DEBUG = os.getenv('DEBUG', 'false').lower()

# Audio buffering settings
MAX_AUDIO_BUFFER_SIZE = 50
AUDIO_FRAME_SEND_INTERVAL = 0.15

# Server settings
GENESYS_LISTEN_HOST = "0.0.0.0"
GENESYS_LISTEN_PORT = int(os.getenv("PORT", "443"))
GENESYS_PATH = "/audiohook"

# OpenAI Translation API settings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# New environment variable for the transcription model.
OPENAI_TRANSCRIPTION_MODEL = os.getenv('OPENAI_TRANSCRIPTION_MODEL', 'whisper-1')

# New environment variable for Genesys API key.
GENESYS_API_KEY = os.getenv('GENESYS_API_KEY')
if not GENESYS_API_KEY:
    raise ValueError("GENESYS_API_KEY not found in environment variables.")

# Genesys rate limiting constants
GENESYS_MSG_RATE_LIMIT = 5
GENESYS_BINARY_RATE_LIMIT = 5
GENESYS_MSG_BURST_LIMIT = 25
GENESYS_BINARY_BURST_LIMIT = 25
GENESYS_RATE_WINDOW = 1.0

# Rate limiting constants 
RATE_LIMIT_MAX_RETRIES = 3
RATE_LIMIT_BASE_DELAY = 3
RATE_LIMIT_WINDOW = 300
RATE_LIMIT_PHASES = [
    {"window": 300, "delay": 3},
    {"window": 600, "delay": 9},
    {"window": float('inf'), "delay": 27}
]

SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "es-ES,en-US")

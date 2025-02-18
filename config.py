import os
from dotenv import load_dotenv

load_dotenv()  # load variables from .env

DEBUG = os.getenv('DEBUG', 'false').lower()

# Audio buffering settings
MAX_AUDIO_BUFFER_SIZE = 50

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

# New environment variable for Genesys Organization ID.
GENESYS_ORG_ID = os.getenv('GENESYS_ORG_ID')
if not GENESYS_ORG_ID:
    raise ValueError("GENESYS_ORG_ID not found in environment variables.")

# New environment variable for Genesys Client Secret.
GENESYS_CLIENT_SECRET = os.getenv('GENESYS_CLIENT_SECRET')
if not GENESYS_CLIENT_SECRET:
    raise ValueError("GENESYS_CLIENT_SECRET not found in environment variables.")

# Genesys rate limiting constants
GENESYS_MSG_RATE_LIMIT = 5
GENESYS_BINARY_RATE_LIMIT = 5
GENESYS_MSG_BURST_LIMIT = 25
GENESYS_BINARY_BURST_LIMIT = 25

# Rate limiting constants 
RATE_LIMIT_MAX_RETRIES = 3

# Transcription Connector language support
SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "es-ES,it-IT,en-US")

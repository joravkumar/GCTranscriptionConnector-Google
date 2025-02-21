import os
from dotenv import load_dotenv

load_dotenv()  # load variables from .env

DEBUG = os.getenv('DEBUG', 'false').lower()

# Audio buffering settings
MAX_AUDIO_BUFFER_SIZE = 50

# Server settings
GENESYS_LISTEN_HOST = "0.0.0.0"
GENESYS_LISTEN_PORT = "443"
GENESYS_PATH = "/audiohook"

# Google Cloud Speech-to-Text API settings
GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
if not GOOGLE_CLOUD_PROJECT:
    raise ValueError("GOOGLE_CLOUD_PROJECT not found in environment variables.")

# Centralized service account credentials JSON key.
# This environment variable should contain the entire JSON key as a string.
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
if not GOOGLE_APPLICATION_CREDENTIALS:
    raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in environment variables.")

# New environment variable to set the speech recognition model.
# Defaults to "chirp_2" if not provided.
GOOGLE_SPEECH_MODEL = os.getenv('GOOGLE_SPEECH_MODEL', 'chirp_2')

# New environment variable for Genesys API key.
GENESYS_API_KEY = os.getenv('GENESYS_API_KEY')
if not GENESYS_API_KEY:
    raise ValueError("GENESYS_API_KEY not found in environment variables.")

# New environment variable for Genesys Organization ID.
GENESYS_ORG_ID = os.getenv('GENESYS_ORG_ID')
if not GENESYS_ORG_ID:
    raise ValueError("GENESYS_ORG_ID not found in environment variables.")

# Genesys rate limiting constants
GENESYS_MSG_RATE_LIMIT = 5
GENESYS_BINARY_RATE_LIMIT = 5
GENESYS_MSG_BURST_LIMIT = 25
GENESYS_BINARY_BURST_LIMIT = 25

# Rate limiting constants 
RATE_LIMIT_MAX_RETRIES = 3

# Transcription Connector language support
SUPPORTED_LANGUAGES = os.getenv("SUPPORTED_LANGUAGES", "es-ES,it-IT,en-US")

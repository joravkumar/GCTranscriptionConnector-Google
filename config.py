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

# Genesys rate limiting constants (to respect Audio Hook limits)
GENESYS_MSG_RATE_LIMIT = 5
GENESYS_BINARY_RATE_LIMIT = 5
GENESYS_MSG_BURST_LIMIT = 25
GENESYS_BINARY_BURST_LIMIT = 25
GENESYS_RATE_WINDOW = 1.0 

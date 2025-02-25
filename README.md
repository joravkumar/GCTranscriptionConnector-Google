# Genesys AudioHook & Transcription Connector Server

This repository contains a production-ready implementation of a Genesys AudioHook & Transcription Connector server that processes real-time audio streams for transcription using the **Google Cloud Speech-to-Text API** and translation using **Google Gemini**. The transcribed and translated text is then injected back into Genesys Cloud via event messages. This server is designed to meet the Genesys AudioHook protocol requirements and supports essential transactions such as session establishment, audio streaming, ping/pong heartbeats, and clean disconnection.

The project is designed to be deployed on **Digital Ocean** (or a similar platform) and integrates with **Google Cloud** for transcription and **Google Gemini** for translation.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Code Structure](#code-structure)
- [Transcription and Translation Processing](#transcription-and-translation-processing)
- [Language Handling](#language-handling)
- [Deployment](#deployment)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Error Handling and Logging](#error-handling-and-logging)
- [Configuration](#configuration)

---

## Overview

The server accepts WebSocket connections from Genesys Cloud (the AudioHook client) and performs the following key operations:

1. **Connection Establishment & Validation:**  
   - Validates incoming HTTPS upgrade requests against required headers (e.g., API key, organization ID).
   - Negotiates a media format (typically PCMU at 8000 Hz).

2. **Session Lifecycle Management:**  
   - Manages the session lifecycle by handling "open", "ping", "close", and other transaction messages.
   - Sends an "opened" message to Genesys Cloud upon successful open transaction, enabling audio streaming.

3. **Real-Time Audio Processing:**  
   - Processes incoming audio frames (in PCMU format) in real time.
   - Converts audio frames from PCMU (u-law) to PCM16 using the Python `audioop` module.

4. **Transcription via Google Cloud Speech-to-Text:**  
   - Sends PCM16 audio to the Google Cloud Speech-to-Text API for transcription in the source language.

5. **Translation via Google Gemini:**  
   - Translates the transcribed text to the destination language using Google Gemini.
   - Uses structured output to ensure only the translated text is returned.

6. **Injection Back into Genesys Cloud:**  
   - Constructs a transcript event message with the translated text.
   - Sends the message back to Genesys Cloud via the WebSocket connection for injection into the conversation.

---

## Architecture

The application is built around the following core components:

- **WebSocket Server:**  
  - Uses the `websockets` library to manage connections and message exchanges with Genesys Cloud.

- **Session Handler (`AudioHookServer`):**  
  - Processes incoming messages, handles transactions (open, ping, close, etc.), and manages rate limiting.
  - Implemented in `audio_hook_server.py`.

- **Audio Processing:**  
  - Converts audio frames from PCMU to PCM16 using `audioop`.
  - Feeds PCM16 audio to Google Cloud Speech-to-Text for transcription.
  - Translates transcribed text using Google Gemini.

- **Transcription and Translation:**  
  - **Transcription:** Uses Google Cloud Speech-to-Text API with streaming recognition for real-time transcription.
  - **Translation:** Uses Google Gemini with structured output to ensure only the translated text is returned.

- **Rate Limiting:**  
  - Implements a custom rate limiter to prevent exceeding Genesys Cloud's messaging rate limits.
  - Defined in `rate_limiter.py`.

- **Environment Configuration:**  
  - Loads configurations (API keys, Google Cloud settings, rate limits) from environment variables.
  - Managed in `config.py`.

---

## Code Structure

- **Procfile**  
  Specifies the command to start the application:
  ```bash
  web: python main.py
  ```

- **main.py**  
  - Main entry point that starts the WebSocket server.
  - Validates incoming connections and delegates handling to `AudioHookServer`.
  - Includes WebSocket handshake validation and health endpoint (`/health`) for Digital Ocean.

- **audio_hook_server.py**  
  - Contains the `AudioHookServer` class, which manages:
    - Session lifecycle (open, ping, close, etc.).
    - Audio frame processing and rate limiting.
    - Transcription and translation event sending back to Genesys Cloud.

- **google_speech_transcription.py**  
  - Implements the `StreamingTranscription` class for real-time transcription using Google Cloud Speech-to-Text.
  - Handles audio conversion from PCMU to PCM16 and feeds it to the API.
  - Includes `normalize_language_code` for BCP-47 language code normalization.

- **google_gemini_translation.py**  
  - Implements the `translate_with_gemini` function for translating text using Google Gemini.
  - Uses structured output (via Pydantic) to ensure only the translation is returned.
  - Handles translation errors and logs them appropriately.

- **rate_limiter.py**  
  - Provides an asynchronous rate limiter (`RateLimiter`) to throttle message sending.
  - Supports Genesys Cloud's rate limits (e.g., 5 messages/sec, 25 burst limit).

- **config.py**  
  - Loads all configuration variables from environment variables.
  - Includes settings for Google Cloud, Google Gemini, Genesys, and rate limiting.

- **utils.py**  
  - Contains helper functions:
    - `format_json`: Pretty-prints JSON for logging.
    - `parse_iso8601_duration`: Parses ISO 8601 duration strings for rate limiting.

- **requirements.txt**  
  - Lists all Python dependencies required for the project.

---

## Transcription and Translation Processing

1. **Receiving Audio:**  
   - Genesys Cloud streams audio frames (binary WebSocket messages) after the open transaction.
   - Each frame is received in `AudioHookServer.handle_audio_frame`.

2. **Real-Time Processing:**  
   - Converts audio frames from PCMU (u-law) to PCM16 using `audioop`.
   - Supports multi-channel audio (e.g., stereo, with external and internal channels).

3. **Transcription:**  
   - Uses Google Cloud Speech-to-Text API with streaming recognition.
   - Feeds PCM16 audio to `StreamingTranscription` instances (one per channel).
   - Retrieves transcription results with word-level timing and confidence scores.

4. **Translation:**  
   - Sends transcribed text to Google Gemini via `translate_with_gemini`.
   - Uses structured output (JSON with Pydantic model) to ensure only the translated text is returned.
   - Logs translation failures and skips sending events if translation fails.

5. **Injection Back into Genesys Cloud:**  
   - Constructs a transcript event message with:
     - Unique transcript ID.
     - Channel identifier (e.g., 0 for external, 1 for internal).
     - Translated text, offset, duration, and confidence.
   - Sends the event to Genesys Cloud via WebSocket for conversation injection.

---

## Language Handling

- **Input Language (Source):**  
  - Chirp 2, the default Google model we use for transcription, doesn't support auto language detection. That is why we have to explicitly provide the input language. Determined from the `customConfig.inputLanguage` field in the "open" message received from Genesys Cloud. You have to configure something like this in the Configuration - Advanced section of your GC Transcription connector integration:
```json 
	{
	  "inputLanguage": "es-es"
	}
```	
  
  - Used for transcription via Google Cloud Speech-to-Text.
  - Defaults to "en-US" if not provided.
  - Normalized to BCP-47 format using `normalize_language_code`.

- **Destination Language:**  
  - Determined from the `language` field in the "open" message.
  - Used as the target language for translation via Google Gemini.
  - Normalized to BCP-47 format.

*Note: Language handling is now entirely driven by the data provided in the open message, eliminating the need for separate environment variables to define the destination language or supported languages.*

---

## Deployment

This project is designed to be deployed on **Digital Ocean** (or a similar platform). It integrates with **Google Cloud** for transcription (Speech-to-Text API) and **Google Gemini** for translation.

---

## Prerequisites

- **Dependencies:**  
  All Python dependencies are listed in `requirements.txt`:
  - `websockets`
  - `aiohttp`
  - `pydub`
  - `python-dotenv`
  - `google-cloud-speech`
  - `google-generativeai`

- **Google Cloud Account:**  
  - Required for Google Cloud Speech-to-Text API access.
  - Set up a service account and download the JSON key.

- **Google Gemini API Key:**  
  - Required for translation services.
  - Obtain from Google AI Studio or similar.

---

## Usage

1. **Local Development:**  
   - Set up your environment variables (you can use a `.env` file).
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```
   - Run the server:
     ```bash
     python main.py
     ```

2. **Deployment on Digital Ocean App Platform:**  
   - Configure environment variables in the App Platform settings.
   - Deploy the application; the Procfile will trigger the start command.

---

## Error Handling and Logging

- **Error Logging:**  
  - Logs detailed debug and error messages for:
    - WebSocket connection issues.
    - Audio processing errors.
    - Transcription and translation failures.
    - Rate limiting events.

- **Transcription and Translation Logging:**  
  - Transcription results and events sent to Genesys are logged at the `INFO` level.
  - Translation failures are logged with details.

- **Graceful Shutdown:**  
  - Handles close transactions by sending a "closed" message to Genesys Cloud.
  - Cleans up session resources (stops transcription threads, cancels tasks).

- **Rate Limiting:**  
  - Implements backoff for 429 errors (rate limit exceeded) from Genesys.
  - Supports retry-after durations from Genesys or HTTP headers.

---

## Configuration

All configurable parameters are defined in `config.py` and loaded from environment variables. Below is a list of required environment variables:

| Variable                          | Description                                                                 | Default     |
|-----------------------------------|-----------------------------------------------------------------------------|-------------|
| `GOOGLE_CLOUD_PROJECT`            | Google Cloud project ID for Speech-to-Text API                              | -           |
| `GOOGLE_APPLICATION_CREDENTIALS`  | JSON key for Google Cloud service account                                   | -           |
| `GOOGLE_SPEECH_MODEL`             | Speech recognition model (e.g., 'chirp_2')                                  | chirp_2     |
| `GOOGLE_TRANSLATION_MODEL`        | Google Gemini model for translation                                         | -           |
| `GEMINI_API_KEY`                  | API key for Google Gemini                                                   | -           |
| `GENESYS_API_KEY`                 | API key for Genesys Cloud Transcription Connector                           | -           |
| `GENESYS_ORG_ID`                  | Genesys Cloud organization ID                                               | -           |
| `DEBUG`                           | Set to "true" for increased logging granularity                             | false       |
| `SUPPORTED_LANGUAGES`             | Comma-separated list of supported input languages                           | es-ES,it-IT |

---

# Genesys AudioHook & Transcription Connector Server

This repository contains a production-ready implementation of a Genesys AudioHook & Transcription Connector server that processes real-time audio streams for transcription using the **Google Cloud Speech-to-Text API** and translation using **Google Gemini**. The transcribed (and optionally translated) text is then injected back into Genesys Cloud via event messages. This server is designed to meet the Genesys AudioHook protocol requirements and supports essential transactions such as session establishment, audio streaming, ping/pong heartbeats, and clean disconnection.

The project is designed to be deployed on **Digital Ocean** (or a similar platform) and integrates with **Google Cloud** for transcription and **Google Gemini** for translation.

---

## Table of Contents

- [Overview](#overview)
- [Use Cases](#use-cases)
- [Architecture](#architecture)
- [Code Structure](#code-structure)
- [Transcription and Translation Processing](#transcription-and-translation-processing)
- [Supported Speech Models](#supported-speech-models)
- [Language Handling](#language-handling)
- [Deployment](#deployment)
  - [Digital Ocean App Platform Configuration](#digital-ocean-app-platform-configuration)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Error Handling and Logging](#error-handling-and-logging)
- [Configuration](#configuration)
- [Issues](#issues)

---

## Overview

The server accepts WebSocket connections from Genesys Cloud (the AudioHook client) and performs the following key operations:

1. **Connection Establishment & Validation:**  
   - Validates incoming HTTPS upgrade requests against required headers (e.g., API key, organization ID).
   - Negotiates a media format (typically PCMU at 8000 Hz).

2. **Session Lifecycle Management:**  
   - Manages the session lifecycle by handling "open", "ping", "close", and other transaction messages.
   - Sends an "opened" message to Genesys Cloud upon successful open transaction, enabling audio streaming.

3. **Real-Time Audio Processing and Control Message Handling:**  
   - Processes incoming audio frames (in PCMU format) in real time.
   - Converts audio frames from PCMU (u-law) to PCM16 using the Python `audioop` module.
   - **Control Message Handling:** The server processes control messages—such as "paused", "discarded", and "resumed"—to adjust the effective audio timeline. This ensures that the computed offsets for transcription events exclude any periods where audio was lost or intentionally paused, aligning with the Genesys AudioHook protocol requirements.

4. **Transcription via Google Cloud Speech-to-Text:**  
   - Sends PCM16 audio to the Google Cloud Speech-to-Text API for transcription in the source language.

5. **Translation via Google Gemini (Optional):**  
   - If enabled via `customConfig.enableTranslation` in the open message, translates the transcribed text to the destination language using Google Gemini (see 'enableTranslation' in 'Language Handling' section).
   - Uses structured output to ensure only the translated text is returned.
   - If disabled or not specified, the original transcript is returned without translation, using the input language.

6. **Injection Back into Genesys Cloud:**  
   - Constructs a transcript event message with the (translated or original) text, including accurate offset and duration values adjusted for any control messages.
   - Sends the message back to Genesys Cloud via the WebSocket connection for injection into the conversation.

---

## Use Cases

This connector is designed to support two primary use cases that address different needs in contact center environments:

### 1. Transcription Only (No Translation)

This use case is ideal when you need an specialized transcription engine different than the native options provided by Genesys Cloud or EVTS.

**Key benefits:**
- In this repo we are leveraging Google's advanced speech recognition capabilities
- Supports languages that might not be available in Genesys' native transcription or EVTS

**Configuration:**
- Set `enableTranslation: false` or omit it in the `customConfig`
- Ensure the `inputLanguage` in `customConfig` matches the language being spoken

This approach maintains the original language throughout the conversation, making it suitable for environments where all systems (including analytics, agent assistance, etc.) support the source language.

### 2. Transcription + Translation

This use case is particularly valuable for enabling advanced Genesys features (like Copilot or Speech & Text Analytics) for languages that aren't directly supported by these tools.

**Example scenario:**
A contact center serves customers who speak a regionally-important language (such as Basque, Zulu, Welsh, etc.) that isn't directly supported by Genesys Copilot or STA. However, these tools do support a widely-used language in that same region (such as Spanish or English).

**How it works:**
1. The customer speaks in their preferred language (e.g., Basque)
2. The connector transcribes the audio in the source language
3. The text is translated to a widely-supported language (e.g., Spanish)
4. Genesys Cloud receives the translated transcript, enabling tools like Copilot and STA to function

**Key benefits:**
- Extends advanced Genesys features to additional languages
- Provides a more inclusive customer experience
- Leverages existing agent language capabilities
- Enables analytics and assistance tools across more languages

**Configuration:**
- Set `enableTranslation: true` in the `customConfig`
- Set `inputLanguage` to the regionally-important language (source)
- The `language` field in the message determines the target language for translation

This use case is especially valuable in regions with linguistic diversity, where contact centers need to support regional languages while leveraging tools optimized for more widely-spoken languages.

---

## Architecture

The application is built around the following core components:

- **WebSocket Server:**  
  - Uses the `websockets` library to manage connections and message exchanges with Genesys Cloud.

- **Session Handler (`AudioHookServer`):**  
  - Processes incoming messages, handles transactions (open, ping, close, etc.), manages rate limiting, and adjusts transcription offsets based on control messages.
  - Implemented in `audio_hook_server.py`.

- **Audio Processing:**  
  - Converts audio frames from PCMU to PCM16 using `audioop`.
  - Feeds PCM16 audio to Google Cloud Speech-to-Text for transcription.
  - Optionally translates transcribed text using Google Gemini.

- **Transcription and Translation:**  
  - **Transcription:** Uses Google Cloud Speech-to-Text API with streaming recognition for real-time transcription.
  - **Translation (Optional):** Uses Google Gemini with structured output to ensure only the translated text is returned. This step is performed only if `customConfig.enableTranslation` is set to true in the open message.

- **Rate Limiting:**  
  - Implements a custom rate limiter to prevent exceeding GC AudioHook's messaging rate limits.
  - Defined in `rate_limiter.py`.

- **Environment Configuration:**  
  - Loads configurations (API keys, Google Cloud settings, rate limits, supported languages, etc.) from environment variables.
  - Managed in `config.py`.

---

## Code Structure

- **Procfile**  
  Specifies the command to start the application:
  ```bash
  web: python main.py
  ```

- **main.py**  
  Main entry point that starts the WebSocket server.
  Validates incoming connections and delegates handling to AudioHookServer.
  Includes WebSocket handshake validation and health endpoint (/health) for Digital Ocean.

- **audio_hook_server.py**  
  Contains the AudioHookServer class, which manages:
  - Session lifecycle (open, ping, close, etc.).
  - Audio frame processing, control message handling, and rate limiting.
  - Transcription and (optionally) translation event sending back to Genesys Cloud.
  - For probe connections, the server sends the list of supported languages (as defined in the SUPPORTED_LANGUAGES environment variable) to Genesys Cloud.
  - The server adjusts transcript offsets based on control messages (`paused`, `discarded`, and `resumed`) to ensure that only the processed audio timeline is considered.

- **google_speech_transcription.py**  
  Implements the StreamingTranscription class for real-time transcription using Google Cloud Speech-to-Text.
  Handles audio conversion from PCMU to PCM16 and feeds it to the API.
  Includes normalize_language_code for BCP-47 language code normalization.

- **google_gemini_translation.py**  
  Implements the translate_with_gemini function for translating text using Google Gemini.
  Uses structured output (via Pydantic) to ensure only the translation is returned.
  Handles translation errors and logs them appropriately.

- **rate_limiter.py**  
  Provides an asynchronous rate limiter (RateLimiter) to throttle message sending.
  Supports Genesys Cloud's rate limits (e.g., 5 messages/sec, 25 burst limit).

- **config.py**  
  Loads all configuration variables from environment variables.
  Includes settings for Google Cloud, Google Gemini, Genesys, rate limiting, and supported languages.

- **utils.py**  
  Contains helper functions:
  - `format_json`: Pretty-prints JSON for logging.
  - `parse_iso8601_duration`: Parses ISO 8601 duration strings for rate limiting.

- **requirements.txt**  
  Lists all Python dependencies required for the project.

---

## Transcription and Translation Processing

- **Receiving Audio:**  
  - Genesys Cloud streams audio frames (binary WebSocket messages) after the open transaction.
  - Each frame is received in `AudioHookServer.handle_audio_frame`.

- **Real-Time Processing:**  
  - Converts audio frames from PCMU (u-law) to PCM16 using `audioop`.
  - Supports multi-channel audio (e.g., stereo, with external and internal channels).

- **Control Message Handling:**  
  - The server processes control messages such as **"paused"**, **"discarded"**, and **"resumed"**.
  - These messages adjust an internal offset (tracked as processed audio samples) so that transcription offsets and durations accurately reflect only the audio that was received (excluding any gaps due to pauses or audio loss).

- **Transcription:**  
  - Uses Google Cloud Speech-to-Text API with streaming recognition.
  - Feeds PCM16 audio to `StreamingTranscription` instances (one per channel).
  - Retrieves transcription results with word-level timing and confidence scores.
  - Adjusts calculated offsets by subtracting the cumulative gap from control messages.

- **Translation (Optional):**  
  - If `customConfig.enableTranslation` is set to true in the open message, the transcribed text is sent to Google Gemini for translation into the destination language.
  - If disabled or not specified, the original transcript is returned without translation, using the input language.
  - Structured output ensures that only the translated (or original) text is returned.
  - Translation failures are logged and skipped.

- **Injection Back into Genesys Cloud:**  
  - Constructs a transcript event message with:
    - Unique transcript ID.
    - Channel identifier (e.g., 0 for external, 1 for internal).
    - Transcribed text with adjusted offsets, duration, and confidence.
  - Sends the event to Genesys Cloud via WebSocket for conversation injection.

---

## Supported Speech Models

This connector supports two Google Cloud Speech-to-Text models:

- **Chirp 2:** 
  - The most advanced model with full feature support, including:
    - Greater performance
    - Word-level confidence scores
    - Limited language support

- **Chirp:**
  - Good model with big language support:
    - Does not support word-level confidence scores (fixed value of 1.0 is used)

The connector automatically adapts to whichever model is specified in the `GOOGLE_SPEECH_MODEL` environment variable, adjusting request parameters and response handling accordingly. When using Chirp, the connector still maintains full compatibility with the Genesys AudioHook protocol by supplying default confidence values where needed.

---

## Language Handling

- **Input Language (Source):**  
  - Chirp 2, the default Google model we use for transcription, doesn't support auto language detection. That is why you must explicitly provide the input language.
  - Determined from the `customConfig.inputLanguage` field in the "open" message received from Genesys Cloud. For example:
    ```json
    {
      "inputLanguage": "es-es",
      "enableTranslation": true
    }
    ```
  - Used for transcription via Google Cloud Speech-to-Text.
  - Defaults to "en-US" if not provided.
  - Normalized to BCP-47 format using `normalize_language_code`.

- **Destination Language:**  
  - Determined from the `language` field in the "open" message.
  - Used as the target language for translation via Google Gemini when translation is enabled.
  - Normalized to BCP-47 format.

- **Supported Languages:**  
  - Defined in the `SUPPORTED_LANGUAGES` environment variable (comma-separated, e.g., "es-ES,it-IT,en-US"). They must be supported by the Google model we are leveraging (for Chirp 2, see: [Google Cloud Speech-to-Text Chirp 2 Model](https://cloud.google.com/speech-to-text/v2/docs/chirp_2-model)).
  - Sent to Genesys Cloud in the "opened" message for probe connections.

- **Translation Toggle:**  
  - The `customConfig.enableTranslation` boolean in the open message controls whether translation is enabled for the session.
  - If disabled or not specified, the server returns the original transcription without translation, using the input language.

---

## Deployment

This project is designed to be deployed on Digital Ocean (or a similar platform). It integrates with Google Cloud for transcription (Speech-to-Text API) and Google Gemini for translation.

### Digital Ocean App Platform Configuration

When deploying this application on Digital Ocean App Platform, you'll need to configure the following settings:

- **HTTP Request Routes**  
  - Route Path: `/audiohook`
  - Preserve Path Prefix: Enabled (check this option to ensure the path will remain `/audiohook` when forwarded to the component)

- **Ports**  
  - Public HTTP Port: 443 (for HTTPS connections)

- **Health Checks**  
  - Path: `/health`
  - Protocol: HTTP

- **Commands**  
  - Build Command: None
  - Run Command: `python main.py`

These settings ensure that:

- The application listens on the correct path (`/audiohook`) for incoming Genesys Cloud AudioHook connections.
- The health check path (`/health`) is properly configured to allow Digital Ocean to monitor the application's status.
- The application starts correctly with the proper run command.

When configuring your Genesys Cloud AudioHook integration, use the full URL provided by Digital Ocean (e.g., `https://startish-app-1gxm4.ondigitalocean.app/audiohook`) as your connector endpoint.

---

## Prerequisites

- **Dependencies:**  
  All Python dependencies are listed in `requirements.txt`:
  - websockets
  - aiohttp
  - pydub
  - python-dotenv
  - google-cloud-speech
  - google-generativeai

- **Google Cloud Account:**  
  - Required for Google Cloud Speech-to-Text API access.
  - Set up a service account and download the JSON key.

- **Google Gemini API Key:**  
  - Required for translation services.
  - Obtain from Google AI Studio or similar.

---

## Usage

- **Local Development:**  
  - Set up your environment variables (you can use a `.env` file).
  - Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
  - Run the server:
    ```bash
    python main.py
    ```

- **Deployment on Digital Ocean App Platform:**  
  - Configure environment variables in the App Platform settings.
  - Set up HTTP routes, health checks, and commands as described in the Digital Ocean App Platform Configuration section.
  - Deploy the application; the Run Command will trigger the start command.

---

## Error Handling and Logging

- **Error Logging:**  
  - Logs detailed debug and error messages for:
    - WebSocket connection issues.
    - Audio processing errors.
    - Transcription and translation failures.
    - Rate limiting events.

- **Transcription and Translation Logging:**  
  - Transcription results and events sent to Genesys are logged at the INFO level.
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

| Variable                        | Description                                                                                   | Default           |
|---------------------------------|-----------------------------------------------------------------------------------------------|-------------------|
| GOOGLE_CLOUD_PROJECT            | Google Cloud project ID for Speech-to-Text API                                                | -                 |
| GOOGLE_APPLICATION_CREDENTIALS  | JSON key for Google Cloud service account                                                    | -                 |
| GOOGLE_SPEECH_MODEL             | Speech recognition model ('chirp_2' or 'chirp')                                               | chirp_2           |
| GOOGLE_TRANSLATION_MODEL        | Google Gemini model for translation                                                           | -                 |
| GEMINI_API_KEY                  | API key for Google Gemini                                                                     | -                 |
| GENESYS_API_KEY                 | API key for Genesys Cloud Transcription Connector                                             | -                 |
| GENESYS_ORG_ID                  | Genesys Cloud organization ID                                                                 | -                 |
| DEBUG                           | Set to "true" for increased logging granularity                                               | false             |
| SUPPORTED_LANGUAGES             | Comma-separated list of supported input languages (e.g., "es-ES,it-IT,en-US")                   | es-ES,it-IT       |


---

## Issues

- **Random numbers in the transcription:**  
  - From time to time some arbitrary numbers show up in the transcription, totally unrelated to the conversation itself. It requires further investigation.

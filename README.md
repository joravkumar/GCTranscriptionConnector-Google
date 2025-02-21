# Genesys AudioHook Transcription Server

This repository contains a production-ready implementation of a Genesys AudioHook server that processes real-time audio streams for transcription using the OpenAI Translation API (currently limited to English). The transcribed text is then injected back into Genesys Cloud via event messages. This server is designed to meet the Genesys AudioHook protocol requirements and supports essential transactions such as session establishment, audio streaming, ping/pong heartbeats, and clean disconnection.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Code Structure](#code-structure)
- [Transcription Processing and Injection](#transcription-processing-and-injection)
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
   The server validates incoming HTTPS upgrade requests against required headers (e.g., API key, organization ID) and negotiates a media format (typically PCMU at 8000 Hz).

2. **Session Lifecycle Management:**  
   It manages the session lifecycle by handling the "open", "ping", "close", and various other transaction messages. A successful open transaction results in an "opened" message being sent back to Genesys Cloud, after which audio frames can be streamed.

3. **Real-Time Audio Processing:**  
   Incoming audio frames (in PCMU format) are processed in real time. Each audio frame is converted from PCMU (u-law) to PCM16, then exported to MP3 using FFmpeg (installed via an Aptfile).

4. **Transcription via OpenAI:**  
   The MP3 audio chunk is then sent to the OpenAI Translation API (which currently only supports English) to obtain a transcript.

5. **Injection Back into Genesys Cloud:**  
   Once a transcript is obtained, the server constructs a transcript event message and sends it over the WebSocket back to Genesys Cloud. This event message follows the Genesys protocol, allowing Genesys Cloud to inject the transcript into the ongoing conversation for purposes such as speech analytics, agent assistance, or further processing.

---

## Architecture

The application is built around the following core components:

- **WebSocket Server:**  
  Uses the `websockets` library to manage connections and message exchanges with Genesys Cloud.

- **Session Handler (`AudioHookServer`):**  
  Responsible for processing incoming messages, handling transactions (open, ping, close, etc.), and managing the rate limiting of messages.

- **Audio Processing and Transcription:**  
  Audio frames received from Genesys Cloud are processed in real time. The conversion from PCMU to PCM16 is performed using the Python `audioop` module and the `pydub` library (which relies on FFmpeg). The processed audio is then sent to OpenAI's translation endpoint to retrieve the transcript.

- **Rate Limiting:**  
  A custom rate limiter ensures that the server does not exceed Genesys Cloud's messaging rate limits.

- **Environment Configuration:**  
  Configurations such as API keys, supported languages, and rate limit settings are loaded from environment variables via the `config.py` file.

---

## Code Structure

- **Procfile**  
  Specifies the command to start the application:
  ```
  web: python oai_middleware.py
  ```

- **oai_middleware.py**  
  The main entry point that starts the WebSocket server, performs connection validations, and delegates connection handling to the session handler.

- **audio_hook_server.py**  
  Contains the `AudioHookServer` class, which implements the session lifecycle. This includes handling open, ping, close, error messages, processing audio frames, and sending transcription event messages back to Genesys Cloud.

- **openai_translation.py**  
  Implements the `translate_audio` function that:
  - Converts raw PCMU audio data to PCM16.
  - Exports the PCM16 data to an MP3 file using FFmpeg.
  - Sends the MP3 file to the OpenAI Translation API to obtain a transcript.
  
- **rate_limiter.py**  
  Provides a basic asynchronous rate limiter used to throttle message sending.

- **config.py**  
  Loads all configuration variables from environment variables. It includes API keys, Genesys rate limit settings, and supported languages.

- **utils.py**  
  Contains helper functions such as `format_json` (for pretty-printing JSON) and `parse_iso8601_duration` (to handle duration strings).

---

## Transcription Processing and Injection

1. **Receiving Audio:**  
   Once the Genesys Cloud client completes the open transaction, it begins streaming audio frames (binary WebSocket messages). Each frame is received in `AudioHookServer.handle_audio_frame`.

2. **Real-Time Processing:**  
   The received audio frame is processed in real time:
   - The frame is converted from PCMU (u-law) to PCM16 using the Python `audioop` module.
   - The PCM16 audio is wrapped into an `AudioSegment` using `pydub`.

3. **Conversion to MP3:**  
   The `AudioSegment.export` method converts the PCM16 audio into an MP3 file. This step relies on FFmpeg, which is installed via the Aptfile.

4. **Calling OpenAI API:**  
   The MP3 file is sent to the OpenAI Translation API by calling `openai.Audio.translations.create`. The API returns a JSON response containing the transcript text.

5. **Injecting Transcription Back to Genesys Cloud:**  
   Upon receiving the transcript, the server builds a "transcript" event message:
   - The message is structured per Genesys Cloud specifications.
   - It includes a unique transcript ID, a channel identifier (typically `external`), a flag indicating whether the transcript is final (in this implementation, it is marked as interim with `isFinal: False`), and the transcript text in the `alternatives` field.
   - The event message is then sent to Genesys Cloud using the `_send_json` method, thereby injecting the transcription back into the Genesys Cloud conversation. Genesys Cloud can then use the transcription for features like speech and text analytics or other WEM features.

---

## Language Handling

Since the OpenAI Translation API currently supports only English, the following language handling occurs:

- **Incoming Language Parameter:**  
  Genesys Cloud may send an initial language preference in the "open" message. However, regardless of the specified language, the transcription obtained from OpenAI will be in English.

- **Injection of Transcription:**  
  When constructing the transcript event message, the server does not perform an explicit language update. Instead, it is expected that Genesys Cloud interprets the English transcript accordingly.  
  *If required by your integration,* you could enhance the system to send an "update" message with the language parameter set to `"en-US"`, ensuring that Genesys Cloud updates the conversation language to English. This is a logical extension if your integration needs to override the client's language settings.

---

## Deployment

This project is specifically designed to be deployed on DigitalOcean as an App Platform service. The deployment can be done using either the provided Dockerfile (recommended) or through buildpacks.

### Dockerfile Deployment

The repository includes a production-ready Dockerfile.

### Digital Ocean App Platform Configuration

To use the Dockerfile deployment method in Digital Ocean App Platform, you need to modify the App Spec configuration:

1. Go to App Settings > App Spec > edit
2. Locate the `services` section
3. Replace:
   ```yaml
   services:
     - environment_slug: python
   ```
   With:
   ```yaml
   services:
     - dockerfile_path: Dockerfile
   ```
4. Leave the rest of the file unaltered

This configuration tells Digital Ocean App Platform to use your Dockerfile for building and deploying the application instead of using the default Python buildpack.

### Alternative Buildpack Deployment

If you prefer using buildpacks instead of Docker:

- **Environment Variables:**  
  The following environment variables must be configured in your deployment environment:

  | Variable | Description | Default |
  |----------|-------------|---------|
  | `OPENAI_TRANSCRIPTION_MODEL` | OpenAI model to use for transcription. Currently only supports "whisper-1" | "whisper-1" |
  | `OPENAI_API_KEY` | Your OpenAI API key for accessing the transcription service | - |
  | `GENESYS_API_KEY` | Must match the API Key configured in the Genesys Cloud Transcription Connector integration | - |
  | `GENESYS_ORG_ID` | The source Genesys Cloud organization ID that has enabled the Transcription Connector integration | - |
  | `SUPPORTED_LANGUAGES` | Comma-separated list of supported input languages (ISO codes) | "es-ES,it-IT,en-US" |
  | `DEBUG` | Set to "true" for increased logging granularity in server logs | "false" |

- **Procfile:**  
  The Procfile specifies the command to start the application:
  ```
  web: python oai_middleware.py
  ```

---

## Prerequisites

- **Python 3.12.8** (as specified in `runtime.txt`)
- **Dependencies:**  
  All Python dependencies are listed in `requirements.txt`:
  - `websockets==15.0`
  - `aiohttp`
  - `pydub`
  - `python-dotenv`
  - `openai==1.63.2`

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
     python oai_middleware.py
     ```

2. **Deployment on DigitalOcean App Platform:**  
   - Ensure the Aptfile is included in your repository root.
   - Configure environment variables in the App Platform settings.
   - Deploy the application; the Procfile will trigger the start command.

---

## Error Handling and Logging

- **Error Logging:**  
  The server is configured to log detailed debug and error messages. This includes issues with WebSocket connections, audio processing, transcription failures (such as missing FFmpeg), and rate limiting.

- **Graceful Shutdown:**  
  The server handles close transactions by sending a "closed" message to Genesys Cloud and cleaning up session resources.

---

## Configuration

All configurable parameters are defined in `config.py`. This includes:
- API keys and organization identifiers.
- Rate limit settings (e.g., `GENESYS_MSG_RATE_LIMIT`, `GENESYS_BINARY_RATE_LIMIT`).
- Supported languages for transcription.
- Server listening host, port, and path.

---

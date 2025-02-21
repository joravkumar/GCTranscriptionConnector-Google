import asyncio
import audioop
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> str:
    if not audio_stream:
        logger.warning("No audio data received for transcription.")
        return ""
    try:
        logger.debug(f"Translating audio chunk of length {len(audio_stream)} bytes")
        # Determine number of channels from negotiated media; default to 1.
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1

        # Convert the accumulated PCMU (u-law) data to PCM16.
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        logger.debug(f"Converted PCMU to PCM16: {len(pcm16_data)} bytes, sample_width=2, frame_rate=8000, channels={channels}")

        # Define a synchronous transcription function.
        def transcribe():
            # Import project ID from config
            from config import GOOGLE_CLOUD_PROJECT
            if not GOOGLE_CLOUD_PROJECT:
                raise ValueError("GOOGLE_CLOUD_PROJECT not configured.")
            client = SpeechClient(client_options=ClientOptions(api_endpoint="us-central1-speech.googleapis.com"))
            config = cloud_speech.RecognitionConfig(
                encoding=cloud_speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=8000,
                language_codes=["en-US"],
                model="chirp_2",
            )
            request = cloud_speech.RecognizeRequest(
                recognizer=f"projects/{GOOGLE_CLOUD_PROJECT}/locations/us-central1/recognizers/_",
                config=config,
                content=pcm16_data,
            )
            response = client.recognize(request=request)
            transcripts = []
            for result in response.results:
                if result.alternatives:
                    transcripts.append(result.alternatives[0].transcript)
            return " ".join(transcripts)

        transcript = await asyncio.to_thread(transcribe)
        logger.debug(f"Received transcript: {transcript}")
        return transcript
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        return ""

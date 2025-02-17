import tempfile
import os
import audioop
from pydub import AudioSegment
import openai
from config import OPENAI_API_KEY, OPENAI_TRANSCRIPTION_MODEL

# Set the API key for the OpenAI SDK.
openai.api_key = OPENAI_API_KEY

async def translate_audio(audio_stream: bytes, negotiated_media: dict, logger) -> str:
    if not audio_stream:
        logger.warning("No audio data received for transcription.")
        return ""
    try:
        # Determine number of channels from negotiated media; default to 1.
        channels = 1
        if negotiated_media and "channels" in negotiated_media:
            channels = len(negotiated_media.get("channels", []))
            if channels == 0:
                channels = 1

        # Convert the accumulated PCMU (u-law) data to PCM16.
        pcm16_data = audioop.ulaw2lin(audio_stream, 2)
        # Create an AudioSegment from the PCM16 data.
        audio_segment = AudioSegment(
            data=pcm16_data,
            sample_width=2,
            frame_rate=8000,
            channels=channels
        )
        # Export the audio segment to an MP3 file.
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
            temp_filename = temp_mp3.name
        audio_segment.export(temp_filename, format="mp3")
        logger.info(f"Exported audio to MP3 file: {temp_filename}")

        # Open the MP3 file and use the OpenAI SDK to create a translation.
        with open(temp_filename, "rb") as audio_file:
            transcript_response = openai.Audio.translations.create(
                model=OPENAI_TRANSCRIPTION_MODEL,
                file=audio_file,
                response_format="json",
                temperature=0
            )
        # Clean up the temporary file.
        os.remove(temp_filename)
        transcript = transcript_response.get("text", "")
        return transcript
    except Exception as e:
        logger.error(f"Error in processing translation: {e}", exc_info=True)
        return ""

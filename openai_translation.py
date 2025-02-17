import asyncio
import tempfile
import aiohttp
import audioop
import os
from pydub import AudioSegment
from config import OPENAI_API_KEY

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

        # Prepare to send the MP3 file to OpenAI's translation endpoint.
        if not OPENAI_API_KEY:
            logger.error("OPENAI_API_KEY not found in config.")
            return ""
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        data_fields = {
            "model": "whisper-1",
            "response_format": "json",
            "temperature": "0"
        }
        async with aiohttp.ClientSession() as session:
            with open(temp_filename, "rb") as f:
                form = aiohttp.FormData()
                form.add_field("file", f, filename="audio.mp3", content_type="audio/mpeg")
                for key, value in data_fields.items():
                    form.add_field(key, value)
                url = "https://api.openai.com/v1/audio/translations"
                async with session.post(url, data=form, headers=headers) as resp:
                    if resp.status != 200:
                        logger.error(f"OpenAI translation API returned status code {resp.status}")
                        text = await resp.text()
                        logger.error(f"Response: {text}")
                        return ""
                    response_json = await resp.json()
        # Clean up the temporary file.
        os.remove(temp_filename)
        transcript = response_json.get("text", "")
        return transcript
    except Exception as e:
        logger.error(f"Error in processing translation: {e}", exc_info=True)
        return ""

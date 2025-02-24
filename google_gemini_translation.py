import asyncio
from google import genai
from config import GEMINI_API_KEY, GOOGLE_TRANSLATION_MODEL

async def translate_with_gemini(text: str, source_language: str, dest_language: str, logger) -> str:
    """
    Translate text from source_language to dest_language using the Gemini API.
    
    Args:
        text (str): The text to translate.
        source_language (str): The source language code (e.g., "en-US").
        dest_language (str): The destination language code (e.g., "es-ES").
        logger: Logger instance for debugging and error logging.
    
    Returns:
        str: The translated text, or the original text if translation fails.
    """
    def sync_translate():
        try:
            client = genai.Client(api_key=GEMINI_API_KEY)
            system_prompt = f"""
            You are a professional translator. Translate the following text from {source_language} to {dest_language}.
            Provide a professional and accurate translation. Respond with only the translated text, nothing else.

            Schema for the response:
            - Type: string
            - Description: The translated text in {dest_language}
            """
            response = client.models.generate_content(
                model=GOOGLE_TRANSLATION_MODEL,
                contents=[system_prompt, text],
                config={'response_mime_type': 'text/plain'}
            )
            translated_text = response.text.strip()
            logger.debug(f"Translated '{text}' to '{translated_text}'")
            return translated_text
        except Exception as e:
            logger.error(f"Error translating text with Gemini API: {e}")
            return text  # Fallback to original text if translation fails

    return await asyncio.to_thread(sync_translate)

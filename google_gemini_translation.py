import asyncio
import google.generativeai as genai
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
        str: The translated text, or None if translation fails.
    """
    logger.info(f"Starting translation from {source_language} to {dest_language} for text: '{text}'")
    
    def sync_translate():
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GOOGLE_TRANSLATION_MODEL)
            system_prompt = f"""
            You are a professional translator. Translate the following text from {source_language} to {dest_language}.
            Provide a professional and accurate translation. Respond with only the translated text, nothing else.

            Schema for the response:
            - Type: string
            - Description: The translated text in {dest_language}
            """
            logger.debug(f"System prompt: {system_prompt}")
            logger.debug(f"Text to translate: '{text}'")
            
            response = model.generate_content([system_prompt, text])
            translated_text = response.text.strip()
            logger.info(f"Translation successful: '{translated_text}'")
            return translated_text
        except Exception as e:
            logger.error(f"Error translating text with Gemini API: {type(e).__name__} - {str(e)}")
            logger.debug("Translation failed, returning None")
            return None  # Indicate failure

    return await asyncio.to_thread(sync_translate)

import asyncio
import google.generativeai as genai
import json
from config import GEMINI_API_KEY, GOOGLE_TRANSLATION_MODEL
from pydantic import BaseModel

# Define a Pydantic model for the structured response
class TranslationResponse(BaseModel):
    translation: str

async def translate_with_gemini(text: str, source_language: str, dest_language: str, logger) -> str:
    """
    Translate text from source_language to dest_language using the Gemini API with structured output.
    
    Args:
        text (str): The text to translate.
        source_language (str): The source language code (e.g., "es-ES").
        dest_language (str): The destination language code (e.g., "en-US").
        logger: Logger instance for debugging and error logging.
    
    Returns:
        str: The translated text, or None if translation fails.
    """
    logger.info(f"Starting translation from {source_language} to {dest_language} for text: '{text}'")
    
    def sync_translate():
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel(GOOGLE_TRANSLATION_MODEL)
            prompt = f"""
            You are a professional translator. Translate the following text from {source_language} to {dest_language}.
            Provide a professional and accurate translation.
            """
            logger.debug(f"Prompt: {prompt}")
            logger.debug(f"Text to translate: '{text}'")
            
            # Use structured output with JSON response
            response = model.generate_content(
                contents=[prompt, text],
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json",
                    response_schema=TranslationResponse
                )
            )
            # Parse the JSON response to extract the translation
            response_json = json.loads(response.text)
            translated_text = response_json.get("translation", None)
            if translated_text:
                logger.info(f"Translation successful: '{translated_text}'")
                return translated_text
            else:
                logger.error("Translation response did not contain 'translation' key")
                return None
        except Exception as e:
            logger.error(f"Error translating text with Gemini API: {type(e).__name__} - {str(e)}")
            logger.debug("Translation failed, returning None")
            return None  # Indicate failure

    return await asyncio.to_thread(sync_translate)

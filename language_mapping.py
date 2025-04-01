def normalize_language_code(lang: str) -> str:
    """
    Normalize language codes to the proper BCP-47 format (e.g. "es-es" -> "es-ES", "en-us" -> "en-US").
    If the language code does not contain a hyphen, return it as is.
    """
    if '-' in lang:
        parts = lang.split('-')
        if len(parts) == 2:
            return f"{parts[0].lower()}-{parts[1].upper()}"
    return lang

def get_openai_language_code(lang: str) -> str:
    """
    Convert a BCP-47 language code (e.g., "es-ES", "en-US") to the corresponding
    ISO 639-1/639-3 language code that OpenAI's speech API accepts.
    
    Args:
        lang: The BCP-47 language code
        
    Returns:
        A language code compatible with OpenAI's speech API
    """
    normalized = normalize_language_code(lang)
    
    # Dictionary mapping BCP-47 codes to OpenAI-compatible language codes
    language_mapping = {
        # Spanish varieties
        "es-ES": "es",
        "es-MX": "es",
        "es-AR": "es",
        "es-CO": "es",
        "es-CL": "es",
        "es-US": "es",
        
        # English varieties
        "en-US": "en",
        "en-GB": "en",
        "en-AU": "en",
        "en-CA": "en",
        "en-IN": "en",
        "en-NZ": "en",
        
        # French varieties
        "fr-FR": "fr",
        "fr-CA": "fr",
        "fr-BE": "fr",
        "fr-CH": "fr",
        
        # German varieties
        "de-DE": "de",
        "de-AT": "de",
        "de-CH": "de",
        
        # Portuguese varieties
        "pt-BR": "pt",
        "pt-PT": "pt",
        
        # Italian
        "it-IT": "it",
        
        # Dutch
        "nl-NL": "nl",
        "nl-BE": "nl",
        
        # Other major languages
        "ja-JP": "ja",
        "ko-KR": "ko",
        "zh-CN": "zh",
        "zh-TW": "zh",
        "ru-RU": "ru",
        "ar-SA": "ar",
        "hi-IN": "hi",
        "tr-TR": "tr",
        "pl-PL": "pl",
        "th-TH": "th",
        "vi-VN": "vi",
        "sv-SE": "sv",
        "da-DK": "da",
        "fi-FI": "fi",
        "no-NO": "no",
        "cs-CZ": "cs",
        "hu-HU": "hu",
        "el-GR": "el",
        "he-IL": "he",
        "id-ID": "id",
        "ro-RO": "ro",
        "zu-ZA": "zu",
        "ms-MY": "ms"
    }
    
    # If the language is in our mapping, return its value
    if normalized in language_mapping:
        return language_mapping[normalized]
    
    # If not in mapping but has a hyphen, return just the language part (first segment)
    if "-" in normalized:
        return normalized.split("-")[0].lower()
    
    # Otherwise return as is (likely already a simple ISO code)
    return normalized.lower()

# Add a list of languages that OpenAI doesn't support by code but can handle via prompts
OPENAI_UNSUPPORTED_LANGUAGE_CODES = {
    "zu": "Zulu",
    "zu-ZA": "Zulu",
    # Add other unsupported languages as needed
}

# Language-specific prompts for unsupported languages
# Include both the native language prompt and the English description
LANGUAGE_SPECIFIC_PROMPTS = {
    "zu": {
        "native": "Humusha ngesizulu (Mzansi Afrika)",
        "english": "Transcribe to Zulu (South Africa)"
    },
    "zu-ZA": {
        "native": "Humusha ngesizulu (Mzansi Afrika)",
        "english": "Transcribe to Zulu (South Africa)"
    },
}

def is_openai_unsupported_language(lang: str) -> bool:
    """
    Check if a language is known to be unsupported by code in OpenAI's API.
    
    Args:
        lang: The language code to check
        
    Returns:
        True if the language is known to be unsupported by code but can be handled via prompt
    """
    normalized = normalize_language_code(lang)
    code = normalized.split('-')[0].lower() if '-' in normalized else normalized.lower()
    
    return code in OPENAI_UNSUPPORTED_LANGUAGE_CODES or normalized in OPENAI_UNSUPPORTED_LANGUAGE_CODES

def get_language_name_for_prompt(lang: str) -> str:
    """
    Get the full language name for a given language code to use in prompts.
    
    Args:
        lang: The language code
        
    Returns:
        The full language name suitable for use in prompts
    """
    normalized = normalize_language_code(lang)
    code = normalized.split('-')[0].lower() if '-' in normalized else normalized.lower()
    
    # Check first for the full normalized code
    if normalized in OPENAI_UNSUPPORTED_LANGUAGE_CODES:
        return OPENAI_UNSUPPORTED_LANGUAGE_CODES[normalized]
    
    # Then check for just the language part
    if code in OPENAI_UNSUPPORTED_LANGUAGE_CODES:
        return OPENAI_UNSUPPORTED_LANGUAGE_CODES[code]
    
    # If not found, return a default based on the code
    return f"Language code {code}"

def get_language_specific_prompt(lang: str) -> str:
    """
    Get a language-specific prompt template for unsupported languages.
    
    Args:
        lang: The language code
        
    Returns:
        A prompt template in the target language if available, otherwise in English
    """
    normalized = normalize_language_code(lang)
    code = normalized.split('-')[0].lower() if '-' in normalized else normalized.lower()
    
    # Look for language-specific prompt template
    if normalized in LANGUAGE_SPECIFIC_PROMPTS:
        return LANGUAGE_SPECIFIC_PROMPTS[normalized]["native"]
    elif code in LANGUAGE_SPECIFIC_PROMPTS:
        return LANGUAGE_SPECIFIC_PROMPTS[code]["native"]
    
    # Fallback to English for unknown languages
    return "This is a customer service call. The customer may be discussing problems with services or products."

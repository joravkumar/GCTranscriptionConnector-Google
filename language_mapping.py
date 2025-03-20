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
    }
    
    # If the language is in our mapping, return its value
    if normalized in language_mapping:
        return language_mapping[normalized]
    
    # If not in mapping but has a hyphen, return just the language part (first segment)
    if "-" in normalized:
        return normalized.split("-")[0].lower()
    
    # Otherwise return as is (likely already a simple ISO code)
    return normalized.lower()

"""
Translation utility module using Google Translate API (googletrans).
Provides a clean interface for translating text to any supported language.
"""
from googletrans import Translator, LANGUAGES

# Singleton translator instance (reuse across requests)
_translator = Translator()

# Curated list of popular languages for the UI dropdown
SUPPORTED_LANGUAGES = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "ur": "Urdu",
    "ar": "Arabic",
    "zh-cn": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "it": "Italian",
    "nl": "Dutch",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "sw": "Swahili",
    "pl": "Polish",
}


def translate_text(text: str, dest_lang: str = "en") -> dict:
    """
    Translate the given text to the destination language.

    Args:
        text: The text to translate.
        dest_lang: ISO 639-1 language code (e.g., 'es', 'fr', 'hi').

    Returns:
        dict with keys: translated_text, src_lang, dest_lang
    """
    if not text or not text.strip():
        return {"translated_text": "", "src_lang": "en", "dest_lang": dest_lang}

    # If destination is English, no translation needed
    if dest_lang == "en":
        return {"translated_text": text, "src_lang": "en", "dest_lang": "en"}

    try:
        result = _translator.translate(text, dest=dest_lang)
        return {
            "translated_text": result.text,
            "src_lang": result.src,
            "dest_lang": dest_lang,
        }
    except Exception as e:
        print(f"[Translation Error] {e}")
        # Fallback: return original text on error
        return {"translated_text": text, "src_lang": "en", "dest_lang": dest_lang}


def translate_batch(texts: list, dest_lang: str = "en") -> list:
    """
    Translate a list of strings to the destination language.
    Iterates through strings individually for maximum stability.

    Args:
        texts: List of strings to translate.
        dest_lang: ISO 639-1 language code.

    Returns:
        List of translated strings.
    """
    if not texts or dest_lang == "en":
        return texts

    try:
        # Translate each string individually to avoid rare batch-merge bugs
        translated = []
        for text in texts:
            if not text or not text.strip():
                translated.append(text)
                continue
            
            res = _translator.translate(text, dest=dest_lang)
            translated.append(res.text)
        return translated
    except Exception as e:
        print(f"[Batch Translation Error] {e}")
        return texts


def get_supported_languages() -> dict:
    """Return the curated dictionary of supported languages."""
    return SUPPORTED_LANGUAGES


def get_all_languages() -> dict:
    """Return all languages supported by the Google Translate API."""
    return dict(LANGUAGES)

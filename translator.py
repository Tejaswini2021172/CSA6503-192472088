"""
Polyglot Neural Translation Engine powered by deep-translator.
Handles multi-language mapping, text chunking, and translation execution.
"""

from typing import Any, Dict, List, Optional
from deep_translator import GoogleTranslator
from deep_translator.exceptions import (
    LanguageNotSupportedException,
    NotValidPayload,
    TranslationNotFound,
)


class TranslationError(Exception):
    """Custom exception raised when translation fails."""
    pass


def get_supported_languages() -> Dict[str, str]:
    """Returns supported languages mapped to their ISO codes."""
    languages: Dict[str, str] = {
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Bengali": "bn",
        "Marathi": "mr",
        "Gujarati": "gu",
        "Kannada": "kn",
        "Malayalam": "ml",
        "Punjabi": "pa",
        "Urdu": "ur",
        "Chinese (Simplified)": "zh-CN",
        "Chinese (Traditional)": "zh-TW",
        "Japanese": "ja",
        "Korean": "ko",
        "Arabic": "ar",
        "Russian": "ru",
        "Portuguese": "pt",
        "Italian": "it",
        "Dutch": "nl",
        "Turkish": "tr",
        "Polish": "pl",
        "Vietnamese": "vi",
        "Thai": "th",
        "Indonesian": "id",
        "Swedish": "sv",
        "Greek": "el",
        "Hebrew": "he",
        "Czech": "cs",
        "Romanian": "ro",
        "Hungarian": "hu",
        "Danish": "da",
        "Finnish": "fi",
        "Norwegian": "no",
        "Ukrainian": "uk",
    }
    return dict(sorted(languages.items()))


def _split_text_into_chunks(text: str, max_chunk_size: int = 4000) -> List[str]:
    """Splits text into chunks to avoid API size limitations."""
    paragraphs = text.split("\n")
    chunks, current_chunk, current_length = [], [], 0

    for para in paragraphs:
        para_len = len(para) + 1
        if current_length + para_len > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [para]
            current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks if chunks else [text]


def translate_text(text: str, *args: Any, **kwargs: Any) -> str:
    """
    Translates text into the requested target language.
    Returns translated string or raises TranslationError.
    """
    target = kwargs.get("target_lang") or kwargs.get("target_language_code") or kwargs.get("target_language")
    if not target and args:
        target = args[0]
    source = kwargs.get("source_lang") or kwargs.get("source_language_code") or "auto"

    if not text or not text.strip():
        raise TranslationError("Translation input text is empty.")
    if not target or not str(target).strip():
        raise TranslationError("Target language was not specified.")

    cleaned_text = text.strip()

    try:
        translator = GoogleTranslator(source=source, target=str(target).strip())

        if len(cleaned_text) > 4000:
            chunks = _split_text_into_chunks(cleaned_text, max_chunk_size=4000)
            translated_chunks = [translator.translate(c) if c.strip() else "" for c in chunks]
            return "\n".join(translated_chunks)

        result = translator.translate(cleaned_text)
        if not result:
            raise TranslationError("Translation returned an empty result.")
        return result

    except LanguageNotSupportedException as e:
        raise TranslationError(f"Unsupported language: {str(e)}") from e
    except NotValidPayload as e:
        raise TranslationError(f"Invalid translation payload: {str(e)}") from e
    except TranslationNotFound as e:
        raise TranslationError(f"Translation not found: {str(e)}") from e
    except Exception as e:
        raise TranslationError(f"Translation failed: {str(e)}") from e
"""
Extractive Summarization Module using Latent Semantic Analysis (LSA).
Leverages Sumy and NLTK for unsupervised mathematical sentence extraction.
"""

from typing import Any, Optional
import nltk
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words


class ExtractiveSummaryError(Exception):
    """Custom exception raised when extractive summarization fails."""
    pass


def _ensure_nltk_resources() -> None:
    """Safely check and download required NLTK tokenization datasets."""
    for resource in ["punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except (LookupError, OSError):
            nltk.download(resource, quiet=True)


_ensure_nltk_resources()


def generate_extractive_summary(text: str, *args: Any, **kwargs: Any) -> str:
    """
    Generates an extractive summary using the LSA algorithm.
    Returns the summary as a string or raises ExtractiveSummaryError on failure.
    """
    count = kwargs.get("sentences_count") or kwargs.get("sentence_count")
    if count is None and args:
        count = args[0]
    count = count if count is not None else 5
    language = kwargs.get("language", "english")

    if not text or not text.strip():
        raise ExtractiveSummaryError("Input text is empty. Please provide valid text.")

    cleaned_text = text.strip()
    words = cleaned_text.split()
    if len(words) < 10:
        return cleaned_text

    try:
        parser = PlaintextParser.from_string(cleaned_text, Tokenizer(language))
        total_sentences = len(parser.document.sentences)
        if total_sentences == 0:
            return cleaned_text

        target_sentences = min(count, total_sentences)
        stemmer = Stemmer(language)
        summarizer = LsaSummarizer(stemmer)

        try:
            summarizer.stop_words = get_stop_words(language)
        except Exception:
            summarizer.stop_words = []

        extracted_sentences = summarizer(parser.document, target_sentences)
        summary_sentences = [str(s).strip() for s in extracted_sentences]

        if not summary_sentences:
            fallback = [str(s).strip() for s in parser.document.sentences[:target_sentences]]
            return " ".join(fallback)

        return " ".join(summary_sentences)

    except Exception as e:
        raise ExtractiveSummaryError(f"Extractive summarization failed: {str(e)}") from e


def extractive_summary(*args: Any, **kwargs: Any) -> str:
    """Alias function for generate_extractive_summary."""
    return generate_extractive_summary(*args, **kwargs)
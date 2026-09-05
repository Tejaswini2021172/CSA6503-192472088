"""
Abstractive Summarization Engine powered by Google Gemini SDK.
Handles prompt construction, granularity controls, and API execution.
"""

import os
from typing import Any, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError


class GeminiSummaryError(Exception):
    """Custom exception raised when Gemini abstractive summarization fails."""
    pass


def generate_gemini_summary(text: str, *args: Any, **kwargs: Any) -> str:
    """
    Generates an abstractive summary using Google Gemini.
    Returns the summary as a string or raises GeminiSummaryError on failure.
    """
    resolved_api_key = kwargs.get("api_key") or os.getenv("GEMINI_API_KEY")
    resolved_length = kwargs.get("length") or kwargs.get("summary_length") or "Standard (Balanced)"
    model_name = kwargs.get("model_name", "gemini-3.6-flash")

    if not text or not text.strip():
        raise GeminiSummaryError("Input document text is empty.")

    if not resolved_api_key or not resolved_api_key.strip():
        raise GeminiSummaryError("Gemini API Key is missing. Configure it in .env or the sidebar.")

    # Configure length constraints
    if "Concise" in resolved_length or resolved_length.lower() == "short":
        length_instruction = (
            "Provide a concise executive summary in 2 focused paragraphs highlighting the core takeaways."
        )
    elif "Detailed" in resolved_length or resolved_length.lower() == "long":
        length_instruction = (
            "Provide a detailed analytical summary in 4 to 5 paragraphs covering all key arguments and nuances."
        )
    else:
        length_instruction = (
            "Provide a balanced and structured summary in 3 paragraphs capturing the main thesis and key findings."
        )

    prompt = f"""You are an expert research analyst.
Summarize the following document accurately:

Document:
\"\"\"
{text.strip()}
\"\"\"

Requirements:
{length_instruction}

Rely strictly on facts grounded in the provided document."""

    try:
        client = genai.Client(api_key=resolved_api_key.strip())
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                top_p=0.95,
            ),
        )

        if not response or not response.text:
            raise GeminiSummaryError("Gemini returned an empty response.")

        return response.text.strip()

    except APIError as e:
        raise GeminiSummaryError(f"Google Gemini API Error: {e.message} (Code: {e.code})") from e
    except Exception as e:
        raise GeminiSummaryError(f"Unexpected error in Gemini summarization: {str(e)}") from e


def gemini_summary(*args: Any, **kwargs: Any) -> str:
    """Alias function for generate_gemini_summary."""
    return generate_gemini_summary(*args, **kwargs)
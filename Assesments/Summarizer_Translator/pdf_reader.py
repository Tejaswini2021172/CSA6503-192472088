"""
PDF Reader and Extraction Engine.
Parses text from multi-page PDFs using PyPDF2 with validation checks.
"""

import io
import re
from typing import Any, Tuple, Union
import PyPDF2


class PDFReadError(Exception):
    """Custom exception raised when a PDF cannot be read or parsed."""
    pass


def extract_text_from_pdf(pdf_file: Union[io.BytesIO, bytes, str, Any], *args: Any, **kwargs: Any) -> str:
    """
    Extracts text from a PDF file stream.
    Returns cleaned text string or raises PDFReadError.
    """
    if pdf_file is None:
        raise PDFReadError("No PDF file provided.")

    try:
        if hasattr(pdf_file, "read"):
            if hasattr(pdf_file, "seek"):
                pdf_file.seek(0)
            pdf_stream = io.BytesIO(pdf_file.read())
        elif isinstance(pdf_file, bytes):
            pdf_stream = io.BytesIO(pdf_file)
        elif isinstance(pdf_file, str):
            pdf_stream = open(pdf_file, "rb")
        else:
            raise PDFReadError("Unsupported file format passed.")

        reader = PyPDF2.PdfReader(pdf_stream)

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise PDFReadError("PDF is password-protected or encrypted.")

        if len(reader.pages) == 0:
            raise PDFReadError("The uploaded PDF contains 0 pages.")

        extracted_pages = []
        for page in reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())
            except Exception:
                continue

        if not extracted_pages:
            raise PDFReadError("No extractable text found. Document may be an image-only scan.")

        combined_text = "\n\n".join(extracted_pages)
        cleaned_text = re.sub(r"\r\n|\r", "\n", combined_text)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
        cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

        return cleaned_text.strip()

    except PyPDF2.errors.PdfReadError as e:
        raise PDFReadError(f"Corrupted PDF file: {str(e)}") from e
    except PDFReadError:
        raise
    except Exception as e:
        raise PDFReadError(f"Failed to read PDF: {str(e)}") from e


def validate_pdf_content(text: str, min_words: int = 15) -> Tuple[bool, str]:
    """Validates whether extracted PDF content is sufficient for summarization."""
    if not text or not text.strip():
        return False, "Document contains no readable text."

    words = text.strip().split()
    if len(words) < min_words:
        return False, f"Extracted content is too short ({len(words)} words)."

    return True, f"PDF text validated successfully ({len(words)} words)."
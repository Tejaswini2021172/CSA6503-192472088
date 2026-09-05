import pdfplumber

def extract_pdf_text(uploaded_file):
    """
    Extract text from all pages of a PDF.
    Returns the extracted text as a string.
    """

    text = ""

    try:
        with pdfplumber.open(uploaded_file) as pdf:

            for page in pdf.pages:

                page_text = page.extract_text()

                if page_text:
                    text += page_text + "\n"

    except Exception as e:
        return f"Error reading PDF: {e}"

    return text
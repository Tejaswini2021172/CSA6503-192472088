from read_pdf import load_pdf

PDF_PATH = r"C:\Users\tejv3\OneDrive\Desktop\RAG\College_Assistant\data\Generative AI and LLMs - Comprehensive Guide.pdf"

text = load_pdf(PDF_PATH)


def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        chunks.append(text[start:start + chunk_size])
        start += chunk_size - overlap

    return chunks


chunks = chunk_text(text)

print(f"Created {len(chunks)} chunks")
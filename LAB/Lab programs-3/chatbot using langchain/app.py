import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_google_genai import ChatGoogleGenerativeAI

# ------------------------------------
# Load API Key
# ------------------------------------
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("Google API Key not found.")

# ------------------------------------
# Load PDF
# ------------------------------------
loader = PyPDFLoader("documents/college_handbook.pdf")

documents = loader.load()

print(f"Loaded {len(documents)} pages.")

# ------------------------------------
# Split Document
# ------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

docs = splitter.split_documents(documents)

print(f"Created {len(docs)} chunks.")

# ------------------------------------
# Create Embeddings
# ------------------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------
# Build Vector Store
# ------------------------------------
vector_db = FAISS.from_documents(
    docs,
    embedding_model
)

retriever = vector_db.as_retriever(
    search_kwargs={"k":3}
)

# ------------------------------------
# Gemini Model
# ------------------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2
)

print("\nCollege Chatbot Ready!")

print("Type 'exit' to quit.\n")

# ------------------------------------
# Chat Loop
# ------------------------------------
while True:

    question = input("You : ")

    if question.lower()=="exit":
        break

    docs = retriever.invoke(question)

    context="\n\n".join(
        doc.page_content for doc in docs
    )

    prompt=f"""
You are a college assistant.

Answer ONLY using the context below.

If the answer is not found,

reply exactly:

'I couldn't find that information in the uploaded document.'

Context:

{context}

Question:

{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nBot :",response.content,"\n")
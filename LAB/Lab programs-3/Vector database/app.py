import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# -----------------------------
# Load Embedding Model
# -----------------------------
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Read Documents
# -----------------------------
with open("sample_sentences.txt", "r", encoding="utf-8") as file:
    documents = [line.strip() for line in file if line.strip()]

print(f"\nLoaded {len(documents)} documents.")

# -----------------------------
# Generate Embeddings
# -----------------------------
print("Generating embeddings...")
embeddings = model.encode(documents)

embeddings = np.array(embeddings).astype("float32")

# -----------------------------
# Create FAISS Index
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("\nFAISS index created successfully.")
print("Total vectors in index:", index.ntotal)

# -----------------------------
# Save Index
# -----------------------------
os.makedirs("faiss_index", exist_ok=True)

faiss.write_index(index, "faiss_index/index.faiss")

with open("faiss_index/documents.pkl", "wb") as f:
    pickle.dump(documents, f)

print("FAISS index saved successfully.")

# -----------------------------
# Load Index Again
# -----------------------------
index = faiss.read_index("faiss_index/index.faiss")

with open("faiss_index/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# -----------------------------
# Semantic Search
# -----------------------------
while True:

    query = input("\nEnter your query (or type exit): ")

    if query.lower() == "exit":
        print("Program terminated.")
        break

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, k=3)

    print("\nTop 3 Similar Results:\n")

    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {documents[idx]}")
        print(f"Distance: {distances[0][i]:.4f}\n")
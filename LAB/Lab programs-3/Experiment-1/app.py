from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pretrained embedding model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Read sentences from file
with open("sample_sentences.txt", "r", encoding="utf-8") as file:
    documents = [line.strip() for line in file if line.strip()]

# Generate embeddings
print("Generating embeddings...")
doc_embeddings = model.encode(documents)

print("\nEmbeddings generated successfully!")
print(f"Total Documents: {len(documents)}")

while True:
    query = input("\nEnter your query (or type 'exit'): ")

    if query.lower() == "exit":
        print("Program terminated.")
        break

    # Generate query embedding
    query_embedding = model.encode([query])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, doc_embeddings)[0]

    # Get best match
    best_index = np.argmax(similarities)

    print("\nMost Similar Sentence:")
    print(documents[best_index])

    print(f"\nSimilarity Score: {similarities[best_index]:.4f}")
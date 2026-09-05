"""
=========================================================
FAISS Indexing and Semantic Search
Model: all-MiniLM-L6-v2
Author: Your Name
=========================================================
"""

import faiss
import numpy as np
import pandas as pd
import time
import os
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Load Metadata and Embeddings
# ---------------------------------------------------------

metadata = pd.read_csv("proposal_metadata.csv")
embeddings = np.load("proposal_embeddings.npy").astype("float32")

print("=" * 60)
print("Metadata Loaded :", len(metadata))
print("Embedding Shape :", embeddings.shape)
print("=" * 60)

# ---------------------------------------------------------
# Build FAISS Index
# ---------------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

print("\nBuilding FAISS Index...")

start = time.time()

index.add(embeddings)

index_time = time.time() - start

print(f"Index Built Successfully")
print(f"Indexing Time : {index_time:.4f} seconds")

# ---------------------------------------------------------
# Save Index
# ---------------------------------------------------------

faiss.write_index(index, "grant_faiss.index")

size_mb = os.path.getsize("grant_faiss.index") / (1024 * 1024)

print(f"Index Size : {size_mb:.2f} MB")

# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------

model = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------
# Semantic Search Queries
# ---------------------------------------------------------

queries = [

    "Deep learning for medical diagnosis",

    "Climate prediction using machine learning",

    "Blockchain based supply chain",

    "Robotics for industrial automation",

    "IoT smart irrigation",

    "Cyber attack detection using AI",

    "Transformer models for education analytics",

    "Renewable energy forecasting",

    "AI fraud detection",

    "Projects using Graph Neural Networks in different application domains"

]

# ---------------------------------------------------------
# Perform Search
# ---------------------------------------------------------

results = []

latencies = []

print("\nRunning Semantic Search...\n")

for query in queries:

    query_embedding = model.encode([query]).astype("float32")

    start = time.time()

    distances, indices = index.search(query_embedding, 5)

    latency = time.time() - start

    latencies.append(latency)

    print("=" * 70)
    print("QUERY :", query)
    print("Latency :", round(latency * 1000, 3), "ms")
    print("-" * 70)

    for rank, idx in enumerate(indices[0], start=1):

        proposal = metadata.iloc[idx]

        print(
            f"{rank}. "
            f"{proposal['Proposal_ID']} | "
            f"{proposal['Title']} | "
            f"{proposal['Domain']}"
        )

        results.append({

            "Query": query,

            "Rank": rank,

            "Proposal_ID": proposal["Proposal_ID"],

            "Title": proposal["Title"],

            "Domain": proposal["Domain"],

            "Methodology": proposal["Methodology"],

            "Distance": float(distances[0][rank - 1])

        })

# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv("faiss_results.csv", index=False)

print("\nResults saved to faiss_results.csv")

# ---------------------------------------------------------
# Performance Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)

print("FAISS PERFORMANCE")

print("=" * 60)

print(f"Dataset Size        : {len(metadata)}")

print(f"Embedding Dimension : {dimension}")

print(f"Indexing Time       : {index_time:.4f} sec")

print(f"Average Query Time  : {np.mean(latencies)*1000:.3f} ms")

print(f"Index Size          : {size_mb:.2f} MB")

print("=" * 60)
performance = {
    "indexing_time": index_time,
    "avg_query_time": np.mean(latencies) * 1000,
    "storage_size": size_mb
}

pd.DataFrame([performance]).to_csv("faiss_performance.csv", index=False)
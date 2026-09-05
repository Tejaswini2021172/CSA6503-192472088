"""
=========================================================
Grant Proposal Similarity Screening
ChromaDB Indexing and Semantic Search
Embedding Model : all-MiniLM-L6-v2
Author : Your Name
=========================================================
"""

import chromadb
import pandas as pd
import numpy as np
import time
import shutil
import os

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

metadata = pd.read_csv("proposal_metadata.csv")
embeddings = np.load("proposal_embeddings.npy")

print("=" * 60)
print("Metadata Loaded Successfully")
print("Total Records :", len(metadata))
print("=" * 60)

# ---------------------------------------------------------
# Create Persistent Database
# ---------------------------------------------------------

DB_PATH = "chroma_db"

if os.path.exists(DB_PATH):
    shutil.rmtree(DB_PATH)

client = chromadb.PersistentClient(path=DB_PATH)

collection = client.create_collection(
    name="grant_proposals"
)

# ---------------------------------------------------------
# Build Index
# ---------------------------------------------------------

print("\nBuilding ChromaDB Index...")

start = time.time()

for i in range(len(metadata)):

    collection.add(

        ids=[metadata.iloc[i]["Proposal_ID"]],

        documents=[metadata.iloc[i]["Summary"]],

        embeddings=[embeddings[i].tolist()],

        metadatas=[

            {

                "title": metadata.iloc[i]["Title"],

                "domain": metadata.iloc[i]["Domain"],

                "methodology": metadata.iloc[i]["Methodology"],

                "funding": metadata.iloc[i]["Funding_Agency"]

            }

        ]
    )

index_time = time.time() - start

print("Index Built Successfully")
print(f"Indexing Time : {index_time:.4f} seconds")

# ---------------------------------------------------------
# Storage Size
# ---------------------------------------------------------

def folder_size(folder):

    total = 0

    for path, dirs, files in os.walk(folder):

        for file in files:

            fp = os.path.join(path, file)

            total += os.path.getsize(fp)

    return total

size_mb = folder_size(DB_PATH) / (1024 * 1024)

print(f"Database Size : {size_mb:.2f} MB")

# ---------------------------------------------------------
# Semantic Queries
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
# Search
# ---------------------------------------------------------

results = []

latencies = []

print("\nRunning Semantic Search...\n")

for query in queries:

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")

    query_embedding = model.encode([query])[0].tolist()

    start = time.time()

    response = collection.query(

        query_embeddings=[query_embedding],

        n_results=5

    )

    latency = time.time() - start

    latencies.append(latency)

    print("=" * 70)
    print("QUERY :", query)
    print("Latency :", round(latency * 1000, 3), "ms")
    print("-" * 70)

    ids = response["ids"][0]
    docs = response["documents"][0]
    metas = response["metadatas"][0]
    distances = response["distances"][0]

    for rank in range(5):

        print(
            f"{rank+1}. "
            f"{ids[rank]} | "
            f"{metas[rank]['title']} | "
            f"{metas[rank]['domain']}"
        )

        results.append({

            "Query": query,

            "Rank": rank + 1,

            "Proposal_ID": ids[rank],

            "Title": metas[rank]["title"],

            "Domain": metas[rank]["domain"],

            "Methodology": metas[rank]["methodology"],

            "Distance": distances[rank]

        })

# ---------------------------------------------------------
# Save Results
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv("chromadb_results.csv", index=False)

print("\nResults saved as chromadb_results.csv")

# ---------------------------------------------------------
# Performance Summary
# ---------------------------------------------------------

print("\n" + "=" * 60)

print("CHROMADB PERFORMANCE")

print("=" * 60)

print(f"Dataset Size        : {len(metadata)}")

print(f"Embedding Dimension : {embeddings.shape[1]}")

print(f"Indexing Time       : {index_time:.4f} sec")

print(f"Average Query Time  : {np.mean(latencies)*1000:.3f} ms")

print(f"Database Size       : {size_mb:.2f} MB")

print("=" * 60)

performance = {
    "indexing_time": index_time,
    "avg_query_time": np.mean(latencies) * 1000,
    "storage_size": size_mb
}

pd.DataFrame([performance]).to_csv("chromadb_performance.csv", index=False)
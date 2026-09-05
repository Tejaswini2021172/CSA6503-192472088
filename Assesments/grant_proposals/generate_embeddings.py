"""
=========================================================
Generate Sentence Embeddings for Grant Proposals
Model: all-MiniLM-L6-v2
Author: Your Name
=========================================================
"""

import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

DATASET_PATH = "grant_proposals_dataset.csv"

df = pd.read_csv(DATASET_PATH)

print(f"Dataset Loaded Successfully")
print(f"Total Records : {len(df)}")

# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------

print("\nLoading Sentence Transformer Model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model Loaded Successfully")

# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------

texts = df["Summary"].tolist()

print("\nGenerating Embeddings...")

start_time = time.time()

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True
)

end_time = time.time()

print("\nEmbedding Generation Completed")

print(f"Embedding Shape : {embeddings.shape}")

print(f"Time Taken : {end_time-start_time:.2f} seconds")

# ---------------------------------------------------------
# Save Embeddings
# ---------------------------------------------------------

np.save("proposal_embeddings.npy", embeddings)

print("\nEmbeddings saved as proposal_embeddings.npy")

# ---------------------------------------------------------
# Save Metadata
# ---------------------------------------------------------

metadata = df[
    [
        "Proposal_ID",
        "Title",
        "Summary",
        "Domain",
        "Methodology",
        "Funding_Agency"
    ]
]

metadata.to_csv("proposal_metadata.csv", index=False)

print("Metadata saved as proposal_metadata.csv")

print("\nProcess Completed Successfully")
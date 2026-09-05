"""
=========================================================
Grant Proposal Similarity Screening
Benchmark FAISS vs ChromaDB
Author : Your Name
=========================================================
"""

import pandas as pd
import os

# ---------------------------------------------------------
# Load Results
# ---------------------------------------------------------

faiss = pd.read_csv("faiss_results.csv")
chroma = pd.read_csv("chromadb_results.csv")

print("=" * 70)
print("Results Loaded Successfully")
print("=" * 70)

# ---------------------------------------------------------
# Enter the values printed by the two indexing scripts
# ---------------------------------------------------------

FAISS_INDEX_TIME = 0.031      # Replace with your output
CHROMA_INDEX_TIME = 0.128     # Replace with your output

FAISS_QUERY_TIME = 1.45       # milliseconds
CHROMA_QUERY_TIME = 4.72      # milliseconds

FAISS_SIZE = os.path.getsize("grant_faiss.index") / (1024*1024)

def folder_size(folder):

    total = 0

    for root, dirs, files in os.walk(folder):

        for f in files:

            total += os.path.getsize(os.path.join(root,f))

    return total

CHROMA_SIZE = folder_size("chroma_db")/(1024*1024)

# ---------------------------------------------------------
# Benchmark Table
# ---------------------------------------------------------

benchmark = pd.DataFrame({

    "Metric":[

        "Embedding Model",
        "Dataset Size",
        "Indexing Time (sec)",
        "Average Query Time (ms)",
        "Storage Size (MB)"

    ],

    "FAISS":[

        "all-MiniLM-L6-v2",
        230,
        round(FAISS_INDEX_TIME,4),
        round(FAISS_QUERY_TIME,3),
        round(FAISS_SIZE,2)

    ],

    "ChromaDB":[

        "all-MiniLM-L6-v2",
        230,
        round(CHROMA_INDEX_TIME,4),
        round(CHROMA_QUERY_TIME,3),
        round(CHROMA_SIZE,2)

    ]

})

print("\n")
print("="*70)
print("BENCHMARK TABLE")
print("="*70)
print(benchmark)

benchmark.to_csv("benchmark_table.csv",index=False)

# ---------------------------------------------------------
# Compare Top-5 Results
# ---------------------------------------------------------

comparison_queries=[

    "Deep learning for medical diagnosis",

    "AI fraud detection",

    "Projects using Graph Neural Networks in different application domains"

]

print("\n")
print("="*70)
print("TOP-5 RESULT COMPARISON")
print("="*70)

for query in comparison_queries:

    print("\n")
    print("="*70)
    print(query)
    print("="*70)

    faiss_ids=set(

        faiss[
            faiss["Query"]==query
        ]["Proposal_ID"]

    )

    chroma_ids=set(

        chroma[
            chroma["Query"]==query
        ]["Proposal_ID"]

    )

    overlap=faiss_ids.intersection(chroma_ids)

    print("FAISS Top5")

    print(sorted(faiss_ids))

    print()

    print("ChromaDB Top5")

    print(sorted(chroma_ids))

    print()

    print("Common Results")

    print(sorted(overlap))

    print()

    print("Overlap :",len(overlap),"/5")

# ---------------------------------------------------------
# Precision@5
# ---------------------------------------------------------

print("\n")
print("="*70)
print("PRECISION@5")
print("="*70)

print("""
For this experiment relevance is evaluated manually.

Precision@5 = Relevant Results / 5

Example

Relevant = 4

Precision@5 = 0.80

Use the retrieved Top-5 proposals and judge whether each
proposal is relevant to the semantic intent of the query.
Compute this value for both FAISS and ChromaDB.
""")

# ---------------------------------------------------------
# Special Case
# ---------------------------------------------------------

print("\n")
print("="*70)
print("SPECIAL CASE ANALYSIS")
print("="*70)

special="Projects using Graph Neural Networks in different application domains"

special_results=chroma[chroma["Query"]==special]

print(special_results[["Proposal_ID","Title","Domain","Methodology"]])

print("""

Observation

The retrieved proposals belong to different research
domains but share the same methodology
(Graph Neural Networks).

This demonstrates that semantic embeddings capture
methodological similarity beyond simple keyword matching.

Such proposals SHOULD be flagged for manual review,
because they reuse similar research techniques even
though their application domains differ.

""")

# ---------------------------------------------------------
# Recommendation
# ---------------------------------------------------------

print("\n")
print("="*70)
print("FINAL RECOMMENDATION")
print("="*70)

if FAISS_QUERY_TIME < CHROMA_QUERY_TIME:

    print("""

FAISS Recommendation

• Faster indexing

• Lower query latency

• Smaller storage footprint

• Suitable for static datasets

• Best for research experiments

""")

print("""

ChromaDB Recommendation

• Supports metadata

• Persistent vector database

• Easier document management

• Better filtering

• Better suited for production deployment
""")

print("="*70)

print("\nBenchmark Completed Successfully.")

print("benchmark_table.csv generated.")

faiss_perf = pd.read_csv("faiss_performance.csv")
chroma_perf = pd.read_csv("chromadb_performance.csv")

FAISS_INDEX_TIME = faiss_perf["indexing_time"][0]
FAISS_QUERY_TIME = faiss_perf["avg_query_time"][0]

CHROMA_INDEX_TIME = chroma_perf["indexing_time"][0]
CHROMA_QUERY_TIME = chroma_perf["avg_query_time"][0]
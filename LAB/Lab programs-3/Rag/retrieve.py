import chromadb

from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./db")

collection = client.get_collection("curriculum")


def retrieve(question, k=3):

    embedding = model.encode(question).tolist()

    result = collection.query(

        query_embeddings=[embedding],

        n_results=k

    )

    return result["documents"][0], result["metadatas"][0]
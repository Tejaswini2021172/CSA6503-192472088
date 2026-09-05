import chromadb

from sentence_transformers import SentenceTransformer

from split_chunks import chunks


model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks).tolist()

client = chromadb.PersistentClient(path="./db")

collection = client.get_or_create_collection("curriculum")


if collection.count() == 0:

    collection.add(

        documents=chunks,

        embeddings=embeddings,

        metadatas=[
            {"chunk": i}
            for i in range(len(chunks))
        ],

        ids=[f"c{i}" for i in range(len(chunks))]

    )


print("Vector Database Ready")
print(collection.count())
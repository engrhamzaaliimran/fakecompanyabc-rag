from pathlib import Path

import chromadb

from langchain_ollama import OllamaEmbeddings


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# =========================================================
# 2. LOAD THE EMBEDDING MODEL
# =========================================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================================
# 3. OPEN THE EXISTING CHROMA DATABASE
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="fakecompanyabc_hr"
)


print(f"Stored chunks: {collection.count()}")


# =========================================================
# 4. USER QUESTION
# =========================================================

query = "How many annual leave days do employees in Germany receive?"


# =========================================================
# 5. CREATE QUERY EMBEDDING
# =========================================================

query_embedding = embedding_model.embed_query(
    f"search_query: {query}"
)


print(f"Query embedding dimension: {len(query_embedding)}")


# =========================================================
# 6. SEARCH CHROMA
# =========================================================

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=["documents", "metadatas", "distances"]
)


# =========================================================
# 7. DISPLAY RETRIEVED CHUNKS
# =========================================================

print()
print("QUESTION:")
print(query)

print()
print("TOP 3 RETRIEVED CHUNKS:")


for i in range(3):

    print()
    print("=" * 70)

    print(f"RANK {i + 1}")

    print("Distance:")
    print(results["distances"][0][i])

    print("Source:")
    print(results["metadatas"][0][i]["source"])

    print("Content:")
    print(results["documents"][0][i])
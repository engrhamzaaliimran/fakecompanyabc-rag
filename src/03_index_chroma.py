from pathlib import Path

import yaml
import chromadb

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# =========================================================
# 2. LOAD MARKDOWN FILE
# =========================================================

def load_markdown_file(file_path: Path) -> Document:

    raw_text = file_path.read_text(encoding="utf-8")

    metadata = {
        "source": file_path.name
    }

    page_content = raw_text

    if raw_text.startswith("---"):

        parts = raw_text.split("---", 2)

        if len(parts) == 3:

            yaml_text = parts[1]
            page_content = parts[2].strip()

            yaml_metadata = yaml.safe_load(yaml_text) or {}

            metadata.update(yaml_metadata)

    return Document(
        page_content=page_content,
        metadata=metadata
    )


# =========================================================
# 3. LOAD FILES
# =========================================================

documents = []

for file_path in sorted(DATASET_DIR.glob("*.md")):

    document = load_markdown_file(file_path)

    documents.append(document)


print(f"Loaded {len(documents)} documents.")


# =========================================================
# 4. CHUNK
# =========================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# =========================================================
# 5. EMBEDDING MODEL
# =========================================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================================
# 6. PREPARE TEXT ONLY FOR EMBEDDING
# =========================================================

embedding_texts = [
    f"search_document: {chunk.page_content}"
    for chunk in chunks
]


# =========================================================
# 7. GENERATE EMBEDDINGS
# =========================================================

vectors = embedding_model.embed_documents(
    embedding_texts
)

print(f"Generated {len(vectors)} embeddings.")
print(f"Embedding dimension: {len(vectors[0])}")


# =========================================================
# 8. CREATE LOCAL CHROMA DATABASE
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_or_create_collection(
    name="fakecompanyabc_hr"
)


# =========================================================
# 9. PREPARE CLEAN DATA FOR CHROMA
# =========================================================

clean_texts = [
    chunk.page_content
    for chunk in chunks
]

metadatas = [
    chunk.metadata
    for chunk in chunks
]

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]


# =========================================================
# 10. STORE IN CHROMA
# =========================================================

collection.add(
    ids=ids,
    documents=clean_texts,
    embeddings=vectors,
    metadatas=metadatas,
)


# =========================================================
# 11. VERIFY
# =========================================================

print()
print("Indexing complete.")
print(f"Stored chunks: {collection.count()}")
print(f"Chroma directory: {CHROMA_DIR}")
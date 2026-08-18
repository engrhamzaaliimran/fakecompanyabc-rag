from pathlib import Path
import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# 1. LOCATE THE DATASET DIRECTORY
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset"


# ---------------------------------------------------------
# 2. LOAD ONE MARKDOWN FILE
# ---------------------------------------------------------

def load_markdown_file(file_path: Path) -> Document:

    raw_text = file_path.read_text(encoding="utf-8")

    metadata = {
        "source": file_path.name
    }

    page_content = raw_text


    # files contain YAML front matter between --- markers.
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


# ---------------------------------------------------------
# 3. LOAD ALL MARKDOWN FILES
# ---------------------------------------------------------

documents = []

for file_path in sorted(DATASET_DIR.glob("*.md")):

    document = load_markdown_file(file_path)

    documents.append(document)


# ---------------------------------------------------------
# 4. INSPECT WHAT WE LOADED
# ---------------------------------------------------------

# print(f"Loaded {len(documents)} documents.\n")

# for document in documents:

#     print("=" * 70)

#     print("SOURCE:")
#     print(document.metadata["source"])

#     print("\nMETADATA:")
#     print(document.metadata)

#     print("\nFIRST 300 CHARACTERS OF CONTENT:")
#     print(document.page_content[:300])

#     print()


# ---------------------------------------------------------
# 5. CREATE THE TEXT SPLITTER
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=120,
)


# ---------------------------------------------------------
# 6. SPLIT DOCUMENTS INTO CHUNKS
# ---------------------------------------------------------

chunks = text_splitter.split_documents(documents)


# ---------------------------------------------------------
# 7. INSPECT CHUNKS
# ---------------------------------------------------------

print("\n")
print("#" * 70)
print(f"Created {len(chunks)} chunks from {len(documents)} documents.")
print("#" * 70)


for index, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 70)

    print(f"CHUNK {index}")

    print("\nSOURCE:")
    print(chunk.metadata["source"])

    print("\nLENGTH:")
    print(len(chunk.page_content))

    print("\nCONTENT:")
    print(chunk.page_content)

    print()
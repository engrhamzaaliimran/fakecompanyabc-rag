from pathlib import Path
import yaml
from langchain_core.documents import Document


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

print(f"Loaded {len(documents)} documents.\n")

for document in documents:

    print("=" * 70)

    print("SOURCE:")
    print(document.metadata["source"])

    print("\nMETADATA:")
    print(document.metadata)

    print("\nFIRST 300 CHARACTERS OF CONTENT:")
    print(document.page_content[:300])

    print()

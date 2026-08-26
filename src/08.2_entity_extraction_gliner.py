#!/usr/bin/env python3
"""
08.2_entity_extraction_gliner.py

Domain-specific entity extraction using GLiNER.

This script is the next stage after 08.1 entity-schema discovery.

Pipeline
--------
selected_entity_schema.csv
        -> read reviewed entity TYPES
        -> convert labels to GLiNER-friendly natural language
                                  |
HR Markdown documents             |
        -> remove YAML frontmatter|
        -> light Markdown cleanup |
        -> RecursiveCharacterTextSplitter (700 / 120)
                                  |
                                  v
                               GLiNER
                                  |
                                  v
                    domain-specific entity mentions
                                  |
                                  v
       graphResults/entity_extraction/gliner_entities.json

Important
---------
- 08.1 discovered/reviewed the entity TYPES.
- 08.2 goes back to the actual document chunks and extracts the
  exact text spans that belong to those types.
- selected_values from selected_entity_schema.csv are NOT used as
  a lookup dictionary. Only the reviewed entity_type column is used
  as the schema given to GLiNER.
- Relationships are NOT extracted here.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


ALLOWED_EXTENSIONS = {".md", ".txt"}


# ---------------------------------------------------------------------
# Text cleaning
# Keep this aligned with 08.1 so schema discovery and GLiNER see the
# same document representation.
# ---------------------------------------------------------------------

def remove_frontmatter(text: str) -> str:
    """
    Remove YAML frontmatter only when the first non-empty line is ---.
    """
    lines = text.splitlines()

    first_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            first_idx = i
            break

    if first_idx is None:
        return text

    if lines[first_idx].strip() != "---":
        return text

    for j in range(first_idx + 1, len(lines)):
        if lines[j].strip() == "---":
            return "\n".join(lines[j + 1:])

    # Malformed frontmatter: keep document unchanged.
    return text


def clean_markdown(text: str) -> str:
    """
    Light Markdown cleanup only.

    Removes presentation syntax while preserving semantic text such as:
    - full-time
    - cross-border
    - company-paid
    - pro-rata
    """
    text = remove_frontmatter(text)

    cleaned_lines = []
    in_code_fence = False

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line.startswith("```"):
            in_code_fence = not in_code_fence
            continue

        if in_code_fence:
            continue

        if not line:
            cleaned_lines.append("")
            continue

        # Markdown horizontal rules.
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Markdown table separator rows.
        if re.fullmatch(
            r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?",
            line,
        ):
            continue

        # Heading markers.
        line = re.sub(r"^\s*#{1,6}\s*", "", line)

        # List markers.
        line = re.sub(r"^\s*[-+*]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)

        # Block quote marker.
        line = re.sub(r"^\s*>\s?", "", line)

        # Markdown links: [text](url) -> text
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)

        # Inline formatting.
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = line.replace("`", "")

        # Turn table cells into separate textual units.
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            line = ". ".join(cells)

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        # Prevent adjacent Markdown lines/headings from merging.
        if not re.search(r"[.!?;:]$", line):
            line += "."

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


# ---------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------

def canonical_to_gliner_label(entity_type: str) -> str:
    """
    Convert our canonical schema label into a natural-language GLiNER label.

    Example:
        EMPLOYEE_TYPE -> employee type
        MONEY_AMOUNT  -> money amount

    GLiNER generally works better with human-readable labels.
    """
    return str(entity_type).strip().lower().replace("_", " ")


def load_entity_schema(schema_path: Path):
    """
    Read the reviewed schema produced by 08.1.

    Only `entity_type` is used for GLiNER extraction.
    `selected_values` are retained as review evidence but are not used
    as a dictionary or whitelist.
    """
    df = pd.read_csv(schema_path)

    if "entity_type" not in df.columns:
        raise ValueError(
            f"{schema_path} must contain an 'entity_type' column."
        )

    canonical_types = []

    for value in df["entity_type"].dropna():
        label = str(value).strip()

        if label and label not in canonical_types:
            canonical_types.append(label)

    if not canonical_types:
        raise ValueError(
            f"No entity types found in {schema_path}."
        )

    canonical_to_gliner = {
        canonical: canonical_to_gliner_label(canonical)
        for canonical in canonical_types
    }

    gliner_to_canonical = {
        gliner_label: canonical
        for canonical, gliner_label in canonical_to_gliner.items()
    }

    return canonical_types, canonical_to_gliner, gliner_to_canonical


# ---------------------------------------------------------------------
# Documents / chunks
# ---------------------------------------------------------------------

def load_documents(input_path: Path):
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(
            path
            for path in input_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in ALLOWED_EXTENSIONS
        )

    if not paths:
        raise FileNotFoundError(
            f"No Markdown/text documents found under: {input_path}"
        )

    for path in paths:
        try:
            raw_text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw_text = path.read_text(
                encoding="utf-8",
                errors="ignore",
            )

        if raw_text.strip():
            yield path, raw_text


def build_chunks(
    input_path: Path,
    chunk_size: int,
    chunk_overlap: int,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for document_index, (path, raw_text) in enumerate(
        load_documents(input_path)
    ):
        cleaned_text = clean_markdown(raw_text)
        split_chunks = splitter.split_text(cleaned_text)

        for local_chunk_index, chunk_text in enumerate(split_chunks):
            chunk_id = (
                f"{path.stem}__chunk_{local_chunk_index:03d}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_index": document_index,
                    "chunk_index": local_chunk_index,
                    "source": str(path),
                    "text": chunk_text,
                }
            )

    return chunks


# ---------------------------------------------------------------------
# GLiNER extraction
# ---------------------------------------------------------------------

def load_gliner_model(model_name: str):
    """
    Import GLiNER lazily so --dry-run can still validate the pipeline
    before the package/model is installed.
    """
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            "GLiNER is not installed.\n"
            "Install it in your virtual environment, for example:\n\n"
            "    pip install gliner\n"
        ) from exc

    print(f"Loading GLiNER model: {model_name}")
    return GLiNER.from_pretrained(model_name)


def normalize_prediction(
    prediction: dict,
    gliner_to_canonical: dict[str, str],
):
    """
    Convert a GLiNER prediction to our stable project schema.
    """
    raw_label = str(prediction.get("label", "")).strip().lower()

    canonical_label = gliner_to_canonical.get(
        raw_label,
        raw_label.upper().replace(" ", "_"),
    )

    result = {
        "text": str(prediction.get("text", "")).strip(),
        "entity_type": canonical_label,
        "gliner_label": raw_label,
        "score": float(prediction.get("score", 0.0)),
    }

    # GLiNER versions commonly return character offsets.
    # Preserve them when available.
    if "start" in prediction:
        result["start"] = int(prediction["start"])

    if "end" in prediction:
        result["end"] = int(prediction["end"])

    return result


def extract_entities(
    model,
    chunks,
    gliner_labels,
    gliner_to_canonical,
    threshold: float,
):
    output_chunks = []
    total_entities = 0

    for i, chunk in enumerate(chunks, start=1):
        print(
            f"[{i:03d}/{len(chunks):03d}] "
            f"{chunk['chunk_id']}"
        )

        predictions = model.predict_entities(
            chunk["text"],
            gliner_labels,
            threshold=threshold,
        )

        entities = [
            normalize_prediction(
                pred,
                gliner_to_canonical,
            )
            for pred in predictions
        ]

        # Stable reading order where offsets are available.
        entities.sort(
            key=lambda item: (
                item.get("start", 10**12),
                -item["score"],
                item["text"].lower(),
            )
        )

        total_entities += len(entities)

        output_chunks.append(
            {
                **chunk,
                "entities": entities,
            }
        )

    return output_chunks, total_entities


# ---------------------------------------------------------------------
# Helpful corpus-wide summary
# ---------------------------------------------------------------------

def build_unique_entity_summary(chunk_results):
    """
    Aggregate repeated extracted spans across overlapping chunks.

    This does NOT remove the original per-chunk detections.
    It provides a convenient corpus-wide summary alongside them.
    """
    grouped = {}

    for chunk in chunk_results:
        for entity in chunk["entities"]:
            key = (
                entity["entity_type"],
                entity["text"].strip().casefold(),
            )

            if key not in grouped:
                grouped[key] = {
                    "text": entity["text"],
                    "entity_type": entity["entity_type"],
                    "max_score": entity["score"],
                    "occurrences": 0,
                    "sources": set(),
                    "chunk_ids": set(),
                }

            row = grouped[key]
            row["occurrences"] += 1
            row["max_score"] = max(
                row["max_score"],
                entity["score"],
            )
            row["sources"].add(chunk["source"])
            row["chunk_ids"].add(chunk["chunk_id"])

    rows = []

    for row in grouped.values():
        rows.append(
            {
                "text": row["text"],
                "entity_type": row["entity_type"],
                "max_score": round(
                    float(row["max_score"]),
                    6,
                ),
                "occurrences": int(row["occurrences"]),
                "sources": sorted(row["sources"]),
                "chunk_ids": sorted(row["chunk_ids"]),
            }
        )

    rows.sort(
        key=lambda row: (
            row["entity_type"],
            row["text"].casefold(),
        )
    )

    return rows


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="../data",
        help=(
            "Markdown/text document or directory. "
            "Default assumes execution from src/: ../data"
        ),
    )

    parser.add_argument(
        "--schema",
        default=(
            "../graphResults/entity_schema_discovery/"
            "selected_entity_schema.csv"
        ),
        help="Reviewed entity schema from 08.1",
    )

    parser.add_argument(
        "--output",
        default=(
            "../graphResults/entity_extraction/"
            "gliner_entities.json"
        ),
        help="JSON output path",
    )

    parser.add_argument(
        "--model",
        default="urchade/gliner_medium-v2.1",
        help="GLiNER model name or local model path",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Minimum GLiNER confidence score",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Load schema and chunks without loading GLiNER. "
            "Useful for checking paths and inputs."
        ),
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    schema_path = Path(args.schema)
    output_path = Path(args.output)

    print("\n=== 08.2 GLiNER ENTITY EXTRACTION ===\n")

    # --------------------------------------------------------------
    # Load reviewed 08.1 schema
    # --------------------------------------------------------------

    (
        canonical_types,
        canonical_to_gliner,
        gliner_to_canonical,
    ) = load_entity_schema(schema_path)

    gliner_labels = list(canonical_to_gliner.values())

    print(
        f"Loaded {len(canonical_types)} reviewed entity types "
        f"from:\n  {schema_path}\n"
    )

    for canonical in canonical_types:
        print(
            f"  {canonical:<25} -> "
            f"{canonical_to_gliner[canonical]}"
        )

    # --------------------------------------------------------------
    # Recreate original cleaned chunks
    # --------------------------------------------------------------

    chunks = build_chunks(
        input_path=input_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(
        f"\nCreated {len(chunks)} chunks "
        f"using {args.chunk_size}/{args.chunk_overlap}."
    )

    if args.dry_run:
        print("\nDry run complete. GLiNER was not loaded.")
        print("\nExample chunk:")
        print("-" * 70)
        print(chunks[0]["text"])
        print("-" * 70)
        return

    # --------------------------------------------------------------
    # GLiNER
    # --------------------------------------------------------------

    model = load_gliner_model(args.model)

    chunk_results, total_entities = extract_entities(
        model=model,
        chunks=chunks,
        gliner_labels=gliner_labels,
        gliner_to_canonical=gliner_to_canonical,
        threshold=args.threshold,
    )

    unique_entities = build_unique_entity_summary(
        chunk_results
    )

    # --------------------------------------------------------------
    # JSON output
    # --------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "metadata": {
            "stage": "08.2",
            "task": "GLiNER domain entity extraction",
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "input": str(input_path),
            "schema_file": str(schema_path),
            "model": args.model,
            "threshold": args.threshold,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "entity_type_count": len(canonical_types),
            "chunk_count": len(chunk_results),
            "entity_detection_count": total_entities,
            "unique_entity_count": len(unique_entities),
        },
        "entity_schema": [
            {
                "entity_type": canonical,
                "gliner_label": canonical_to_gliner[
                    canonical
                ],
            }
            for canonical in canonical_types
        ],
        "unique_entities": unique_entities,
        "chunks": chunk_results,
    }

    output_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== COMPLETE ===")
    print(
        f"Raw entity detections: {total_entities}"
    )
    print(
        f"Unique entity/type pairs: "
        f"{len(unique_entities)}"
    )
    print(f"Saved JSON:\n  {output_path}")

    print(
        "\nThe JSON keeps both:\n"
        "  1. corpus-wide unique entity summary\n"
        "  2. original chunk-level entities + context\n"
        "\nThe chunk-level structure is intentionally preserved "
        "for the later relationship-extraction stage."
    )


if __name__ == "__main__":
    main()

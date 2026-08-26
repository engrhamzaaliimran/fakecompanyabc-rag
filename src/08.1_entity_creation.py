#!/usr/bin/env python3
"""
08.1_entity_creation.py

Automatic entity-schema discovery — simplified cleaning revision.

Pipeline
--------
Documents
  -> remove YAML frontmatter only
  -> light Markdown cleanup
  -> RecursiveCharacterTextSplitter (700 / 120)
  -> spaCy NER + noun phrase candidates
  -> light normalization + deduplication
  -> Ollama nomic-embed-text embeddings
  -> PCA (default: 30 dimensions)
  -> HDBSCAN
  -> cluster summaries + 2-D PCA visualization

Current focus:
Get clean semantic candidate concepts without over-cleaning.
GLiNER, relationships and graph construction come later.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ollama
import pandas as pd
import spacy
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA


ALLOWED_EXTENSIONS = {".txt", ".md"}

LEADING_DETERMINERS = {
    "a",
    "an",
    "the",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "any",
    "another",
}


def remove_frontmatter(text: str) -> str:
    """
    Remove YAML frontmatter only when the document actually starts with ---.

    Example:
        ---
        company: FakeCompanyABC
        policy_id: HR-EXP-0308
        ...
        ---
        # Policy title

    Everything between the first two --- markers is dropped.
    """
    lines = text.splitlines()

    # Find first non-empty line.
    first_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            first_idx = i
            break

    if first_idx is None:
        return text

    if lines[first_idx].strip() != "---":
        return text

    # Find closing frontmatter marker.
    for j in range(first_idx + 1, len(lines)):
        if lines[j].strip() == "---":
            return "\n".join(lines[j + 1:])

    # If opening marker exists but no closing marker, leave text unchanged.
    return text


def clean_markdown(text: str) -> str:
    """
    Light Markdown cleanup.

    Only remove presentation syntax; keep semantic content intact.
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

        # Remove horizontal rules.
        if re.fullmatch(r"[-*_]{3,}", line):
            continue

        # Remove Markdown table separator rows.
        if re.fullmatch(
            r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?",
            line,
        ):
            continue

        # Strip heading markers but keep heading text.
        line = re.sub(r"^\s*#{1,6}\s*", "", line)

        # Strip list markers but keep list content.
        line = re.sub(r"^\s*[-+*]\s+", "", line)
        line = re.sub(r"^\s*\d+[.)]\s+", "", line)

        # Blockquote marker.
        line = re.sub(r"^\s*>\s?", "", line)

        # Markdown links: [text](url) -> text
        line = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", line)

        # Remove formatting markers only.
        line = line.replace("**", "")
        line = line.replace("__", "")
        line = line.replace("`", "")

        # Convert table cells into separate textual units.
        # Example:
        # | Germany | EUR 35 |
        # becomes:
        # Germany. EUR 35.
        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            cells = [cell for cell in cells if cell]
            line = ". ".join(cells)

        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        # Prevent adjacent Markdown lines/headings from being merged by spaCy.
        if not re.search(r"[.!?;:]$", line):
            line += "."

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def clean_surface(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?()[]{}<>\"'`*_#|")


def join_normalized_tokens(tokens) -> str:
    """
    Normalize while preserving hyphenated compounds.

    Examples:
        full-time employees       -> full-time employee
        pro-rata basis            -> pro-rata basis
        company-paid transition days -> company-paid transition day
        cross-border remote work  -> cross-border remote work
    """
    tokens = [
        t for t in tokens
        if not t.is_space
    ]

    # Remove leading determiners only.
    while tokens and (
        tokens[0].lower_ in LEADING_DETERMINERS
        or tokens[0].pos_ == "DET"
    ):
        tokens.pop(0)

    if not tokens:
        return ""

    parts = []

    for token in tokens:
        text = token.text

        # Preserve hyphens as connectors.
        if text in {"-", "–", "—"}:
            parts.append("-")
            continue

        # Skip possessive marker itself, but do NOT globally remove semantic words.
        if text in {"'s", "’s"}:
            continue

        # Singularize nouns and normalize verbs.
        if token.pos_ in {"NOUN", "PROPN", "VERB", "AUX"}:
            value = token.lemma_.lower().strip()
        else:
            value = token.lower_.strip()

        if not value or value == "-pron-":
            value = token.lower_

        parts.append(value)

    if not parts:
        return ""

    # Rebuild phrase while preserving hyphenated compounds.
    phrase = ""
    for part in parts:
        if part == "-":
            phrase = phrase.rstrip() + "-"
        else:
            if phrase and not phrase.endswith("-"):
                phrase += " "
            phrase += part

    phrase = re.sub(r"\s+", " ", phrase).strip()
    phrase = clean_surface(phrase)

    return phrase


def is_valid_candidate(phrase: str) -> bool:
    """
    Minimal filtering only:
    - must contain alphabetic content
    - must not be pure punctuation/number
    - must be a reasonable length
    """
    if not phrase:
        return False

    if len(phrase) < 2 or len(phrase) > 120:
        return False

    if not any(ch.isalpha() for ch in phrase):
        return False

    return True


def split_noun_chunk_around_entities(chunk):
    """
    Remove spaCy named-entity spans from noun chunks so a location/currency
    does not contaminate the surrounding domain concept.

    Hyphens are NOT treated as separators.

    Example:
        Germany-based full-time employee
        -> Germany handled by NER
        -> based full-time employee residual

    We then remove a leading 'based' residual in normalization.
    """
    segments = []
    current = []

    for token in chunk:
        # Named-entity token = boundary.
        if token.ent_iob_ != "O":
            if current:
                segments.append(current)
                current = []
            continue

        # Strong punctuation boundaries only.
        # Keep hyphens because they are semantic compound connectors.
        if token.is_punct and token.text not in {"-", "–", "—"}:
            if current:
                segments.append(current)
                current = []
            continue

        current.append(token)

    if current:
        segments.append(current)

    return segments


def normalize_segment(tokens) -> str:
    """
    Light noun-concept normalization after NER tokens are removed.
    """
    # Remove possessive owners, e.g. employee's manager -> manager.
    tokens = [t for t in tokens if t.dep_ != "poss"]

    # If a location was removed from "Germany-based employee",
    # the residual may start with "-based" or "based".
    while tokens and (
        tokens[0].text in {"-", "–", "—"}
        or tokens[0].lower_ == "based"
    ):
        tokens.pop(0)

    phrase = join_normalized_tokens(tokens)

    # One extra cleanup for leading "based ".
    phrase = re.sub(r"^based\s+", "", phrase)

    return phrase


def extract_candidates(doc):
    """
    Two candidate streams:

    1) spaCy generic NER
       Germany, Pakistan, Belgium, dates, organizations, etc.

    2) noun concepts with named-entity spans removed
       full-time employee, annual leave, manager approval, etc.
    """
    results = []

    # --- spaCy NER candidates ---
    for ent in doc.ents:
        phrase = join_normalized_tokens(list(ent))
        surface = clean_surface(ent.text)

        if not is_valid_candidate(phrase):
            continue

        results.append(
            {
                "surface": surface,
                "normalized": phrase,
                "kind": "spacy_ner",
                "spacy_label": ent.label_,
            }
        )

    # --- noun phrase candidates ---
    for chunk in doc.noun_chunks:
        for segment in split_noun_chunk_around_entities(chunk):
            phrase = normalize_segment(segment)

            if not is_valid_candidate(phrase):
                continue

            # Require at least one substantive noun/adjective/proper noun.
            meaningful = [
                t for t in segment
                if t.is_alpha
                and not t.is_stop
                and t.pos_ in {"NOUN", "PROPN", "ADJ"}
            ]
            if not meaningful:
                continue

            surface = clean_surface(" ".join(t.text for t in segment))

            results.append(
                {
                    "surface": surface,
                    "normalized": phrase,
                    "kind": "noun_chunk",
                    "spacy_label": "",
                }
            )

    return results


def load_documents(input_path: Path):
    if input_path.is_file():
        paths = [input_path]
    else:
        paths = sorted(
            p
            for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS
        )

    if not paths:
        raise FileNotFoundError(
            f"No .txt or .md files found under: {input_path}"
        )

    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")

        if text.strip():
            yield path, text


def aggregate_candidates(raw_candidates):
    """
    Deduplicate by normalized phrase.
    """
    grouped = {}

    for item in raw_candidates:
        key = item["normalized"]

        if key not in grouped:
            grouped[key] = {
                "phrase": key,
                "count": 0,
                "surface_forms": Counter(),
                "candidate_sources": set(),
                "spacy_labels": Counter(),
            }

        group = grouped[key]
        group["count"] += 1
        group["surface_forms"][item["surface"]] += 1
        group["candidate_sources"].add(item["kind"])

        if item["spacy_label"]:
            group["spacy_labels"][item["spacy_label"]] += 1

    rows = []

    for group in grouped.values():
        rows.append(
            {
                "phrase": group["phrase"],
                "count": group["count"],
                "surface_examples": " | ".join(
                    text for text, _ in group["surface_forms"].most_common(5)
                ),
                "candidate_sources": "|".join(
                    sorted(group["candidate_sources"])
                ),
                "spacy_labels": "|".join(
                    label
                    for label, _ in group["spacy_labels"].most_common()
                ),
            }
        )

    return rows


def embed_with_nomic(phrases, batch_size=64):
    all_embeddings = []

    for start in range(0, len(phrases), batch_size):
        batch = phrases[start:start + batch_size]

        response = ollama.embed(
            model="nomic-embed-text",
            input=batch,
        )

        all_embeddings.extend(response["embeddings"])

    embeddings = np.asarray(all_embeddings, dtype=np.float32)

    # Defensive normalization.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    return embeddings


def representative_terms(
    original_embeddings,
    phrases,
    labels,
    top_n=10,
):
    summaries = []

    for cluster_id in sorted(set(labels)):
        if cluster_id < 0:
            continue

        indices = np.where(labels == cluster_id)[0]
        cluster_embeddings = original_embeddings[indices]

        centroid = cluster_embeddings.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-12

        scores = cluster_embeddings @ centroid
        ranked = indices[np.argsort(scores)[::-1]]

        representatives = [
            phrases[i] for i in ranked[:top_n]
        ]

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(indices)),
                "representative_terms": " | ".join(representatives),
            }
        )

    return summaries


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="../data",
        help="Input .txt/.md file or directory (default: ../data)",
    )
    parser.add_argument(
        "--output",
        default="./entity_schema_output",
    )

    # Existing RAG baseline.
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--chunk-overlap", type=int, default=120)

    # Fixed lower-dimensional clustering baseline.
    parser.add_argument("--pca-components", type=int, default=30)

    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_candidates = []
    cleaned_documents = []
    chunk_count = 0

    print("Removing frontmatter and extracting candidates...")

    for path, raw_text in load_documents(input_path):
        clean_text = clean_markdown(raw_text)

        cleaned_documents.append(
            {
                "file": str(path),
                "cleaned_text": clean_text,
            }
        )

        for chunk_text in splitter.split_text(clean_text):
            doc = nlp(chunk_text)
            raw_candidates.extend(extract_candidates(doc))
            chunk_count += 1

    if not raw_candidates:
        raise RuntimeError("No candidates were extracted.")

    # Diagnostic outputs.
    pd.DataFrame(cleaned_documents).to_csv(
        output_dir / "cleaned_documents.csv",
        index=False,
    )

    pd.DataFrame(raw_candidates).to_csv(
        output_dir / "raw_normalized_candidates.csv",
        index=False,
    )

    candidate_df = pd.DataFrame(
        aggregate_candidates(raw_candidates)
    )

    # Keep:
    # - all NER candidates
    # - multi-word noun concepts
    # - repeated single-word noun concepts
    keep_mask = []

    for _, row in candidate_df.iterrows():
        is_ner = "spacy_ner" in row["candidate_sources"]
        is_multiword = len(str(row["phrase"]).split()) >= 2
        repeated = int(row["count"]) >= 2

        keep_mask.append(is_ner or is_multiword or repeated)

    candidate_df = candidate_df[
        np.asarray(keep_mask)
    ].reset_index(drop=True)

    if len(candidate_df) < args.min_cluster_size:
        raise RuntimeError(
            f"Only {len(candidate_df)} candidate concepts remain."
        )

    candidate_df.to_csv(
        output_dir / "clean_candidates.csv",
        index=False,
    )

    phrases = candidate_df["phrase"].tolist()

    print(
        f"Embedding {len(phrases)} candidate concepts "
        "with nomic-embed-text..."
    )

    embeddings = embed_with_nomic(
        phrases,
        batch_size=args.batch_size,
    )

    max_components = min(
        args.pca_components,
        embeddings.shape[0],
        embeddings.shape[1],
    )

    print(
        f"PCA: {embeddings.shape[1]}D -> "
        f"{max_components}D..."
    )

    pca = PCA(n_components=max_components)
    reduced = pca.fit_transform(embeddings)

    retained_variance = float(
        pca.explained_variance_ratio_.sum()
    )

    print(
        f"PCA retained variance: "
        f"{retained_variance * 100:.2f}%"
    )

    print("HDBSCAN...")

    clusterer = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(reduced)

    candidate_df["cluster_id"] = labels

    candidate_df.to_csv(
        output_dir / "clustered_candidates.csv",
        index=False,
    )

    summaries = representative_terms(
        embeddings,
        phrases,
        labels,
        top_n=10,
    )

    summary_df = pd.DataFrame(summaries)

    summary_df.to_csv(
        output_dir / "cluster_summary.csv",
        index=False,
    )

    # ---------------------------
    # Visualization outputs
    # ---------------------------
    #
    # Important:
    # HDBSCAN clusters in the PCA-reduced space above (default 30D).
    # The first two principal components are used only for visualization.

    if reduced.shape[1] >= 2:
        coords = reduced[:, :2]
    else:
        coords = PCA(n_components=2).fit_transform(embeddings)

    points_df = candidate_df.copy()
    points_df["pc1"] = coords[:, 0]
    points_df["pc2"] = coords[:, 1]

    points_df.to_csv(
        output_dir / "pca_points.csv",
        index=False,
    )

    # Representative term lookup for each cluster.
    if not summary_df.empty:
        summary_df["top_term"] = (
            summary_df["representative_terms"]
            .str.split(r"\s*\|\s*", regex=True)
            .str[0]
        )
    else:
        summary_df["top_term"] = []

    cluster_lookup = summary_df[
        ["cluster_id", "size", "top_term", "representative_terms"]
    ].copy()

    cluster_lookup.to_csv(
        output_dir / "cluster_lookup.csv",
        index=False,
    )

    # ---------------------------
    # Pretty PCA cluster plot
    # ---------------------------

    plt.figure(figsize=(14, 10))

    actual_clusters = sorted(set(labels) - {-1})

    # Use a larger qualitative colormap.
    # Colors repeat only when cluster count exceeds palette size.
    cmap = plt.get_cmap("tab20")
    cluster_colors = {
        cluster_id: cmap(i % 20)
        for i, cluster_id in enumerate(actual_clusters)
    }

    # Plot noise first in light gray.
    noise_mask = labels == -1
    if np.any(noise_mask):
        plt.scatter(
            coords[noise_mask, 0],
            coords[noise_mask, 1],
            s=24,
            color="lightgray",
            alpha=0.45,
            edgecolors="none",
            label="Noise",
        )

    # Plot each real cluster.
    for cluster_id in actual_clusters:
        mask = labels == cluster_id

        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=52,
            color=cluster_colors[cluster_id],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.4,
        )

    # Cluster centroids are much easier to read than many point labels.
    if actual_clusters:
        centroid_rows = []

        for cluster_id in actual_clusters:
            mask = labels == cluster_id

            centroid_rows.append(
                {
                    "cluster_id": cluster_id,
                    "pc1": float(coords[mask, 0].mean()),
                    "pc2": float(coords[mask, 1].mean()),
                }
            )

        centroid_df = pd.DataFrame(centroid_rows)

        lookup_map = {
            int(row["cluster_id"]): row["top_term"]
            for _, row in cluster_lookup.iterrows()
        }

        for _, row in centroid_df.iterrows():
            cluster_id = int(row["cluster_id"])
            x = row["pc1"]
            y = row["pc2"]

            top_term = str(
                lookup_map.get(
                    cluster_id,
                    f"Cluster {cluster_id}",
                )
            )

            if len(top_term) > 24:
                top_term = top_term[:21] + "..."

            label = f"C{cluster_id}: {top_term}"

            plt.scatter(
                [x],
                [y],
                s=180,
                color=cluster_colors[cluster_id],
                edgecolors="black",
                linewidths=1.0,
                zorder=5,
            )

            plt.annotate(
                label,
                (x, y),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=8,
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.75,
                ),
            )

    plt.title(
        "PCA cluster map: spaCy → nomic-embed-text → PCA → HDBSCAN"
    )
    plt.xlabel("Principal component 1")
    plt.ylabel("Principal component 2")
    plt.tight_layout()

    pretty_plot_path = output_dir / "clusters_pca_pretty.png"

    plt.savefig(
        pretty_plot_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close()

    # ---------------------------
    # Cluster-size summary plot
    # ---------------------------

    if not cluster_lookup.empty:
        size_df = cluster_lookup.sort_values(
            ["size", "cluster_id"],
            ascending=[False, True],
        ).copy()

        def build_cluster_label(row):
            top_term = str(row["top_term"])

            if len(top_term) > 36:
                top_term = top_term[:33] + "..."

            return f"C{int(row['cluster_id'])}: {top_term}"

        size_df["display_label"] = size_df.apply(
            build_cluster_label,
            axis=1,
        )

        plt.figure(
            figsize=(
                12,
                max(7, 0.32 * len(size_df) + 2),
            )
        )

        y_positions = np.arange(len(size_df))

        plt.barh(
            y_positions,
            size_df["size"],
        )

        plt.yticks(
            y_positions,
            size_df["display_label"],
            fontsize=8,
        )

        plt.gca().invert_yaxis()

        plt.xlabel("Cluster size")
        plt.ylabel("Cluster")
        plt.title(
            "HDBSCAN cluster sizes with representative concept"
        )
        plt.tight_layout()

        size_plot_path = output_dir / "cluster_size_summary.png"

        plt.savefig(
            size_plot_path,
            dpi=180,
            bbox_inches="tight",
        )
        plt.close()

    else:
        size_plot_path = None

    # ---------------------------
    # Final console summary
    # ---------------------------

    n_clusters = len(actual_clusters)
    n_noise = int(np.sum(labels == -1))

    print("\n=== RESULT ===")
    print(f"Chunks processed:      {chunk_count}")
    print(f"Raw candidates:        {len(raw_candidates)}")
    print(f"Unique candidates:     {len(candidate_df)}")
    print(f"PCA dimensions:        {reduced.shape[1]}")
    print(
        f"PCA variance retained: "
        f"{retained_variance * 100:.2f}%"
    )
    print(f"HDBSCAN clusters:      {n_clusters}")
    print(
        f"Noise concepts:        {n_noise} "
        f"({n_noise / len(labels):.1%})"
    )

    if not summary_df.empty:
        print("\n=== CLUSTER SUMMARY ===")
        print(
            summary_df[
                ["cluster_id", "size", "representative_terms"]
            ].to_string(index=False)
        )

    print("\nInspect cleaning first:")
    print(f"  {output_dir / 'cleaned_documents.csv'}")
    print(f"  {output_dir / 'clean_candidates.csv'}")

    print("\nCluster outputs:")
    print(f"  {output_dir / 'clustered_candidates.csv'}")
    print(f"  {output_dir / 'cluster_summary.csv'}")
    print(f"  {output_dir / 'cluster_lookup.csv'}")
    print(f"  {output_dir / 'pca_points.csv'}")

    print("\nVisualizations:")
    print(f"  {pretty_plot_path}")

    if size_plot_path is not None:
        print(f"  {size_plot_path}")


if __name__ == "__main__":
    main()

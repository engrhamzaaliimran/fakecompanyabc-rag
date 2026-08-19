import json
from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_ground_truth_simple.json"
)

CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# =========================================================
# 2. LOAD GROUND TRUTH
# =========================================================

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8"
) as file:

    ground_truth = json.load(file)


questions = ground_truth["questions"]

print(
    f"Loaded {len(questions)} evaluation questions."
)


# =========================================================
# 3. LOAD EMBEDDING MODEL
# =========================================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================================
# 4. OPEN EXISTING CHROMA DATABASE
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="fakecompanyabc_hr"
)

print(
    f"Chroma contains {collection.count()} chunks."
)


# =========================================================
# 5. K VALUES TO TEST
# =========================================================

K_VALUES = [1, 3, 5]

MAX_K = max(K_VALUES)


# =========================================================
# 6. METRIC STORAGE
# =========================================================

metrics = {}

for k in K_VALUES:

    metrics[k] = {
        "hits": [],
        "recalls": [],
        "precisions": []
    }


reciprocal_ranks = []


# =========================================================
# 7. RUN EVERY QUESTION
# =========================================================

for item in questions:

    question_id = item["id"]

    question = item["question"]

    evidence_items = [
        evidence["exact_text"]
        for evidence in item["evidence"]
    ]


    # -----------------------------------------------------
    # CREATE QUERY EMBEDDING
    # -----------------------------------------------------

    query_embedding = embedding_model.embed_query(
        f"search_query: {question}"
    )


    # -----------------------------------------------------
    # RETRIEVE TOP MAX_K CHUNKS
    # -----------------------------------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=MAX_K,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )


    retrieved_documents = results["documents"][0]


    # -----------------------------------------------------
    # FIND WHICH GOLD EVIDENCE ITEMS ARE PRESENT
    # -----------------------------------------------------

    relevant_flags = []

    evidence_found_per_chunk = []

    for document in retrieved_documents:

        found_evidence = []

        for evidence in evidence_items:

            if evidence in document:

                found_evidence.append(evidence)

        evidence_found_per_chunk.append(
            found_evidence
        )

        relevant_flags.append(
            len(found_evidence) > 0
        )


    # -----------------------------------------------------
    # CALCULATE METRICS FOR EACH K
    # -----------------------------------------------------

    for k in K_VALUES:

        top_k_flags = relevant_flags[:k]

        top_k_evidence = (
            evidence_found_per_chunk[:k]
        )


        # -------------------------
        # HIT@K
        # -------------------------

        hit = int(
            any(top_k_flags)
        )


        # -------------------------
        # PRECISION@K
        # -------------------------

        relevant_chunks = sum(
            top_k_flags
        )

        precision = (
            relevant_chunks / k
        )


        # -------------------------
        # RECALL@K
        # -------------------------

        evidence_found = set()

        for chunk_evidence in top_k_evidence:

            evidence_found.update(
                chunk_evidence
            )


        recall = (
            len(evidence_found)
            /
            len(evidence_items)
        )


        metrics[k]["hits"].append(hit)

        metrics[k]["precisions"].append(
            precision
        )

        metrics[k]["recalls"].append(
            recall
        )


    # -----------------------------------------------------
    # RECIPROCAL RANK
    # -----------------------------------------------------

    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(
        relevant_flags,
        start=1
    ):

        if is_relevant:

            reciprocal_rank = 1 / rank

            break


    reciprocal_ranks.append(
        reciprocal_rank
    )


    # -----------------------------------------------------
    # PRINT PER-QUESTION RESULT
    # -----------------------------------------------------

    print()
    print("=" * 70)

    print(
        f"{question_id}: {question}"
    )

    print(
        f"Expected: {item['expected_answer']}"
    )

    print(
        f"First relevant reciprocal rank: "
        f"{reciprocal_rank:.3f}"
    )


    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        relevant = relevant_flags[
            rank - 1
        ]

        marker = (
            "RELEVANT"
            if relevant
            else "NOT RELEVANT"
        )

        source = (
            results["metadatas"][0]
            [rank - 1]
            .get("source", "unknown")
        )

        distance = (
            results["distances"][0]
            [rank - 1]
        )

        print(
            f"  Rank {rank}: "
            f"{marker} | "
            f"{source} | "
            f"distance={distance:.4f}"
        )


# =========================================================
# 8. AGGREGATE RESULTS
# =========================================================

print()
print()
print("#" * 70)
print("RETRIEVAL BENCHMARK RESULTS")
print("#" * 70)


for k in K_VALUES:

    hit_at_k = (
        sum(metrics[k]["hits"])
        /
        len(questions)
    )

    recall_at_k = (
        sum(metrics[k]["recalls"])
        /
        len(questions)
    )

    precision_at_k = (
        sum(metrics[k]["precisions"])
        /
        len(questions)
    )


    print()

    print(f"Hit@{k}:       {hit_at_k:.3f}")

    print(
        f"Recall@{k}:    "
        f"{recall_at_k:.3f}"
    )

    print(
        f"Precision@{k}: "
        f"{precision_at_k:.3f}"
    )


mrr = (
    sum(reciprocal_ranks)
    /
    len(reciprocal_ranks)
)


print()
print(f"MRR:         {mrr:.3f}")
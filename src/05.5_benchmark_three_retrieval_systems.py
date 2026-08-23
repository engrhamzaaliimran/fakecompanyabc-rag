import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from sentence_transformers import CrossEncoder
except ImportError as exc:
    raise ImportError(
        "sentence-transformers is required for the reranker.\n"
        "Install it inside your existing venv with:\n"
        "pip install sentence-transformers"
    ) from exc


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_ground_truth_100_hard.json"
)

RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CSV_OUTPUT = RESULTS_DIR / "benchmark_100_three_systems.csv"
JSON_OUTPUT = RESULTS_DIR / "benchmark_100_three_systems.json"


# =========================================================
# 2. BENCHMARK SETTINGS
# =========================================================

# Final number of chunks evaluated / passed downstream.
FINAL_K = 3

# The reranker gets a slightly larger pool and then chooses Top-3.
RERANK_CANDIDATE_K = 5

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Keep the exact TF-IDF settings already used in the previous experiment.
MIN_LEXICAL_SCORE = 0.08
MAX_AMBIGUOUS_SCORE_SPREAD = 0.05
RELATIVE_SCORE_THRESHOLD = 0.55
MAX_CANDIDATE_FILES = 2


# =========================================================
# 3. LOAD THE 100-QUESTION GROUND TRUTH
# =========================================================

if not GROUND_TRUTH_FILE.exists():
    raise FileNotFoundError(
        f"Ground truth not found:\n{GROUND_TRUTH_FILE}\n\n"
        "Place retrieval_ground_truth_100_hard.json inside evaluation/."
    )

with GROUND_TRUTH_FILE.open("r", encoding="utf-8") as file:
    ground_truth = json.load(file)

questions = ground_truth["questions"]

print("=" * 100)
print("RUNNING 100-QUESTION RETRIEVAL BENCHMARK")
print("=" * 100)


# =========================================================
# 4. LOAD POLICY FILES FOR TF-IDF
# =========================================================

policy_documents = []
policy_sources = []

for file_path in sorted(DATASET_DIR.glob("*.md")):
    if file_path.name.lower() == "readme.md":
        continue

    policy_documents.append(
        file_path.read_text(encoding="utf-8")
    )
    policy_sources.append(file_path.name)

# Keep startup output minimal.


# =========================================================
# 5. BUILD TF-IDF FILE REPRESENTATION
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.8,
)

document_tfidf = vectorizer.fit_transform(
    policy_documents
)



# =========================================================
# 6. LOAD CHROMA + LOCAL MODELS
# =========================================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="fakecompanyabc_hr"
)

reranker = CrossEncoder(
    RERANKER_MODEL,
    device="cpu",
)


# =========================================================
# 7. TF-IDF CANDIDATE FILE SELECTION
# =========================================================

def get_candidate_sources(question):
    query_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        query_vector,
        document_tfidf
    )[0]

    ranked = sorted(
        zip(policy_sources, similarities),
        key=lambda item: item[1],
        reverse=True,
    )

    best_source, best_score = ranked[0]
    worst_source, worst_score = ranked[-1]

    # Weak lexical signal -> full dense retrieval.
    if best_score < MIN_LEXICAL_SCORE:
        return None, ranked, "weak_signal"

    # Nearly equal scores -> TF-IDF is not separating policies.
    score_spread = best_score - worst_score

    if score_spread <= MAX_AMBIGUOUS_SCORE_SPREAD:
        return None, ranked, "ambiguous_signal"

    candidates = []

    for source, score in ranked:
        if score <= 0:
            continue

        if score >= best_score * RELATIVE_SCORE_THRESHOLD:
            candidates.append(source)

        if len(candidates) >= MAX_CANDIDATE_FILES:
            break

    return candidates, ranked, "filtered"


# =========================================================
# 8. DENSE SEARCH
# =========================================================

def dense_search(query_embedding, n_results, candidate_sources=None):
    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": n_results,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if candidate_sources is not None:
        query_args["where"] = {
            "source": {
                "$in": candidate_sources
            }
        }

    results = collection.query(**query_args)

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


# =========================================================
# 9. CROSS-ENCODER RERANKING
# =========================================================

def rerank(question, retrieval_result):
    documents = retrieval_result["documents"]
    metadatas = retrieval_result["metadatas"]
    distances = retrieval_result["distances"]

    pairs = [
        (question, document)
        for document in documents
    ]

    # Raw CrossEncoder scores are used only for ordering.
    # Higher score = the reranker considers the pair more relevant.
    scores = reranker.predict(
        pairs,
        show_progress_bar=False,
    )

    order = sorted(
        range(len(scores)),
        key=lambda index: float(scores[index]),
        reverse=True,
    )

    return {
        "documents": [
            documents[index]
            for index in order
        ],
        "metadatas": [
            metadatas[index]
            for index in order
        ],
        "distances": [
            distances[index]
            for index in order
        ],
        "rerank_scores": [
            float(scores[index])
            for index in order
        ],
    }


# =========================================================
# 10. EVALUATION
# =========================================================

def evaluate_result(result, question_item, k):
    documents = result["documents"][:k]
    metadatas = result["metadatas"][:k]

    evidence_items = question_item["evidence"]

    # New ground truth contains relevant_sources.
    # Fall back to evidence sources if needed.
    relevant_sources = set(
        question_item.get(
            "relevant_sources",
            [
                evidence["source"]
                for evidence in evidence_items
            ],
        )
    )

    relevant_flags = []
    source_flags = []
    found_evidence = set()

    for document, metadata in zip(
        documents,
        metadatas
    ):
        source = metadata.get("source", "")

        source_flags.append(
            source in relevant_sources
        )

        chunk_relevant = False

        for evidence_index, evidence in enumerate(
            evidence_items
        ):
            correct_source = (
                source == evidence["source"]
            )

            exact_text_found = (
                evidence["exact_text"] in document
            )

            if correct_source and exact_text_found:
                chunk_relevant = True
                found_evidence.add(
                    evidence_index
                )

        relevant_flags.append(
            chunk_relevant
        )

    # If fewer than K chunks are returned, use the actual
    # number retrieved as the denominator.
    actual_k = max(1, len(documents))

    hit_at_1 = int(
        bool(relevant_flags)
        and relevant_flags[0]
    )

    hit_at_k = int(
        any(relevant_flags)
    )

    precision_at_k = (
        sum(relevant_flags)
        / actual_k
    )

    recall_at_k = (
        len(found_evidence)
        / len(evidence_items)
    )

    source_precision_at_k = (
        sum(source_flags)
        / actual_k
    )

    reciprocal_rank = 0.0
    first_relevant_rank = None

    for rank, relevant in enumerate(
        relevant_flags,
        start=1
    ):
        if relevant:
            reciprocal_rank = 1.0 / rank
            first_relevant_rank = rank
            break

    return {
        "hit1": hit_at_1,
        "hitk": hit_at_k,
        "precision": precision_at_k,
        "recall": recall_at_k,
        "source_precision": source_precision_at_k,
        "rr": reciprocal_rank,
        "first_rank": first_relevant_rank,
        "relevant_flags": relevant_flags,
    }


# =========================================================
# 11. METRIC HELPERS
# =========================================================

SYSTEM_NAMES = [
    "dense_only",
    "tfidf_dense",
    "tfidf_dense_reranker",
]

metrics = {
    name: defaultdict(list)
    for name in SYSTEM_NAMES
}

difficulty_metrics = {
    name: defaultdict(
        lambda: defaultdict(list)
    )
    for name in SYSTEM_NAMES
}

rows = []

fallback_counts = {
    "weak_signal": 0,
    "ambiguous_signal": 0,
    "filtered": 0,
}

embedding_times = []


def store_metrics(system_name, evaluation, difficulty):
    for key in [
        "hit1",
        "hitk",
        "precision",
        "recall",
        "source_precision",
        "rr",
    ]:
        metrics[system_name][key].append(
            evaluation[key]
        )

        difficulty_metrics[
            system_name
        ][difficulty][key].append(
            evaluation[key]
        )


def mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


# =========================================================
# 12. RUN THE 3-SYSTEM BENCHMARK
# =========================================================


benchmark_start = time.perf_counter()

for number, item in enumerate(
    questions,
    start=1
):
    question_id = item["id"]
    question = item["question"]
    difficulty = item.get(
        "difficulty",
        "unknown"
    )

    # -----------------------------------------------------
    # 12.1 Embed the query ONCE.
    #
    # The same query embedding is reused for all 3 systems,
    # so quality comparison is fair and the benchmark does
    # not waste CPU by embedding each question three times.
    # -----------------------------------------------------

    embedding_start = time.perf_counter()

    query_embedding = embedding_model.embed_query(
        f"search_query: {question}"
    )

    embedding_times.append(
        time.perf_counter()
        - embedding_start
    )

    # -----------------------------------------------------
    # SYSTEM A: BASELINE = DENSE ONLY
    # -----------------------------------------------------

    start = time.perf_counter()

    dense_result = dense_search(
        query_embedding=query_embedding,
        n_results=FINAL_K,
        candidate_sources=None,
    )

    dense_time = (
        time.perf_counter()
        - start
    )

    dense_eval = evaluate_result(
        dense_result,
        item,
        FINAL_K,
    )

    store_metrics(
        "dense_only",
        dense_eval,
        difficulty,
    )

    # -----------------------------------------------------
    # TF-IDF policy selection shared by systems B and C
    # -----------------------------------------------------

    tfidf_start = time.perf_counter()

    (
        candidate_sources,
        lexical_ranking,
        tfidf_mode,
    ) = get_candidate_sources(
        question
    )

    tfidf_time = (
        time.perf_counter()
        - tfidf_start
    )

    fallback_counts[tfidf_mode] += 1

    # -----------------------------------------------------
    # SYSTEM B: BASELINE + TF-IDF
    # -----------------------------------------------------

    start = time.perf_counter()

    tfidf_dense_result = dense_search(
        query_embedding=query_embedding,
        n_results=FINAL_K,
        candidate_sources=candidate_sources,
    )

    tfidf_dense_time = (
        time.perf_counter()
        - start
        + tfidf_time
    )

    tfidf_dense_eval = evaluate_result(
        tfidf_dense_result,
        item,
        FINAL_K,
    )

    store_metrics(
        "tfidf_dense",
        tfidf_dense_eval,
        difficulty,
    )

    # -----------------------------------------------------
    # SYSTEM C: BASELINE + TF-IDF + RERANKER
    #
    # Dense first retrieves Top-5.
    # CrossEncoder reorders them.
    # Evaluation is still on final Top-3.
    # -----------------------------------------------------

    start = time.perf_counter()

    reranker_candidates = dense_search(
        query_embedding=query_embedding,
        n_results=RERANK_CANDIDATE_K,
        candidate_sources=candidate_sources,
    )

    reranked_result = rerank(
        question,
        reranker_candidates,
    )

    reranker_time = (
        time.perf_counter()
        - start
        + tfidf_time
    )

    reranker_eval = evaluate_result(
        reranked_result,
        item,
        FINAL_K,
    )

    store_metrics(
        "tfidf_dense_reranker",
        reranker_eval,
        difficulty,
    )

    # -----------------------------------------------------
    # Save per-question comparison
    # -----------------------------------------------------

    rows.append({
        "id": question_id,
        "difficulty": difficulty,
        "question": question,

        "tfidf_mode": tfidf_mode,
        "tfidf_candidates": (
            "FULL_DENSE"
            if candidate_sources is None
            else ",".join(candidate_sources)
        ),

        "dense_first_rank":
            dense_eval["first_rank"],

        "tfidf_dense_first_rank":
            tfidf_dense_eval["first_rank"],

        "reranker_first_rank":
            reranker_eval["first_rank"],

        "dense_hit1":
            dense_eval["hit1"],

        "tfidf_dense_hit1":
            tfidf_dense_eval["hit1"],

        "reranker_hit1":
            reranker_eval["hit1"],

        "dense_recall3":
            dense_eval["recall"],

        "tfidf_dense_recall3":
            tfidf_dense_eval["recall"],

        "reranker_recall3":
            reranker_eval["recall"],

        "dense_ms_post_embedding":
            round(dense_time * 1000, 3),

        "tfidf_dense_ms_post_embedding":
            round(tfidf_dense_time * 1000, 3),

        "reranker_ms_post_embedding":
            round(reranker_time * 1000, 3),
    })

    # No per-question console output.
    # Detailed per-question results are still saved to CSV.


benchmark_total_time = (
    time.perf_counter()
    - benchmark_start
)


# =========================================================
# 13. SUMMARY
# =========================================================

def system_summary(system_name):
    return {
        "Hit@1": mean(
            metrics[system_name]["hit1"]
        ),
        "Hit@3": mean(
            metrics[system_name]["hitk"]
        ),
        "Recall@3": mean(
            metrics[system_name]["recall"]
        ),
        "Evidence Precision@3": mean(
            metrics[system_name]["precision"]
        ),
        "Source Precision@3": mean(
            metrics[system_name]["source_precision"]
        ),
        "MRR@3": mean(
            metrics[system_name]["rr"]
        ),
    }


summaries = {
    system_name: system_summary(
        system_name
    )
    for system_name in SYSTEM_NAMES
}


print()
print()
print("#" * 100)
print("FINAL 100-QUESTION BENCHMARK")
print("#" * 100)

header = (
    f"{'Metric':<24}"
    f"{'Dense Only':>16}"
    f"{'Dense + TF-IDF':>18}"
    f"{'+ Reranker':>16}"
)

print()
print(header)
print("-" * len(header))

metric_order = [
    "Hit@1",
    "Hit@3",
    "Recall@3",
    "Evidence Precision@3",
    "Source Precision@3",
    "MRR@3",
]

for metric_name in metric_order:
    print(
        f"{metric_name:<24}"
        f"{summaries['dense_only'][metric_name]:>16.3f}"
        f"{summaries['tfidf_dense'][metric_name]:>18.3f}"
        f"{summaries['tfidf_dense_reranker'][metric_name]:>16.3f}"
    )


# =========================================================
# 14. DIFFICULTY BREAKDOWN
# =========================================================

print()
print("#" * 100)
print("BREAKDOWN BY DIFFICULTY")
print("#" * 100)

difficulties = sorted(
    {
        item.get("difficulty", "unknown")
        for item in questions
    }
)

for difficulty in difficulties:
    count = sum(
        1
        for item in questions
        if item.get(
            "difficulty",
            "unknown"
        ) == difficulty
    )

    print()
    print(
        f"{difficulty.upper()} "
        f"({count} questions)"
    )

    print(
        f"{'System':<28}"
        f"{'Hit@1':>10}"
        f"{'Hit@3':>10}"
        f"{'Recall@3':>12}"
        f"{'MRR@3':>10}"
    )

    for system_name in SYSTEM_NAMES:
        group = (
            difficulty_metrics[
                system_name
            ][difficulty]
        )

        print(
            f"{system_name:<28}"
            f"{mean(group['hit1']):>10.3f}"
            f"{mean(group['hitk']):>10.3f}"
            f"{mean(group['recall']):>12.3f}"
            f"{mean(group['rr']):>10.3f}"
        )


# =========================================================
# 15. TF-IDF FALLBACK BEHAVIOUR
# =========================================================

print()
print("#" * 100)
print("TF-IDF PREFILTER BEHAVIOUR")
print("#" * 100)

print(
    f"Filtered normally:  "
    f"{fallback_counts['filtered']}"
)

print(
    f"Weak-signal fallback: "
    f"{fallback_counts['weak_signal']}"
)

print(
    f"Ambiguous fallback:   "
    f"{fallback_counts['ambiguous_signal']}"
)


# =========================================================
# 16. SPEED INFORMATION
# =========================================================

print()
print("#" * 100)
print("CPU-FRIENDLY SPEED INFORMATION")
print("#" * 100)

print(
    "Query embedding is performed once per question and "
    "shared across all three systems."
)

print(
    f"Average Nomic query embedding: "
    f"{mean(embedding_times) * 1000:.2f} ms"
)

for key, label in [
    (
        "dense_ms_post_embedding",
        "Dense only post-embedding"
    ),
    (
        "tfidf_dense_ms_post_embedding",
        "TF-IDF + Dense post-embedding"
    ),
    (
        "reranker_ms_post_embedding",
        "TF-IDF + Dense + Reranker post-embedding"
    ),
]:
    print(
        f"Average {label}: "
        f"{mean([row[key] for row in rows]):.2f} ms"
    )

print(
    f"Total benchmark wall time: "
    f"{benchmark_total_time:.2f} s"
)


# =========================================================
# 17. RANK-CHANGE ANALYSIS
# =========================================================

print()
print("#" * 100)
print("RERANKER FIRST-RELEVANT-RANK CHANGES")
print("#" * 100)

improved = 0
worsened = 0
same = 0

for row in rows:
    before = row["tfidf_dense_first_rank"]
    after = row["reranker_first_rank"]

    # Treat None as worse than any finite rank.
    before_value = (
        before
        if before is not None
        else 999
    )

    after_value = (
        after
        if after is not None
        else 999
    )

    if after_value < before_value:
        improved += 1
    elif after_value > before_value:
        worsened += 1
    else:
        same += 1

print(f"Improved questions: {improved}")
print(f"Worsened questions: {worsened}")
print(f"Unchanged questions: {same}")


# =========================================================
# 18. SAVE CSV
# =========================================================

with CSV_OUTPUT.open(
    "w",
    encoding="utf-8",
    newline=""
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=rows[0].keys(),
    )

    writer.writeheader()
    writer.writerows(rows)


# =========================================================
# 19. SAVE JSON SUMMARY
# =========================================================

json_result = {
    "benchmark": {
        "ground_truth": GROUND_TRUTH_FILE.name,
        "questions": len(questions),
        "final_k": FINAL_K,
        "rerank_candidate_k": RERANK_CANDIDATE_K,
        "reranker_model": RERANKER_MODEL,
    },
    "systems": summaries,
    "tfidf_fallback_counts": fallback_counts,
    "timing": {
        "average_query_embedding_ms":
            mean(embedding_times) * 1000,
        "average_dense_only_post_embedding_ms":
            mean([
                row["dense_ms_post_embedding"]
                for row in rows
            ]),
        "average_tfidf_dense_post_embedding_ms":
            mean([
                row["tfidf_dense_ms_post_embedding"]
                for row in rows
            ]),
        "average_reranker_post_embedding_ms":
            mean([
                row["reranker_ms_post_embedding"]
                for row in rows
            ]),
        "total_wall_seconds":
            benchmark_total_time,
    },
    "reranker_rank_changes": {
        "improved": improved,
        "worsened": worsened,
        "unchanged": same,
    },
}

with JSON_OUTPUT.open(
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        json_result,
        file,
        indent=2,
    )


print()
print("#" * 100)
print("FILES SAVED")
print("#" * 100)
print(CSV_OUTPUT)
print(JSON_OUTPUT)
import json
from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Use whichever ground-truth filename exists in your repo
GROUND_TRUTH_FILES = [
    PROJECT_ROOT / "evaluation" / "retrieval_ground_truth.json",
    PROJECT_ROOT / "evaluation" / "retrieval_ground_truth_simple.json",
]


# =========================================================
# 2. SETTINGS
# =========================================================

TOP_K = 3

# If TF-IDF cannot find a meaningful lexical match,
# use the full Chroma collection.
MIN_LEXICAL_SCORE = 0.08

# If all policy TF-IDF scores are too similar, the lexical
# stage is ambiguous and should not decide which files to keep.
# In that rare case, fall back to full semantic/dense retrieval.
MAX_AMBIGUOUS_SCORE_SPREAD = 0.05

# Keep another policy if its score is reasonably close
# to the best policy.
RELATIVE_SCORE_THRESHOLD = 0.55

# For now allow maximum two policy files.
# This is useful for cross-policy questions.
MAX_CANDIDATE_FILES = 2


# =========================================================
# 3. FIND GROUND TRUTH FILE
# =========================================================

ground_truth_file = None

for path in GROUND_TRUTH_FILES:
    if path.exists():
        ground_truth_file = path
        break

if ground_truth_file is None:
    raise FileNotFoundError(
        "Could not find retrieval ground truth JSON."
    )


# =========================================================
# 4. LOAD POLICY MARKDOWN FILES
# =========================================================

documents = []
sources = []

for file_path in sorted(DATASET_DIR.glob("*.md")):

    # README is not a real policy
    if file_path.name.lower() == "readme.md":
        continue

    text = file_path.read_text(
        encoding="utf-8"
    )

    documents.append(text)
    sources.append(file_path.name)


print(f"Loaded {len(documents)} policy files.")

for source in sources:
    print(f"  - {source}")


# =========================================================
# 5. BUILD TF-IDF REPRESENTATION
# =========================================================

vectorizer = TfidfVectorizer(

    lowercase=True,

    # remove common words:
    # the, is, of, and...
    stop_words="english",

    # single words + two-word phrases
    ngram_range=(1, 2),

    # ignore terms appearing in almost every policy
    max_df=0.8,
)

document_tfidf = vectorizer.fit_transform(
    documents
)

print()
print(
    f"TF-IDF vocabulary: "
    f"{len(vectorizer.get_feature_names_out())} terms"
)


# =========================================================
# 6. LOAD CHROMA + EMBEDDING MODEL
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

print(
    f"Chroma chunks: {collection.count()}"
)


# =========================================================
# 7. TF-IDF POLICY SELECTION
# =========================================================

def get_candidate_sources(question):

    # Convert user question to same TF-IDF space
    query_vector = vectorizer.transform(
        [question]
    )

    # Compare question with every policy file
    similarities = cosine_similarity(
        query_vector,
        document_tfidf
    )[0]

    ranked = sorted(
        zip(sources, similarities),
        key=lambda x: x[1],
        reverse=True
    )

    best_source, best_score = ranked[0]
    worst_source, worst_score = ranked[-1]

    # -----------------------------------------------------
    # Weak lexical result -> DO NOT FILTER
    # -----------------------------------------------------

    if best_score < MIN_LEXICAL_SCORE:

        return None, ranked

    # -----------------------------------------------------
    # Ambiguous lexical result -> DO NOT FILTER
    # -----------------------------------------------------
    #
    # Example:
    #   policy_a = 0.21
    #   policy_b = 0.20
    #   policy_c = 0.19
    #   policy_d = 0.18
    #   policy_e = 0.17
    #
    # Here TF-IDF is not clearly separating the policies.
    # Even though the best score is above MIN_LEXICAL_SCORE,
    # it is safer to let semantic dense retrieval search the
    # full collection.
    # -----------------------------------------------------

    score_spread = best_score - worst_score

    if score_spread <= MAX_AMBIGUOUS_SCORE_SPREAD:

        return None, ranked

    # -----------------------------------------------------
    # Strong lexical result -> select candidate files
    # -----------------------------------------------------

    candidates = []

    for source, score in ranked:

        if score <= 0:
            continue

        # Keep sources reasonably close to best result
        if score >= (
            best_score
            * RELATIVE_SCORE_THRESHOLD
        ):

            candidates.append(source)

        if len(candidates) >= MAX_CANDIDATE_FILES:
            break

    return candidates, ranked


# =========================================================
# 8. DENSE RETRIEVAL
# =========================================================

def retrieve(question, candidate_sources):

    query_embedding = (
        embedding_model.embed_query(
            f"search_query: {question}"
        )
    )

    # -----------------------------------------------------
    # No TF-IDF filter -> full dense retrieval
    # -----------------------------------------------------

    if candidate_sources is None:

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=TOP_K,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    # -----------------------------------------------------
    # TF-IDF candidates -> dense retrieval only there
    # -----------------------------------------------------

    else:

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=TOP_K,

            where={
                "source": {
                    "$in": candidate_sources
                }
            },

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

    return (
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )


# =========================================================
# 9. LOAD GROUND TRUTH
# =========================================================

with open(
    ground_truth_file,
    "r",
    encoding="utf-8"
) as file:

    ground_truth = json.load(file)


questions = ground_truth["questions"]

print()
print(
    f"Loaded {len(questions)} "
    f"ground-truth questions."
)


# =========================================================
# 10. METRIC STORAGE
# =========================================================

hit_scores = []
precision_scores = []
recall_scores = []
reciprocal_ranks = []


# =========================================================
# 11. RUN ALL QUESTIONS
# =========================================================

for item in questions:

    question_id = item["id"]
    question = item["question"]
    evidence_items = item["evidence"]


    # =====================================================
    # 11.1 TF-IDF PREFILTER
    # =====================================================

    candidates, lexical_ranking = (
        get_candidate_sources(question)
    )


    # =====================================================
    # 11.2 DENSE RETRIEVAL
    # =====================================================

    documents_found, metadatas, distances = (
        retrieve(
            question,
            candidates
        )
    )


    # =====================================================
    # 11.3 EVALUATE RETRIEVED CHUNKS
    # =====================================================

    relevant_flags = []

    found_evidence = set()


    for document, metadata in zip(
        documents_found,
        metadatas
    ):

        source = metadata.get(
            "source",
            ""
        )

        chunk_relevant = False


        for evidence_index, evidence in enumerate(
            evidence_items
        ):

            correct_source = (
                source
                == evidence["source"]
            )

            exact_text_found = (
                evidence["exact_text"]
                in document
            )

            if (
                correct_source
                and exact_text_found
            ):

                chunk_relevant = True

                found_evidence.add(
                    evidence_index
                )


        relevant_flags.append(
            chunk_relevant
        )


    # =====================================================
    # 11.4 HIT@3
    # =====================================================

    hit = int(
        any(relevant_flags)
    )

    hit_scores.append(hit)


    # =====================================================
    # 11.5 PRECISION@3
    # =====================================================

    precision = (
        sum(relevant_flags)
        / TOP_K
    )

    precision_scores.append(
        precision
    )


    # =====================================================
    # 11.6 RECALL@3
    # =====================================================

    recall = (
        len(found_evidence)
        / len(evidence_items)
    )

    recall_scores.append(
        recall
    )


    # =====================================================
    # 11.7 RECIPROCAL RANK
    # =====================================================

    reciprocal_rank = 0

    for rank, relevant in enumerate(
        relevant_flags,
        start=1
    ):

        if relevant:

            reciprocal_rank = (
                1 / rank
            )

            break

    reciprocal_ranks.append(
        reciprocal_rank
    )


    # =====================================================
    # 12. PRINT QUICK DIAGNOSTIC
    # =====================================================

    print()
    print("=" * 90)

    print(
        f"{question_id}: {question}"
    )

    print()

    print("TF-IDF ranking:")

    for source, score in lexical_ranking:

        print(
            f"  {source:<25} "
            f"{score:.4f}"
        )


    print()

    if candidates is None:

        best_score = lexical_ranking[0][1]
        worst_score = lexical_ranking[-1][1]
        score_spread = best_score - worst_score

        if best_score < MIN_LEXICAL_SCORE:
            reason = (
                f"weak lexical signal "
                f"(best={best_score:.4f} < {MIN_LEXICAL_SCORE:.4f})"
            )
        elif score_spread <= MAX_AMBIGUOUS_SCORE_SPREAD:
            reason = (
                f"ambiguous lexical signal "
                f"(spread={score_spread:.4f} <= "
                f"{MAX_AMBIGUOUS_SCORE_SPREAD:.4f})"
            )
        else:
            reason = "lexical fallback"

        print(
            "Prefilter: FALLBACK TO FULL DENSE RETRIEVAL "
            f"[{reason}]"
        )

    else:

        print(
            f"Prefilter candidates: "
            f"{candidates}"
        )


    print()

    print("Dense Top-3:")

    for rank, (
        metadata,
        distance,
        relevant
    ) in enumerate(
        zip(
            metadatas,
            distances,
            relevant_flags
        ),
        start=1
    ):

        print(
            f"  R{rank}: "
            f"{metadata.get('source')} | "
            f"distance={distance:.4f} | "
            f"relevant={relevant}"
        )


    print(
        f"Hit@3={hit} | "
        f"Recall@3={recall:.3f} | "
        f"Precision@3={precision:.3f}"
    )


# =========================================================
# 13. FINAL RESULTS
# =========================================================

number_questions = len(questions)

hit_at_3 = (
    sum(hit_scores)
    / number_questions
)

recall_at_3 = (
    sum(recall_scores)
    / number_questions
)

precision_at_3 = (
    sum(precision_scores)
    / number_questions
)

mrr_at_3 = (
    sum(reciprocal_ranks)
    / number_questions
)


print()
print()
print("#" * 90)

print("FINAL TF-IDF + DENSE RETRIEVAL RESULTS")

print("#" * 90)

print(
    f"Hit@3:       {hit_at_3:.3f}"
)

print(
    f"Recall@3:    {recall_at_3:.3f}"
)

print(
    f"Precision@3: {precision_at_3:.3f}"
)

print(
    f"MRR@3:       {mrr_at_3:.3f}"
)

print()
print("Previous dense-only baseline:")
print("Hit@3:       1.000")
print("Recall@3:    1.000")
print("Precision@3: 0.367")
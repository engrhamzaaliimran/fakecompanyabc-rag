import json
from pathlib import Path

import chromadb
import yaml

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"

GROUND_TRUTH_FILE = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_ground_truth_simple.json"
)


# =========================================================
# 2. BENCHMARK SETTINGS
# =========================================================

# Different chunking configurations we want to compare.
CHUNK_CONFIGURATIONS = [
    {
        "chunk_size": 400,
        "chunk_overlap": 80,
    },
    {
        "chunk_size": 700,
        "chunk_overlap": 120,
    },
    {
        "chunk_size": 1000,
        "chunk_overlap": 150,
    },
]


# We retrieve Top-5 once.
# From the same ranking we calculate metrics at 1, 3 and 5.
K_VALUES = [1, 3, 5]

MAX_K = max(K_VALUES)


# =========================================================
# 3. LOAD ONE MARKDOWN FILE
# =========================================================

def load_markdown_file(file_path: Path) -> Document:

    raw_text = file_path.read_text(
        encoding="utf-8"
    )

    metadata = {
        "source": file_path.name
    }

    page_content = raw_text


    # ---------------------------------------------
    # Separate YAML metadata from actual content
    # ---------------------------------------------

    if raw_text.startswith("---"):

        parts = raw_text.split(
            "---",
            2
        )

        if len(parts) == 3:

            yaml_text = parts[1]

            page_content = (
                parts[2].strip()
            )

            yaml_metadata = (
                yaml.safe_load(yaml_text)
                or {}
            )

            metadata.update(
                yaml_metadata
            )


    return Document(
        page_content=page_content,
        metadata=metadata
    )


# =========================================================
# 4. LOAD ALL SOURCE DOCUMENTS
# =========================================================

def load_documents():

    documents = []

    for file_path in sorted(
        DATASET_DIR.glob("*.md")
    ):

        document = load_markdown_file(
            file_path
        )

        documents.append(
            document
        )

    return documents


# =========================================================
# 5. MAKE METADATA SAFE FOR CHROMA
# =========================================================

def prepare_metadata(metadata):

    prepared = {}

    for key, value in metadata.items():

        # Chroma works cleanly with simple values.
        if isinstance(
            value,
            (str, int, float, bool)
        ) or value is None:

            prepared[key] = value

        else:

            # Lists/dicts etc. are converted to strings.
            prepared[key] = json.dumps(
                value
            )

    return prepared


# =========================================================
# 6. CHECK WHETHER A RETRIEVED CHUNK
#    CONTAINS GROUND-TRUTH EVIDENCE
# =========================================================

def find_evidence_in_chunk(
    document,
    metadata,
    evidence_items
):

    found_evidence = []


    for evidence in evidence_items:

        gold_text = evidence[
            "exact_text"
        ]

        gold_source = evidence[
            "source"
        ]


        retrieved_source = metadata.get(
            "source"
        )


        # We require:
        #
        # 1. correct source file
        # 2. exact gold evidence inside chunk
        #
        # This decides whether this retrieved
        # chunk is relevant for the question.

        correct_source = (
            retrieved_source
            == gold_source
        )

        contains_evidence = (
            gold_text in document
        )


        if (
            correct_source
            and contains_evidence
        ):

            found_evidence.append(
                gold_text
            )


    return found_evidence


# =========================================================
# 7. EVALUATE ONE CHUNKING CONFIGURATION
# =========================================================

def evaluate_configuration(
    documents,
    questions,
    embedding_model,
    chunk_size,
    chunk_overlap
):

    print()
    print("#" * 70)

    print(
        f"TESTING: "
        f"chunk_size={chunk_size}, "
        f"chunk_overlap={chunk_overlap}"
    )

    print("#" * 70)


    # =====================================================
    # 7.1 CHUNK SOURCE DOCUMENTS
    # =====================================================

    text_splitter = (
        RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    )


    chunks = (
        text_splitter
        .split_documents(documents)
    )


    print(
        f"Created {len(chunks)} chunks."
    )


    # =====================================================
    # 7.2 PREPARE DOCUMENT TEXT FOR EMBEDDING
    # =====================================================

    embedding_texts = [

        f"search_document: "
        f"{chunk.page_content}"

        for chunk in chunks
    ]


    # =====================================================
    # 7.3 CREATE DOCUMENT EMBEDDINGS
    # =====================================================

    vectors = (
        embedding_model
        .embed_documents(
            embedding_texts
        )
    )


    print(
        f"Generated {len(vectors)} embeddings."
    )


    # =====================================================
    # 7.4 CREATE TEMPORARY CHROMA DATABASE
    # =====================================================

    # This database exists only in memory.
    #
    # We do NOT use the persistent RAG database here
    # because each experiment has different chunks.

    client = chromadb.EphemeralClient()


    # Give each configuration its own collection name.
    #
    # Example:
    # benchmark_400_80
    # benchmark_700_120

    collection_name = (
        f"benchmark_"
        f"{chunk_size}_"
        f"{chunk_overlap}"
    )


    collection = (
        client.create_collection(
            name=collection_name
        )
    )


    # =====================================================
    # 7.5 PREPARE DATA FOR CHROMA
    # =====================================================

    clean_texts = [

        chunk.page_content

        for chunk in chunks
    ]


    metadatas = [

        prepare_metadata(
            chunk.metadata
        )

        for chunk in chunks
    ]


    ids = [

        f"chunk_{i}"

        for i in range(
            len(chunks)
        )
    ]


    # =====================================================
    # 7.6 STORE THIS CONFIGURATION
    # =====================================================

    collection.add(
        ids=ids,
        documents=clean_texts,
        embeddings=vectors,
        metadatas=metadatas
    )


    # =====================================================
    # 7.7 STORAGE FOR METRICS
    # =====================================================

    metrics = {}

    for k in K_VALUES:

        metrics[k] = {

            "hits": [],

            "recalls": [],

            "precisions": []
        }


    reciprocal_ranks = []


    # Keep per-question results so we can
    # inspect failures later.

    question_results = []


    # =====================================================
    # 7.8 RUN EVERY GROUND-TRUTH QUESTION
    # =====================================================

    for item in questions:

        question_id = item["id"]

        question = item["question"]

        evidence_items = item[
            "evidence"
        ]


        # -------------------------------------------------
        # A. CREATE QUERY EMBEDDING
        # -------------------------------------------------

        query_embedding = (
            embedding_model
            .embed_query(
                f"search_query: "
                f"{question}"
            )
        )


        # -------------------------------------------------
        # B. RETRIEVE TOP MAX_K
        # -------------------------------------------------

        results = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=MAX_K,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )


        retrieved_documents = (
            results["documents"][0]
        )

        retrieved_metadatas = (
            results["metadatas"][0]
        )

        retrieved_distances = (
            results["distances"][0]
        )


        # -------------------------------------------------
        # C. DETERMINE WHICH CHUNKS ARE RELEVANT
        # -------------------------------------------------

        relevant_flags = []

        evidence_found_per_chunk = []


        for (
            document,
            metadata
        ) in zip(
            retrieved_documents,
            retrieved_metadatas
        ):

            found_evidence = (
                find_evidence_in_chunk(
                    document=document,
                    metadata=metadata,
                    evidence_items=evidence_items
                )
            )


            evidence_found_per_chunk.append(
                found_evidence
            )


            relevant_flags.append(

                len(found_evidence) > 0

            )


        # =================================================
        # 7.9 CALCULATE HIT / PRECISION / RECALL
        # =================================================

        per_question_metrics = {}


        for k in K_VALUES:

            top_k_flags = (
                relevant_flags[:k]
            )

            top_k_evidence = (
                evidence_found_per_chunk[:k]
            )


            # ---------------------------------------------
            # HIT@K
            # ---------------------------------------------
            #
            # Did Top-K contain at least
            # one relevant chunk?

            hit = int(
                any(top_k_flags)
            )


            # ---------------------------------------------
            # PRECISION@K
            # ---------------------------------------------
            #
            # How many retrieved chunks
            # were actually relevant?

            relevant_chunks = sum(
                top_k_flags
            )


            precision = (
                relevant_chunks
                /
                k
            )


            # ---------------------------------------------
            # RECALL@K
            # ---------------------------------------------
            #
            # How much of the gold evidence
            # was found in Top-K?

            evidence_found = set()


            for chunk_evidence in (
                top_k_evidence
            ):

                evidence_found.update(
                    chunk_evidence
                )


            total_gold_evidence = (
                len(evidence_items)
            )


            recall = (

                len(evidence_found)
                /
                total_gold_evidence

            )


            # Store values for final average

            metrics[k]["hits"].append(
                hit
            )

            metrics[k][
                "precisions"
            ].append(
                precision
            )

            metrics[k][
                "recalls"
            ].append(
                recall
            )


            per_question_metrics[k] = {
                "hit": hit,
                "precision": precision,
                "recall": recall
            }


        # =================================================
        # 7.10 RECIPROCAL RANK
        # =================================================
        #
        # Find the FIRST relevant retrieved chunk.

        reciprocal_rank = 0.0


        for (
            rank,
            is_relevant
        ) in enumerate(
            relevant_flags,
            start=1
        ):

            if is_relevant:

                reciprocal_rank = (
                    1 / rank
                )

                break


        reciprocal_ranks.append(
            reciprocal_rank
        )


        # =================================================
        # 7.11 STORE PER-QUESTION INFORMATION
        # =================================================

        ranking = []


        for i in range(
            len(retrieved_documents)
        ):

            ranking.append({

                "rank": i + 1,

                "relevant":
                    relevant_flags[i],

                "source":
                    retrieved_metadatas[i]
                    .get(
                        "source",
                        "unknown"
                    ),

                "distance":
                    retrieved_distances[i]
            })


        question_results.append({

            "id": question_id,

            "question": question,

            "metrics":
                per_question_metrics,

            "reciprocal_rank":
                reciprocal_rank,

            "ranking":
                ranking
        })


    # =====================================================
    # 8. CALCULATE FINAL AVERAGES
    # =====================================================

    final_metrics = {}


    for k in K_VALUES:

        hit_at_k = (

            sum(
                metrics[k]["hits"]
            )
            /
            len(questions)

        )


        recall_at_k = (

            sum(
                metrics[k]["recalls"]
            )
            /
            len(questions)

        )


        precision_at_k = (

            sum(
                metrics[k][
                    "precisions"
                ]
            )
            /
            len(questions)

        )


        final_metrics[k] = {

            "hit": hit_at_k,

            "recall": recall_at_k,

            "precision":
                precision_at_k
        }


    # =====================================================
    # 9. MRR@MAX_K
    # =====================================================

    mrr = (

        sum(reciprocal_ranks)
        /
        len(reciprocal_ranks)

    )


    # =====================================================
    # 10. PRINT RESULTS FOR THIS CONFIGURATION
    # =====================================================

    print()

    print(
        "RETRIEVAL RESULTS"
    )

    print("-" * 70)


    for k in K_VALUES:

        print()

        print(
            f"Hit@{k}:        "
            f"{final_metrics[k]['hit']:.3f}"
        )

        print(
            f"Recall@{k}:     "
            f"{final_metrics[k]['recall']:.3f}"
        )

        print(
            f"Precision@{k}:  "
            f"{final_metrics[k]['precision']:.3f}"
        )


    print()

    print(
        f"MRR@{MAX_K}:        "
        f"{mrr:.3f}"
    )


    return {

        "chunk_size":
            chunk_size,

        "chunk_overlap":
            chunk_overlap,

        "number_of_chunks":
            len(chunks),

        "metrics":
            final_metrics,

        "mrr":
            mrr,

        "question_results":
            question_results
    }


# =========================================================
# 11. MAIN
# =========================================================

# Load fixed ground truth

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8"
) as file:

    ground_truth = json.load(
        file
    )


questions = ground_truth[
    "questions"
]


print(
    f"Loaded {len(questions)} "
    f"evaluation questions."
)


# Load source corpus once

documents = load_documents()


print(
    f"Loaded {len(documents)} "
    f"source documents."
)


# Load embedding model once

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================================
# 12. TEST EVERY CHUNKING CONFIGURATION
# =========================================================

all_results = []


for configuration in (
    CHUNK_CONFIGURATIONS
):

    result = evaluate_configuration(

        documents=documents,

        questions=questions,

        embedding_model=
            embedding_model,

        chunk_size=
            configuration[
                "chunk_size"
            ],

        chunk_overlap=
            configuration[
                "chunk_overlap"
            ]
    )


    all_results.append(
        result
    )


# =========================================================
# 13. FINAL COMPARISON
# =========================================================

print()
print()
print("#" * 90)

print(
    "CHUNKING CONFIGURATION COMPARISON"
)

print("#" * 90)


header = (

    f"{'Size':>6} "
    f"{'Overlap':>8} "
    f"{'Chunks':>7} "
    f"{'Hit@1':>8} "
    f"{'Recall@1':>10} "
    f"{'Hit@3':>8} "
    f"{'Recall@3':>10} "
    f"{'Prec@3':>8} "
    f"{'MRR@5':>8}"

)


print(header)

print("-" * len(header))


for result in all_results:

    print(

        f"{result['chunk_size']:>6} "

        f"{result['chunk_overlap']:>8} "

        f"{result['number_of_chunks']:>7} "

        f"{result['metrics'][1]['hit']:>8.3f} "

        f"{result['metrics'][1]['recall']:>10.3f} "

        f"{result['metrics'][3]['hit']:>8.3f} "

        f"{result['metrics'][3]['recall']:>10.3f} "

        f"{result['metrics'][3]['precision']:>8.3f} "

        f"{result['mrr']:>8.3f}"

    )


# =========================================================
# 14. SHOW HIT@1 FAILURES
# =========================================================

print()
print()
print("#" * 90)

# print(
#     "HIT@1 FAILURES BY CONFIGURATION"
# )

# print("#" * 90)


# for result in all_results:

#     print()

#     print(
#         f"chunk_size="
#         f"{result['chunk_size']}, "
#         f"overlap="
#         f"{result['chunk_overlap']}"
#     )


#     failures = [

#         question_result

#         for question_result
#         in result[
#             "question_results"
#         ]

#         if (
#             question_result[
#                 "metrics"
#             ][1]["hit"]
#             == 0
#         )
#     ]


#     if not failures:

#         print(
#             "  No Hit@1 failures."
#         )

#         continue


#     for failure in failures:

#         print()

#         print(
#             f"  {failure['id']}: "
#             f"{failure['question']}"
#         )


#         print(
#             "  Top result:"
#         )


#         top_result = (
#             failure["ranking"][0]
#         )


#         print(
#             f"    Source: "
#             f"{top_result['source']}"
#         )

#         print(
#             f"    Distance: "
#             f"{top_result['distance']:.4f}"
#         )

#         print(
#             f"    Relevant: "
#             f"{top_result['relevant']}"
#         )
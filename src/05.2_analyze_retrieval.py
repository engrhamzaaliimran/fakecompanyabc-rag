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
# 2. SETTINGS
# =========================================================

TOP_K = 3


# =========================================================
# 3. LOAD GROUND TRUTH
# =========================================================

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8"
) as file:

    ground_truth = json.load(file)


questions = ground_truth["questions"]


print(
    f"Loaded {len(questions)} "
    f"ground-truth questions."
)


# =========================================================
# 4. LOAD EMBEDDING MODEL
# =========================================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =========================================================
# 5. OPEN EXISTING CHROMA DATABASE
# =========================================================

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="fakecompanyabc_hr"
)


print(
    f"Chroma contains "
    f"{collection.count()} chunks."
)


# =========================================================
# 6. ANALYZE EVERY QUESTION
# =========================================================

for item in questions:

    question_id = item["id"]
    question = item["question"]

    expected_answer = item[
        "expected_answer"
    ]

    evidence_items = item[
        "evidence"
    ]


    # =====================================================
    # 6.1 QUERY EMBEDDING
    # =====================================================

    query_embedding = (
        embedding_model.embed_query(
            f"search_query: {question}"
        )
    )


    # =====================================================
    # 6.2 RETRIEVE TOP-3
    # =====================================================

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


    retrieved_documents = (
        results["documents"][0]
    )

    retrieved_metadatas = (
        results["metadatas"][0]
    )

    retrieved_distances = (
        results["distances"][0]
    )


    # =====================================================
    # 7. PRINT QUESTION
    # =====================================================

    print()
    print()
    print("#" * 90)

    print(
        f"{question_id}: {question}"
    )

    print("#" * 90)


    # =====================================================
    # 8. PRINT GROUND TRUTH
    # =====================================================

    print()
    print("GROUND TRUTH")
    print("-" * 90)

    print(
        f"Expected answer:"
    )

    print(
        expected_answer
    )


    print()

    print(
        "Required evidence:"
    )


    for evidence_number, evidence in enumerate(
        evidence_items,
        start=1
    ):

        print()

        print(
            f"Evidence {evidence_number}"
        )

        print(
            f"Source: "
            f"{evidence['source']}"
        )

        if "section" in evidence:

            print(
                f"Section: "
                f"{evidence['section']}"
            )

        print(
            "Exact text:"
        )

        print(
            evidence["exact_text"]
        )


    # =====================================================
    # 9. PRINT RETRIEVED TOP-3
    # =====================================================

    print()
    print()
    print("TOP-3 RETRIEVED CHUNKS")
    print("=" * 90)


    for rank in range(
        len(retrieved_documents)
    ):

        document = (
            retrieved_documents[rank]
        )

        metadata = (
            retrieved_metadatas[rank]
        )

        distance = (
            retrieved_distances[rank]
        )

        retrieved_source = (
            metadata.get(
                "source",
                "unknown"
            )
        )


        # =================================================
        # 9.1 CHECK SOURCE MATCH
        # =================================================

        gold_sources = {

            evidence["source"]

            for evidence
            in evidence_items

        }


        source_match = (
            retrieved_source
            in gold_sources
        )


        # =================================================
        # 9.2 CHECK EXACT GOLD EVIDENCE
        # =================================================

        matched_evidence = []


        for evidence in evidence_items:

            correct_source = (
                retrieved_source
                == evidence["source"]
            )

            exact_text_present = (

                evidence["exact_text"]
                in document

            )


            if (
                correct_source
                and exact_text_present
            ):

                matched_evidence.append(
                    evidence["exact_text"]
                )


        # =================================================
        # 9.3 CURRENT BENCHMARK RELEVANCE
        # =================================================

        relevant = (
            len(matched_evidence) > 0
        )


        # =================================================
        # 9.4 PRINT RETRIEVAL
        # =================================================

        print()
        print("-" * 90)

        print(
            f"RANK {rank + 1}"
        )

        print(
            f"Source: {retrieved_source}"
        )

        print(
            f"Distance: {distance:.4f}"
        )

        print(
            f"Source matches ground truth: "
            f"{source_match}"
        )

        print(
            f"Exact evidence found: "
            f"{len(matched_evidence) > 0}"
        )

        print(
            "Current evaluator judgment: "
            + (
                "RELEVANT"
                if relevant
                else "NOT RELEVANT"
            )
        )


        # =================================================
        # 9.5 EXPLAIN WHY
        # =================================================

        if relevant:

            print()

            print(
                "Matched gold evidence:"
            )

            for matched in (
                matched_evidence
            ):

                print(
                    f"- {matched}"
                )


        elif source_match:

            print()

            print(
                "NOTE: The retrieved chunk "
                "comes from a ground-truth "
                "source file, but does not "
                "contain the exact required "
                "gold evidence."
            )


        else:

            print()

            print(
                "NOTE: This chunk is from "
                "a different source than "
                "the ground-truth evidence."
            )


        # =================================================
        # 9.6 FULL RETRIEVED CHUNK
        # =================================================

        print()
        print(
            "Retrieved chunk:"
        )

        print()

        print(document)


    print()
    print("=" * 90)
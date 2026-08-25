#!/usr/bin/env python3

"""
07_evaluate_rag_tfidf_ragas.py

Robust generation evaluation for FakeCompanyABC TF-IDF-routed + dense RAG.

Generator:
    qwen3:0.6b

Judge:
    qwen2.5:1.5b-instruct

Retrieval pipeline:
    1. TF-IDF policy-file routing
    2. Nomic query embedding
    3. Chroma dense retrieval restricted to routed policy files
    4. Top-K chunks passed to the generator

Evaluation:
    1. RAGAS Faithfulness
    2. RAGAS Factual Correctness (F1)
    3. Nomic semantic similarity between generated and reference answer

Important robustness features:
    - Each RAGAS metric gets a completely fresh Ollama judge.
    - Each RAGAS metric gets its own evaluate() call.
    - Parser failures are retried once with another fresh judge.
    - Failure of one metric does NOT discard other metric scores.
    - Results are saved after EVERY question.
    - Failed metrics are stored as None with the error message.
    - Final summary reports valid-score coverage.
    - --resume continues an interrupted run.

Normal full run:
    python 07_evaluate_rag_tfidf_ragas.py

Smoke test:
    python 07_evaluate_rag_tfidf_ragas.py --limit 3

Resume interrupted run:
    python 07_evaluate_rag_tfidf_ragas.py --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
import traceback
import warnings

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

CHROMA_PATH = PROJECT_ROOT / "chroma_db"

DATASET_DIR = PROJECT_ROOT / "dataset"

COLLECTION_NAME = "fakecompanyabc_hr"

GROUND_TRUTH_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "retrieval_ground_truth_100_hard.json"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)


# =============================================================================
# MODELS
# =============================================================================

DEFAULT_GENERATOR_MODEL = "qwen3:0.6b"

DEFAULT_JUDGE_MODEL = "qwen2.5:1.5b-instruct"

EMBEDDING_MODEL = "nomic-embed-text"

TOP_K = 3


# =============================================================================
# TF-IDF POLICY ROUTING
# =============================================================================

# Same routing idea used by the final hybrid retriever:
# first identify the most likely policy file(s), then perform dense
# retrieval only inside those candidate files.

MIN_LEXICAL_SCORE = 0.08
MAX_AMBIGUOUS_SCORE_SPREAD = 0.05
RELATIVE_SCORE_THRESHOLD = 0.55
MAX_CANDIDATE_FILES = 2


# =============================================================================
# RAGAS / CPU SETTINGS
# =============================================================================

RAGAS_TIMEOUT_SECONDS = 600

# Keep RAGAS executor conservative on CPU.
RAGAS_MAX_WORKERS = 1

# RAGAS-level retry.
# We additionally do ONE manual retry only for parser failures.
RAGAS_MAX_RETRIES = 1

# Number of complete attempts for a metric when parsing fails.
PARSER_ATTEMPTS = 2


# =============================================================================
# REMOVE DEPRECATION WARNING NOISE
# =============================================================================

warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
)


# =============================================================================
# IMPORTS
# =============================================================================

try:
    import chromadb

except ImportError:

    print("ERROR: chromadb is not installed.")
    print("Run:")
    print("    pip install chromadb")

    sys.exit(1)


try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

except ImportError:

    print("ERROR: scikit-learn is not installed.")
    print("Run:")
    print("    pip install scikit-learn")

    sys.exit(1)


try:
    from langchain_ollama import (
        ChatOllama,
        OllamaEmbeddings,
    )

except ImportError:

    print("ERROR: langchain-ollama is not installed.")
    print("Run:")
    print("    pip install langchain-ollama")

    sys.exit(1)


try:

    from ragas import (
        EvaluationDataset,
        evaluate,
    )

    from ragas.llms import (
        LangchainLLMWrapper,
    )

    from ragas.metrics import (
        Faithfulness,
        FactualCorrectness,
    )

    from ragas.run_config import (
        RunConfig,
    )

except ImportError as error:

    print("ERROR importing RAGAS:")
    print(error)

    print()
    print("Try:")
    print("    pip install -U ragas")

    sys.exit(1)


# =============================================================================
# COMMAND-LINE ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Robust generation evaluation for "
            "FakeCompanyABC local RAG."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Evaluate only the first N ground-truth questions."
        ),
    )

    parser.add_argument(
        "--generator-model",
        type=str,
        default=DEFAULT_GENERATOR_MODEL,
        help=(
            "Ollama generator model. "
            f"Default: {DEFAULT_GENERATOR_MODEL}"
        ),
    )

    parser.add_argument(
        "--judge-model",
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help=(
            "Ollama RAGAS judge. "
            f"Default: {DEFAULT_JUDGE_MODEL}"
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
        help=(
            "Number of retrieved chunks. "
            f"Default: {TOP_K}"
        ),
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=RAGAS_TIMEOUT_SECONDS,
        help=(
            "RAGAS timeout per metric operation. "
            f"Default: {RAGAS_TIMEOUT_SECONDS}"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from existing results for this "
            "generator/judge/Top-K configuration."
        ),
    )

    return parser.parse_args()


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def separator(
    char: str = "=",
    width: int = 100,
):

    print(char * width)


def format_score(
    score: Optional[float],
) -> str:

    if score is None:
        return "None"

    return f"{score:.4f}"


def safe_float(
    value: Any,
) -> Optional[float]:

    if value is None:
        return None

    try:

        value = float(value)

        if math.isnan(value):
            return None

        return value

    except (
        TypeError,
        ValueError,
    ):

        return None


# =============================================================================
# RESULT FILE NAMES
# =============================================================================

def safe_model_name(
    model: str,
) -> str:

    return (
        model
        .replace(":", "_")
        .replace("/", "_")
    )


def get_result_paths(
    generator_model: str,
    judge_model: str,
    top_k: int,
):

    generator_safe = safe_model_name(
        generator_model
    )

    judge_safe = safe_model_name(
        judge_model
    )

    stem = (
        "generation_eval_tfidf_dense_"
        f"gen_{generator_safe}_"
        f"judge_{judge_safe}_"
        f"k{top_k}"
    )

    return {

        "csv":
            RESULTS_DIR
            / f"{stem}.csv",

        "json":
            RESULTS_DIR
            / f"{stem}.json",

        "jsonl":
            RESULTS_DIR
            / f"{stem}.jsonl",
    }


# =============================================================================
# GROUND TRUTH
# =============================================================================

def load_ground_truth() -> List[Dict[str, Any]]:

    if not GROUND_TRUTH_PATH.exists():

        raise FileNotFoundError(
            "Ground-truth file not found:\n"
            f"{GROUND_TRUTH_PATH}"
        )

    with GROUND_TRUTH_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    if isinstance(data, list):

        questions = data

    elif isinstance(data, dict):

        questions = None

        for key in [
            "questions",
            "data",
            "samples",
            "items",
            "ground_truth",
        ]:

            if (
                key in data
                and isinstance(
                    data[key],
                    list,
                )
            ):

                questions = data[key]
                break

        if questions is None:

            raise ValueError(
                "Could not find a question list "
                "inside ground-truth JSON."
            )

    else:

        raise ValueError(
            "Unsupported ground-truth JSON structure."
        )

    valid = []

    for item in questions:

        if not isinstance(
            item,
            dict,
        ):
            continue

        if "question" not in item:
            continue

        if "expected_answer" not in item:
            continue

        valid.append(
            item
        )

    if not valid:

        raise ValueError(
            "No valid question + expected_answer "
            "records were found."
        )

    return valid


# =============================================================================
# TF-IDF POLICY ROUTER
# =============================================================================

def create_tfidf_router():

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            "Dataset directory not found:\n"
            f"{DATASET_DIR}"
        )

    policy_documents = []
    policy_sources = []

    for file_path in sorted(
        DATASET_DIR.glob("*.md")
    ):

        # Do not route to the dataset README.
        if file_path.name.lower() == "readme.md":
            continue

        policy_documents.append(
            file_path.read_text(
                encoding="utf-8"
            )
        )

        policy_sources.append(
            file_path.name
        )

    if not policy_documents:

        raise RuntimeError(
            "No Markdown policy files were found in:\n"
            f"{DATASET_DIR}"
        )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_df=0.8,
    )

    document_tfidf = (
        vectorizer.fit_transform(
            policy_documents
        )
    )

    print(
        f"TF-IDF router loaded: "
        f"{len(policy_sources)} policy files."
    )

    return {
        "vectorizer":
            vectorizer,

        "document_tfidf":
            document_tfidf,

        "policy_sources":
            policy_sources,
    }


def get_candidate_sources(
    question: str,
    tfidf_router,
):

    vectorizer = (
        tfidf_router[
            "vectorizer"
        ]
    )

    document_tfidf = (
        tfidf_router[
            "document_tfidf"
        ]
    )

    policy_sources = (
        tfidf_router[
            "policy_sources"
        ]
    )

    query_vector = (
        vectorizer.transform(
            [question]
        )
    )

    similarities = (
        sklearn_cosine_similarity(
            query_vector,
            document_tfidf,
        )[0]
    )

    ranked = sorted(
        zip(
            policy_sources,
            similarities,
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    best_source, best_score = (
        ranked[0]
    )

    positive_scores = [
        score
        for _, score
        in ranked
        if score > 0
    ]

    # -----------------------------------------------------------------
    # Weak lexical evidence:
    # fall back to full-corpus dense retrieval.
    # -----------------------------------------------------------------

    if best_score < MIN_LEXICAL_SCORE:

        return (
            None,
            ranked,
            "weak_signal_full_corpus",
        )

    # -----------------------------------------------------------------
    # Ambiguous lexical evidence:
    # if the top policies are almost tied, do not over-filter.
    # Fall back to full-corpus dense retrieval.
    # -----------------------------------------------------------------

    if len(positive_scores) >= 2:

        second_best_score = (
            positive_scores[1]
        )

        score_spread = (
            best_score
            - second_best_score
        )

        if (
            score_spread
            <= MAX_AMBIGUOUS_SCORE_SPREAD
        ):

            return (
                None,
                ranked,
                "ambiguous_signal_full_corpus",
            )

    # -----------------------------------------------------------------
    # Strong signal:
    # keep up to MAX_CANDIDATE_FILES policies whose lexical score is
    # sufficiently close to the best policy.
    # -----------------------------------------------------------------

    candidates = []

    for source, score in ranked:

        if score <= 0:
            continue

        if (
            score
            >= best_score
            * RELATIVE_SCORE_THRESHOLD
        ):

            candidates.append(
                source
            )

        if (
            len(candidates)
            >= MAX_CANDIDATE_FILES
        ):

            break

    if not candidates:

        return (
            None,
            ranked,
            "no_candidate_full_corpus",
        )

    return (
        candidates,
        ranked,
        "tfidf_filtered",
    )


# =============================================================================
# RAG COMPONENTS
# =============================================================================

def create_rag_components(
    generator_model: str,
):

    print(
        "Loading Nomic embedding interface..."
    )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    print(
        f"Loading generator interface: "
        f"{generator_model}"
    )

    generator_kwargs = {

        "model":
            generator_model,

        "temperature":
            0,
    }

    # Qwen3 supports reasoning mode.
    # Disable it because the final RAG response is what we need.
    if (
        "qwen3" in
        generator_model.lower()
    ):

        generator_kwargs[
            "reasoning"
        ] = False

    generator = ChatOllama(
        **generator_kwargs
    )

    print(
        "Opening Chroma database..."
    )

    chroma_client = (
        chromadb.PersistentClient(
            path=str(
                CHROMA_PATH
            )
        )
    )

    collection = (
        chroma_client.get_collection(
            name=COLLECTION_NAME
        )
    )

    return (
        embeddings,
        generator,
        collection,
    )


# =============================================================================
# RETRIEVAL
# =============================================================================

def retrieve(
    question: str,
    embeddings,
    collection,
    tfidf_router,
    top_k: int,
) -> Dict[str, Any]:

    # -----------------------------------------------------------------
    # Stage 1: TF-IDF policy-file routing
    # -----------------------------------------------------------------

    (
        candidate_sources,
        lexical_ranking,
        tfidf_mode,
    ) = get_candidate_sources(
        question=question,
        tfidf_router=tfidf_router,
    )

    # -----------------------------------------------------------------
    # Stage 2: Nomic dense query embedding
    # -----------------------------------------------------------------

    query_text = (
        f"search_query: {question}"
    )

    query_embedding = (
        embeddings.embed_query(
            query_text
        )
    )

    query_args = {

        "query_embeddings": [
            query_embedding
        ],

        "n_results":
            top_k,

        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    # -----------------------------------------------------------------
    # Stage 3: dense retrieval only inside TF-IDF-selected files.
    #
    # If TF-IDF is weak/ambiguous, candidate_sources is None and the
    # query deliberately falls back to full-corpus dense retrieval.
    # -----------------------------------------------------------------

    if candidate_sources is not None:

        if len(candidate_sources) == 1:

            query_args["where"] = {
                "source":
                    candidate_sources[0]
            }

        else:

            query_args["where"] = {
                "source": {
                    "$in":
                        candidate_sources
                }
            }

    try:

        result = collection.query(
            **query_args
        )

    except Exception as error:

        # Defensive fallback:
        # if metadata filtering fails because an older Chroma DB stores
        # source metadata differently, do not crash a long evaluation.
        # Instead, retry full-corpus dense retrieval and record the mode.
        if candidate_sources is None:
            raise

        print(
            "WARNING: TF-IDF metadata filter failed; "
            "falling back to full-corpus dense retrieval."
        )

        print(
            f"Filter error: "
            f"{type(error).__name__}: {error}"
        )

        tfidf_mode = (
            "filter_error_full_corpus"
        )

        result = collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=top_k,

            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

    documents = (
        result.get(
            "documents",
            [[]],
        )[0]
    )

    metadatas = (
        result.get(
            "metadatas",
            [[]],
        )[0]
    )

    distances = (
        result.get(
            "distances",
            [[]],
        )[0]
    )

    contexts = [

        str(document)

        for document
        in documents

        if document
    ]

    sources = []

    for metadata in metadatas:

        if not metadata:

            sources.append(
                None
            )

            continue

        source = (
            metadata.get("source")
            or metadata.get("filename")
            or metadata.get("file")
            or metadata.get("document")
        )

        sources.append(
            source
        )

    # Keep lexical ranking compact and JSON/CSV friendly.
    lexical_ranking_serializable = [

        {
            "source":
                source,

            "score":
                round(
                    float(score),
                    6,
                ),
        }

        for source, score
        in lexical_ranking
    ]

    return {

        "contexts":
            contexts,

        "sources":
            sources,

        "distances":
            distances,

        "candidate_sources":
            candidate_sources,

        "tfidf_mode":
            tfidf_mode,

        "lexical_ranking":
            lexical_ranking_serializable,
    }


# =============================================================================
# GENERATION
# =============================================================================

def generate_answer(
    question: str,
    contexts: List[str],
    generator,
) -> str:

    context_text = "\n\n".join(

        [
            (
                f"[Context {index + 1}]\n"
                f"{context}"
            )

            for index, context
            in enumerate(contexts)
        ]
    )

    prompt = f"""
You are an HR policy assistant for FakeCompanyABC.

Answer the question using ONLY the retrieved company-policy context.

Rules:
- Use only information contained in the context.
- Do not use outside knowledge.
- Do not invent company policies or facts.
- If the context is insufficient, clearly say so.
- Give a direct and concise answer.
- Do not output chain-of-thought.
- Do not explain internal reasoning.

RETRIEVED POLICY CONTEXT:

{context_text}

QUESTION:

{question}

FINAL ANSWER:
""".strip()

    response = generator.invoke(
        prompt
    )

    if hasattr(
        response,
        "content",
    ):

        answer = response.content

    else:

        answer = str(
            response
        )

    return answer.strip()


# =============================================================================
# FRESH RAGAS JUDGE
# =============================================================================

def create_fresh_judge(
    judge_model: str,
):

    """
    IMPORTANT:

    Every individual RAGAS metric call receives a NEW
    ChatOllama instance and a NEW LangchainLLMWrapper.

    This avoids reusing an async Ollama client after
    ragas.evaluate() closes its event loop.
    """

    judge = ChatOllama(

        model=judge_model,

        temperature=0,

        # Prevent excessively long judge outputs.
        num_predict=512,
    )

    return LangchainLLMWrapper(
        judge
    )


# =============================================================================
# CREATE ONE METRIC
# =============================================================================

def create_metric(
    metric_name: str,
    evaluator_llm,
):

    if (
        metric_name
        == "faithfulness"
    ):

        return Faithfulness(
            llm=evaluator_llm
        )

    if (
        metric_name
        == "factual_correctness"
    ):

        return FactualCorrectness(
            llm=evaluator_llm,
            mode="f1",
        )

    raise ValueError(
        f"Unknown metric: {metric_name}"
    )


# =============================================================================
# EXTRACT RAGAS SCORE
# =============================================================================

def extract_ragas_score(
    dataframe,
    metric_name: str,
) -> Optional[float]:

    if len(dataframe) == 0:
        return None

    row = dataframe.iloc[0]

    # Exact name first.
    if (
        metric_name
        in dataframe.columns
    ):

        return safe_float(
            row[metric_name]
        )

    # Handle names such as:
    #
    # factual_correctness(mode=f1)
    #
    for column in dataframe.columns:

        if column.startswith(
            metric_name + "("
        ):

            return safe_float(
                row[column]
            )

    return None


# =============================================================================
# DETECT PARSER ERROR
# =============================================================================

def is_parser_error(
    error: Exception,
) -> bool:

    class_name = (
        type(error).__name__.lower()
    )

    message = str(
        error
    ).lower()

    parser_terms = [

        "parser",

        "parse",

        "outputparser",

        "ragasoutputparser",

        "validationerror",
    ]

    text = (
        class_name
        + " "
        + message
    )

    return any(
        term in text
        for term in parser_terms
    )


# =============================================================================
# ONE ISOLATED RAGAS METRIC
# =============================================================================

def evaluate_ragas_metric(
    metric_name: str,
    question: str,
    contexts: List[str],
    answer: str,
    reference: str,
    judge_model: str,
    timeout: int,
) -> Tuple[
    Optional[float],
    Optional[str],
    float,
]:

    """
    Evaluate exactly ONE metric.

    A completely fresh judge is created for EVERY attempt.

    Parser failure:
        retry once.

    Other failure:
        record failure immediately and continue to next metric.
    """

    sample = {

        "user_input":
            question,

        "retrieved_contexts":
            contexts,

        "response":
            answer,

        "reference":
            reference,
    }

    dataset = (
        EvaluationDataset.from_list(
            [sample]
        )
    )

    start_time = time.time()

    last_error = None

    for attempt in range(
        1,
        PARSER_ATTEMPTS + 1,
    ):

        try:

            # -------------------------------------------------------------
            # FRESH JUDGE FOR THIS ATTEMPT
            # -------------------------------------------------------------

            evaluator_llm = (
                create_fresh_judge(
                    judge_model
                )
            )

            metric = create_metric(
                metric_name,
                evaluator_llm,
            )

            run_config = RunConfig(

                timeout=timeout,

                max_workers=(
                    RAGAS_MAX_WORKERS
                ),

                max_retries=(
                    RAGAS_MAX_RETRIES
                ),
            )

            result = evaluate(

                dataset=dataset,

                metrics=[
                    metric
                ],

                run_config=run_config,

                raise_exceptions=True,

                show_progress=False,
            )

            dataframe = (
                result.to_pandas()
            )

            score = extract_ragas_score(
                dataframe,
                metric_name,
            )

            if score is None:

                raise RuntimeError(
                    f"{metric_name} returned "
                    "no numeric score."
                )

            elapsed = (
                time.time()
                - start_time
            )

            return (
                score,
                None,
                elapsed,
            )

        except Exception as error:

            last_error = error

            elapsed = (
                time.time()
                - start_time
            )

            parser_failure = (
                is_parser_error(
                    error
                )
            )

            print()

            print(
                f"    Attempt {attempt} failed:"
            )

            print(
                f"    {type(error).__name__}: "
                f"{error}"
            )

            # -------------------------------------------------------------
            # Retry ONLY parser failures
            # -------------------------------------------------------------

            if (
                parser_failure
                and attempt
                < PARSER_ATTEMPTS
            ):

                print(
                    "    Parser failure detected."
                )

                print(
                    "    Retrying with a fresh "
                    "judge instance..."
                )

                continue

            # No retry for timeout, connection errors, etc.
            break

    elapsed = (
        time.time()
        - start_time
    )

    error_text = (

        f"{type(last_error).__name__}: "
        f"{last_error}"

        if last_error
        else
        "Unknown evaluation failure"
    )

    return (
        None,
        error_text,
        elapsed,
    )


# =============================================================================
# COSINE SIMILARITY
# =============================================================================

def cosine_similarity(
    vector_a: List[float],
    vector_b: List[float],
) -> float:

    dot = sum(
        a * b
        for a, b
        in zip(
            vector_a,
            vector_b,
        )
    )

    norm_a = math.sqrt(
        sum(
            value * value
            for value in vector_a
        )
    )

    norm_b = math.sqrt(
        sum(
            value * value
            for value in vector_b
        )
    )

    if (
        norm_a == 0
        or norm_b == 0
    ):

        return 0.0

    return (
        dot
        / (
            norm_a
            * norm_b
        )
    )


# =============================================================================
# REFERENCE SEMANTIC SIMILARITY
# =============================================================================

def semantic_similarity(
    answer: str,
    reference: str,
    embeddings,
) -> float:

    """
    Symmetric semantic comparison.

    Both generated answer and reference answer use the same
    Nomic document-style prefix.
    """

    answer_embedding = (
        embeddings.embed_query(
            "search_document: "
            + answer
        )
    )

    reference_embedding = (
        embeddings.embed_query(
            "search_document: "
            + reference
        )
    )

    return cosine_similarity(
        answer_embedding,
        reference_embedding,
    )


# =============================================================================
# SUMMARY
# =============================================================================

def build_summary(
    rows: List[Dict[str, Any]],
) -> Dict[str, Any]:

    summary = {}

    metric_names = [

        "faithfulness",

        "factual_correctness",

        "semantic_similarity",
    ]

    total = len(
        rows
    )

    for metric_name in metric_names:

        values = [

            row.get(
                metric_name
            )

            for row in rows

            if row.get(
                metric_name
            ) is not None
        ]

        valid = len(
            values
        )

        failed = (
            total
            - valid
        )

        mean = (

            sum(values)
            / valid

            if valid
            else None
        )

        summary[
            metric_name
        ] = {

            "mean":
                mean,

            "valid":
                valid,

            "failed":
                failed,

            "total":
                total,

            "coverage":
                (
                    valid / total
                    if total
                    else 0.0
                ),
        }

    return summary


# =============================================================================
# SAVE CHECKPOINT
# =============================================================================

def save_results(
    rows: List[Dict[str, Any]],
    generator_model: str,
    judge_model: str,
    top_k: int,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = get_result_paths(
        generator_model,
        judge_model,
        top_k,
    )

    summary = build_summary(
        rows
    )

    # -------------------------------------------------------------------------
    # CSV
    # -------------------------------------------------------------------------

    csv_rows = []

    for row in rows:

        flat = dict(
            row
        )

        flat[
            "retrieved_contexts"
        ] = json.dumps(

            flat.get(
                "retrieved_contexts",
                [],
            ),

            ensure_ascii=False,
        )

        flat[
            "retrieved_sources"
        ] = json.dumps(

            flat.get(
                "retrieved_sources",
                [],
            ),

            ensure_ascii=False,
        )

        flat[
            "retrieval_distances"
        ] = json.dumps(

            flat.get(
                "retrieval_distances",
                [],
            ),

            ensure_ascii=False,
        )

        flat[
            "tfidf_candidate_sources"
        ] = json.dumps(

            flat.get(
                "tfidf_candidate_sources",
                None,
            ),

            ensure_ascii=False,
        )

        flat[
            "tfidf_lexical_ranking"
        ] = json.dumps(

            flat.get(
                "tfidf_lexical_ranking",
                [],
            ),

            ensure_ascii=False,
        )

        csv_rows.append(
            flat
        )

    if csv_rows:

        with paths["csv"].open(
            "w",
            encoding="utf-8",
            newline="",
        ) as file:

            writer = csv.DictWriter(

                file,

                fieldnames=list(
                    csv_rows[0].keys()
                ),
            )

            writer.writeheader()

            writer.writerows(
                csv_rows
            )

    # -------------------------------------------------------------------------
    # JSON
    # -------------------------------------------------------------------------

    output = {

        "updated_at":
            datetime.now().isoformat(),

        "evaluation_type":
            "generation",

        "retrieval_pipeline":
            "tfidf_policy_routing_then_dense_chroma",

        "generator_model":
            generator_model,

        "judge_model":
            judge_model,

        "embedding_model":
            EMBEDDING_MODEL,

        "top_k":
            top_k,

        "tfidf_routing": {
            "min_lexical_score":
                MIN_LEXICAL_SCORE,

            "max_ambiguous_score_spread":
                MAX_AMBIGUOUS_SCORE_SPREAD,

            "relative_score_threshold":
                RELATIVE_SCORE_THRESHOLD,

            "max_candidate_files":
                MAX_CANDIDATE_FILES,
        },

        "ground_truth_file":
            str(
                GROUND_TRUTH_PATH
            ),

        "completed_questions":
            len(rows),

        "metrics": [

            "faithfulness",

            "factual_correctness",

            "semantic_similarity",
        ],

        "summary":
            summary,

        "results":
            rows,
    }

    with paths["json"].open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(

            output,

            file,

            indent=2,

            ensure_ascii=False,
        )

    # -------------------------------------------------------------------------
    # JSONL
    # -------------------------------------------------------------------------

    with paths["jsonl"].open(
        "w",
        encoding="utf-8",
    ) as file:

        for row in rows:

            file.write(

                json.dumps(
                    row,
                    ensure_ascii=False,
                )

                + "\n"
            )

    return (
        paths,
        summary,
    )


# =============================================================================
# RESUME
# =============================================================================

def load_previous_results(
    generator_model: str,
    judge_model: str,
    top_k: int,
) -> List[Dict[str, Any]]:

    paths = get_result_paths(
        generator_model,
        judge_model,
        top_k,
    )

    json_path = (
        paths["json"]
    )

    if not json_path.exists():

        print(
            "No existing checkpoint found."
        )

        return []

    try:

        with json_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        rows = data.get(
            "results",
            [],
        )

        print(
            f"Loaded {len(rows)} "
            "previously completed questions."
        )

        return rows

    except Exception as error:

        print(
            "WARNING: could not load "
            "previous checkpoint:"
        )

        print(error)

        return []


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    questions = (
        load_ground_truth()
    )

    if args.limit is not None:

        if args.limit <= 0:

            raise ValueError(
                "--limit must be greater than 0."
            )

        questions = (
            questions[
                :args.limit
            ]
        )

    # =========================================================================
    # HEADER
    # =========================================================================

    separator()

    print(
        "FAKECOMPANYABC TF-IDF + DENSE GENERATION EVALUATION"
    )

    separator()

    print(
        f"Questions requested:    "
        f"{len(questions)}"
    )

    print(
        f"Generator:              "
        f"{args.generator_model}"
    )

    print(
        f"RAGAS judge:            "
        f"{args.judge_model}"
    )

    print(
        f"Embedding:              "
        f"{EMBEDDING_MODEL}"
    )

    print(
        f"Retrieval pipeline:     "
        f"TF-IDF routing + dense Chroma"
    )

    print(
        f"Retrieval Top-K:        "
        f"{args.top_k}"
    )

    print(
        f"RAGAS timeout:          "
        f"{args.timeout} s"
    )

    print(
        f"Parser attempts:        "
        f"{PARSER_ATTEMPTS}"
    )

    print()

    print(
        "Metrics:"
    )

    print(
        "  1. RAGAS Faithfulness"
    )

    print(
        "  2. RAGAS Factual Correctness (F1)"
    )

    print(
        "  3. Nomic Semantic Similarity"
    )

    print()

    print(
        "Checkpointing: after EVERY question"
    )

    print(
        "Metric failures are isolated and "
        "do not stop the run."
    )

    # =========================================================================
    # RAG COMPONENTS
    # =========================================================================

    (
        embeddings,
        generator,
        collection,
    ) = create_rag_components(
        args.generator_model
    )

    tfidf_router = (
        create_tfidf_router()
    )

    print()

    print(
        "RAG components loaded successfully."
    )

    print(
        "Retrieval pipeline: "
        "TF-IDF policy routing -> "
        "Nomic/Chroma dense Top-K"
    )

    # =========================================================================
    # RESUME
    # =========================================================================

    rows: List[
        Dict[str, Any]
    ] = []

    if args.resume:

        rows = load_previous_results(

            args.generator_model,

            args.judge_model,

            args.top_k,
        )

    completed_ids = {

        str(
            row.get("id")
        )

        for row in rows
    }

    remaining_questions = []

    for index, item in enumerate(
        questions,
        start=1,
    ):

        question_id = str(
            item.get(
                "id",
                f"q{index:03d}",
            )
        )

        if (
            question_id
            in completed_ids
        ):

            continue

        remaining_questions.append(
            (
                index,
                item,
            )
        )

    print()

    print(
        f"Already completed: "
        f"{len(completed_ids)}"
    )

    print(
        f"Remaining:         "
        f"{len(remaining_questions)}"
    )

    total_start = (
        time.time()
    )

    # =========================================================================
    # EVALUATE QUESTIONS
    # =========================================================================

    for (
        original_index,
        item,
    ) in remaining_questions:

        question_start = (
            time.time()
        )

        question_id = str(
            item.get(
                "id",
                f"q{original_index:03d}",
            )
        )

        question = str(
            item[
                "question"
            ]
        ).strip()

        reference = str(
            item[
                "expected_answer"
            ]
        ).strip()

        separator("-")

        print(
            f"[{original_index}/{len(questions)}] "
            f"{question_id}: "
            f"{question}"
        )

        # =====================================================================
        # RETRIEVAL
        # =====================================================================

        print()

        print(
            "Retrieving contexts..."
        )

        retrieval_start = (
            time.time()
        )

        retrieval = retrieve(

            question=question,

            embeddings=embeddings,

            collection=collection,

            tfidf_router=tfidf_router,

            top_k=args.top_k,
        )

        retrieval_time = (
            time.time()
            - retrieval_start
        )

        contexts = (
            retrieval[
                "contexts"
            ]
        )

        print(
            f"Retrieved {len(contexts)} "
            f"chunks in {retrieval_time:.1f} s."
        )

        print(
            f"TF-IDF mode: "
            f"{retrieval['tfidf_mode']}"
        )

        print(
            f"Candidate policies: "
            f"{retrieval['candidate_sources']}"
        )

        if retrieval[
            "lexical_ranking"
        ]:

            top_lexical = (
                retrieval[
                    "lexical_ranking"
                ][:3]
            )

            print(
                f"Top TF-IDF ranking: "
                f"{top_lexical}"
            )

        # =====================================================================
        # GENERATION
        # =====================================================================

        print()

        print(
            f"Generating with "
            f"{args.generator_model}..."
        )

        generation_start = (
            time.time()
        )

        answer = generate_answer(

            question=question,

            contexts=contexts,

            generator=generator,
        )

        generation_time = (
            time.time()
            - generation_start
        )

        print()

        print(
            "Generated answer:"
        )

        print(answer)

        print()

        print(
            f"Generation time: "
            f"{generation_time:.1f} s"
        )

        print()

        print(
            "Reference answer:"
        )

        print(reference)

        # =====================================================================
        # FAITHFULNESS
        # =====================================================================

        print()

        print(
            f"Evaluating Faithfulness "
            f"with {args.judge_model}..."
        )

        (
            faithfulness,
            faithfulness_error,
            faithfulness_time,
        ) = evaluate_ragas_metric(

            metric_name=(
                "faithfulness"
            ),

            question=question,

            contexts=contexts,

            answer=answer,

            reference=reference,

            judge_model=(
                args.judge_model
            ),

            timeout=args.timeout,
        )

        print(
            f"  Faithfulness = "
            f"{format_score(faithfulness)} "
            f"({faithfulness_time:.1f} s)"
        )

        if faithfulness_error:

            print(
                f"  Error: "
                f"{faithfulness_error}"
            )

        # =====================================================================
        # FACTUAL CORRECTNESS
        # =====================================================================

        print()

        print(
            f"Evaluating Factual Correctness "
            f"with {args.judge_model}..."
        )

        (
            factual_correctness,
            factual_error,
            factual_time,
        ) = evaluate_ragas_metric(

            metric_name=(
                "factual_correctness"
            ),

            question=question,

            contexts=contexts,

            answer=answer,

            reference=reference,

            judge_model=(
                args.judge_model
            ),

            timeout=args.timeout,
        )

        print(
            f"  Factual Correctness = "
            f"{format_score(factual_correctness)} "
            f"({factual_time:.1f} s)"
        )

        if factual_error:

            print(
                f"  Error: "
                f"{factual_error}"
            )

        # =====================================================================
        # SEMANTIC SIMILARITY
        # =====================================================================

        print()

        print(
            "Calculating Nomic semantic similarity..."
        )

        semantic_start = (
            time.time()
        )

        semantic_error = None

        try:

            similarity = (
                semantic_similarity(
                    answer=answer,
                    reference=reference,
                    embeddings=embeddings,
                )
            )

        except Exception as error:

            similarity = None

            semantic_error = (
                f"{type(error).__name__}: "
                f"{error}"
            )

        semantic_time = (
            time.time()
            - semantic_start
        )

        print(
            f"  Semantic Similarity = "
            f"{format_score(similarity)} "
            f"({semantic_time:.1f} s)"
        )

        if semantic_error:

            print(
                f"  Error: "
                f"{semantic_error}"
            )

        # =====================================================================
        # STORE QUESTION RESULT
        # =====================================================================

        question_time = (
            time.time()
            - question_start
        )

        row = {

            "id":
                question_id,

            "question":
                question,

            "reference_answer":
                reference,

            "generated_answer":
                answer,

            "retrieved_contexts":
                contexts,

            "retrieved_sources":
                retrieval[
                    "sources"
                ],

            "retrieval_distances":
                retrieval[
                    "distances"
                ],

            "tfidf_mode":
                retrieval[
                    "tfidf_mode"
                ],

            "tfidf_candidate_sources":
                retrieval[
                    "candidate_sources"
                ],

            "tfidf_lexical_ranking":
                retrieval[
                    "lexical_ranking"
                ],

            "retrieval_time_seconds":
                round(
                    retrieval_time,
                    3,
                ),

            "generation_time_seconds":
                round(
                    generation_time,
                    3,
                ),

            "faithfulness":
                faithfulness,

            "faithfulness_error":
                faithfulness_error,

            "faithfulness_time_seconds":
                round(
                    faithfulness_time,
                    3,
                ),

            "factual_correctness":
                factual_correctness,

            "factual_correctness_error":
                factual_error,

            "factual_correctness_time_seconds":
                round(
                    factual_time,
                    3,
                ),

            "semantic_similarity":
                similarity,

            "semantic_similarity_error":
                semantic_error,

            "semantic_similarity_time_seconds":
                round(
                    semantic_time,
                    3,
                ),

            "question_time_seconds":
                round(
                    question_time,
                    3,
                ),
        }

        rows.append(
            row
        )

        # =====================================================================
        # CHECKPOINT AFTER EVERY QUESTION
        # =====================================================================

        (
            paths,
            summary,
        ) = save_results(

            rows=rows,

            generator_model=(
                args.generator_model
            ),

            judge_model=(
                args.judge_model
            ),

            top_k=args.top_k,
        )

        print()

        print(
            "QUESTION RESULT:"
        )

        print(
            f"  Faithfulness        = "
            f"{format_score(faithfulness)}"
        )

        print(
            f"  Factual Correctness = "
            f"{format_score(factual_correctness)}"
        )

        print(
            f"  Semantic Similarity = "
            f"{format_score(similarity)}"
        )

        print()

        print(
            f"Question completed in "
            f"{question_time:.1f} s"
        )

        print(
            f"Checkpoint saved "
            f"({len(rows)} completed)."
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    total_elapsed = (
        time.time()
        - total_start
    )

    (
        paths,
        summary,
    ) = save_results(

        rows=rows,

        generator_model=(
            args.generator_model
        ),

        judge_model=(
            args.judge_model
        ),

        top_k=args.top_k,
    )

    print()

    separator()

    print(
        "FINAL GENERATION EVALUATION SUMMARY"
    )

    separator()

    for (
        label,
        metric_name,
    ) in [

        (
            "Faithfulness",
            "faithfulness",
        ),

        (
            "Factual Correctness",
            "factual_correctness",
        ),

        (
            "Semantic Similarity",
            "semantic_similarity",
        ),
    ]:

        metric_summary = (
            summary[
                metric_name
            ]
        )

        mean = (
            metric_summary[
                "mean"
            ]
        )

        valid = (
            metric_summary[
                "valid"
            ]
        )

        total = (
            metric_summary[
                "total"
            ]
        )

        failed = (
            metric_summary[
                "failed"
            ]
        )

        print(
            f"{label:<25}"
            f"{format_score(mean):<10}"
            f"valid={valid}/{total}  "
            f"failed={failed}"
        )

    print()

    print(
        f"Completed questions: "
        f"{len(rows)}"
    )

    print(
        f"Session evaluation time: "
        f"{total_elapsed / 60:.2f} minutes"
    )

    print()

    separator()

    print(
        "RESULT FILES"
    )

    separator()

    print(
        paths["csv"]
    )

    print(
        paths["json"]
    )

    print(
        paths["jsonl"]
    )

    print()

    print(
        "If the run was interrupted, continue with:"
    )

    print()

    print(
        "python 07_evaluate_rag_tfidf_ragas.py --resume"
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
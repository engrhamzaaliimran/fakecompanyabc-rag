#!/usr/bin/env python3
"""
09_compare_four_generation_evaluations.py

Compare four generation-evaluation cases for FakeCompanyABC RAG:

1. Dense baseline + RAGAS/Qwen2.5
2. Dense baseline + GPT-5.6 Sol
3. TF-IDF-routed dense + RAGAS/Qwen2.5
4. TF-IDF-routed dense + GPT-5.6 Sol

The script focuses on two different questions:

A) Did TF-IDF routing improve the RAG pipeline?
   - Compare Dense vs TF-IDF under the SAME judge.
   - Use paired per-question deltas because the same 100 questions are used.

B) How much do the two judges agree?
   - Compare RAGAS/Qwen2.5 vs GPT-5.6 Sol within each pipeline.
   - Report correlation, MAE, and agreement tolerances.

Additional metrics:
- Nomic Semantic Similarity: Dense vs TF-IDF
- GPT-5.6 Sol Answer Relevancy: Dense vs TF-IDF

Example:

python 09_compare_four_generation_evaluations.py \
    --dense-ragas ../evaluation/results/dense_baseline/ragas_qwen25_generation_100.csv \
    --dense-gpt ../evaluation/results/dense_baseline/gpt56_sol_generation_100.csv \
    --tfidf-ragas ../evaluation/results/tfidf_dense/ragas_qwen25_generation_100.csv \
    --tfidf-gpt ../evaluation/results/tfidf_dense/gpt56_sol_generation_100.csv \
    --output-dir ../evaluation/results/final/four_way_comparison
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

EPS = 1e-12
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 42

COMMON_METRICS = [
    "faithfulness",
    "factual_correctness",
]

PRETTY_METRICS = {
    "faithfulness": "Faithfulness",
    "factual_correctness": "Factual Correctness",
    "semantic_similarity": "Semantic Similarity",
    "answer_relevancy": "Answer Relevancy",
}

CASE_LABELS = {
    "dense_qwen": "Dense + RAGAS/Qwen2.5",
    "dense_gpt": "Dense + GPT-5.6 Sol",
    "tfidf_qwen": "TF-IDF + RAGAS/Qwen2.5",
    "tfidf_gpt": "TF-IDF + GPT-5.6 Sol",
}


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Four-way generation evaluation comparison."
    )

    parser.add_argument(
        "--dense-ragas",
        required=True,
        help="Dense baseline CSV evaluated with RAGAS/Qwen2.5.",
    )

    parser.add_argument(
        "--dense-gpt",
        required=True,
        help="Dense baseline CSV with GPT-5.6 Sol judge scores.",
    )

    parser.add_argument(
        "--tfidf-ragas",
        required=True,
        help="TF-IDF + dense CSV evaluated with RAGAS/Qwen2.5.",
    )

    parser.add_argument(
        "--tfidf-gpt",
        required=True,
        help="TF-IDF + dense CSV with GPT-5.6 Sol judge scores.",
    )

    parser.add_argument(
        "--output-dir",
        default="four_way_generation_comparison",
        help="Directory for generated CSVs and plots.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of largest improvements/regressions to export.",
    )

    return parser.parse_args()


# =============================================================================
# BASIC HELPERS
# =============================================================================

def require_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
    label: str,
):

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}"
        )


def read_csv(
    path: str,
    label: str,
) -> pd.DataFrame:

    dataframe = pd.read_csv(path)

    require_columns(
        dataframe,
        ["id", "question"],
        label,
    )

    dataframe["id"] = dataframe["id"].astype(str)

    if dataframe["id"].duplicated().any():
        duplicated = (
            dataframe.loc[
                dataframe["id"].duplicated(),
                "id",
            ]
            .tolist()
        )
        raise ValueError(
            f"{label} contains duplicate IDs: {duplicated[:10]}"
        )

    return dataframe


def numeric(
    series: pd.Series,
) -> pd.Series:

    return pd.to_numeric(
        series,
        errors="coerce",
    )


def safe_mean(
    series: pd.Series,
) -> float:

    values = numeric(series).dropna()

    if values.empty:
        return np.nan

    return float(values.mean())


def normalize_text(value) -> str:

    if pd.isna(value):
        return ""

    return " ".join(
        str(value).split()
    )


def parse_jsonish(value):

    if value is None:
        return None

    if isinstance(
        value,
        (list, dict, tuple),
    ):
        return value

    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    for parser in (
        json.loads,
        ast.literal_eval,
    ):
        try:
            return parser(text)
        except Exception:
            pass

    return text


# =============================================================================
# ALIGN / VALIDATE
# =============================================================================

def align_by_id(
    dataframe: pd.DataFrame,
    ids: List[str],
    label: str,
) -> pd.DataFrame:

    indexed = dataframe.set_index("id")

    missing = [
        question_id
        for question_id in ids
        if question_id not in indexed.index
    ]

    if missing:
        raise ValueError(
            f"{label} is missing IDs: {missing[:10]}"
        )

    extra = [
        question_id
        for question_id in indexed.index
        if question_id not in set(ids)
    ]

    if extra:
        print(
            f"WARNING: {label} contains "
            f"{len(extra)} extra IDs; ignoring them."
        )

    return (
        indexed
        .loc[ids]
        .reset_index()
    )


def validate_same_questions(
    frames: Dict[str, pd.DataFrame],
):

    base_label = next(iter(frames))
    base = frames[base_label]

    ids = base["id"].tolist()

    for label, dataframe in frames.items():

        if dataframe["id"].tolist() != ids:
            raise ValueError(
                f"Question IDs are not aligned for {label}."
            )

        mismatch = (
            base["question"]
            .map(normalize_text)
            != dataframe["question"]
            .map(normalize_text)
        )

        if mismatch.any():

            examples = (
                base.loc[
                    mismatch,
                    ["id", "question"],
                ]
                .head(5)
            )

            raise ValueError(
                f"Question text differs in {label}.\n"
                f"Examples:\n{examples}"
            )


def validate_gpt_matches_pipeline(
    ragas_df: pd.DataFrame,
    gpt_df: pd.DataFrame,
    pipeline_name: str,
):

    shared = [
        column
        for column in [
            "question",
            "reference_answer",
            "generated_answer",
            "retrieved_contexts",
        ]
        if (
            column in ragas_df.columns
            and column in gpt_df.columns
        )
    ]

    for column in shared:

        left = ragas_df[column].map(
            normalize_text
        )

        right = gpt_df[column].map(
            normalize_text
        )

        mismatch_count = int(
            (left != right).sum()
        )

        if mismatch_count:

            raise ValueError(
                f"{pipeline_name}: RAGAS and GPT files "
                f"differ in {mismatch_count} rows for "
                f"'{column}'. These files do not describe "
                f"the exact same pipeline outputs."
            )


# =============================================================================
# BOOTSTRAP / PAIRED EFFECTS
# =============================================================================

def bootstrap_mean_ci(
    values: np.ndarray,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float]:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    if len(values) == 1:
        return (
            float(values[0]),
            float(values[0]),
        )

    rng = np.random.default_rng(seed)

    means = np.empty(
        samples,
        dtype=float,
    )

    n = len(values)

    # Batched loop avoids allocating a very large matrix.
    for index in range(samples):

        sampled = rng.choice(
            values,
            size=n,
            replace=True,
        )

        means[index] = sampled.mean()

    low, high = np.quantile(
        means,
        [0.025, 0.975],
    )

    return (
        float(low),
        float(high),
    )


def paired_effect_row(
    judge: str,
    metric: str,
    dense_values: pd.Series,
    tfidf_values: pd.Series,
) -> Dict[str, object]:

    dense = numeric(dense_values)
    tfidf = numeric(tfidf_values)

    valid = (
        dense.notna()
        & tfidf.notna()
    )

    dense = dense[valid]
    tfidf = tfidf[valid]

    delta = (
        tfidf
        - dense
    )

    ci_low, ci_high = (
        bootstrap_mean_ci(
            delta.to_numpy()
        )
    )

    improved = int(
        (delta > EPS).sum()
    )

    worsened = int(
        (delta < -EPS).sum()
    )

    unchanged = int(
        (delta.abs() <= EPS).sum()
    )

    return {
        "judge":
            judge,

        "metric":
            metric,

        "dense_mean":
            float(dense.mean()),

        "tfidf_mean":
            float(tfidf.mean()),

        "mean_delta_tfidf_minus_dense":
            float(delta.mean()),

        "median_delta_tfidf_minus_dense":
            float(delta.median()),

        "bootstrap_95ci_low":
            ci_low,

        "bootstrap_95ci_high":
            ci_high,

        "improved_questions":
            improved,

        "worsened_questions":
            worsened,

        "unchanged_questions":
            unchanged,

        "valid_pairs":
            int(valid.sum()),
    }


# =============================================================================
# JUDGE AGREEMENT
# =============================================================================

def judge_agreement_row(
    pipeline: str,
    metric: str,
    qwen_values: pd.Series,
    gpt_values: pd.Series,
) -> Dict[str, object]:

    qwen = numeric(qwen_values)
    gpt = numeric(gpt_values)

    valid = (
        qwen.notna()
        & gpt.notna()
    )

    qwen = qwen[valid]
    gpt = gpt[valid]

    difference = (
        qwen
        - gpt
    )

    if len(qwen) >= 2:
        pearson = qwen.corr(
            gpt,
            method="pearson",
        )

        spearman = qwen.corr(
            gpt,
            method="spearman",
        )
    else:
        pearson = np.nan
        spearman = np.nan

    return {
        "pipeline":
            pipeline,

        "metric":
            metric,

        "qwen_mean":
            float(qwen.mean()),

        "gpt56_mean":
            float(gpt.mean()),

        "pearson":
            float(pearson)
            if pd.notna(pearson)
            else np.nan,

        "spearman":
            float(spearman)
            if pd.notna(spearman)
            else np.nan,

        "mean_absolute_difference":
            float(
                difference.abs().mean()
            ),

        "mean_qwen_minus_gpt56":
            float(
                difference.mean()
            ),

        "agreement_within_0.10":
            float(
                (difference.abs() <= 0.10)
                .mean()
            ),

        "agreement_within_0.25":
            float(
                (difference.abs() <= 0.25)
                .mean()
            ),

        "valid_pairs":
            int(valid.sum()),
    }


# =============================================================================
# PLOT HELPERS
# =============================================================================

def save_mean_bar(
    labels: List[str],
    values: List[float],
    title: str,
    ylabel: str,
    path: Path,
):

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    positions = np.arange(
        len(labels)
    )

    bars = ax.bar(
        positions,
        values,
    )

    ax.set_xticks(
        positions
    )

    ax.set_xticklabels(
        labels,
        rotation=15,
        ha="right",
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    for bar, value in zip(
        bars,
        values,
    ):

        if np.isfinite(value):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                value + 0.015,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_pipeline_scatter(
    dense: pd.Series,
    tfidf: pd.Series,
    ids: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    annotate_threshold: float = 0.50,
):

    x = numeric(dense)
    y = numeric(tfidf)

    valid = (
        x.notna()
        & y.notna()
    )

    x = x[valid]
    y = y[valid]

    question_ids = (
        ids[valid]
        .astype(str)
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 7.5)
    )

    ax.scatter(
        x,
        y,
        alpha=0.65,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlim(
        -0.03,
        1.03,
    )

    ax.set_ylim(
        -0.03,
        1.03,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.20,
    )

    deltas = (
        y
        - x
    )

    for question_id, x_value, y_value, delta in zip(
        question_ids,
        x,
        y,
        deltas,
    ):

        if abs(delta) >= annotate_threshold:

            ax.annotate(
                question_id,
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_judge_scatter(
    qwen: pd.Series,
    gpt: pd.Series,
    ids: pd.Series,
    title: str,
    path: Path,
    annotate_threshold: float = 0.50,
):

    x = numeric(qwen)
    y = numeric(gpt)

    valid = (
        x.notna()
        & y.notna()
    )

    x = x[valid]
    y = y[valid]

    question_ids = (
        ids[valid]
        .astype(str)
    )

    fig, ax = plt.subplots(
        figsize=(7.5, 7.5)
    )

    ax.scatter(
        x,
        y,
        alpha=0.65,
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlim(
        -0.03,
        1.03,
    )

    ax.set_ylim(
        -0.03,
        1.03,
    )

    ax.set_xlabel(
        "RAGAS + Qwen2.5"
    )

    ax.set_ylabel(
        "GPT-5.6 Sol Judge"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.20,
    )

    difference = (
        y
        - x
    )

    for question_id, x_value, y_value, delta in zip(
        question_ids,
        x,
        y,
        difference,
    ):

        if abs(delta) >= annotate_threshold:

            ax.annotate(
                question_id,
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_delta_bar(
    dataframe: pd.DataFrame,
    delta_column: str,
    title: str,
    path: Path,
    top_n: int = 20,
):

    plot_df = (
        dataframe[
            ["id", delta_column]
        ]
        .dropna()
        .copy()
    )

    plot_df[
        "abs_delta"
    ] = (
        plot_df[
            delta_column
        ]
        .abs()
    )

    plot_df = (
        plot_df
        .sort_values(
            "abs_delta",
            ascending=False,
        )
        .head(top_n)
        .sort_values(
            delta_column
        )
    )

    fig, ax = plt.subplots(
        figsize=(9, 8)
    )

    positions = np.arange(
        len(plot_df)
    )

    ax.barh(
        positions,
        plot_df[
            delta_column
        ],
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        plot_df["id"]
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_xlabel(
        "TF-IDF score − Dense score"
    )

    ax.set_title(
        title
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    for position, value in zip(
        positions,
        plot_df[
            delta_column
        ],
    ):

        offset = (
            0.015
            if value >= 0
            else -0.015
        )

        ax.text(
            value + offset,
            position,
            f"{value:+.2f}",
            va="center",
            ha=(
                "left"
                if value >= 0
                else "right"
            ),
            fontsize=8,
        )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    dense_ragas = read_csv(
        args.dense_ragas,
        "Dense RAGAS/Qwen2.5",
    )

    dense_gpt = read_csv(
        args.dense_gpt,
        "Dense GPT-5.6 Sol",
    )

    tfidf_ragas = read_csv(
        args.tfidf_ragas,
        "TF-IDF RAGAS/Qwen2.5",
    )

    tfidf_gpt = read_csv(
        args.tfidf_gpt,
        "TF-IDF GPT-5.6 Sol",
    )

    # -------------------------------------------------------------------------
    # Required metric columns
    # -------------------------------------------------------------------------

    require_columns(
        dense_ragas,
        [
            "faithfulness",
            "factual_correctness",
            "semantic_similarity",
        ],
        "Dense RAGAS/Qwen2.5",
    )

    require_columns(
        tfidf_ragas,
        [
            "faithfulness",
            "factual_correctness",
            "semantic_similarity",
        ],
        "TF-IDF RAGAS/Qwen2.5",
    )

    require_columns(
        dense_gpt,
        [
            "gpt56_faithfulness",
            "gpt56_factual_correctness",
            "gpt56_answer_relevancy",
        ],
        "Dense GPT-5.6 Sol",
    )

    require_columns(
        tfidf_gpt,
        [
            "gpt56_faithfulness",
            "gpt56_factual_correctness",
            "gpt56_answer_relevancy",
        ],
        "TF-IDF GPT-5.6 Sol",
    )

    # -------------------------------------------------------------------------
    # Align all files to dense baseline order.
    # -------------------------------------------------------------------------

    ids = dense_ragas[
        "id"
    ].tolist()

    dense_gpt = align_by_id(
        dense_gpt,
        ids,
        "Dense GPT-5.6 Sol",
    )

    tfidf_ragas = align_by_id(
        tfidf_ragas,
        ids,
        "TF-IDF RAGAS/Qwen2.5",
    )

    tfidf_gpt = align_by_id(
        tfidf_gpt,
        ids,
        "TF-IDF GPT-5.6 Sol",
    )

    frames = {
        "dense_ragas":
            dense_ragas,

        "dense_gpt":
            dense_gpt,

        "tfidf_ragas":
            tfidf_ragas,

        "tfidf_gpt":
            tfidf_gpt,
    }

    validate_same_questions(
        frames
    )

    validate_gpt_matches_pipeline(
        dense_ragas,
        dense_gpt,
        "Dense baseline",
    )

    validate_gpt_matches_pipeline(
        tfidf_ragas,
        tfidf_gpt,
        "TF-IDF pipeline",
    )

    # -------------------------------------------------------------------------
    # FOUR-CASE SUMMARY
    # -------------------------------------------------------------------------

    four_case_rows = []

    four_case_rows.extend([
        {
            "case":
                CASE_LABELS[
                    "dense_qwen"
                ],
            "pipeline":
                "Dense baseline",
            "judge":
                "RAGAS + Qwen2.5",
            "faithfulness":
                safe_mean(
                    dense_ragas[
                        "faithfulness"
                    ]
                ),
            "factual_correctness":
                safe_mean(
                    dense_ragas[
                        "factual_correctness"
                    ]
                ),
            "semantic_similarity":
                safe_mean(
                    dense_ragas[
                        "semantic_similarity"
                    ]
                ),
            "answer_relevancy":
                np.nan,
        },
        {
            "case":
                CASE_LABELS[
                    "dense_gpt"
                ],
            "pipeline":
                "Dense baseline",
            "judge":
                "GPT-5.6 Sol",
            "faithfulness":
                safe_mean(
                    dense_gpt[
                        "gpt56_faithfulness"
                    ]
                ),
            "factual_correctness":
                safe_mean(
                    dense_gpt[
                        "gpt56_factual_correctness"
                    ]
                ),
            "semantic_similarity":
                np.nan,
            "answer_relevancy":
                safe_mean(
                    dense_gpt[
                        "gpt56_answer_relevancy"
                    ]
                ),
        },
        {
            "case":
                CASE_LABELS[
                    "tfidf_qwen"
                ],
            "pipeline":
                "TF-IDF + Dense",
            "judge":
                "RAGAS + Qwen2.5",
            "faithfulness":
                safe_mean(
                    tfidf_ragas[
                        "faithfulness"
                    ]
                ),
            "factual_correctness":
                safe_mean(
                    tfidf_ragas[
                        "factual_correctness"
                    ]
                ),
            "semantic_similarity":
                safe_mean(
                    tfidf_ragas[
                        "semantic_similarity"
                    ]
                ),
            "answer_relevancy":
                np.nan,
        },
        {
            "case":
                CASE_LABELS[
                    "tfidf_gpt"
                ],
            "pipeline":
                "TF-IDF + Dense",
            "judge":
                "GPT-5.6 Sol",
            "faithfulness":
                safe_mean(
                    tfidf_gpt[
                        "gpt56_faithfulness"
                    ]
                ),
            "factual_correctness":
                safe_mean(
                    tfidf_gpt[
                        "gpt56_factual_correctness"
                    ]
                ),
            "semantic_similarity":
                np.nan,
            "answer_relevancy":
                safe_mean(
                    tfidf_gpt[
                        "gpt56_answer_relevancy"
                    ]
                ),
        },
    ])

    four_case_summary = pd.DataFrame(
        four_case_rows
    )

    four_case_summary.to_csv(
        output_dir
        / "comparison_summary_four_cases.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # PER-QUESTION FOUR-WAY TABLE
    # -------------------------------------------------------------------------

    per_question = pd.DataFrame({
        "id":
            ids,

        "question":
            dense_ragas[
                "question"
            ],

        "reference_answer":
            dense_ragas.get(
                "reference_answer",
                pd.Series(
                    [""] * len(ids)
                ),
            ),

        "dense_generated_answer":
            dense_ragas.get(
                "generated_answer",
                pd.Series(
                    [""] * len(ids)
                ),
            ),

        "tfidf_generated_answer":
            tfidf_ragas.get(
                "generated_answer",
                pd.Series(
                    [""] * len(ids)
                ),
            ),

        "dense_qwen_faithfulness":
            numeric(
                dense_ragas[
                    "faithfulness"
                ]
            ),

        "dense_qwen_factual_correctness":
            numeric(
                dense_ragas[
                    "factual_correctness"
                ]
            ),

        "dense_semantic_similarity":
            numeric(
                dense_ragas[
                    "semantic_similarity"
                ]
            ),

        "dense_gpt56_faithfulness":
            numeric(
                dense_gpt[
                    "gpt56_faithfulness"
                ]
            ),

        "dense_gpt56_factual_correctness":
            numeric(
                dense_gpt[
                    "gpt56_factual_correctness"
                ]
            ),

        "dense_gpt56_answer_relevancy":
            numeric(
                dense_gpt[
                    "gpt56_answer_relevancy"
                ]
            ),

        "tfidf_qwen_faithfulness":
            numeric(
                tfidf_ragas[
                    "faithfulness"
                ]
            ),

        "tfidf_qwen_factual_correctness":
            numeric(
                tfidf_ragas[
                    "factual_correctness"
                ]
            ),

        "tfidf_semantic_similarity":
            numeric(
                tfidf_ragas[
                    "semantic_similarity"
                ]
            ),

        "tfidf_gpt56_faithfulness":
            numeric(
                tfidf_gpt[
                    "gpt56_faithfulness"
                ]
            ),

        "tfidf_gpt56_factual_correctness":
            numeric(
                tfidf_gpt[
                    "gpt56_factual_correctness"
                ]
            ),

        "tfidf_gpt56_answer_relevancy":
            numeric(
                tfidf_gpt[
                    "gpt56_answer_relevancy"
                ]
            ),
    })

    # Pipeline deltas under the same judge.
    per_question[
        "delta_qwen_faithfulness"
    ] = (
        per_question[
            "tfidf_qwen_faithfulness"
        ]
        - per_question[
            "dense_qwen_faithfulness"
        ]
    )

    per_question[
        "delta_qwen_factual_correctness"
    ] = (
        per_question[
            "tfidf_qwen_factual_correctness"
        ]
        - per_question[
            "dense_qwen_factual_correctness"
        ]
    )

    per_question[
        "delta_gpt56_faithfulness"
    ] = (
        per_question[
            "tfidf_gpt56_faithfulness"
        ]
        - per_question[
            "dense_gpt56_faithfulness"
        ]
    )

    per_question[
        "delta_gpt56_factual_correctness"
    ] = (
        per_question[
            "tfidf_gpt56_factual_correctness"
        ]
        - per_question[
            "dense_gpt56_factual_correctness"
        ]
    )

    per_question[
        "delta_gpt56_answer_relevancy"
    ] = (
        per_question[
            "tfidf_gpt56_answer_relevancy"
        ]
        - per_question[
            "dense_gpt56_answer_relevancy"
        ]
    )

    per_question[
        "delta_semantic_similarity"
    ] = (
        per_question[
            "tfidf_semantic_similarity"
        ]
        - per_question[
            "dense_semantic_similarity"
        ]
    )

    # Judge deltas within each pipeline.
    per_question[
        "dense_qwen_minus_gpt56_faithfulness"
    ] = (
        per_question[
            "dense_qwen_faithfulness"
        ]
        - per_question[
            "dense_gpt56_faithfulness"
        ]
    )

    per_question[
        "dense_qwen_minus_gpt56_factual"
    ] = (
        per_question[
            "dense_qwen_factual_correctness"
        ]
        - per_question[
            "dense_gpt56_factual_correctness"
        ]
    )

    per_question[
        "tfidf_qwen_minus_gpt56_faithfulness"
    ] = (
        per_question[
            "tfidf_qwen_faithfulness"
        ]
        - per_question[
            "tfidf_gpt56_faithfulness"
        ]
    )

    per_question[
        "tfidf_qwen_minus_gpt56_factual"
    ] = (
        per_question[
            "tfidf_qwen_factual_correctness"
        ]
        - per_question[
            "tfidf_gpt56_factual_correctness"
        ]
    )

    # Did the actual RAG output change?
    if (
        "generated_answer"
        in dense_ragas.columns
        and "generated_answer"
        in tfidf_ragas.columns
    ):

        per_question[
            "generated_answer_changed"
        ] = (
            dense_ragas[
                "generated_answer"
            ]
            .map(normalize_text)
            != tfidf_ragas[
                "generated_answer"
            ]
            .map(normalize_text)
        )

    if (
        "retrieved_contexts"
        in dense_ragas.columns
        and "retrieved_contexts"
        in tfidf_ragas.columns
    ):

        per_question[
            "retrieved_contexts_changed"
        ] = (
            dense_ragas[
                "retrieved_contexts"
            ]
            .map(normalize_text)
            != tfidf_ragas[
                "retrieved_contexts"
            ]
            .map(normalize_text)
        )

    if (
        "retrieved_sources"
        in dense_ragas.columns
        and "retrieved_sources"
        in tfidf_ragas.columns
    ):

        per_question[
            "retrieved_sources_changed"
        ] = (
            dense_ragas[
                "retrieved_sources"
            ]
            .map(normalize_text)
            != tfidf_ragas[
                "retrieved_sources"
            ]
            .map(normalize_text)
        )

    if (
        "tfidf_mode"
        in tfidf_ragas.columns
    ):

        per_question[
            "tfidf_mode"
        ] = (
            tfidf_ragas[
                "tfidf_mode"
            ]
        )

    if (
        "tfidf_candidate_sources"
        in tfidf_ragas.columns
    ):

        per_question[
            "tfidf_candidate_sources"
        ] = (
            tfidf_ragas[
                "tfidf_candidate_sources"
            ]
        )

    per_question.to_csv(
        output_dir
        / "per_question_four_way_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # PAIRED PIPELINE EFFECTS
    # -------------------------------------------------------------------------

    effect_rows = [
        paired_effect_row(
            judge="RAGAS + Qwen2.5",
            metric="faithfulness",
            dense_values=dense_ragas[
                "faithfulness"
            ],
            tfidf_values=tfidf_ragas[
                "faithfulness"
            ],
        ),

        paired_effect_row(
            judge="RAGAS + Qwen2.5",
            metric="factual_correctness",
            dense_values=dense_ragas[
                "factual_correctness"
            ],
            tfidf_values=tfidf_ragas[
                "factual_correctness"
            ],
        ),

        paired_effect_row(
            judge="GPT-5.6 Sol",
            metric="faithfulness",
            dense_values=dense_gpt[
                "gpt56_faithfulness"
            ],
            tfidf_values=tfidf_gpt[
                "gpt56_faithfulness"
            ],
        ),

        paired_effect_row(
            judge="GPT-5.6 Sol",
            metric="factual_correctness",
            dense_values=dense_gpt[
                "gpt56_factual_correctness"
            ],
            tfidf_values=tfidf_gpt[
                "gpt56_factual_correctness"
            ],
        ),

        paired_effect_row(
            judge="GPT-5.6 Sol",
            metric="answer_relevancy",
            dense_values=dense_gpt[
                "gpt56_answer_relevancy"
            ],
            tfidf_values=tfidf_gpt[
                "gpt56_answer_relevancy"
            ],
        ),

        paired_effect_row(
            judge="Nomic embedding",
            metric="semantic_similarity",
            dense_values=dense_ragas[
                "semantic_similarity"
            ],
            tfidf_values=tfidf_ragas[
                "semantic_similarity"
            ],
        ),
    ]

    paired_effects = pd.DataFrame(
        effect_rows
    )

    paired_effects.to_csv(
        output_dir
        / "paired_pipeline_effects.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # JUDGE AGREEMENT
    # -------------------------------------------------------------------------

    agreement_rows = []

    for pipeline_name, ragas_df, gpt_df in [
        (
            "Dense baseline",
            dense_ragas,
            dense_gpt,
        ),
        (
            "TF-IDF + Dense",
            tfidf_ragas,
            tfidf_gpt,
        ),
    ]:

        agreement_rows.append(
            judge_agreement_row(
                pipeline=pipeline_name,
                metric="faithfulness",
                qwen_values=ragas_df[
                    "faithfulness"
                ],
                gpt_values=gpt_df[
                    "gpt56_faithfulness"
                ],
            )
        )

        agreement_rows.append(
            judge_agreement_row(
                pipeline=pipeline_name,
                metric="factual_correctness",
                qwen_values=ragas_df[
                    "factual_correctness"
                ],
                gpt_values=gpt_df[
                    "gpt56_factual_correctness"
                ],
            )
        )

    judge_agreement = pd.DataFrame(
        agreement_rows
    )

    judge_agreement.to_csv(
        output_dir
        / "judge_agreement_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # TF-IDF ROUTING SUMMARY
    # -------------------------------------------------------------------------

    if (
        "tfidf_mode"
        in tfidf_ragas.columns
    ):

        routing_summary = (
            tfidf_ragas[
                "tfidf_mode"
            ]
            .fillna("missing")
            .value_counts(
                dropna=False
            )
            .rename_axis(
                "tfidf_mode"
            )
            .reset_index(
                name="count"
            )
        )

        routing_summary[
            "fraction"
        ] = (
            routing_summary[
                "count"
            ]
            / len(tfidf_ragas)
        )

        routing_summary.to_csv(
            output_dir
            / "tfidf_routing_summary.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # CHANGE SUMMARY
    # -------------------------------------------------------------------------

    change_rows = []

    for column, label in [
        (
            "generated_answer_changed",
            "Generated answer changed",
        ),
        (
            "retrieved_contexts_changed",
            "Retrieved contexts changed",
        ),
        (
            "retrieved_sources_changed",
            "Retrieved sources changed",
        ),
    ]:

        if column in per_question.columns:

            changed_count = int(
                per_question[
                    column
                ].sum()
            )

            change_rows.append({
                "comparison":
                    label,

                "changed_questions":
                    changed_count,

                "unchanged_questions":
                    int(
                        len(per_question)
                        - changed_count
                    ),

                "total_questions":
                    len(per_question),

                "changed_fraction":
                    (
                        changed_count
                        / len(per_question)
                    ),
            })

    if change_rows:

        pd.DataFrame(
            change_rows
        ).to_csv(
            output_dir
            / "pipeline_change_summary.csv",
            index=False,
        )

    # -------------------------------------------------------------------------
    # TOP GPT IMPROVEMENTS / REGRESSIONS
    # -------------------------------------------------------------------------

    export_columns = [
        column
        for column in [
            "id",
            "question",
            "reference_answer",
            "dense_generated_answer",
            "tfidf_generated_answer",
            "dense_gpt56_faithfulness",
            "tfidf_gpt56_faithfulness",
            "delta_gpt56_faithfulness",
            "dense_gpt56_factual_correctness",
            "tfidf_gpt56_factual_correctness",
            "delta_gpt56_factual_correctness",
            "dense_gpt56_answer_relevancy",
            "tfidf_gpt56_answer_relevancy",
            "delta_gpt56_answer_relevancy",
            "tfidf_mode",
            "tfidf_candidate_sources",
        ]
        if column in per_question.columns
    ]

    (
        per_question
        .sort_values(
            "delta_gpt56_factual_correctness",
            ascending=False,
        )
        .head(args.top_n)[
            export_columns
        ]
        .to_csv(
            output_dir
            / "top_gpt56_factual_improvements.csv",
            index=False,
        )
    )

    (
        per_question
        .sort_values(
            "delta_gpt56_factual_correctness",
            ascending=True,
        )
        .head(args.top_n)[
            export_columns
        ]
        .to_csv(
            output_dir
            / "top_gpt56_factual_regressions.csv",
            index=False,
        )
    )

    (
        per_question
        .sort_values(
            "delta_gpt56_faithfulness",
            ascending=False,
        )
        .head(args.top_n)[
            export_columns
        ]
        .to_csv(
            output_dir
            / "top_gpt56_faithfulness_improvements.csv",
            index=False,
        )
    )

    (
        per_question
        .sort_values(
            "delta_gpt56_faithfulness",
            ascending=True,
        )
        .head(args.top_n)[
            export_columns
        ]
        .to_csv(
            output_dir
            / "top_gpt56_faithfulness_regressions.csv",
            index=False,
        )
    )

    # -------------------------------------------------------------------------
    # TOP JUDGE DISAGREEMENTS FOR EACH PIPELINE
    # -------------------------------------------------------------------------

    for prefix, faith_delta, factual_delta in [
        (
            "dense",
            "dense_qwen_minus_gpt56_faithfulness",
            "dense_qwen_minus_gpt56_factual",
        ),
        (
            "tfidf",
            "tfidf_qwen_minus_gpt56_faithfulness",
            "tfidf_qwen_minus_gpt56_factual",
        ),
    ]:

        temp = per_question.copy()

        temp[
            "absolute_difference"
        ] = (
            temp[
                factual_delta
            ]
            .abs()
        )

        (
            temp
            .sort_values(
                "absolute_difference",
                ascending=False,
            )
            .head(args.top_n)
            .to_csv(
                output_dir
                / f"top_{prefix}_factual_judge_disagreements.csv",
                index=False,
            )
        )

        temp = per_question.copy()

        temp[
            "absolute_difference"
        ] = (
            temp[
                faith_delta
            ]
            .abs()
        )

        (
            temp
            .sort_values(
                "absolute_difference",
                ascending=False,
            )
            .head(args.top_n)
            .to_csv(
                output_dir
                / f"top_{prefix}_faithfulness_judge_disagreements.csv",
                index=False,
            )
        )

    # -------------------------------------------------------------------------
    # PLOTS: FOUR CASES
    # -------------------------------------------------------------------------

    case_labels = (
        four_case_summary[
            "case"
        ]
        .tolist()
    )

    save_mean_bar(
        labels=case_labels,
        values=(
            four_case_summary[
                "faithfulness"
            ]
            .tolist()
        ),
        title=(
            "Faithfulness — Four Evaluation Cases"
        ),
        ylabel="Mean score",
        path=(
            output_dir
            / "01_four_case_faithfulness.png"
        ),
    )

    save_mean_bar(
        labels=case_labels,
        values=(
            four_case_summary[
                "factual_correctness"
            ]
            .tolist()
        ),
        title=(
            "Factual Correctness — Four Evaluation Cases"
        ),
        ylabel="Mean score",
        path=(
            output_dir
            / "02_four_case_factual_correctness.png"
        ),
    )

    # -------------------------------------------------------------------------
    # PLOTS: ADDITIONAL METRICS
    # -------------------------------------------------------------------------

    save_mean_bar(
        labels=[
            "Dense baseline",
            "TF-IDF + Dense",
        ],
        values=[
            safe_mean(
                dense_ragas[
                    "semantic_similarity"
                ]
            ),
            safe_mean(
                tfidf_ragas[
                    "semantic_similarity"
                ]
            ),
        ],
        title=(
            "Nomic Semantic Similarity — Pipeline Comparison"
        ),
        ylabel="Mean score",
        path=(
            output_dir
            / "03_semantic_similarity_dense_vs_tfidf.png"
        ),
    )

    save_mean_bar(
        labels=[
            "Dense baseline",
            "TF-IDF + Dense",
        ],
        values=[
            safe_mean(
                dense_gpt[
                    "gpt56_answer_relevancy"
                ]
            ),
            safe_mean(
                tfidf_gpt[
                    "gpt56_answer_relevancy"
                ]
            ),
        ],
        title=(
            "GPT-5.6 Sol Answer Relevancy — Pipeline Comparison"
        ),
        ylabel="Mean score",
        path=(
            output_dir
            / "04_answer_relevancy_dense_vs_tfidf.png"
        ),
    )

    # -------------------------------------------------------------------------
    # PLOTS: DENSE VS TF-IDF UNDER SAME JUDGE
    # -------------------------------------------------------------------------

    save_pipeline_scatter(
        dense=dense_gpt[
            "gpt56_factual_correctness"
        ],
        tfidf=tfidf_gpt[
            "gpt56_factual_correctness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "GPT-5.6 Sol Factual Correctness: Dense vs TF-IDF"
        ),
        xlabel="Dense baseline",
        ylabel="TF-IDF + Dense",
        path=(
            output_dir
            / "05_gpt56_factual_dense_vs_tfidf.png"
        ),
    )

    save_pipeline_scatter(
        dense=dense_gpt[
            "gpt56_faithfulness"
        ],
        tfidf=tfidf_gpt[
            "gpt56_faithfulness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "GPT-5.6 Sol Faithfulness: Dense vs TF-IDF"
        ),
        xlabel="Dense baseline",
        ylabel="TF-IDF + Dense",
        path=(
            output_dir
            / "06_gpt56_faithfulness_dense_vs_tfidf.png"
        ),
    )

    save_pipeline_scatter(
        dense=dense_ragas[
            "factual_correctness"
        ],
        tfidf=tfidf_ragas[
            "factual_correctness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "RAGAS/Qwen2.5 Factual Correctness: Dense vs TF-IDF"
        ),
        xlabel="Dense baseline",
        ylabel="TF-IDF + Dense",
        path=(
            output_dir
            / "07_qwen_factual_dense_vs_tfidf.png"
        ),
    )

    save_pipeline_scatter(
        dense=dense_ragas[
            "faithfulness"
        ],
        tfidf=tfidf_ragas[
            "faithfulness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "RAGAS/Qwen2.5 Faithfulness: Dense vs TF-IDF"
        ),
        xlabel="Dense baseline",
        ylabel="TF-IDF + Dense",
        path=(
            output_dir
            / "08_qwen_faithfulness_dense_vs_tfidf.png"
        ),
    )

    # -------------------------------------------------------------------------
    # PLOTS: JUDGE AGREEMENT WITHIN EACH PIPELINE
    # -------------------------------------------------------------------------

    save_judge_scatter(
        qwen=dense_ragas[
            "factual_correctness"
        ],
        gpt=dense_gpt[
            "gpt56_factual_correctness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "Dense Baseline — Factual Correctness Judge Agreement"
        ),
        path=(
            output_dir
            / "09_dense_factual_judge_agreement.png"
        ),
    )

    save_judge_scatter(
        qwen=tfidf_ragas[
            "factual_correctness"
        ],
        gpt=tfidf_gpt[
            "gpt56_factual_correctness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "TF-IDF + Dense — Factual Correctness Judge Agreement"
        ),
        path=(
            output_dir
            / "10_tfidf_factual_judge_agreement.png"
        ),
    )

    save_judge_scatter(
        qwen=dense_ragas[
            "faithfulness"
        ],
        gpt=dense_gpt[
            "gpt56_faithfulness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "Dense Baseline — Faithfulness Judge Agreement"
        ),
        path=(
            output_dir
            / "11_dense_faithfulness_judge_agreement.png"
        ),
    )

    save_judge_scatter(
        qwen=tfidf_ragas[
            "faithfulness"
        ],
        gpt=tfidf_gpt[
            "gpt56_faithfulness"
        ],
        ids=dense_ragas[
            "id"
        ],
        title=(
            "TF-IDF + Dense — Faithfulness Judge Agreement"
        ),
        path=(
            output_dir
            / "12_tfidf_faithfulness_judge_agreement.png"
        ),
    )

    # -------------------------------------------------------------------------
    # PLOTS: LARGEST GPT PIPELINE EFFECTS
    # -------------------------------------------------------------------------

    save_delta_bar(
        dataframe=per_question,
        delta_column=(
            "delta_gpt56_factual_correctness"
        ),
        title=(
            "Largest GPT-5.6 Factual Changes After TF-IDF Routing"
        ),
        path=(
            output_dir
            / "13_largest_gpt56_factual_pipeline_changes.png"
        ),
        top_n=20,
    )

    save_delta_bar(
        dataframe=per_question,
        delta_column=(
            "delta_gpt56_faithfulness"
        ),
        title=(
            "Largest GPT-5.6 Faithfulness Changes After TF-IDF Routing"
        ),
        path=(
            output_dir
            / "14_largest_gpt56_faithfulness_pipeline_changes.png"
        ),
        top_n=20,
    )

    # -------------------------------------------------------------------------
    # TERMINAL SUMMARY
    # -------------------------------------------------------------------------

    print()
    print("=" * 92)
    print("FOUR-WAY GENERATION EVALUATION")
    print("=" * 92)
    print()

    display = (
        four_case_summary[
            [
                "case",
                "faithfulness",
                "factual_correctness",
                "semantic_similarity",
                "answer_relevancy",
            ]
        ]
        .copy()
    )

    for column in [
        "faithfulness",
        "factual_correctness",
        "semantic_similarity",
        "answer_relevancy",
    ]:

        display[
            column
        ] = (
            display[
                column
            ]
            .map(
                lambda value:
                    ""
                    if pd.isna(value)
                    else f"{value:.4f}"
            )
        )

    print(
        display.to_string(
            index=False
        )
    )

    print()
    print("-" * 92)
    print("PAIRED EFFECT OF TF-IDF ROUTING")
    print("-" * 92)
    print()

    effect_display = (
        paired_effects[
            [
                "judge",
                "metric",
                "dense_mean",
                "tfidf_mean",
                "mean_delta_tfidf_minus_dense",
                "bootstrap_95ci_low",
                "bootstrap_95ci_high",
                "improved_questions",
                "worsened_questions",
                "unchanged_questions",
            ]
        ]
        .copy()
    )

    for column in [
        "dense_mean",
        "tfidf_mean",
        "mean_delta_tfidf_minus_dense",
        "bootstrap_95ci_low",
        "bootstrap_95ci_high",
    ]:

        effect_display[
            column
        ] = (
            effect_display[
                column
            ]
            .map(
                lambda value:
                    ""
                    if pd.isna(value)
                    else f"{value:.4f}"
            )
        )

    print(
        effect_display.to_string(
            index=False
        )
    )

    print()
    print("-" * 92)
    print("JUDGE AGREEMENT")
    print("-" * 92)
    print()

    agreement_display = (
        judge_agreement.copy()
    )

    for column in [
        "qwen_mean",
        "gpt56_mean",
        "pearson",
        "spearman",
        "mean_absolute_difference",
        "mean_qwen_minus_gpt56",
        "agreement_within_0.10",
        "agreement_within_0.25",
    ]:

        agreement_display[
            column
        ] = (
            agreement_display[
                column
            ]
            .map(
                lambda value:
                    ""
                    if pd.isna(value)
                    else f"{value:.4f}"
            )
        )

    print(
        agreement_display.to_string(
            index=False
        )
    )

    if (
        "generated_answer_changed"
        in per_question.columns
    ):

        changed = int(
            per_question[
                "generated_answer_changed"
            ]
            .sum()
        )

        print()
        print(
            f"Generated answer changed on "
            f"{changed}/{len(per_question)} questions."
        )

    if (
        "tfidf_mode"
        in tfidf_ragas.columns
    ):

        print()
        print("TF-IDF routing modes:")

        print(
            tfidf_ragas[
                "tfidf_mode"
            ]
            .value_counts(
                dropna=False
            )
            .to_string()
        )

    print()
    print("=" * 92)
    print(f"Results written to: {output_dir.resolve()}")
    print("=" * 92)


if __name__ == "__main__":
    main()

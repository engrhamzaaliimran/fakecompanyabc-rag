# FakeCompanyABC RAG

A fully local Retrieval-Augmented Generation (RAG) practice project built around fictional HR policies. I used the project to revisit RAG from first principles, measure retrieval instead of only looking at final answers, progressively improve the pipeline, and then evaluate the end-to-end system with both a local LLM judge and a stronger secondary judge.. I also wanted to keep the project completely free, so I deliberately avoided paid APIs and stayed with local, open-source tools and models throughout.

All company names, policies, questions, answers, and data in this repository are **completely fictional** and exist only for learning and experimentation.

## Project Status - 25 August 2026

The **classical RAG stage is complete** for this repository.

The project now includes:

- dense vector retrieval with `nomic-embed-text` + ChromaDB
- deterministic retrieval evaluation with exact source evidence
- chunking experiments
- manual retrieval-failure analysis
- TF-IDF policy-file routing before dense search
- a harder 100-question retrieval benchmark
- optional cross-encoder reranking
- grounded answer generation with a deliberately small `qwen3:0.6b` model
- end-to-end generation evaluation with RAGAS + a local Qwen judge
- a second evaluation of the exact same saved answers using GPT-5.6 Sol
- comparison of retrieval-pipeline effects and judge disagreement
- manual validation on 5 questions, where I compared both judges against my own faithfulness and factual-correctness judgments

The next stage of the repository will start at **08** and add **GraphRAG / Neo4j** using the same fictional policies and benchmark questions so that graph-based retrieval can be compared with the existing RAG approaches.

---

## Final Classical RAG Architecture

The practical pipeline selected after the retrieval experiments is:

```text
User question
    -> TF-IDF policy-file routing
    -> nomic-embed-text query embedding
    -> ChromaDB dense retrieval inside candidate policy file(s)
    -> Top-3 policy chunks
    -> qwen3:0.6b grounded generation
    -> final answer
```

A cross-encoder reranker was also benchmarked:

```text
TF-IDF routing
    -> dense Top-5
    -> cross-encoder/ms-marco-MiniLM-L6-v2
    -> final Top-3
```

It produced the best retrieval ranking metrics, but its CPU cost was much higher. For the end-to-end generation experiments I therefore compared **dense-only RAG** against the more practical **TF-IDF + dense RAG** pipeline without the reranker.

### Main Local Stack

| Component | Choice |
|---|---|
| Runtime | Ollama |
| Embeddings | `nomic-embed-text` |
| Vector store | ChromaDB |
| Policy routing | TF-IDF (`scikit-learn`) |
| Optional reranker | `cross-encoder/ms-marco-MiniLM-L6-v2` |
| Final generation model | `qwen3:0.6b` |
| Local generation judge | `qwen2.5:1.5b-instruct` through RAGAS |
| Secondary external judge | `GPT-5.6 Sol` |
| Manual judge check | 5-question human spot-check of faithfulness and factual correctness |
| Main orchestration/interfaces | LangChain |
| Development machine | Ubuntu 22.04.5 LTS, CPU-only (Intel 13th Gen Intel® Core™ i5-13420H, 16GB RAM) |

I intentionally used a very small **0.6B** generator. I originally started with Qwen3 1.7B, but when I later introduced a larger local judge model, the evaluation became too slow and started timing out on my CPU-only machine. I therefore reduced the generator size and adjusted the model selection to keep the full evaluation pipeline practical. At the same time, this gave me an additional experiment: I wanted to see how well the RAG application could perform with a much smaller language model.

---

## Final Results at a Glance

### 1. Retrieval Benchmark - 100 Questions

The final retrieval benchmark uses `retrieval_ground_truth_100_hard.json`: **100 questions**, including **64 hard**, **36 medium**, **51 multi-evidence**, and **19 multi-source** questions.

Question Difficulty

The difficulty labels are specific to this benchmark and are based mainly on how much reasoning and evidence retrieval a question requires. They are not universal definitions of RAG difficulty.

* Easy — A direct lookup from a single policy, usually requiring one piece of evidence and using wording close to the source text.
**Example**: “How many annual leave days does a full-time employee in Germany receive?”
The answer can be found directly in the annual leave policy.
* Medium — Usually still answerable from one main policy, but the question may be paraphrased, scenario-based, involve a threshold, location rule, date, exception, or require combining two pieces of evidence.
**Example**: “An employee has German nationality but their contractual work location is Pakistan. How many annual leave days do they receive?”
The system must understand that the policy uses contractual location, not nationality, and then retrieve the Pakistan leave entitlement.
* Hard — Requires more reasoning, multiple evidence items, cross-policy relationships, temporal distinctions, negative conditions, or distinguishing between very similar rules. Some questions require evidence from more than one policy.
**Example**: “After returning from parental leave, which policy governs remote work and how many remote days per week may a Germany-based employee normally have?”
The system must connect the parental leave policy with the remote work policy and retrieve the applicable Germany rule.

All systems are evaluated at final **Top-3**. The reranker receives Top-5 candidates and reorders them before Top-3 evaluation.

| Metric | Dense Only | TF-IDF + Dense | TF-IDF + Dense + Reranker |
|---|---:|---:|---:|
| Hit@1 | 0.730 | 0.790 | **0.840** |
| Hit@3 | 0.990 | **1.000** | **1.000** |
| Recall@3 | 0.938 | 0.938 | **0.947** |
| Evidence Precision@3 | 0.420 | 0.413 | **0.420** |
| Source Precision@3 | 0.847 | 0.950 | **0.963** |
| MRR@3 | 0.850 | 0.888 | **0.917** |

Final measured CPU timing:

| Stage | Average time |
|---|---:|
| Nomic query embedding | 64.32 ms |
| Dense retrieval after embedding | 1.76 ms |
| TF-IDF + dense after embedding | 2.97 ms |
| TF-IDF + dense + reranker after embedding | 401.08 ms |

The main retrieval conclusion is:

> **TF-IDF + dense gives the best accuracy/speed trade-off:** The reranker gives the best ranking quality, but the additional latency is large enough that I treat it as an optional accuracy-focused stage.

### 2. End-to-End Generation Benchmark - 100 Questions

For generation I used the same 100 questions and the same Top-3 context size.

The generator was fixed to:

```text
qwen3:0.6b
```

The main local evaluator used:

```text
RAGAS
  + qwen2.5:1.5b-instruct as LLM judge
  + nomic-embed-text for semantic answer similarity
```

I then evaluated the **same saved generated answers** with GPT-5.6 Sol as a second judge. GPT-5.6 Sol did not regenerate the answers; it only evaluated the already saved outputs.

#### Final Four-Case Comparison

| Pipeline | Judge | Faithfulness | Factual Correctness | Semantic Similarity | Answer Relevancy |
|---|---|---:|---:|---:|---:|
| Dense | RAGAS + Qwen2.5 | 0.801 | 0.652 | 0.779 | - |
| TF-IDF + Dense | RAGAS + Qwen2.5 | 0.754 | 0.668 | 0.778 | - |
| Dense | GPT-5.6 Sol | 0.831 | 0.753 | - | 0.912 |
| TF-IDF + Dense | GPT-5.6 Sol | **0.848** | **0.775** | - | **0.921** |


The retrieval pipeline changed retrieved contexts for **36/100 questions** and changed the generated answer for **32/100 questions**. TF-IDF routing selected candidate policy files for **92 questions**, used ambiguous-signal full-corpus fallback for **7**, and weak-signal full-corpus fallback for **1**. All these results commparison can be done with 07.1_compare_four_generation_evaluations.py present in ./evaluation/results

The paired improvements are modest and the two judges differ in their assessments. I therefore treat these generation numbers as **diagnostic evidence**, not as an absolute measure of truth. The judge-comparison experiment is useful because it shows why an LLM-as-a-judge should itself be checked rather than directly trusted. I later checked few questions where there was a complete disagreement in both judge LLMs.


### Manual Judge Validation - 5 Questions

The local and external judges do **not** agree perfectly. I manually checked **5 representative questions**.

For each of those 5 questions, I looked at:

- the user question
- the retrieved context
- the generated answer
- the reference answer
- the **faithfulness** judgment
- the **factual correctness** judgment

I then compared my own manual judgment with the two automated judges:

```text
Local judge:
qwen2.5:1.5b-instruct
used through RAGAS

Secondary judge:
GPT-5.6 Sol
```

In this small manual spot-check, **GPT-5.6 Sol aligned much better with my own judgments than `qwen2.5:1.5b-instruct`**.

This does not prove that GPT-5.6 Sol is always a perfect judge. The manual sample is small, only 5 questions. However, it gave me a practical reason to trust the stronger judge more for interpretation of the final generation results. Therefore, I would say **The quality of an LLM-as-a-judge matters. A small judge can itself become an evaluation bottleneck.**

---

## What I Consider Finished

At this point the normal RAG work in this repository covers the complete path:

```text
Dense baseline
    -> deterministic retrieval ground truth
    -> chunk-size evaluation
    -> manual failure analysis
    -> TF-IDF policy routing
    -> harder 100-question benchmark
    -> reranker comparison
    -> grounded generation
    -> local RAGAS evaluation
    -> secondary stronger-judge evaluation
    -> judge/pipeline comparison
```

The sections below keep the full progression and intermediate experiments.

---

# Detailed Project Progression


## Initial Setup

The project started with a fully local, open-source stack because I wanted to understand each RAG component separately before putting everything together.

The first source files were:

| File | Purpose |
|---|---|
| `01_test_embedding.py` | Check that `nomic-embed-text` is installed and returns embeddings correctly. |
| `02_load_and_chunk.py` | Load the Markdown dataset and experiment with chunking. |
| `03_index_chroma.py` | Chunk the documents, generate embeddings, and persist them in ChromaDB. |
| `04_retrieve.py` | Run a simple fixed-query Top-3 dense retrieval test. No benchmark and no LLM at this stage. |
| `05_evaluate_retrieval.py` | Evaluate the first dense retriever using the initial ground truth. |
| `05.1_evaluate_retrieval_chunking.py` | Compare different chunk-size and overlap configurations. |
| `05.2_analyze_retrieval.py` | Print ground truth and actual Top-3 chunks for manual failure analysis. |
| `05.3_tdidf_retrieval.py` | Add the TF-IDF policy-file prefilter before dense retrieval. |
| `05.4_tfidf_prefilter_evaluate_updated.py` | Evaluate the improved TF-IDF fallback/filtering logic. |
| `05.5_benchmark_three_retrieval_systems.py` | Compare dense-only, TF-IDF + dense, and TF-IDF + dense + reranker on the 100-question benchmark. |
| `06_generate_answer.py` | Interactive grounded RAG answer generation with the deliberately small `qwen3:0.6b` model. |
| `07_evaluate_rag_ragas.py` | Run the 100-question dense-RAG generation baseline and evaluate it with a local RAGAS/Qwen judge. |
| `07_evaluate_rag_tfidf_ragas.py` | Run the same 100-question generation benchmark with TF-IDF policy routing before dense retrieval. |

The document embedding and query embedding use the Nomic retrieval prefixes:

```text
Documents/chunks -> search_document:
Questions        -> search_query:
```

The prefixes are only supplied to the embedding model. The clean chunk text is what is stored in Chroma.

---

## Evaluation 1 - Initial Dense Retrieval

Before adding the LLM, I wanted to know whether the retriever itself was actually finding the required information.

### Initial Ground Truth

I created the first ground truth with the help of ChatGPT using the fictional FakeCompanyABC policy files. It contained questions at easy, medium and hard levels.

For each question, I stored:

- the question
- expected answer
- source file
- relevant section
- exact supporting text

I also manually checked a sample of the questions against the source policy files.

The purpose of the ground truth was to evaluate retrieval before generation. The first metrics were:

- **Hit@K** - whether at least one relevant chunk appears inside Top-K
- **Recall@K** - how much of the required evidence was found
- **Precision@K** - how many returned chunks contained the exact required evidence
- **MRR** - how early the first relevant chunk appeared

### Initial Results

| Metric | K=1 | K=3 | K=5 |
|---|---:|---:|---:|
| Hit@K | 0.800 | 1.000 | 1.000 |
| Recall@K | 0.775 | 1.000 | 1.000 |
| Precision@K | 0.800 | 0.367 | 0.220 |
| MRR | - | - | 0.883 |

The required information was normally retrieved by Top-3, but precision dropped as K increased because additional chunks were returned.

This gave me a working baseline, but I did not want to move directly to the LLM without understanding why the extra retrieved chunks were often not useful.

---

## Evaluation 2 - Chunking

I compared multiple chunk sizes using `05.1_evaluate_retrieval_chunking.py`.

| Size | Overlap | Chunks | Hit@1 | Recall@1 | Hit@3 | Recall@3 | Prec@3 | MRR@5 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 400 | 80 | 38 | 0.800 | 0.725 | 1.000 | 0.950 | 0.383 | 0.892 |
| **700** | **120** | **22** | **0.800** | **0.775** | **1.000** | **1.000** | **0.367** | **0.883** |
| 1000 | 150 | 16 | 0.700 | 0.650 | 1.000 | 1.000 | 0.367 | 0.825 |

I selected **700 chunk size with 120 overlap**.

The 400/80 configuration had slightly better evidence precision and MRR, but 700/120 achieved complete Recall@3 while keeping the number of chunks smaller. The 1000/150 chunks were too broad and reduced first-rank performance.

At this point the main question became:

```text
The required evidence is usually present.
Why are irrelevant chunks still entering Top-3?
```

---

## Evaluation 3 - Manual Retrieval Failure Analysis

Instead of directly adding reranking, I created `05.2_analyze_retrieval.py` to inspect the actual retrieved chunks beside the ground truth.

The repeated pattern was:

```text
Dense retrieval
  -> required evidence is usually found
  -> additional chunks sometimes come from unrelated policy files
```

Examples included Annual Leave questions retrieving Parental Leave chunks and Expense questions retrieving unrelated policy chunks.

So the problem was not simply "semantic retrieval is failing".

A better description was:

```text
Dense retrieval searches the whole corpus
  -> semantically related policy language overlaps
  -> wrong-policy chunks can enter Top-3
```

This suggested improving the **search space** before trying to improve chunk ranking.

---

## Evaluation 4 - TF-IDF Policy Prefilter

I wanted a lightweight policy-selection stage without manually writing routing rules such as:

```text
receipt     -> expense_policy.md
remote work -> remote_work.md
hotel       -> travel_policy.md
```

I therefore used TF-IDF over the complete policy files.

TF-IDF is used only for **policy-level candidate selection**. Nomic embeddings are still responsible for semantic chunk retrieval.

The flow became:

```text
Question
  -> TF-IDF scores policy files
  -> select likely policy file(s), or fall back to full corpus
  -> dense semantic retrieval inside that search space
  -> Top-3
```

### TF-IDF Configuration

```text
MIN_LEXICAL_SCORE = 0.08
MAX_AMBIGUOUS_SCORE_SPREAD = 0.05
RELATIVE_SCORE_THRESHOLD = 0.55
MAX_CANDIDATE_FILES = 2
```

The vectorizer uses:

```text
lowercase = True
stop_words = english
ngram_range = (1, 2) - use both single words and two-word phrases as features
max_df = 0.8
```

The current logic is:

```text
1. Score every policy file with TF-IDF.

2. If the best score is below 0.08:
   -> do not filter
   -> use full dense retrieval.

3. If TF-IDF scores are too similar:
   best score - lowest score <= 0.05
   -> do not filter
   -> use full dense retrieval.

4. Otherwise keep files scoring at least 55% of the best score.

5. Keep at most two policy files.

6. Run Nomic + Chroma retrieval inside the selected search space.
```

The thresholds are currently heuristics. They are not probabilities and were not learned by TF-IDF.

Keeping up to two policy files is important because some questions legitimately need more than one policy.

---

## Evaluation 5 - What TF-IDF Actually Improved

On the first 20-question benchmark, evidence-level metrics alone did not show the full improvement.

| Metric | Dense Only | TF-IDF + Dense |
|---|---:|---:|
| Hit@3 | 1.000 | 1.000 |
| Recall@3 | 1.000 | 1.000 |
| Evidence Precision@3 | 0.367 | 0.367 |
| Source Precision@3 | ~0.750 | ~0.967 |
| Wrong-policy chunks in Top-3 | 15 / 60 | 2 / 60 |
| Questions containing wrong-policy chunks | 11 / 20 | 1 / 20 |

Wrong-policy chunks dropped from **15/60 to 2/60**, around an **87% reduction**.

A simplified example:

| Question | Dense Only | TF-IDF + Dense |
|---|---|---|
| Q01 | Annual + Annual + **Parental** | Annual + Annual + Annual |
| Q02 | Annual + Annual + **Parental** | Annual + Annual + Annual |
| Q03 | Annual + **Parental** + Annual | Annual + Annual + Annual |
| Q04 | Annual + **Parental + Parental** | Annual + Annual + Annual |
| Q06 | Expense + **Parental + Parental** | Expense + Expense + Expense |

### Why Evidence Precision@3 Did Not Improve

For many of the original questions, only one chunk contains the exact required evidence.

A good result can therefore be:

```text
R1 -> exact required evidence
R2 -> correct policy, different section
R3 -> correct policy, different section
```

Evidence Precision@3 is still:

```text
1 / 3 = 0.333
```

A worse result can be:

```text
R1 -> exact required evidence
R2 -> wrong policy
R3 -> wrong policy
```

and the same metric is still:

```text
1 / 3 = 0.333
```

This is why I added **Source Precision@3**.

Source Precision asks whether the final chunks come from policy files that are legitimately relevant to the question. It measures policy-space cleanliness, while evidence precision remains stricter and checks exact answer evidence.

An important example was a question where the Expense Policy points the employee to the Travel Policy for hotel limits. A Travel chunk can be logically useful even when the exact evidence sentence in the ground truth is stored in Expense. This later motivated adding `relevant_sources` to the harder benchmark.

---

## Evaluation 6 - Ranking Before Adding a Reranker

After TF-IDF improved file selection, I checked where the first exact evidence appeared in the original 20-question Top-3.

| First correct evidence rank | Questions | Count |
|---|---|---:|
| R1 | Q01-Q06, Q08, Q10-Q15, Q17, Q19, Q20 | 16 / 20 |
| R2 | Q07, Q18 | 2 / 20 |
| R3 | Q09, Q16 | 2 / 20 |

| Ranking measure | Result |
|---|---:|
| Correct evidence at R1 | 16 / 20 = 80% |
| Correct evidence within R2 | 18 / 20 = 90% |
| Correct evidence within R3 | 20 / 20 = 100% |
| MRR@3 | 0.883 |

This showed that the 20-question benchmark had become too easy for evaluating a reranker. The correct evidence was already inside Top-3 for every question.

Instead of tuning a reranker on this small set and declaring success, I created a larger, harder benchmark.

---

## Evaluation 7 - Harder 100-Question Ground Truth

The new file is:

`evaluation/retrieval_ground_truth_100_hard.json`

The 100-question ground truth was created **with the help of ChatGPT** using the fictional policy documents as the source.

It contains:

| Property | Count |
|---|---:|
| Questions | 100 |
| Medium | 36 |
| Hard | 64 |
| Multi-evidence questions | 51 |
| Multi-source questions | 19 |

The questions include paraphrases, scenarios, threshold boundaries, negative conditions, location rules, approval conditions, current-versus-historical values and cross-policy questions.

The ground truth does **not** contain chunk IDs. It is designed to remain independent of:

- chunk size
- chunk overlap
- splitter implementation
- embedding model
- vector database
- Top-K
- reranker

For each question, the benchmark can contain:

```text
id
category
difficulty
challenge_types
question
expected_answer
relevant_sources
evidence
```

Each evidence item stores its source, section and exact supporting text.

Every `evidence.exact_text` string was programmatically verified as a verbatim substring of its named policy Markdown file.

The `expected_answer` is **not used to score retrieval**. It is stored for interpretation and for the later generation benchmark.

### Evidence Relevance vs Source Relevance

The harder benchmark separates two concepts:

```text
Evidence relevance:
Does this chunk contain exact required evidence?

Source relevance:
Does this chunk come from a policy that is legitimately relevant to the question?
```

This is why the final benchmark reports both **Evidence Precision@3** and **Source Precision@3**.

This benchmark is synthetic/ChatGPT-assisted. Exact evidence existence was verified programmatically, but that is different from claiming that every semantic label has been independently human-validated.

---

## Evaluation 8 - Cross-Encoder Reranking

After policy-file routing was improved, the next target was **chunk-level ranking**.

I tested:

`cross-encoder/ms-marco-MiniLM-L6-v2`

The reranker does **not** select policy files and does **not** retrieve new chunks.

Its job is only:

```text
question + already retrieved candidate chunks
  -> score each (question, chunk) pair
  -> reorder the candidates
```

The reranker system retrieves five dense candidates because it needs a larger pool than the final evaluation set:

```text
TF-IDF policy selection
  -> Nomic + Chroma Top-5
  -> cross-encoder reranking
  -> keep final Top-3
  -> evaluate at K=3
```

This distinction is important. If the correct policy was filtered out before dense retrieval, the reranker cannot recover it.

---

## Evaluation 9 - Final Three-System Retrieval Benchmark

The final benchmark compares three systems using the **same 100 questions** and the **same Nomic query embedding per question**.

### System A - Dense Only

```text
Question -> Nomic -> Chroma full corpus -> Top-3
```

### System B - TF-IDF + Dense

```text
Question -> TF-IDF policy selection -> Nomic/Chroma -> Top-3
```

### System C - TF-IDF + Dense + Reranker

```text
Question -> TF-IDF policy selection -> Nomic/Chroma Top-5
         -> cross-encoder reranking -> final Top-3
```

All final metrics are evaluated at **K=3**.

### Results

| Metric | Dense Only | Dense + TF-IDF | Dense + TF-IDF + Reranker |
|---|---:|---:|---:|
| Hit@1 | 0.730 | 0.790 | **0.840** |
| Hit@3 | 0.990 | **1.000** | **1.000** |
| Recall@3 | 0.938 | 0.938 | **0.947** |
| Evidence Precision@3 | 0.420 | 0.413 | **0.420** |
| Source Precision@3 | 0.847 | 0.950 | **0.963** |
| MRR@3 | 0.850 | 0.888 | **0.917** |

### TF-IDF Prefilter Behaviour

| Behaviour | Questions |
|---|---:|
| Filtered normally | 99 |
| Weak-signal fallback | 1 |
| Ambiguous fallback | 0 |

### Reranker Rank Changes

| Change in first relevant rank | Questions |
|---|---:|
| Improved | 11 |
| Worsened | 7 |
| Unchanged | 82 |

The reranker therefore improves the aggregate ranking, but it is not automatically better for every query.

---

## Understanding the Final Metrics

### TF-IDF Effect

The clearest policy-routing improvement is:

```text
Source Precision@3
Dense Only       = 0.847
TF-IDF + Dense   = 0.950
```

TF-IDF is responsible for this main policy-level improvement because it restricts the dense search space.

It also improves ranking:

```text
Hit@1: 0.730 -> 0.790
MRR@3: 0.850 -> 0.888
```

### Reranker Effect

The reranker further improves:

```text
Hit@1: 0.790 -> 0.840
MRR@3: 0.888 -> 0.917
Recall@3: 0.938 -> 0.947
```

Source Precision@3 also increases slightly:

```text
0.950 -> 0.963
```

This does **not** mean the reranker is selecting better policy files.

Source Precision is calculated on the **final reranked Top-3**. If TF-IDF selected two candidate policies and dense retrieval returned chunks from both, the reranker can move chunks from the more relevant source into the final Top-3 and push other chunks below rank 3.

So:

```text
TF-IDF -> policy/file-level routing
Dense retrieval -> candidate chunk retrieval
Cross-encoder -> candidate chunk ordering
Evaluator -> judges the final Top-3
```

---

## CPU Latency Trade-Off

Because this project runs on CPU, I measured the cost of each design.

The query embedding is calculated once and reused across the three systems.

| Stage | Average time |
|---|---:|
| Nomic query embedding | 64.32 ms |
| Dense only after embedding | 1.76 ms |
| TF-IDF + dense after embedding | 2.97 ms |
| TF-IDF + dense + reranker after embedding | 401.08 ms |

Approximate end-to-end retrieval time including the shared query embedding is therefore around:

```text
Dense only               ~66 ms
TF-IDF + dense           ~67 ms
TF-IDF + dense + rerank ~465 ms
```

The TF-IDF stage gives a large policy-routing improvement for almost no additional latency.

The cross-encoder gives the best ranking quality, but at a much larger CPU cost.

For the current small corpus, **TF-IDF + dense is my preferred accuracy/speed configuration**, while the reranker remains an optional accuracy-focused stage.

---

## Current Retrieval Architecture

The retrieval work now has clear responsibilities:

| Component | Responsibility |
|---|---|
| TF-IDF | Decide which policy file(s) are likely worth searching. |
| Nomic | Convert the query into a semantic embedding. |
| ChromaDB | Retrieve candidate chunks from the allowed search space. |
| Cross-encoder | Optionally reorder already retrieved chunks. |
| Retrieval evaluator | Measure exact evidence retrieval, source quality and ranking. |

Current practical flow:

```text
Question
  -> TF-IDF policy routing
  -> Nomic embedding
  -> Chroma dense retrieval
  -> optional cross-encoder reranking
  -> Top-3 context
```

---

## Retrieval Conclusion

The main lesson from the retrieval stage is that adding more RAG components is not automatically an improvement.

The progression was driven by observed failures:

```text
Dense baseline
  -> measure retrieval
  -> test chunking
  -> inspect failures
  -> identify wrong-policy noise
  -> add TF-IDF policy routing
  -> build a harder benchmark
  -> test reranking
  -> compare accuracy and latency
```

The final benchmark shows:

- **Dense only** is a useful simple baseline.
- **TF-IDF + dense** gives the strongest practical accuracy/speed trade-off on the current CPU-only setup.
- **TF-IDF + dense + reranker** gives the best ranking metrics, but the extra CPU latency is substantial.

At this point, I consider the retrieval side sufficiently benchmarked for the first version.

---

## Evaluation 10 - Grounded Generation with a Small Model

After retrieval was sufficiently benchmarked, I added the generation stage.

For the final classical-RAG experiments I deliberately used:

```text
qwen3:0.6b
```

This is a weak/small local model compared with common production LLMs. That was intentional for two reasons:

1. the project runs on a CPU-only machine, so model size matters;
2. a small generator makes retrieval and grounding quality more visible instead of allowing a large model to answer from its own background knowledge.

The interactive generation flow in `06_generate_answer.py` is:

```text
Question
  -> TF-IDF policy routing
  -> Nomic query embedding
  -> Chroma Top-3
  -> build labelled context
  -> qwen3:0.6b
  -> grounded answer
```

The generation prompt restricts the assistant to the supplied policy context, asks it not to invent rules or numbers, asks it to abstain when context is insufficient, and avoids silently assuming missing conditions such as location, employment type, dates, or approval status.

Following is the prompt: 
"You are an HR assistant for the fictional company FakeCompanyABC.

Answer the user's question using only the information in the provided context.

Rules:
- Do not use outside knowledge.
- Do not invent company rules, numbers, conditions, dates, approvals, or exceptions.
- If the context does not contain enough information to answer the question,
  say: "I cannot answer this from the provided company policies."
- Keep the answer concise but include all information needed to answer the question.
- Do not assume conditions that the user did not provide,
  such as employment type, location, approval status, or dates.
- If the answer depends on a missing condition, explain the
  condition rather than choosing one silently.
- If the question is ambiguous and the context contains
  multiple possible interpretations, say what needs to be
  clarified."
---

## Evaluation 11 - Dense RAG Generation Baseline

`07_evaluate_rag_ragas.py` runs the full 100-question benchmark using dense retrieval without TF-IDF routing.

```text
Question
  -> Nomic embedding
  -> Chroma full-corpus dense Top-3
  -> qwen3:0.6b generation
  -> evaluation
```

The local evaluator uses `qwen2.5:1.5b-instruct` through RAGAS.

The evaluated local metrics are:

- **Faithfulness** - whether generated claims are supported by retrieved context.
- **Factual Correctness** - whether the generated answer matches the reference answer semantically/factually.
- **Semantic Similarity** - Nomic embedding cosine similarity between generated and reference answers.

The final dense baseline used in the four-way comparison produced:

| Metric | Dense RAG |
|---|---:|
| Faithfulness | 0.801 |
| Factual Correctness | 0.652 |
| Semantic Similarity | 0.779 |

The evaluator is designed conservatively for local CPU execution: individual RAGAS metrics are evaluated separately, parser failures can be retried, results are checkpointed after each question, and interrupted runs can be resumed.

---

## Evaluation 12 - TF-IDF + Dense End-to-End RAG

`07_evaluate_rag_tfidf_ragas.py` repeats the generation benchmark with the selected policy-routing strategy in front of dense retrieval.

```text
Question
  -> TF-IDF over whole policy files
  -> candidate policy file(s)
  -> Nomic query embedding
  -> Chroma dense Top-3 restricted to candidates
  -> qwen3:0.6b generation
  -> local RAGAS evaluation
```

The routing configuration is the same idea used in the retrieval benchmark:

```text
MIN_LEXICAL_SCORE = 0.08
MAX_AMBIGUOUS_SCORE_SPREAD = 0.05
RELATIVE_SCORE_THRESHOLD = 0.55
MAX_CANDIDATE_FILES = 2
```

Weak or ambiguous lexical signals fall back to full-corpus dense retrieval rather than forcing a potentially wrong policy filter.

The 100-question generation run produced:

| Metric | TF-IDF + Dense RAG |
|---|---:|
| Faithfulness - local Qwen judge | 0.754 |
| Factual Correctness - local Qwen judge | 0.668 |
| Semantic Similarity - Nomic | 0.778 |

The important point is not that every metric increased. TF-IDF changed the retrieval context for only part of the benchmark, and the small local judge itself has limitations. This motivated a second-judge comparison.

---

## Evaluation 13 - Secondary GPT-5.6 Sol Judge

I kept the generated answers fixed and evaluated the same 100 dense-RAG answers and the same 100 TF-IDF+dense-RAG answers again with **GPT-5.6 Sol**.

The second judge scored:

- faithfulness to retrieved context
- factual correctness against the reference answer
- answer relevancy to the original question

This is a **secondary external evaluation**. It is included to study judge agreement, not to claim that the external judge is infallible.

### GPT-5.6 Sol Results

| Metric | Dense | TF-IDF + Dense |
|---|---:|---:|
| Faithfulness | 0.831 | **0.848** |
| Factual Correctness | 0.753 | **0.775** |
| Answer Relevancy | 0.912 | **0.921** |

Under this stronger judge, TF-IDF + dense improves all three scored generation metrics slightly.


### Manual Spot-Check of the Two Judges

I also manually reviewed **5 questions** and compared both automated judges against my own assessment for:

- faithfulness to the retrieved context
- factual correctness against the reference answer

The judges were:

```text
Local RAGAS judge:
qwen2.5:1.5b-instruct

Secondary external judge:
GPT-5.6 Sol
```

On this 5-question human spot-check, **GPT-5.6 Sol matched my manual judgments much better than the local `qwen2.5:1.5b-instruct` judge**.

I therefore treat the local judge as the reproducible baseline evaluator, but I place more confidence in GPT-5.6 Sol when interpreting nuanced generation-quality cases. I still keep both sets of results because the disagreement itself is useful evidence about evaluator quality.

The GPT-5.6 Sol files are saved evaluation artifacts. They are **not part of the fully local reproducible stack**. The local Qwen/RAGAS evaluation remains the reproducible judge pipeline.

---

## Evaluation Artifacts

The repository stores both aggregate and per-question results so that the final averages can be traced back to individual questions.

Important result files include:

```text
evaluation/
  retrieval_ground_truth_simple.json
  retrieval_ground_truth_100_hard.json
  results/
        07.1_compare_four_generation_evaluations.py
        generation_eval_gen_qwen3_0.6b_judge_qwen2.5_1.5b-instruct_k3.csv
        generation_eval_tfidf_dense_gen_qwen3_0.6b_judge_qwen2.5_1.5b-instruct_k3.csv
        gpt56_sol_generation_evaluation_100.csv
        gpt56_sol_tfidf_dense_generation_100.csv
```

---

## Final Classical RAG Conclusion

This project started as a small RAG refresher, but the most useful part became the evaluation process.

The main lessons were:

- retrieval should be evaluated before blaming the generator;
- a simple routing layer can be more cost-effective than immediately adding a heavier reranker;
- exact evidence metrics and source-level metrics reveal different retrieval problems;
- a reranker can improve ranking while still being unattractive on CPU because of latency;
- a very small generator can still produce useful grounded answers when context is good;
- end-to-end generation scores can change when retrieval changes even if retrieval recall looks similar;
- LLM-as-a-judge evaluation is useful, but judge disagreement must be treated as part of the result; in a 5-question manual spot-check, `GPT-5.6 Sol` aligned much better with my judgments than `qwen2.5:1.5b-instruct`.

The final classical RAG experiments are therefore preserved as a baseline for the next stage.

---

## Next Stage - GraphRAG / Neo4j

The next work in this same repository will add GraphRAG rather than starting a separate project.

The goal is to use the same fictional HR policies and, where appropriate, the same benchmark questions so that the comparison is meaningful.

The intended comparison becomes:

```text
Dense RAG
    vs
TF-IDF + Dense RAG
    vs
GraphRAG / Neo4j
```

GraphRAG work will begin with the **08** source-file series.

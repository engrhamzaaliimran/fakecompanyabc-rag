# FakeCompanyABC RAG

A small local Retrieval-Augmented Generation (RAG) project built for hands-on practice and to refresh practical RAG concepts.

All company names, HR policies, documents, and data used in this repository are **completely fictional** and created only for learning and experimentation.

## Project Status - 23 August 2026

At this point, I have completed the **retrieval side** of the first version of this project and benchmarked several retrieval designs before moving to LLM generation.

The current local stack is:

- **Ollama** as the local model runtime
- **nomic-embed-text** for query/document embeddings
- **ChromaDB** as the persistent vector store
- **TF-IDF** as a lightweight policy-file prefilter
- **cross-encoder/ms-marco-MiniLM-L6-v2** as an optional chunk reranker
- **Qwen3 1.7B** as the local generation model for the next stage
- **LangChain** for model/document interfaces
- **CPU-only Ubuntu machine**, so latency and model size matter

The indexed corpus currently contains the fictional FakeCompanyABC HR policies. With the selected chunking configuration of **700 characters with 120 overlap**, the current Chroma index contains **22 chunks**.

The retrieval work progressed in stages:

1. Started with simple dense retrieval using Nomic embeddings and Chroma.
2. Built an initial 20-question ground truth and added Hit@K, Recall@K, Precision@K and MRR evaluation.
3. Compared multiple chunk sizes and selected 700/120.
4. Manually inspected retrieval failures instead of immediately adding more models.
5. Found that the main problem was often **wrong-policy chunks entering the Top-3**, even when the required evidence was already retrieved.
6. Added a TF-IDF policy-file prefilter to reduce the search space before dense retrieval.
7. Added **Source Precision@3** because exact-evidence Precision@3 alone could not show the improvement in policy-level cleanliness.
8. Created a harder **100-question ground-truth benchmark with the help of ChatGPT** using the fictional policy files as the source.
9. Added a cross-encoder reranker and compared three complete retrieval systems on the same 100 questions.
10. Measured both retrieval quality and CPU latency before deciding which design is practically useful.

The harder 100-question benchmark contains **36 medium and 64 hard questions**, including **51 multi-evidence** and **19 multi-source** questions. Every stored `exact_text` evidence string was programmatically checked to exist verbatim in its named Markdown policy file. The benchmark is source based rather than chunk-ID based, so it is not tied to one specific chunking or retrieval configuration.

### Current Main Benchmark

All three systems are evaluated at **Top-3**. The reranker receives **Top-5 dense candidates** and chooses the final Top-3.

| Metric | Dense Only | Dense + TF-IDF | Dense + TF-IDF + Reranker |
|---|---:|---:|---:|
| Hit@1 | 0.730 | 0.790 | **0.840** |
| Hit@3 | 0.990 | **1.000** | **1.000** |
| Recall@3 | 0.938 | 0.938 | **0.947** |
| Evidence Precision@3 | 0.420 | 0.413 | **0.420** |
| Source Precision@3 | 0.847 | 0.950 | **0.963** |
| MRR@3 | 0.850 | 0.888 | **0.917** |

On this CPU-only setup, the average timings were:

| Stage | Average time |
|---|---:|
| Nomic query embedding | 69.69 ms |
| Dense retrieval after embedding | 1.55 ms |
| TF-IDF + dense after embedding | 2.89 ms |
| TF-IDF + dense + reranker after embedding | 403.14 ms |

My current conclusion is that **TF-IDF + dense retrieval gives the best accuracy/speed trade-off** for this small local corpus. The reranker gives the best ranking numbers, especially Hit@1 and MRR, but adds a large CPU cost. I therefore consider the reranker useful as an optional accuracy-focused stage rather than something that must always be enabled.

The current retrieval flow is:

```text
Question
  -> TF-IDF policy-file prefilter
  -> Nomic query embedding
  -> Chroma dense retrieval
  -> optional cross-encoder reranking
  -> final Top-3 context
```

**The complete progression of the project, each retrieval evaluation, the reasoning behind every change, and the intermediate results are documented below.**

The next major stage is **generation**: pass the retrieved context to Qwen3 1.7B, constrain the model to the retrieved evidence, and then evaluate generation separately from retrieval.

---

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
ngram_range = (1, 2)
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
| Nomic query embedding | 69.69 ms |
| Dense only after embedding | 1.55 ms |
| TF-IDF + dense after embedding | 2.89 ms |
| TF-IDF + dense + reranker after embedding | 403.14 ms |

Approximate end-to-end retrieval time including the shared query embedding is therefore around:

```text
Dense only               ~71 ms
TF-IDF + dense           ~73 ms
TF-IDF + dense + rerank ~473 ms
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

## Current Conclusion

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

## Next Step - Generation

The next stage is to pass the retrieved Top-3 context to **Qwen3 1.7B** and evaluate the generation stage separately.

The planned generation evaluation will focus on questions such as:

```text
Was the correct evidence retrieved?
Did the LLM use only that evidence?
Did it answer the question correctly?
Did it invent unsupported information?
Did it include all required evidence for multi-evidence questions?
Can it abstain when the retrieved context does not contain the answer?
```

This separation is important because a final wrong answer can come from two different failures:

```text
Wrong/missing context -> retrieval failure
Correct context but wrong answer -> generation failure
```

The retrieval benchmark in this README is intended to make that distinction measurable before moving further into the generation side of RAG.
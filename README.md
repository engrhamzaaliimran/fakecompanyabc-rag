# FakeCompanyABC RAG

A small local Retrieval-Augmented Generation (RAG) project built for hands-on practice and to refresh practical RAG concepts.

All company names, HR policies, documents, and data used in this repository are **completely fictional** and created only for learning and experimentation.

## Initial Setup

The project starts with a fully local, open-source stack:

* **Ollama** — local model runtime

* **Qwen3 1.7B** — generative language model

* **nomic-embed-text** — embedding model

* **Langchain** — RAG framework

My Ubuntu machine is **CPU-only**, so I intentionally started with relatively lightweight models that can run locally without a GPU.

I started by revisiting simple RAG concepts. Then progressively kept building up. The following code files do the following:

- 01_test_embedding.py -- Just check if nomic-embed-text is installed and working

- 02_load_and_chunk.py -- Just load the dataset and chunk in 700 size with 120 as overlap. Initial first chunking process.

- 03_index_chroma.py -- Loads the chunks generated in the above step into a ChromaDB database. It has the chunking code in it. The reason for keeping it this way was that if someone uses this project for learning, he/she should learn each step with clean code.

- 04_retrieve.py -- here I fetch the top-3 chunks based on fixed string. So, at this step, there is no benchmarking and no LLM.

## Benchmarking Retrieval

Since, in the stable state of source file 04, the retrieval was tested and it was working. Before moving towards the LLM. Once needs to benchmark the retrieval, and I did so after this stage.

### Ground Truth

I created a ground truth with the help of ChatGPT. I asked it to make questions from the FakeCompanyABC policy files with easy, medium and hard levels. For each question, the JSON file also has the expected answer, the file where the answer should be, the related section, and the exact text where the answer is written.

For the levels, I used a simple idea:

Easy: the answer is directly written in one place, like number of leave days or hotel limit.

Medium: the answer needs some more understanding of the rule or condition, for example whether manager approval is enough or which policy should be used.

Hard: the answer can need information from more than one place, or there can be similar values that can confuse retrieval, for example current and old hotel limits.

I manually checked a few questions by opening the respective files and confirming the answers and exact text.

The idea is to use this ground truth to test the retriever before adding the LLM. I want to see if the correct information is retrieved and how high it appears in the results. For this I plan to use Hit@K, Recall@K, Precision@K and MRR.

### Results

| Metric | K=1 | K=3 | K=5 |
|---|---:|---:|---:|
| Hit@K | 0.800 | 1.000 | 1.000 |
| Recall@K | 0.775 | 1.000 | 1.000 |
| Precision@K | 0.800 | 0.367 | 0.220 |
| MRR | - | - | 0.883 |

The results make sense because increasing `K` retrieves more chunks, so the chance of finding all relevant information increases. After `K=3`, recall does not improve further, while precision drops because more irrelevant chunks are also retrieved.

At this stage, I have a baseline retrieval system with reasonably good results. Next, I want to experiment with a few hyperparameters and retrieval strategies, such as different chunking approaches, metadata filtering, and reranking, to see how much I can improve the performance. The dataset is relatively simple, so extensive optimization may not be necessary, but since this is a practice project, I want to get my hands dirty :), with these techniques and understand their impact.

## Chunking Evaluation

Different chunk sizes and overlaps were evaluated using `05.1_evaluate_retrieval_chunking.py`.

| Size | Overlap | Chunks | Hit@1 | Recall@1 | Hit@3 | Recall@3 | Prec@3 | MRR@5 |
| ------: | ------: | -----: | --------: | --------: | --------: | --------: | --------: | --------: |
| 400 | 80 | 38 | 0.800 | 0.725 | 1.000 | 0.950 | 0.383 | 0.892 |
| **700** | **120** | **22** | **0.800** | **0.775** | **1.000** | **1.000** | **0.367** | **0.883** |
| 1000 | 150 | 16 | 0.700 | 0.650 | 1.000 | 1.000 | 0.367 | 0.825 |

I selected **700 chunk size with 120 overlap** as the current winner. It gives complete retrieval coverage at Top-3 (`Hit@3 = 1.0`, `Recall@3 = 1.0`) with good ranking performance (`MRR@5 = 0.883`).

The main area to improve now is `Precision@3 = 0.367`, which means the required information is retrieved, but some irrelevant chunks are also included.

## Rethinking Retrieval

My next target was therefore to improve `Precision@3`.

Before directly changing the retrieval pipeline, I manually inspected all 20 questions together with their ground truth and actual Top-3 retrieved chunks using `05.2_analyze_retrieval.py`.

This made the problem much clearer.

```text

Dense Retrieval

      ↓

Required evidence was usually found 

      ↓

But chunks from unrelated policy files

were also entering the Top-3 results 

```

For example, some Annual Leave questions were retrieving chunks from the Parental Leave Policy, while some Expense questions were also retrieving unrelated policy files.

So the first target became:

```text

Reduce retrieval from unrelated files

              ↓

Give dense retrieval a cleaner search space

```

Initially, I was thinking about metadata filtering. However, instead of manually defining which query should search which policy file, I wanted to see if the correct policy could be identified from the words present in the documents themselves.

This led me towards a **Bag-of-Words style approach**.

A normal Bag-of-Words approach treats words mostly based on their occurrence, but some words are much more useful than others. Common words can appear in many policy documents while some words or phrases are much more specific to one policy.

This led me to **TF-IDF (Term Frequency-Inverse Document Frequency)**.

In simple terms:

```text

TF  → How important is a word inside one document?

IDF → How uncommon is that word across all documents?

TF-IDF = TF × IDF

```

So instead of manually creating rules like:

```text

receipt     → expense_policy.md

remote work → remote_work.md

hotel       → travel_policy.md

```

TF-IDF learns useful words and phrases directly from the Markdown policy files.

## TF-IDF + Dense Retrieval

The TF-IDF prefiltering and two-stage retrieval approach is implemented and evaluated in `05.3_tdidf_retrieval.py`.

I then changed the retrieval flow to a two-stage approach:

```text

                         Question

                            │

                            ▼

                     TF-IDF Search

                            │

                  strong lexical match?

                      /           \

                    YES            NO

                     │              │

                     ▼              ▼

             Candidate policy    Full corpus

                  files

                     │              │

                     └──────┬───────┘

                            ▼

                     Dense Retrieval

                  using Nomic embeddings

                            │

                            ▼

                          Top-3

```

First, TF-IDF compares the question with all policy files and returns a similarity score for each file.

I added a small lexical-confidence policy before applying the file filter.

The current logic is:

```text

TF-IDF scores all policy files

        ↓

Is the best score below 0.08?

        ↓ YES

Fallback to full dense retrieval

        ↓ NO

Are all TF-IDF scores almost the same?

(best score - lowest score <= 0.05)

        ↓ YES

Fallback to full dense retrieval

        ↓ NO

Keep files with a score at least

55% of the best TF-IDF score

        ↓

Maximum 2 candidate policy files

        ↓

Dense retrieval inside those files

```

The value `0.08` is currently a heuristic minimum lexical score. It is not a probability or a value learned automatically by TF-IDF. If the best TF-IDF score is below this value, I do not trust the lexical stage enough to restrict the search space.

I also added an ambiguity check. Even if the best score is above `0.08`, TF-IDF should not filter the search if all policy files receive almost the same score. For example:

```text

Policy A   0.21

Policy B   0.20

Policy C   0.19

Policy D   0.18

Policy E   0.17

```

Here the best score is high enough, but TF-IDF is not really separating one policy from the others. In this rare case, the system falls back to full dense retrieval and relies on semantic matching instead.

If there is a useful lexical separation, one or more likely policy files are selected. A second file is kept when its score is at least `55%` of the best score, with a maximum of two candidate policy files.

This is important for cross-policy questions. For example, a question involving both parental leave and remote work can keep both policy files instead of forcing retrieval into only one file.

The purpose is not to replace dense retrieval. TF-IDF only helps decide the search space. Nomic embeddings are still used for the actual chunk-level semantic retrieval.

In the current 20-question benchmark, neither fallback condition was triggered. All questions produced a sufficiently strong and separated TF-IDF result. The fallback is included as a safety mechanism for future or more ambiguous queries.

## Initial Improvement

The difference became much clearer when I compared the **source files** of the Top-3 chunks before and after adding TF-IDF.

| Metric | Dense Only | TF-IDF + Dense |
|---|---:|---:|
| Hit@3 | 1.000 | 1.000 |
| Recall@3 | 1.000 | 1.000 |
| Evidence Precision@3 | 0.367 | 0.367 |
| Source Precision@3 | ~0.750 | ~0.967 |
| Wrong-policy chunks in Top-3 | 15 / 60 | 2 / 60 |
| Questions containing wrong-policy chunks | 11 / 20 | 1 / 20 |

So the number of wrong-policy chunks reduced from:

```text

15 / 60

   ↓

 2 / 60

```

which is around an **87% reduction in wrong-policy retrievals**.

Some examples:

| Question | Dense Only | TF-IDF + Dense |
|---|---|---|
| Q01 | Annual + Annual + **Parental** | Annual + Annual + Annual |
| Q02 | Annual + Annual + **Parental** | Annual + Annual + Annual |
| Q03 | Annual + **Parental** + Annual | Annual + Annual + Annual |
| Q04 | Annual + **Parental + Parental** | Annual + Annual + Annual |
| Q06 | Expense + **Parental + Parental** | Expense + Expense + Expense |

An interesting case is Q09. The ground truth points to the Expense Policy sentence which tells the employee to use the Business Travel Policy for hotel limits. Therefore, the strict evaluator considers Travel Policy chunks a source mismatch even though they are logically useful for answering the question.

## Why is Precision@3 Still 0.367?

At first, this looked strange.

The source-file retrieval clearly improved, but the original evidence-based `Precision@3` remained:

```text

Precision@3 = 0.367

```

The reason is how the current ground truth defines a relevant chunk.

For many questions, only **one chunk contains the exact required evidence**.

Therefore, even a good Top-3 retrieval can look like:

```text

R1 → Exact required evidence 

R2 → Correct policy, different section

R3 → Correct policy, different section

```

The evaluator still calculates:

```text

Precision@3 = 1 / 3 = 0.333

```

Previously, the result could have been:

```text

R1 → Exact required evidence 

R2 → Wrong policy 

R3 → Wrong policy 

```

which also gives:

```text

Precision@3 = 1 / 3 = 0.333

```

So the existing evidence-level `Precision@3` cannot show the improvement between these two situations.

This is why I also looked at **Source Precision@3**.

```text

Source Precision@3

Dense Only       ≈ 0.750

TF-IDF + Dense   ≈ 0.967

```

At the same time:

```text

Hit@3    = 1.000

Recall@3 = 1.000

```

So the required evidence was still retrieved, while the probability of sending completely unrelated policy information to the later stages of the RAG pipeline was greatly reduced.

The current 20-question benchmark is still relatively small, so these results should be considered an initial validation. A larger and more difficult test set will be useful later to check how well this improvement generalizes.

## Next Rethinking Step

The file-selection problem is now greatly reduced.

The remaining problem can be simplified as:

```text

Correct policy file selected 

            ↓

But sometimes a less relevant chunk

inside that file ranks higher 

```

For example, the system may correctly identify `travel_policy.md` for a question about a 2025 hotel limit, but a general hotel example or the 2026 section can still rank above the exact 2025 evidence.

So the next target is **chunk-level ranking**.

```text

Question

   ↓

TF-IDF Policy Selection

   ↓

Candidate Policy Files

   ↓

Dense Retrieval

   ↓

Improve / Rerank Candidate Chunks

   ↓

Keep only the most useful context

   ↓

Final Top-K

```

The idea is to remove less relevant chunks even when they are coming from the correct policy file.

Once I am satisfied with this retrieval stage, I will move towards the **generation part of the RAG pipeline**, where the final retrieved context will be passed to the local LLM to generate a grounded answer.
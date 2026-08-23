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

                      /           \\

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

### Current Ranking of Correct Evidence

Before adding reranking, I also checked where the first correct ground-truth evidence currently appears in the Top-3 results.

| First Correct Evidence Rank | Questions | Count |

|---|---|---:|

| R1 | Q01–Q06, Q08, Q10–Q15, Q17, Q19, Q20 | 16 / 20 |

| R2 | Q07, Q18 | 2 / 20 |

| R3 | Q09, Q16 | 2 / 20 |

This means:

| Ranking Measure | Result |

|---|---:|

| Correct evidence at R1 | 16 / 20 = 80% |

| Correct evidence within R2 | 18 / 20 = 90% |

| Correct evidence within R3 | 20 / 20 = 100% |

| MRR@3 | 0.883 |

So the retriever is already ranking the correct evidence relatively high. In 16 out of 20 questions, the first correct evidence is already at Rank 1.

This makes the next reranking goal more specific:

\> Can a reranker move Q07's exact evidence from R2 to R1 and Q16's exact evidence from R3 to R1, while not damaging the 16 questions that are already correct at R1?

The goal is therefore not simply to retrieve more chunks. The required evidence is already present in the Top-3 for all questions. The next goal is to improve the ordering of those candidate chunks so that the most useful evidence appears earlier.

## Harder 100-Question Retrieval Benchmark

The first 20-question ground truth was useful for building and debugging the retrieval pipeline, but after the previous improvements it became too easy for testing reranking properly. The required evidence was already present inside the Top-3 for all 20 questions.

So before adding a reranker, I created a second and more difficult ground-truth file:

`retrieval_ground_truth_100_hard.json`

The 100-question ground-truth dataset was created with the help of ChatGPT using the fictional policy documents as the source. I then used the source files to verify that the `exact_text` evidence stored for every question exists verbatim in the corresponding Markdown policy file.

This benchmark contains:

- 100 questions
- 36 medium questions
- 64 hard questions
- 51 questions with more than one required evidence item
- 19 questions where more than one policy source can be relevant

The questions include paraphrasing, scenarios, thresholds, negative conditions, current-versus-historical values, policy references and cross-policy questions.

The ground truth is still **source based** rather than chunk-ID based. Therefore, it is not tied to one particular chunk size, overlap, embedding model, vector database or Top-K value.

For every question I keep:

- the question
- the expected answer
- the exact supporting evidence
- the source file containing that evidence
- the section where the evidence appears
- `relevant_sources`

The `expected_answer` is not used in the current retrieval benchmark. It will be useful later when I start evaluating the generation part.

The exact evidence is used to check whether the retrieved chunk actually contains the information required for the answer.

I added `relevant_sources` separately because I noticed an evaluation problem in the earlier 20-question benchmark. A policy can sometimes be useful for answering a question even when the exact answer sentence is located in another policy.

So now I keep two ideas separate:

```text
Evidence relevance
    ↓
Does this chunk contain the exact required evidence?

Source relevance
    ↓
Does this chunk come from a policy that is legitimately relevant
to the question?
```

This lets me calculate both:

- **Evidence Precision@3**
- **Source Precision@3**

All `exact_text` evidence strings in the 100-question ground truth were checked programmatically against their source Markdown files.

## Testing a Cross-Encoder Reranker

After improving policy-file selection with TF-IDF, the remaining question was whether chunk ordering inside the selected search space could also be improved.

For this I tested:

`cross-encoder/ms-marco-MiniLM-L6-v2`

The reranker does **not** retrieve new documents and does **not** select policy files.

The flow is:

```text
Question
   ↓
TF-IDF policy selection
   ↓
Candidate policy file(s)
   ↓
Nomic + Chroma dense retrieval
   ↓
Top-5 candidate chunks
   ↓
Cross-encoder reranking
   ↓
Final Top-3
```

The cross-encoder receives pairs like:

```text
(question, chunk 1)
(question, chunk 2)
(question, chunk 3)
(question, chunk 4)
(question, chunk 5)
```

and gives each pair a relevance score.

The five chunks are then reordered according to those scores and the final Top-3 is evaluated.

So the reranker only changes the ordering of chunks that were already retrieved.

## Three-System Benchmark

I then compared the following three retrieval systems using exactly the same 100-question benchmark:

```text
A. Dense Only
   Nomic → Chroma → Top-3

B. TF-IDF + Dense
   TF-IDF policy selection
        ↓
   Nomic + Chroma
        ↓
   Top-3

C. TF-IDF + Dense + Reranker
   TF-IDF policy selection
        ↓
   Nomic + Chroma
        ↓
   Top-5 candidates
        ↓
   Cross-encoder reranking
        ↓
   Final Top-3
```

The Nomic query embedding is calculated only once per question and reused across the three systems.

The final evaluation stays at **K=3** for all systems.

### 100-Question Results

| Metric | Dense Only | Dense + TF-IDF | Dense + TF-IDF + Reranker |
|---|---:|---:|---:|
| Hit@1 | 0.730 | 0.790 | **0.840** |
| Hit@3 | 0.990 | **1.000** | **1.000** |
| Recall@3 | 0.938 | 0.938 | **0.947** |
| Evidence Precision@3 | 0.420 | 0.413 | **0.420** |
| Source Precision@3 | 0.847 | 0.950 | **0.963** |
| MRR@3 | 0.850 | 0.888 | **0.917** |
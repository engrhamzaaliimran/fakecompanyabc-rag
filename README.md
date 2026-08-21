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
-  04_retrieve.py -- here I fetch the top-3 chunks based on fixed string. So, at this step, there is no benchmarking and no LLM. 

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

|    Size | Overlap | Chunks |     Hit@1 |  Recall@1 |     Hit@3 |  Recall@3 |    Prec@3 |     MRR@5 |
| ------: | ------: | -----: | --------: | --------: | --------: | --------: | --------: | --------: |
|     400 |      80 |     38 |     0.800 |     0.725 |     1.000 |     0.950 |     0.383 |     0.892 |
| **700** | **120** | **22** | **0.800** | **0.775** | **1.000** | **1.000** | **0.367** | **0.883** |
|    1000 |     150 |     16 |     0.700 |     0.650 |     1.000 |     1.000 |     0.367 |     0.825 |

I selected **700 chunk size with 120 overlap** as the current winner. It gives complete retrieval coverage at Top-3 (`Hit@3 = 1.0`, `Recall@3 = 1.0`) with good ranking performance (`MRR@5 = 0.883`).

The main area to improve now is `Precision@3 = 0.367`, which means the required information is retrieved, but some irrelevant chunks are also included.

### Next Step

**700/120 Baseline → Metadata Filtering → Re-evaluate Retrieval** and then maybe also **rethinking**.

The next step is to use metadata filtering and check if `Precision@3` can be improved while keeping `Hit@3` and `Recall@3` at `1.0`.

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

## Benchmarking retrieval
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
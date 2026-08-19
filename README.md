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
- 02_load_and_chunk.py -- Just load the dataset and chunk in 700 size with 120 as overlap. Initial first chunking process.]
- 03_index_chroma.py -- Loads the chunks generated in the above step into a ChromaDB database. It has the chunk code in it. The reason for keeping it this way was that if someone uses this project for learning, he/she should learn each step with clean code.
-  04_retrieve.py -- here I fetch the top-3 chunks based on fixed string. So, at this step, there is no benchmarking and no LLM. 

Each file also has a metadata at the top of it and which I plan to use later in metadata filtering
Just keep in mind this readme will also be loaded since code is looking for md files

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

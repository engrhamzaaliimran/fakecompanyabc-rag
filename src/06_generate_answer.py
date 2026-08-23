from pathlib import Path

import chromadb
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "dataset"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"


# =========================================================
# 2. RETRIEVAL SETTINGS
# =========================================================

TOP_K = 3

# Keep the same TF-IDF settings used in the retrieval benchmark.
MIN_LEXICAL_SCORE = 0.08
MAX_AMBIGUOUS_SCORE_SPREAD = 0.05
RELATIVE_SCORE_THRESHOLD = 0.55
MAX_CANDIDATE_FILES = 2


# =========================================================
# 3. LOAD POLICY FILES FOR TF-IDF
# =========================================================

policy_documents = []
policy_sources = []

for file_path in sorted(DATASET_DIR.glob("*.md")):
    if file_path.name.lower() == "readme.md":
        continue

    policy_documents.append(
        file_path.read_text(encoding="utf-8")
    )
    policy_sources.append(file_path.name)


# =========================================================
# 4. BUILD TF-IDF POLICY REPRESENTATION
# =========================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_df=0.8,
)

document_tfidf = vectorizer.fit_transform(
    policy_documents
)


# =========================================================
# 5. LOAD CHROMA + LOCAL MODELS
# =========================================================

embedding_model = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="qwen3:1.7b",
    temperature=0,
)

client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = client.get_collection(
    name="fakecompanyabc_hr"
)


# =========================================================
# 6. TF-IDF POLICY SELECTION
# =========================================================

def get_candidate_sources(question):
    query_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        query_vector,
        document_tfidf
    )[0]

    ranked = sorted(
        zip(policy_sources, similarities),
        key=lambda item: item[1],
        reverse=True,
    )

    best_source, best_score = ranked[0]
    worst_source, worst_score = ranked[-1]

    # Weak lexical signal -> do not restrict the dense search.
    if best_score < MIN_LEXICAL_SCORE:
        return None, ranked, "weak_signal"

    # If TF-IDF cannot clearly separate the files,
    # fall back to full-corpus dense retrieval.
    score_spread = best_score - worst_score

    if score_spread <= MAX_AMBIGUOUS_SCORE_SPREAD:
        return None, ranked, "ambiguous_signal"

    candidates = []

    for source, score in ranked:
        if score <= 0:
            continue

        if score >= best_score * RELATIVE_SCORE_THRESHOLD:
            candidates.append(source)

        if len(candidates) >= MAX_CANDIDATE_FILES:
            break

    return candidates, ranked, "filtered"


# =========================================================
# 7. SEMANTIC DENSE RETRIEVAL
# =========================================================

def retrieve(question, candidate_sources):
    query_embedding = embedding_model.embed_query(
        f"search_query: {question}"
    )

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": TOP_K,
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if candidate_sources is not None:
        query_args["where"] = {
            "source": {
                "$in": candidate_sources
            }
        }

    results = collection.query(**query_args)

    return {
        "documents": results["documents"][0],
        "metadatas": results["metadatas"][0],
        "distances": results["distances"][0],
    }


# =========================================================
# 8. FORMAT RETRIEVED CHUNKS AS LLM CONTEXT
# =========================================================

def build_context(retrieval_result):
    context_parts = []

    for index, (document, metadata) in enumerate(
        zip(
            retrieval_result["documents"],
            retrieval_result["metadatas"],
        ),
        start=1,
    ):
        source = metadata.get("source", "unknown")

        context_parts.append(
            f"[Context {index} | Source: {source}]\n"
            f"{document}"
        )

    return "\n\n".join(context_parts)


# =========================================================
# 9. GENERATE A GROUNDED ANSWER
# =========================================================

def generate_answer(question, context):
    system_instruction = """
You are an HR assistant for the fictional company FakeCompanyABC.

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
  clarified.

""".strip()

    user_prompt = f"""
CONTEXT
-------
{context}

QUESTION
--------
{question}

ANSWER
------
""".strip()

    response = llm.invoke(
        [
            SystemMessage(content=system_instruction),
            HumanMessage(content=user_prompt),
        ]
    )

    return response.content


# =========================================================
# 10. RUN ONE QUESTION
# =========================================================

question = input("Enter your question: ").strip()

if not question:
    raise ValueError("Question cannot be empty.")

candidate_sources, lexical_ranking, tfidf_mode = (
    get_candidate_sources(question)
)

retrieval_result = retrieve(
    question,
    candidate_sources,
)

context = build_context(
    retrieval_result
)

answer = generate_answer(
    question,
    context,
)


# =========================================================
# 11. DISPLAY RESULT
# =========================================================

print()
print("=" * 80)
print("TF-IDF ROUTING")
print("=" * 80)

print(f"Mode: {tfidf_mode}")

if candidate_sources is None:
    print("Candidate policies: FULL CORPUS")
else:
    print(
        "Candidate policies: "
        + ", ".join(candidate_sources)
    )

print()
print("=" * 80)
print("RETRIEVED TOP-3")
print("=" * 80)

for rank, (metadata, distance) in enumerate(
    zip(
        retrieval_result["metadatas"],
        retrieval_result["distances"],
    ),
    start=1,
):
    print(
        f"R{rank}: "
        f"{metadata.get('source', 'unknown')} "
        f"| distance={distance:.4f}"
    )

print()
print("=" * 80)
print("GENERATED ANSWER")
print("=" * 80)
print(answer)

print()
print("=" * 80)
print("RETRIEVED SOURCES")
print("=" * 80)

seen_sources = []

for metadata in retrieval_result["metadatas"]:
    source = metadata.get("source", "unknown")

    if source not in seen_sources:
        seen_sources.append(source)

for source in seen_sources:
    print(f"- {source}")
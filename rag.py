
import json
from pathlib import Path

import faiss
import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "vector_store"

INDEX_PATH = INDEX_DIR / "company_policy.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen2.5:3b"

SIMILARITY_THRESHOLD = 0.70
TOP_K = 3

FALLBACK_ANSWER = (
    "I don't have enough information in the provided documents."
)

if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "Vector store files are missing. Run 'python ingest.py' first."
    )

index = faiss.read_index(str(INDEX_PATH))
documents = json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def embed_question(question: str) -> np.ndarray:
    """Create a local embedding for the user's question."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={
            "model": EMBED_MODEL,
            "input": question,
        },
        timeout=120,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings", [])

    if not embeddings:
        raise RuntimeError("Ollama returned an empty question embedding.")

    query_embedding = np.asarray(
        [embeddings[0]],
        dtype="float32",
    )

    if query_embedding.shape[1] != index.d:
        raise RuntimeError(
            "Question embedding dimensions do not match the FAISS index. "
            "Run 'python ingest.py' again using the same embedding model."
        )

    faiss.normalize_L2(query_embedding)
    return query_embedding


def answer_question(question: str) -> dict:
    """Retrieve relevant chunks and generate an answer locally."""

    query_embedding = embed_question(question)

    scores, indices = index.search(query_embedding, TOP_K)

    print("Question:", question)
    print("Similarity scores:", scores[0])

    retrieved_documents = []

    for score, document_index in zip(scores[0], indices[0]):
        if document_index < 0:
            continue

        if float(score) < SIMILARITY_THRESHOLD:
            continue

        retrieved_documents.append(documents[int(document_index)])

    print(
        "Accepted chunks:",
        [doc["chunk_id"] for doc in retrieved_documents],
    )

    if not retrieved_documents:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    context = "\n\n".join(
        f"Source: {doc['source']}\n"
        f"Chunk ID: {doc['chunk_id']}\n"
        f"Content: {doc['text']}"
        for doc in retrieved_documents
    )

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's exact question using only the context below.
Do not invent facts or use outside knowledge.
If the answer is not explicitly present, respond exactly:
"{FALLBACK_ANSWER}"

Mention the relevant source filename in your answer.

Context:
{context}

User question:
{question}
"""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()

    answer = response.json().get("message", {}).get("content", "").strip()

    if not answer:
        raise RuntimeError("Ollama returned an empty answer.")

    sources = [
        {
            "chunk_id": doc["chunk_id"],
            "source": doc["source"],
        }
        for doc in retrieved_documents
    ]

    return {
        "answer": answer,
        "sources": sources,
    }

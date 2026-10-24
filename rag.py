
import json
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

# Load the vector store using paths relative to this file.
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "vector_store"

INDEX_PATH = INDEX_DIR / "company_policy.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

if not INDEX_PATH.exists() or not METADATA_PATH.exists():
    raise FileNotFoundError(
        "Vector store files are missing. Run 'python ingest.py' first."
    )

index = faiss.read_index(str(INDEX_PATH))

documents = json.loads(
    METADATA_PATH.read_text(encoding="utf-8")
)

# Initial threshold; tune it using evaluation results.
SIMILARITY_THRESHOLD = 0.70
TOP_K = 3

FALLBACK_ANSWER = (
    "I don't have enough information in the provided documents."
)


def answer_question(question: str) -> dict:
    """Retrieve relevant document chunks and generate a grounded answer."""

    # 1. Convert the user's question into an embedding.
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    if not result.embeddings:
        raise RuntimeError("Failed to generate question embedding.")

    query_embedding = np.array(
        [result.embeddings[0].values],
        dtype="float32",
    )

    faiss.normalize_L2(query_embedding)

    # 2. Retrieve the top matching chunks.
    scores, indices = index.search(query_embedding, TOP_K)

    print("Question:", question)
    print("Similarity scores:", scores[0])

    # 3. Filter chunks below the similarity threshold.
    retrieved_documents = []

    for score, document_index in zip(scores[0], indices[0]):
        if document_index < 0:
            continue

        if float(score) < SIMILARITY_THRESHOLD:
            continue

        retrieved_documents.append(
            documents[int(document_index)]
        )

    print(
        "Accepted chunks:",
        [doc["chunk_id"] for doc in retrieved_documents],
    )

    # 4. If no chunks are relevant enough, do not call Gemini to generate.
    if not retrieved_documents:
        return {
            "answer": FALLBACK_ANSWER,
            "sources": [],
        }

    # 5. Build context using accepted chunks only.
    context = "\n\n".join(
        f"Source: {document['source']}\n"
        f"Chunk ID: {document['chunk_id']}\n"
        f"Content: {document['text']}"
        for document in retrieved_documents
    )

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's exact question using only the provided context.
Do not invent facts or use outside knowledge.
If the answer is not explicitly present in the context, say:
"{FALLBACK_ANSWER}"

Include the relevant source filename in your answer.

Context:
{context}

User question:
{question}
"""

    # 6. Generate a grounded answer.
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("The AI returned an empty response.")

    # 7. Return the answer and only the sources used as context.
    sources = [
        {
            "chunk_id": document["chunk_id"],
            "source": document["source"],
        }
        for document in retrieved_documents
    ]

    return {
        "answer": response.text,
        "sources": sources,
    }

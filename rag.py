
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

INDEX_DIR = Path("vector_store")

index = faiss.read_index(
    str(INDEX_DIR / "company_policy.index")
)

documents = json.loads(
    (INDEX_DIR / "metadata.json").read_text(encoding="utf-8")
)


def answer_question(question: str) -> dict:
    # 1. Convert the user's question into an embedding.
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        ),
    )

    query_embedding = np.array(
        [result.embeddings[0].values],
        dtype="float32",
    )

    faiss.normalize_L2(query_embedding)

    # 2. Retrieve the three most relevant document chunks.
    scores, indices = index.search(query_embedding, 3)

    retrieved_documents = []

    for document_index in indices[0]:
        if document_index < 0:
            continue

        retrieved_documents.append(documents[int(document_index)])

    if not retrieved_documents:
        return {
            "answer": "I don't have enough information in the provided documents.",
            "sources": [],
        }

    # 3. Build the context for Gemini.
    context = "\n\n".join(
        f"Source: {document['source']}\n"
        f"Chunk ID: {document['chunk_id']}\n"
        f"Content: {document['text']}"
        for document in retrieved_documents
    )

    prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using only the provided context.
Answer the exact question asked. Do not answer a different question.
If the answer is not explicitly present, say:
"I don't have enough information in the provided documents."

Do not invent facts. Include the relevant source filename.

Context:
{context}

User question:
{question}
"""

    # 4. Generate a grounded answer.
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError("The AI returned an empty response.")

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

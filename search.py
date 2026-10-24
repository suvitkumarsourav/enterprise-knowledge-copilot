
import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

index_dir = Path("vector_store")

index = faiss.read_index(
    str(index_dir / "company_policy.index")
)

documents = json.loads(
    (index_dir / "metadata.json").read_text(encoding="utf-8")
)

question = "How many unused paid leave days can be carried forward?"

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


scores, indices = index.search(query_embedding, 3)

retrieved_documents = []

for score, document_index in zip(scores[0], indices[0]):
    if document_index < 0:
        continue

    document = documents[int(document_index)]

    retrieved_documents.append(document)

context = "\n\n".join(
    f"Source: {document['source']}\n"
    f"Chunk ID: {document['chunk_id']}\n"
    f"Content: {document['text']}"
    for document in retrieved_documents
)

prompt = f"""
You are an enterprise knowledge assistant.

Answer the user's question using only the provided context.
If the answer is not present in the context, say:
"I don't have enough information in the provided documents."

Include the source filename in your answer.

Context:
{context}

User question:
{question}
"""

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt,
)

print(f"Question: {question}")
print(f"\nAnswer: {response.text}")
print("\nRetrieved sources:")

for document in retrieved_documents:
    print(f"- {document['source']} ({document['chunk_id']})")



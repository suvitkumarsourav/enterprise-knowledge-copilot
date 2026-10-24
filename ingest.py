
from pathlib import Path

document_path = Path("data/company_policy.txt")

text = document_path.read_text(encoding="utf-8")



def split_into_chunks(text):
    chunks = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]
    return chunks




chunks = split_into_chunks(text)

documents = []

for index, chunk in enumerate(chunks, start=1):
    document = {
        "chunk_id": f"policy_{index}",
        "source": document_path.name,
        "text": chunk,
    }

    documents.append(document)

print(f"Total chunks: {len(documents)}")

for document in documents:
    print(f"\nID: {document['chunk_id']}")
    print(f"Source: {document['source']}")
    print(f"Text: {document['text']}")


from dotenv import load_dotenv
import os
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

for document in documents:
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=document["text"],
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_DOCUMENT"
        ),
    )

    document["embedding"] = result.embeddings[0].values

    print(
        f"{document['chunk_id']}: "
        f"embedding generated "
        f"({len(document['embedding'])} numbers)"
    )



import math


def cosine_similarity(vector_a, vector_b):
    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


question = "How many paid leaves do employees get?"

query_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY"
    ),
)

query_embedding = query_result.embeddings[0].values

ranked_documents = sorted(
    documents,
    key=lambda document: cosine_similarity(
        query_embedding,
        document["embedding"],
    ),
    reverse=True,
)

print("\nMost relevant chunks:")

for document in ranked_documents[:3]:
    score = cosine_similarity(
        query_embedding,
        document["embedding"],
    )

    print(
        f"{document['chunk_id']} | "
        f"Similarity: {score:.4f} | "
        f"{document['text']}"
    )



import faiss
import numpy as np

# Create a matrix containing all document embeddings.
embedding_matrix = np.array(
    [document["embedding"] for document in documents],
    dtype="float32",
)

# Normalize vectors so inner product represents cosine similarity.
faiss.normalize_L2(embedding_matrix)

# Create a FAISS index for cosine similarity search.
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embedding_matrix)

# Generate the user's question embedding.
question = "How many paid leaves do employees get?"

query_result = client.models.embed_content(
    model="gemini-embedding-001",
    contents=question,
    config=types.EmbedContentConfig(
        task_type="RETRIEVAL_QUERY"
    ),
)

query_embedding = np.array(
    [query_result.embeddings[0].values],
    dtype="float32",
)

faiss.normalize_L2(query_embedding)

# Retrieve the 3 most similar document chunks.
scores, indices = index.search(query_embedding, 3)

print("\nTop matching chunks:")

for score, document_index in zip(scores[0], indices[0]):
    document = documents[int(document_index)]

    print(f"\nChunk ID: {document['chunk_id']}")
    print(f"Source: {document['source']}")
    print(f"Similarity: {score:.4f}")
    print(f"Text: {document['text']}")



    

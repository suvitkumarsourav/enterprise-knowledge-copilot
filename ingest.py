
import json
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

import os

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vector_store"

VECTOR_DIR.mkdir(parents=True, exist_ok=True)


def load_documents():
    """Read every .txt file from the data directory."""
    documents = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        # Split text into non-empty lines as initial chunks.
        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()

            if not line:
                continue

            documents.append(
                {
                    "chunk_id": f"{file_path.stem}_{line_number}",
                    "source": file_path.name,
                    "text": line,
                }
            )

    return documents


def main():
    documents = load_documents()

    if not documents:
        raise ValueError(
            f"No non-empty .txt documents found in {DATA_DIR}"
        )

    embeddings = []

    for document in documents:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=document["text"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            ),
        )

        if not result.embeddings:
            raise RuntimeError(
                f"Embedding failed for {document['chunk_id']}"
            )

        embeddings.append(result.embeddings[0].values)

    vectors = np.array(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)

    # Inner product on normalized vectors gives cosine similarity.
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(
        index,
        str(VECTOR_DIR / "company_policy.index"),
    )

    (VECTOR_DIR / "metadata.json").write_text(
        json.dumps(documents, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Documents processed: {len(set(d['source'] for d in documents))}")
    print(f"Chunks indexed: {len(documents)}")
    print("FAISS index and metadata saved successfully.")


if __name__ == "__main__":
    main()

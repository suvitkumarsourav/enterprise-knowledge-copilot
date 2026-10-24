
import json
from pathlib import Path

import faiss
import numpy as np
import requests

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
VECTOR_DIR = BASE_DIR / "vector_store"

VECTOR_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"


def load_documents():
    """Read every non-empty line from all .txt files."""
    documents = []

    for file_path in sorted(DATA_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8").strip()

        if not text:
            print(f"Skipping empty file: {file_path.name}")
            continue

        for line_number, line in enumerate(text.splitlines(), start=1):
            line = line.strip()

            if line:
                documents.append(
                    {
                        "chunk_id": f"{file_path.stem}_{line_number}",
                        "source": file_path.name,
                        "text": line,
                    }
                )

    return documents


def embed_text(text):
    """Generate a local embedding using Ollama."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBED_MODEL,
            "input": text,
        },
        timeout=120,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings", [])

    if not embeddings:
        raise RuntimeError("Ollama returned an empty embedding.")

    return embeddings[0]


def main():
    documents = load_documents()

    if not documents:
        raise ValueError(f"No non-empty .txt documents found in {DATA_DIR}")

    embeddings = []

    for document in documents:
        vector = embed_text(document["text"])
        embeddings.append(vector)
        print(f"Embedded: {document['chunk_id']}")

    vectors = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(vectors)

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
    print("Local FAISS index and metadata saved successfully.")


if __name__ == "__main__":
    main()

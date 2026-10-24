# Enterprise Knowledge Copilot

An AI-powered enterprise knowledge assistant that answers questions from company policy documents using Retrieval-Augmented Generation (RAG).

## Overview

Enterprise Knowledge Copilot helps users retrieve relevant information from company documents and generate context-based answers through a simple web interface.

The application uses a locally hosted Large Language Model (LLM), semantic embeddings, and FAISS vector search to retrieve relevant document chunks. It also displays source filenames and chunk IDs to help users verify answers.

## Features

- ChatGPT-style web interface built with HTML, CSS, and JavaScript.
- FastAPI backend with REST endpoints.
- Retrieval-Augmented Generation (RAG) for document-based question answering.
- FAISS vector search for retrieving relevant document chunks.
- Local embeddings using Nomic Embed Text.
- Local answer generation using Qwen 2.5:3B through Ollama.
- Source document and chunk ID references.
- Fallback response when relevant information is not found.
- Automated tests using Pytest.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic |
| FastAPI | REST API backend |
| Ollama | Local LLM runtime |
| Qwen 2.5:3B | Answer generation |
| Nomic Embed Text | Text embeddings |
| FAISS | Vector similarity search |
| HTML, CSS, JavaScript | Chatbot frontend |
| Pytest | Automated testing |

## How It Works

1. Text documents are loaded and split into chunks.
2. Nomic Embed Text converts document chunks into vector embeddings.
3. FAISS indexes the vectors for similarity search.
4. A user submits a question through the chatbot interface.
5. The question is embedded and matched against relevant document chunks.
6. The retrieved context is sent to the local Qwen model through Ollama.
7. The generated answer and source references are returned to the frontend.

## Prerequisites

- Python 3.10 or a compatible version for the installed dependencies.
- Git.
- Ollama.
- The `nomic-embed-text` and `qwen2.5:3b` Ollama models.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/suvitkumarsourav/enterprise-knowledge-copilot.git
cd enterprise-knowledge-copilot
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the required Ollama models

Make sure Ollama is installed and running.

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
```

### 5. Prepare the document index

Place your supported text documents in the project's `data/` directory, then run:

```bash
python ingest.py
```

This generates the FAISS index and metadata required for retrieval.

### 6. Start the application

```bash
python -m uvicorn main:app --reload --port 8000
```

Open the chatbot:

http://127.0.0.1:8000/

API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Serves the chatbot interface |
| GET | `/health` | Checks API health |
| POST | `/ask` | Answers a question using the RAG pipeline |
| GET | `/docs` | Opens interactive API documentation |

Example request to `/ask`:

```json
{
  "question": "How many days per week can employees work from home?"
}
```

## Example

**Question:** How many days per week can employees work from home?

**Answer:** Employees can work from home up to 2 days per week with manager approval.

**Source:** `work_from_home.txt`

The answer depends on the documents indexed in the local knowledge base.

## Running Tests

```bash
pytest -q
```

## Project Status

The local RAG pipeline, FastAPI endpoints, and chatbot interface have been implemented and tested with sample company policy questions.

Cloud deployment is not currently included.

## Future Improvements

- Support for PDF and DOCX document ingestion.
- Improved source previews and document navigation.
- Conversation history.
- Authentication and access control.
- Cloud deployment.

## Security Note

Do not commit API keys, passwords, private company documents, or other secrets to the repository. Keep local environment files and sensitive data out of version control.

## License

No license has been specified yet. Add a suitable license if you intend to distribute or reuse this project publicly.

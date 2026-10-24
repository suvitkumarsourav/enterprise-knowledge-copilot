
import unittest.mock as mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import rag


def mock_embedding_response():
    """Return a fake local embedding with the correct FAISS dimensions."""
    result = MagicMock()
    result.json.return_value = {
        "embeddings": [[0.1] * rag.index.d]
    }
    result.raise_for_status.return_value = None
    return result


def mock_chat_response(answer):
    """Return a fake Ollama chat response."""
    result = MagicMock()
    result.json.return_value = {
        "message": {"content": answer}
    }
    result.raise_for_status.return_value = None
    return result


def test_low_similarity_chunks_are_rejected():
    fake_index = MagicMock()
    fake_index.d = rag.index.d
    fake_index.search.return_value = (
        np.array([[0.50, 0.60, 0.65]], dtype="float32"),
        np.array([[0, 1, 2]], dtype="int64"),
    )

    with (
        patch.object(rag, "index", fake_index),
        patch.object(
            rag,
            "embed_question",
            return_value=np.array([[0.1] * fake_index.d], dtype="float32"),
        ),
        patch.object(rag.requests, "post") as mock_post,
    ):
        result = rag.answer_question("Unrelated question")

    assert result["answer"] == rag.FALLBACK_ANSWER
    assert result["sources"] == []
    mock_post.assert_not_called()


def test_high_similarity_chunks_are_accepted():
    fake_index = MagicMock()
    fake_index.d = rag.index.d
    fake_index.search.return_value = (
        np.array([[0.80, 0.75, 0.50]], dtype="float32"),
        np.array([[1, 3, 0]], dtype="int64"),
    )

    answer = "Employees receive 20 days of paid leave per calendar year."

    with (
        patch.object(rag, "index", fake_index),
        patch.object(
            rag,
            "embed_question",
            return_value=np.array([[0.1] * fake_index.d], dtype="float32"),
        ),
        patch.object(rag.requests, "post") as mock_post,
    ):
        mock_post.side_effect = [
            mock_chat_response(answer)
        ]

        result = rag.answer_question("How much paid leave is provided?")

    assert result["answer"] == answer
    assert len(result["sources"]) == 2
    assert result["sources"][0]["chunk_id"] == rag.documents[1]["chunk_id"]
    assert result["sources"][1]["chunk_id"] == rag.documents[3]["chunk_id"]
    mock_post.assert_called_once()


def test_work_from_home_document_source():
    fake_index = MagicMock()
    fake_index.d = rag.index.d
    fake_index.search.return_value = (
        np.array([[0.85, 0.60, 0.50]], dtype="float32"),
        np.array([[6, 1, 2]], dtype="int64"),
    )

    answer = (
        "Employees can work from home up to 2 days per week "
        "with manager approval. Source: work_from_home.txt"
    )

    with (
        patch.object(rag, "index", fake_index),
        patch.object(
            rag,
            "embed_question",
            return_value=np.array([[0.1] * fake_index.d], dtype="float32"),
        ),
        patch.object(rag.requests, "post") as mock_post,
    ):
        mock_post.return_value = mock_chat_response(answer)

        result = rag.answer_question(
            "How many days per week can employees work from home?"
        )

    assert result["sources"][0]["source"] == "work_from_home.txt"
    assert result["sources"][0]["chunk_id"] == "work_from_home_3"
    assert len(result["sources"]) == 1
    mock_post.assert_called_once()


def test_embed_question_uses_local_ollama():
    fake_response = mock_embedding_response()

    with patch.object(rag.requests, "post", return_value=fake_response) as mock_post:
        embedding = rag.embed_question("What is the leave policy?")

    assert embedding.shape == (1, rag.index.d)
    mock_post.assert_called_once()
    assert mock_post.call_args.args[0].endswith("/api/embed")
    assert mock_post.call_args.kwargs["json"]["model"] == "nomic-embed-text"


def test_empty_local_embedding_raises_error():
    fake_response = MagicMock()
    fake_response.json.return_value = {"embeddings": []}
    fake_response.raise_for_status.return_value = None

    with patch.object(rag.requests, "post", return_value=fake_response):
        with pytest.raises(RuntimeError, match="empty question embedding"):
            rag.embed_question("Any question")

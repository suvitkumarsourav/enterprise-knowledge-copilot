
from unittest.mock import MagicMock, patch

import numpy as np

import rag


def test_low_similarity_chunks_are_rejected():
    fake_index = MagicMock()
    fake_index.search.return_value = (
        np.array([[0.50, 0.60, 0.65]], dtype="float32"),
        np.array([[0, 1, 2]], dtype="int64"),
    )

    with (
        patch.object(rag, "index", fake_index),
        patch.object(
            rag.client.models,
            "embed_content",
        ) as mock_embed,
        patch.object(
            rag.client.models,
            "generate_content",
        ) as mock_generate,
    ):
        mock_embedding_result = MagicMock()
        mock_embedding_result.embeddings = [
            MagicMock(values=[0.1] * 3072)
        ]
        mock_embed.return_value = mock_embedding_result

        result = rag.answer_question(
            "What is the company's maternity leave policy?"
        )

    assert result["answer"] == rag.FALLBACK_ANSWER
    assert result["sources"] == []
    mock_generate.assert_not_called()


def test_high_similarity_chunks_are_accepted():
    fake_index = MagicMock()
    fake_index.search.return_value = (
        np.array([[0.80, 0.75, 0.50]], dtype="float32"),
        np.array([[1, 3, 0]], dtype="int64"),
    )

    with (
        patch.object(rag, "index", fake_index),
        patch.object(
            rag.client.models,
            "embed_content",
        ) as mock_embed,
        patch.object(
            rag.client.models,
            "generate_content",
        ) as mock_generate,
    ):
        mock_embedding_result = MagicMock()
        mock_embedding_result.embeddings = [
            MagicMock(values=[0.1] * 3072)
        ]
        mock_embed.return_value = mock_embedding_result

        mock_response = MagicMock()
        mock_response.text = (
            "Employees receive 20 days of paid leave. "
            "Source: company_policy.txt"
        )
        mock_generate.return_value = mock_response

        result = rag.answer_question(
            "How many paid leave days do employees receive?"
        )

    assert len(result["sources"]) == 2
    assert [s["chunk_id"] for s in result["sources"]] == [
        rag.documents[1]["chunk_id"],
        rag.documents[3]["chunk_id"],
    ]
    mock_generate.assert_called_once()



def test_work_from_home_document_source():
    fake_index = MagicMock()
    fake_index.search.return_value = (
        np.array([[0.85, 0.60, 0.50]], dtype="float32"),
        np.array([[6, 1, 2]], dtype="int64"),
    )

    with (
        patch.object(rag, "index", fake_index),
        patch.object(
            rag.client.models,
            "embed_content",
        ) as mock_embed,
        patch.object(
            rag.client.models,
            "generate_content",
        ) as mock_generate,
    ):
        mock_embedding_result = MagicMock()
        mock_embedding_result.embeddings = [
            MagicMock(values=[0.1] * 3072)
        ]
        mock_embed.return_value = mock_embedding_result

        mock_response = MagicMock()
        mock_response.text = (
            "Employees can work from home up to 2 days per week "
            "with manager approval. Source: work_from_home.txt"
        )
        mock_generate.return_value = mock_response

        result = rag.answer_question(
            "How many days per week can employees work from home?"
        )

    assert result["sources"][0]["source"] == "work_from_home.txt"
    assert result["sources"][0]["chunk_id"] == "work_from_home_3"
    assert len(result["sources"]) == 1
    mock_generate.assert_called_once()

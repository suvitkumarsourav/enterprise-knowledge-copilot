
from fastapi.testclient import TestClient

from main import app


from unittest.mock import MagicMock, patch


def test_successful_ai_response():
    mock_response = MagicMock()
    mock_response.text = "Generative AI creates new content."

    with patch(
        "main.client.models.generate_content",
        return_value=mock_response,
    ) as mock_generate:
        response = client.post(
            "/ask",
            json={"question": "What is Generative AI?"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "question": "What is Generative AI?",
        "answer": "Generative AI creates new content.",
    }

    mock_generate.assert_called_once()


client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Question cannot be empty"
    }


def test_missing_question():
    response = client.post("/ask", json={})

    assert response.status_code == 422


from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app

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
    response = client.post(
        "/ask",
        json={},
    )

    assert response.status_code == 422


def test_successful_ai_response():
    with patch("main.answer_question") as mock_answer:
        mock_answer.return_value = {
            "answer": "Employees receive 20 days of paid leave.",
            "sources": [
                {
                    "chunk_id": "policy_2",
                    "source": "company_policy.txt",
                }
            ],
        }

        response = client.post(
            "/ask",
            json={
                "question": "How many paid leave days do employees get?"
            },
        )

    assert response.status_code == 200

    data = response.json()

    assert data["question"] == (
        "How many paid leave days do employees get?"
    )
    assert data["answer"] == (
        "Employees receive 20 days of paid leave."
    )
    assert data["sources"][0]["chunk_id"] == "policy_2"
    assert data["sources"][0]["source"] == "company_policy.txt"

    mock_answer.assert_called_once_with(
        "How many paid leave days do employees get?"
    )



def test_gemini_quota_exceeded():
    from google.genai.errors import ClientError
    from unittest.mock import patch

    error = ClientError(
        429,
        {
            "error": {
                "code": 429,
                "message": "RESOURCE_EXHAUSTED: quota exceeded",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )

    with patch("main.answer_question", side_effect=error):
        response = client.post(
            "/ask",
            json={"question": "What is the leave policy?"},
        )

    assert response.status_code == 429
    assert response.json() == {
        "detail": (
            "Gemini API quota exceeded. "
            "Check your quota or retry after it resets."
        )
    }

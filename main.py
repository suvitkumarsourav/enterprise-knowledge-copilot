
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.genai.errors import ClientError

from rag import answer_question


app = FastAPI(
    title="Enterprise Knowledge Copilot",
    description="An AI-powered enterprise knowledge assistant",
    version="0.2.0",
)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.post("/ask")
def ask_ai(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty",
        )

    try:
        result = answer_question(question)

        return {
            "question": question,
            "answer": result["answer"],
            "sources": result["sources"],
        }

    except HTTPException:
        raise

    except ClientError as error:
        error_message = str(error)
        print(f"Gemini API error: {error_message}")

        if (
            "429" in error_message
            or "RESOURCE_EXHAUSTED" in error_message
        ):
            raise HTTPException(
                status_code=429,
                detail=(
                    "Gemini API quota exceeded. "
                    "Check your quota or retry after it resets."
                ),
            )

        raise HTTPException(
            status_code=502,
            detail="AI service request failed. Check server logs.",
        )

    except Exception as error:
        print(f"RAG service error: {error}")

        raise HTTPException(
            status_code=502,
            detail="Knowledge assistant request failed. Check server logs.",
        )

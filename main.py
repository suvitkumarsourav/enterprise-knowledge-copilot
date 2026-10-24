
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

client = genai.Client(api_key=api_key)

app = FastAPI(
    title="Enterprise Knowledge Copilot",
    description="An AI-powered enterprise knowledge assistant",
    version="0.1.0",
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
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=question,
        )

        if not response.text:
            raise HTTPException(
                status_code=502,
                detail="The AI returned an empty response",
            )

        return {
            "question": question,
            "answer": response.text,
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI service request failed. Check the server logs.",
        )

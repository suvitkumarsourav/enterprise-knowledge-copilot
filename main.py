

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from pydantic import BaseModel

from rag import answer_question



app = FastAPI(
    title="Enterprise Knowledge Copilot",
    description="An AI-powered enterprise knowledge assistant",
    version="0.3.0",
)

BASE_DIR = Path(__file__).resolve().parent

app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "static"),
    name="static",
)

@app.get("/")
def home():
    return FileResponse(BASE_DIR / "static" / "index.html")



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

    except Exception as error:
        # Log the actual error on the server, not in the API response.
        print(f"Knowledge assistant error: {error}")

        raise HTTPException(
            status_code=502,
            detail=(
                "Knowledge assistant request failed. "
                "Check server logs."
            ),
        ) from error

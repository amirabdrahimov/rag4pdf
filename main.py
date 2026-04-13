from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag4pdf import PdfRagAssistant, Settings


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question to ask about the indexed PDFs")
    top_k: int = Field(4, ge=1, le=20, description="Number of retrieved chunks to include")
    llm_model: str | None = Field(None, description="Optional Ollama model override")


class SourceItem(BaseModel):
    source: str
    page: int | None
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem]


class HealthResponse(BaseModel):
    status: str
    initialized: bool
    index_status: str | None = None
    model: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    assistant = PdfRagAssistant(settings=settings)
    assistant.initialize()
    app.state.assistant = assistant
    yield


app = FastAPI(
    title="rag4pdf",
    version="1.0.0",
    description="FastAPI service for querying PDF content with retrieval-augmented generation.",
    lifespan=lifespan,
)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {"message": "rag4pdf API is running"}


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    assistant = getattr(app.state, "assistant", None)
    return HealthResponse(
        status="ok" if assistant is not None else "starting",
        initialized=assistant is not None,
        index_status=getattr(assistant, "_index_status", None),
        model=getattr(getattr(assistant, "settings", None), "ollama_model", None),
    )


@app.post("/ask", response_model=AskResponse, tags=["rag"])
def ask(payload: AskRequest) -> AskResponse:
    assistant = getattr(app.state, "assistant", None)
    if assistant is None:
        raise HTTPException(status_code=503, detail="Assistant is not initialized yet.")

    result: dict[str, Any] = assistant.answer(payload.question, k=payload.top_k, llm_model=payload.llm_model)
    return AskResponse(
        answer=result["answer"],
        sources=[SourceItem(**source) for source in result.get("sources", [])],
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

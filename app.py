"""
CatalogMind — Layer 12: Full production app.py.

Replaces the Layer 1 stub with the real orchestrator pipeline.
The /chat route now runs the full stack:
  request → validate schema → orchestrator.handle() → response

Design decisions:
  - The route itself is intentionally thin — all business logic
    lives in agent/orchestrator.py. app.py only handles:
      * Schema validation (Pydantic)
      * Error coercion to schema-safe responses
      * FAISS index + model pre-load on startup

  - lifespan() pre-loads FAISS store and embedding model so the
    first /chat request isn't slow. The evaluator allows up to
    2 minutes on /health for cold start — we use that budget here.

  - /health returns {"status":"ok"} only when FAISS is loaded.
    Returns {"status":"not_ready"} with HTTP 200 during startup
    so the evaluator's health poll can differentiate states.

  - Messages are converted from Pydantic models to plain dicts
    before passing to the orchestrator — keeps downstream layers
    free of FastAPI/Pydantic imports.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas.chat_schema import ChatRequest, ChatResponse, HealthResponse, Recommendation
from agent.orchestrator import handle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info("CatalogMind startup — pre-loading FAISS store...")
        from retrieval.faiss_store import store
        store.load()
        logger.info("FAISS store loaded: %d items", len(store.catalog))

        logger.info("Pre-loading embedding model...")
        from ingestion.embeddings import get_model
        get_model()
        logger.info("CatalogMind is ready")
    except Exception as e:
        logger.error("Startup failed: %s", e)
    yield
    logger.info("CatalogMind shutting down")


app = FastAPI(
    title="CatalogMind — Conversational SHL Assessment Recommender",
    lifespan=lifespan,
)


def _schema_safe_fallback(reply: str) -> JSONResponse:
    """Return HTTP 200 with valid schema on any error."""
    return JSONResponse(
        status_code=200,
        content={"reply": reply, "recommendations": [], "end_of_conversation": False},
    )


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return _schema_safe_fallback(
        "I couldn't understand that request. Please make sure 'messages' "
        "is a non-empty list of {role, content} objects and try again."
    )


@app.exception_handler(Exception)
async def fallback_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", type(exc).__name__, exc_info=True)
    return _schema_safe_fallback(
        "Something went wrong on our end. Please try rephrasing your request."
    )


@app.get("/health", response_model=HealthResponse)
def health() -> JSONResponse:
    from retrieval.faiss_store import store
    if store.is_ready:
        return JSONResponse(status_code=200, content={"status": "ok"})
    return JSONResponse(status_code=200, content={"status": "not_ready"})


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> JSONResponse:
    # Convert Pydantic models → plain dicts for downstream layers
    messages = [
        {"role": m.role.value, "content": m.content}
        for m in request.messages
    ]

    result = handle(messages)

    response = ChatResponse(
        reply=result["reply"],
        recommendations=[
            Recommendation(**rec) if isinstance(rec, dict) else rec
            for rec in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )

    return JSONResponse(status_code=200, content=response.model_dump())
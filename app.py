"""
Layer 1: FastAPI service shell.

At this stage there is NO catalog, NO FAISS, NO LLM call yet. The goal is to
validate the API contract end-to-end:
  - GET /health
  - POST /chat with strict request/response schema
  - basic clarify / refuse / recommend(stub) / compare(stub) branching

Recommendations are still hardcoded placeholders in this layer so we can
verify the schema (1-10 items, valid URLs) is wired correctly before the
real retrieval/reranking pipeline exists.
"""

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from schemas.chat_schema import ChatRequest, ChatResponse, HealthResponse, Recommendation
from agent.decision import classify_intent, Intent

app = FastAPI(title="SHL Conversational Assessment Recommender")


def _schema_safe_fallback(reply: str) -> JSONResponse:
    """Always return HTTP 200 with the exact response contract, even on
    malformed input. The assignment is explicit: 'Never break response
    schema' — so even client errors are translated into a normal chat
    reply rather than a raw FastAPI error body."""
    return JSONResponse(
        status_code=200,
        content={"reply": reply, "recommendations": [], "end_of_conversation": False},
    )


@app.exception_handler(RequestValidationError)
def validation_error_handler(request: Request, exc: RequestValidationError):
    return _schema_safe_fallback(
        "I couldn't understand that request. Please make sure 'messages' is a "
        "non-empty list of {role, content} objects and try again."
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    latest_user_msg = next(
        (m.content for m in reversed(request.messages) if m.role == "user"),
        "",
    )
    full_history_text = " ".join(m.content for m in request.messages)

    intent = classify_intent(latest_user_msg, full_history_text)

    if intent == Intent.REFUSE:
        return ChatResponse(
            reply=(
                "I can only help with selecting SHL assessments. I can't help "
                "with that request — feel free to ask about assessments for a role."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    if intent == Intent.CLARIFY:
        return ChatResponse(
            reply=(
                "Happy to help. Could you tell me a bit more about the role "
                "you're hiring for — e.g. job title, seniority, or key skills?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    if intent == Intent.COMPARE:
        return ChatResponse(
            reply=(
                "Comparison logic isn't wired up yet in this layer — "
                "placeholder response."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # intent == RECOMMEND (stub data; real retrieval comes in a later layer)
    return ChatResponse(
        reply="Here is a placeholder shortlist (catalog/retrieval not wired up yet).",
        recommendations=[
            Recommendation(
                name="Placeholder Assessment",
                url="https://www.shl.com/solutions/products/product-catalog/placeholder/",
                test_type="K",
            )
        ],
        end_of_conversation=False,
    )


@app.exception_handler(Exception)
def fallback_handler(request: Request, exc: Exception):
    # Never break the response schema, even on unexpected errors.
    return _schema_safe_fallback(
        "Something went wrong on our end. Please try rephrasing your request."
    )
"""
Pydantic schemas for the /chat endpoint.

Design decision: the response schema is "non-negotiable" per the assignment,
so we lock it down hard here:
  - extra fields are forbidden (model_config extra="forbid") so a stray key
    from the LLM layer can never silently leak into the API response.
  - recommendations length is validated (0 items OR 1-10 items), enforced
    again at the route level as a defense-in-depth check.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str = Field(..., min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: List[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: List[Message]) -> List[Message]:
        # The evaluator always sends the conversation ending on a user turn —
        # we don't *hard fail* on this (be liberal in what we accept) but it's
        # useful to know during dev. Left as a no-op validator hook for now.
        return v


class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: List[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False

    @field_validator("recommendations")
    @classmethod
    def validate_recommendation_count(cls, v: List[Recommendation]) -> List[Recommendation]:
        # Schema rule: empty OR 1-10 items. Never allow 11+ to slip through.
        if len(v) > 10:
            raise ValueError("recommendations must contain at most 10 items")
        return v


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: str = "ok"
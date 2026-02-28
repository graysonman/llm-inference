from pydantic import BaseModel, Field
from typing import Literal, Optional, List, Dict, Any


ChatMode = Literal["single", "refine"]


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=200_000)
    max_new_tokens: int = Field(160, ge=1, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    mode: ChatMode = Field("single")
    refine_steps: int = Field(1, ge=1, le=3)  # keep small for CPU
    critique_temperature: float = Field(0.2, ge=0.0, le=2.0)


class ChatResponse(BaseModel):
    response: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    model: str
    request_id: str

    # Transformer mechanics at the boundary
    context_window: int
    context_used_pct: float
    model_type: str
    attention_masking: str
    attention_heads: Optional[int] = None
    hidden_size: Optional[int] = None
    estimated_attention_ops: Optional[int] = None
    total_tokens: int
    output_to_input_ratio: float
    refined: bool = False
    original_response: Optional[str] = None
    critique: Optional[str] = None
    refine_steps_used: int = 0


EvalCriteria = Literal["accuracy", "clarity", "reasoning", "factuality", "overall"]


class EvaluateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=200_000)
    response: str = Field(..., min_length=1, max_length=200_000)
    criteria: List[EvalCriteria] = Field(default_factory=lambda: ["overall"])


class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(..., ge=1, le=10)
    rationale: str


class EvaluateResponse(BaseModel):
    request_id: str
    model: str
    scores: List[CriterionScore]
    latency_ms: int
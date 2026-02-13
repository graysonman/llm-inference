from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=20_000)
    max_new_tokens: int = Field(160, ge=1, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    top_p: float = Field(0.95, ge=0.0, le=1.0)


class ChatResponse(BaseModel):
    response: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    model: str
    request_id: str

    context_window: int
    context_used_pct: float
    model_type: str                 # "decoder-only" | "encoder-decoder" | "unknown"
    attention_masking: str          # "causal" | "none" | "unknown"
    attention_heads: int | None = None
    hidden_size: int | None = None
    estimated_attention_ops: int | None = None  
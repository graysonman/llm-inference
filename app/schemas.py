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
    cache_hit: bool = False


EvalCriteria = Literal["accuracy", "clarity", "reasoning", "factuality", "overall"]
DatasetType = Literal["rag_corpus", "eval_set"]


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
    run_id: Optional[str] = None


class EmbeddingsRequest(BaseModel):
    input: str | List[str]
    model: Optional[str] = None
    normalize: bool = True


class EmbeddingItem(BaseModel):
    object: Literal["embedding"] = "embedding"
    index: int
    embedding: List[float]


class EmbeddingsResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[EmbeddingItem]
    model: str
    usage: Dict[str, int]
    request_id: str


class RagQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=200_000)
    dataset_id: str = Field(..., min_length=1, max_length=200)
    top_k: int = Field(5, ge=1, le=20)
    max_new_tokens: int = Field(220, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=2.0)


class RagQueryResponse(BaseModel):
    response: str
    retrieved_chunks: List[Dict[str, Any]]
    request_id: str


class DatasetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    type: DatasetType = "eval_set"
    records: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetCreateResponse(BaseModel):
    dataset_id: str
    name: str
    type: DatasetType
    status: Literal["processing", "ready", "failed"] = "ready"
    records_count: int
    created_at: int
    request_id: str


class BatchEvalCreateRequest(BaseModel):
    dataset_id: str = Field(..., min_length=1, max_length=200)
    criteria: List[EvalCriteria] = Field(default_factory=lambda: ["overall"])


class BatchEvalCreateResponse(BaseModel):
    run_id: str
    status: Literal["queued"] = "queued"
    request_id: str


class EvalRunSummary(BaseModel):
    run_id: str
    prompt: str
    response: str
    criteria: List[EvalCriteria]
    scores: List[CriterionScore]
    model: str
    latency_ms: int
    created_at: int


class BatchEvalStatusResponse(BaseModel):
    run_id: str
    dataset_id: str
    status: Literal["queued", "running", "completed", "failed"]
    progress: Dict[str, int]
    criteria: List[EvalCriteria]
    created_at: int
    updated_at: int
    request_id: str


class RagIndexStatusResponse(BaseModel):
    index_id: str
    dataset_id: str
    status: Literal["ready", "building", "not_found"]
    chunk_count: int
    updated_at: int
    request_id: str


class DatasetListItem(BaseModel):
    dataset_id: str
    name: str
    type: DatasetType
    status: Literal["processing", "ready", "failed"]
    record_count: int
    created_at: int


class DatasetListResponse(BaseModel):
    data: List[DatasetListItem]
    next_cursor: Optional[str] = None
    request_id: str


class DatasetGetResponse(BaseModel):
    dataset_id: str
    name: str
    type: DatasetType
    status: Literal["processing", "ready", "failed"]
    record_count: int
    error: Optional[str] = None
    created_at: int
    updated_at: int
    request_id: str


class BatchEvalResultResponse(BaseModel):
    batch_eval_id: str
    status: Literal["queued", "running", "completed", "failed"]
    summary: Dict[str, Any]
    completed_at: Optional[int] = None
    request_id: str


class BatchEvalFailureItem(BaseModel):
    record_index: int
    criterion: str
    score: int
    reason: str
    record: Dict[str, Any]


class BatchEvalFailuresResponse(BaseModel):
    batch_eval_id: str
    data: List[BatchEvalFailureItem]
    count: int
    request_id: str


class BatchEvalDistributionResponse(BaseModel):
    batch_eval_id: str
    criterion: str
    buckets: Dict[str, int]
    summary: Dict[str, float | int]
    request_id: str

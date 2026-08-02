from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class EvalSummaryConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_tokens: int | None = Field(default=0)
    repeat_penalty: float | None = Field(default=1)
    top_k: int | None = Field(default=0)
    top_p: float | None = Field(default=0)
    frequency_penalty: float | None = Field(default=0)
    presence_penalty: float | None = Field(default=0)
    prompt: str | None = Field(default=None)
    context_template: str | None = Field(default=None)
    no_match_prefix: str | None = Field(default=None)
    temperature: float | None = Field(default=0.3)
    seed: int | None = Field(default=0)
    max_completion_tokens: int | None = Field(default=2048)


class EvalTaskParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str | None = Field(default=None)
    knowledge_base_id: str
    vector_threshold: float | None = Field(default=0.5)
    keyword_threshold: float | None = Field(default=0.3)
    embedding_top_k: int | None = Field(default=10)
    vector_database: str | None = Field(default=None)
    rerank_model_id: str | None = Field(default=None)
    rerank_top_k: int | None = Field(default=5)
    rerank_threshold: float | None = Field(default=0.7)
    chat_model_id: str
    summary_config: EvalSummaryConfig | None = Field(default=None)
    fallback_strategy: str | None = Field(default=None)
    fallback_response: str | None = Field(default=None)


class EvalTask(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    tenant_id: int
    dataset_id: str
    start_time: datetime
    status: int
    total: int | None = Field(default=0)
    finished: int | None = Field(default=0)


class EvalRetrievalMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    precision: float
    recall: float
    ndcg3: float
    ndcg10: float
    mrr: float
    map: float


class EvalGenerationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    bleu1: float
    bleu2: float
    bleu4: float
    rouge1: float
    rouge2: float
    rougel: float


class EvalMetric(BaseModel):
    model_config = ConfigDict(frozen=True)

    retrieval_metrics: EvalRetrievalMetrics | None = Field(default=None)
    generation_metrics: EvalGenerationMetrics | None = Field(default=None)


class EvaluationGetResponseData(BaseModel):
    model_config = ConfigDict(frozen=True)

    task: EvalTask
    params: EvalTaskParams
    metric: EvalMetric | None = Field(default=None)


class EvaluationCreateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_id: str
    knowledge_base_id: str
    chat_id: str
    rerank_id: str


class EvaluationGetQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    task_id: str


__all__ = [
    "EvalGenerationMetrics",
    "EvalMetric",
    "EvalRetrievalMetrics",
    "EvalSummaryConfig",
    "EvalTask",
    "EvalTaskParams",
    "EvaluationCreateRequest",
    "EvaluationGetQuery",
    "EvaluationGetResponseData",
]

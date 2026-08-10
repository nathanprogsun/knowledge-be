"""Evaluation seams and DTOs — the injectable surfaces and value objects.

The evaluation service depends on narrow slices of the KB, document, and
chat domains plus two caller-facing DTOs. Every dependency is declared
here as a structural ``Protocol`` (or a dataclass) so tests can supply
tiny in-memory fakes and the web layer can wire the live services
without a circular import.

Scope of this module
--------------------

- ``EvaluationCreateQuery`` — caller-facing create input.
- ``EvaluationParams`` — mutable per-task pipeline parameters that the
  QA runner fills with structured artefacts after each turn.
- ``KnowledgeBaseCreator`` / ``KnowledgeBaseReader`` /
  ``KnowledgeBaseDeleter`` / ``KnowledgeBaseInfoLike`` — KB seams.
- ``KnowledgeFactoryLike`` — knowledge-ingestion seam.
- ``QARunner`` — chat-pipeline seam.
- ``_UpdateResult`` — the storage update callback's return contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from src.common.exception import ValidationError
from src.core.contracts.evaluation import EvalMetric, EvalTask
from src.core.contracts.knowledge import Knowledge
from src.core.evaluation.metric_hook import ChatResponseLike, SearchHitLike

# ── Injectable seams ──────────────────────────────────────────────────


@runtime_checkable
class KnowledgeBaseInfoLike(Protocol):
    """The KB projection the evaluation flow reads back from a clone.

    Structural: the KB service's ``KnowledgeBaseInfo`` view satisfies
    it. Only the model bindings matter to the evaluation flow.
    """

    embedding_model_id: str
    summary_model_id: str


@runtime_checkable
class KnowledgeBaseCreator(Protocol):
    """Subset of the KB service the evaluation flow needs.

    The seam lets tests substitute a one-line stub that returns a fresh
    knowledge-base id without touching the storage layer.
    """

    async def create_knowledge_base(
        self,
        *,
        name: str,
        description: str,
        embedding_model_id: str,
        summary_model_id: str,
    ) -> str: ...


@runtime_checkable
class KnowledgeBaseDeleter(Protocol):
    """Subset of the KB service for the deferred cleanup path."""

    async def delete_knowledge_base(
        self,
        *,
        knowledge_base_id: str,
    ) -> bool: ...


@runtime_checkable
class KnowledgeBaseReader(Protocol):
    """Subset of the KB service used when cloning an existing KB."""

    async def get_knowledge_base_by_id(
        self,
        *,
        knowledge_base_id: str,
    ) -> KnowledgeBaseInfoLike: ...


@runtime_checkable
class KnowledgeFactoryLike(Protocol):
    """Subset of the document service the evaluation flow needs.

    The production factory creates a single ``passage`` knowledge entry
    from the dataset's passage list and indexes it synchronously so the
    retrieval index is ready before the QA loop starts.
    """

    async def create_knowledge_from_passages(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        passages: list[str],
    ) -> Knowledge: ...

    async def delete_knowledge(
        self,
        *,
        tenant_id: int,
        knowledge_id: str,
    ) -> bool: ...


@runtime_checkable
class QARunner(Protocol):
    """Subset of the chat / session service the evaluation loop needs.

    Mirrors the upstream ``KnowledgeQAByEvent`` entry point: run the RAG
    pipeline against ``query`` + ``knowledge_base_id`` and leave the
    structured artefacts (``search_result`` / ``rerank_result`` /
    ``chat_response``) on ``params`` for the metric hook to consume.
    """

    async def knowledge_qa(
        self,
        *,
        tenant_id: int,
        params: EvaluationParams,
    ) -> None: ...


# ── DTOs ──────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class EvaluationCreateQuery:
    """Caller-facing input for :meth:`EvaluationService.create`.

    The four ids mirror the upstream ``Evaluation(ctx, datasetID,
    knowledgeBaseID, chatModelID, rerankModelID)`` signature: an empty
    string falls back to a default value resolved against the tenant's
    model registry.
    """

    dataset_id: str = ""
    knowledge_base_id: str = ""
    chat_model_id: str = ""
    rerank_model_id: str = ""

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValidationError(
                code="evaluation.dataset_id_required",
                message="dataset_id cannot be empty",
            )


@dataclass(slots=True)
class EvaluationParams:
    """Mutable per-task pipeline parameters (mirrors upstream ``ChatManage``).

    The QA runner mutates ``search_result``, ``rerank_result``, and
    ``chat_response`` after the pipeline runs so the metric hook can
    read the structured artefacts without an extra return channel.
    ``query`` / ``rewrite_query`` are likewise set per turn by the
    fan-out loop. The dataclass is intentionally NOT frozen — the
    upstream ``ChatManage`` is a mutable struct that the pipeline
    fills in place; :meth:`clone` provides the per-worker isolation.
    """

    knowledge_base_id: str
    chat_model_id: str
    rerank_model_id: str
    query: str = ""
    rewrite_query: str = ""
    search_result: list[SearchHitLike] = field(default_factory=list)
    rerank_result: list[SearchHitLike] = field(default_factory=list)
    chat_response: ChatResponseLike | None = None

    def clone(self) -> EvaluationParams:
        """Return a fresh copy with the mutable defaults re-created.

        Mirrors the upstream ``ChatManage.Clone`` contract: the
        structured result lists are copied so two workers running the
        same task never share state.
        """
        return EvaluationParams(
            knowledge_base_id=self.knowledge_base_id,
            chat_model_id=self.chat_model_id,
            rerank_model_id=self.rerank_model_id,
            query=self.query,
            rewrite_query=self.rewrite_query,
            search_result=list(self.search_result),
            rerank_result=list(self.rerank_result),
            chat_response=self.chat_response,
        )


#: What an :meth:`EvaluationMemoryStorage.update` callback may return: a
#: single replacement artefact, a tuple of them, or ``None`` to signal
#: "no replacement".
_UpdateResult = (
    EvalTask
    | EvaluationParams
    | EvalMetric
    | tuple[EvalTask | EvaluationParams | EvalMetric, ...]
    | None
)


__all__ = [
    "EvaluationCreateQuery",
    "EvaluationParams",
    "KnowledgeBaseCreator",
    "KnowledgeBaseDeleter",
    "KnowledgeBaseInfoLike",
    "KnowledgeBaseReader",
    "KnowledgeFactoryLike",
    "QARunner",
    "_UpdateResult",
]

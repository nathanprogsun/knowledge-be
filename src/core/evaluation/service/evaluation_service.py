"""Offline evaluation service — orchestrate dataset runs and metric collection.

Mirrors the upstream ``EvaluationService`` entry points. The service
owns the fan-out runner over the dataset's QA pairs, the model /
knowledge-base default resolution, the durable row creation, and the
metric aggregation the result endpoint returns.

The injectable seams (``KnowledgeBaseCreator`` / ``KnowledgeBaseReader`` /
``KnowledgeBaseDeleter`` / ``KnowledgeFactoryLike`` / ``QARunner``) and
the two DTOs live in :mod:`service.seams`; the in-memory run-progress
registry lives in :mod:`service.storage`. The web layer wires the live
seams via :func:`src.core.evaluation.factory.build_evaluation_service`.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

from src.common.exception import ValidationError
from src.core.contracts.evaluation import (
    EvalGenerationMetrics,
    EvalMetric,
    EvalRetrievalMetrics,
    EvalTask,
    EvalTaskParams,
    EvaluationGetResponseData,
)
from src.core.evaluation.dataset import (
    DEFAULT_DATASET_ID,
    DatasetServiceLike,
    QAPair,
    get_passage_list,
)
from src.core.evaluation.metric_hook import HookMetric, MetricResult
from src.core.evaluation.service.seams import (
    EvaluationCreateQuery,
    EvaluationParams,
    KnowledgeBaseCreator,
    KnowledgeBaseDeleter,
    KnowledgeBaseReader,
    KnowledgeFactoryLike,
    QARunner,
)
from src.core.evaluation.service.storage import (
    _STATUS_FAILED,
    _STATUS_PENDING,
    _STATUS_RUNNING,
    _STATUS_SUCCESS,
    EvaluationMemoryStorage,
    _now,
)
from src.core.infra.models.service.model_service import ModelService
from src.db.dao.evaluation_repository import (
    EvaluationDatasetRepository,
    EvaluationMetricRepository,
    EvaluationRepository,
    EvaluationRunRepository,
)
from src.db.models.evaluation import (
    EVAL_STATUS_FAILED,
    EVAL_STATUS_PENDING,
    EVAL_STATUS_RUNNING,
    EVAL_STATUS_SUCCESS,
    Evaluation,
    EvaluationRun,
)

logger = logging.getLogger(__name__)


#: Default rerank top-k the service applies when the caller does not
#: override it on the params payload.
_DEFAULT_RERANK_TOP_K = 5

#: Model-type strings used for default resolution against the tenant's
#: model registry. Matches the upstream ``types.ModelType`` values.
_EMBEDDING_MODEL_TYPE = "Embedding"
_KNOWLEDGE_QA_MODEL_TYPE = "KnowledgeQA"
_RERANK_MODEL_TYPE = "Rerank"


class EvaluationService:
    """Per-request evaluation facade.

    Mirrors the upstream ``EvaluationService`` entry points:
    ``create`` registers a task, persists the initial rows, and kicks
    off a background run; ``get`` returns the latest snapshot for
    ``task_id``; ``run_dataset`` is the synchronous body of the
    background task.

    The runner / KB / knowledge seams are injected so tests can swap
    them for in-memory fakes without a live database, model registry,
    or pipeline engine.
    """

    def __init__(
        self,
        *,
        tenant_id: int,
        evaluation_repo: EvaluationRepository,
        dataset_repo: EvaluationDatasetRepository,
        run_repo: EvaluationRunRepository,
        metric_repo: EvaluationMetricRepository,
        model_service: ModelService,
        dataset_service: DatasetServiceLike,
        kb_creator: KnowledgeBaseCreator | None = None,
        kb_reader: KnowledgeBaseReader | None = None,
        kb_deleter: KnowledgeBaseDeleter | None = None,
        knowledge_factory: KnowledgeFactoryLike | None = None,
        qa_runner: QARunner | None = None,
        max_workers: int = 4,
        memory_storage: EvaluationMemoryStorage | None = None,
    ) -> None:
        if tenant_id <= 0:
            raise ValidationError(
                code="evaluation.invalid_tenant_id",
                message="tenant_id must be positive",
            )
        self._tenant_id = tenant_id
        self._evaluation_repo = evaluation_repo
        self._dataset_repo = dataset_repo
        self._run_repo = run_repo
        self._metric_repo = metric_repo
        self._model_service = model_service
        self._dataset_service = dataset_service
        self._kb_creator = kb_creator
        self._kb_reader = kb_reader
        self._kb_deleter = kb_deleter
        self._knowledge_factory = knowledge_factory
        self._qa_runner = qa_runner
        self._max_workers = max(1, max_workers)
        self._memory = memory_storage or EvaluationMemoryStorage()
        # Strong references to in-flight background runs so the event
        # loop does not garbage-collect a task mid-execution.
        self._background_tasks: set[asyncio.Task[None]] = set()

    @property
    def tenant_id(self) -> int:
        """The caller's active workspace id."""
        return self._tenant_id

    # ── Public API ──────────────────────────────────────────────────

    async def create(
        self,
        query: EvaluationCreateQuery,
    ) -> EvaluationGetResponseData:
        """Register a new evaluation and kick off the background run.

        Returns a snapshot suitable for the create response so the
        caller can poll ``task.id`` immediately.
        """
        chat_model_id = query.chat_model_id
        rerank_model_id = query.rerank_model_id
        if not chat_model_id:
            chat_model_id = await self._pick_default_model(
                model_type=_KNOWLEDGE_QA_MODEL_TYPE,
            )
            if not chat_model_id:
                raise ValidationError(
                    code="evaluation.no_chat_model",
                    message="no default knowledge-QA model available",
                )
        if not rerank_model_id:
            rerank_model_id = await self._pick_default_model(
                model_type=_RERANK_MODEL_TYPE,
            )

        dataset_id = query.dataset_id or DEFAULT_DATASET_ID
        task_id = generate_task_id(
            task_type="evaluation",
            tenant_id=self._tenant_id,
            business_id=dataset_id,
        )
        start = _now()
        task = EvalTask(
            id=task_id,
            tenant_id=self._tenant_id,
            dataset_id=dataset_id,
            start_time=start,
            status=_STATUS_PENDING,
            total=0,
            finished=0,
        )
        params = EvaluationParams(
            knowledge_base_id=query.knowledge_base_id,
            chat_model_id=chat_model_id,
            rerank_model_id=rerank_model_id,
        )
        await self._persist_initial(
            task=task,
            params=params,
            dataset_id=dataset_id,
        )
        self._memory.register(task=task, params=params)

        background_task = asyncio.create_task(
            self._run_in_background(
                task_id=task_id,
                knowledge_base_id=query.knowledge_base_id,
                dataset_id=dataset_id,
            ),
        )
        self._background_tasks.add(background_task)
        background_task.add_done_callback(self._background_tasks.discard)
        return self._snapshot(task=task, params=params, metric=None)

    async def get(self, task_id: str) -> EvaluationGetResponseData:
        """Return the latest snapshot for ``task_id``.

        Raises :class:`NotFoundError` when ``task_id`` is unknown to the
        SQL repository; tenant mismatches surface as
        :class:`ValidationError`.
        """
        if not task_id.strip():
            raise ValidationError(
                code="evaluation.task_id_required",
                message="task_id is required",
            )
        row = await self._evaluation_repo.get_by_id(task_id)
        if row.tenant_id != self._tenant_id:
            raise ValidationError(
                code="evaluation.tenant_mismatch",
                message="task does not belong to the active tenant",
            )
        task = EvalTask(
            id=row.id,
            tenant_id=row.tenant_id,
            dataset_id=row.dataset_id,
            start_time=row.started_at or _now(),
            status=_status_to_int(row.status),
            total=row.total,
            finished=row.finished,
        )
        params = self._memory.get_params(task_id) or self._params_from_row(row)
        metric = self._metric_from_memory(task_id)
        return self._snapshot(task=task, params=params, metric=metric)

    # ── Background runner ───────────────────────────────────────────

    async def _run_in_background(
        self,
        *,
        task_id: str,
        knowledge_base_id: str,
        dataset_id: str,
    ) -> None:
        """Run the dataset, update in-memory state, persist outcomes."""
        try:
            self._set_status(task_id, _STATUS_RUNNING)
            params = self._memory.get_params(task_id)
            if params is None:
                logger.warning("evaluation task %s vanished from memory", task_id)
                return

            resolved_kb = await self._resolve_knowledge_base_id(
                knowledge_base_id=knowledge_base_id or params.knowledge_base_id,
            )
            if params.knowledge_base_id != resolved_kb:
                resolved_params = EvaluationParams(
                    knowledge_base_id=resolved_kb,
                    chat_model_id=params.chat_model_id,
                    rerank_model_id=params.rerank_model_id,
                    query=params.query,
                    rewrite_query=params.rewrite_query,
                    search_result=params.search_result,
                    rerank_result=params.rerank_result,
                    chat_response=params.chat_response,
                )
                self._memory.replace_params(task_id, resolved_params)
            await self.run_dataset(
                task_id=task_id,
                dataset_id=dataset_id,
                knowledge_base_id=resolved_kb,
            )
            self._set_status(task_id, _STATUS_SUCCESS)
        except Exception as exc:
            logger.exception("evaluation task %s failed: %s", task_id, exc)
            self._memory.set_status(
                task_id,
                status=_STATUS_FAILED,
                error_msg=str(exc),
            )

    async def run_dataset(
        self,
        *,
        task_id: str,
        dataset_id: str,
        knowledge_base_id: str,
    ) -> EvalMetric:
        """Run the dataset end-to-end and return the averaged metric.

        Public for tests; the production entry point is :meth:`create`.
        Cleans up the temporary knowledge entry + KB regardless of
        outcome.
        """
        if self._qa_runner is None:
            raise ValidationError(
                code="evaluation.qa_runner_required",
                message="no QA runner wired for evaluation",
            )
        if self._knowledge_factory is None:
            raise ValidationError(
                code="evaluation.knowledge_factory_required",
                message="no knowledge factory wired for evaluation",
            )

        qa_pairs = await self._dataset_service.get_dataset_by_id(dataset_id)
        total = len(qa_pairs)
        self._memory.update(
            task_id,
            lambda task, params, metric: (
                task.model_copy(update={"total": total}) if task is not None else None
            ),
        )

        passages = get_passage_list(qa_pairs)
        knowledge = await self._knowledge_factory.create_knowledge_from_passages(
            tenant_id=self._tenant_id,
            knowledge_base_id=knowledge_base_id,
            passages=passages,
        )

        metric_bundle: EvalMetric | None = None
        cleanup_error: Exception | None = None
        try:
            metric_bundle = await self._evaluate_pairs(
                task_id=task_id,
                qa_pairs=qa_pairs,
                knowledge_base_id=knowledge_base_id,
            )
        finally:
            try:
                await self._knowledge_factory.delete_knowledge(
                    tenant_id=self._tenant_id,
                    knowledge_id=knowledge.id,
                )
            except Exception as exc:
                logger.warning("failed to delete evaluation knowledge: %s", exc)
                cleanup_error = exc
            if self._kb_deleter is not None:
                try:
                    await self._kb_deleter.delete_knowledge_base(
                        knowledge_base_id=knowledge_base_id,
                    )
                except Exception as exc:
                    logger.warning("failed to delete evaluation KB: %s", exc)
                    cleanup_error = cleanup_error or exc

        if cleanup_error is not None:
            raise cleanup_error
        assert metric_bundle is not None
        return metric_bundle

    async def _evaluate_pairs(
        self,
        *,
        task_id: str,
        qa_pairs: list[QAPair],
        knowledge_base_id: str,
    ) -> EvalMetric:
        """Fan out per-pair QA work and aggregate the metric bundle."""
        runner = self._qa_runner
        if runner is None:
            raise ValidationError(
                code="evaluation.qa_runner_required",
                message="no QA runner wired for evaluation",
            )
        hook = HookMetric(capacity=len(qa_pairs))
        counter_lock = asyncio.Lock()
        counter = {"finished": 0}
        semaphore = asyncio.Semaphore(self._max_workers)

        async def _run_one(index: int, qa_pair: QAPair) -> None:
            base = self._memory.get_params(task_id)
            if base is None:
                return
            params = base.clone()
            params.query = qa_pair.question
            params.rewrite_query = qa_pair.question
            await runner.knowledge_qa(
                tenant_id=self._tenant_id,
                params=params,
            )
            hook.record_init(index)
            hook.record_qa_pair(index, qa_pair)
            hook.record_search_result(index, params.search_result)
            hook.record_rerank_result(index, params.rerank_result)
            hook.record_chat_response(index, params.chat_response)
            hook.record_finish(index)
            async with counter_lock:
                counter["finished"] += 1
            snapshot = hook.metric_result()
            finished = counter["finished"]
            metric_dto = _eval_metric_from_result(snapshot)
            self._memory.store_metric(task_id, metric_dto)
            self._memory.update(
                task_id,
                lambda task, params, metric: (
                    task.model_copy(update={"finished": finished}) if task is not None else None
                ),
            )

        async def _gated(index: int, qa_pair: QAPair) -> None:
            async with semaphore:
                await _run_one(index, qa_pair)

        await asyncio.gather(*(_gated(i, pair) for i, pair in enumerate(qa_pairs)))
        return _eval_metric_from_result(hook.metric_result())

    # ── Default resolution ──────────────────────────────────────────

    async def _resolve_knowledge_base_id(
        self,
        *,
        knowledge_base_id: str,
    ) -> str:
        """Create (or clone) a knowledge base when the caller passed one.

        Mirrors the upstream branch that builds a temporary evaluation
        KB from the model's default embedding + summary ids when the
        caller did not provide one, or clones an existing KB's model
        binding when they did. Returns the resolved KB id; the cleanup
        of the temporary KB is the caller's responsibility (handled
        inside :meth:`run_dataset`).
        """
        if not knowledge_base_id:
            if self._kb_creator is None:
                raise ValidationError(
                    code="evaluation.kb_creator_required",
                    message=("knowledge_base_id is required when no KB creator is wired"),
                )
            embedding_id, llm_id = await self._default_kb_models()
            return await self._kb_creator.create_knowledge_base(
                name="evaluation",
                description="evaluation",
                embedding_model_id=embedding_id,
                summary_model_id=llm_id,
            )

        if self._kb_reader is None or self._kb_creator is None:
            raise ValidationError(
                code="evaluation.kb_reader_required",
                message="knowledge_base_id was supplied but no KB reader is wired",
            )
        existing = await self._kb_reader.get_knowledge_base_by_id(
            knowledge_base_id=knowledge_base_id,
        )
        return await self._kb_creator.create_knowledge_base(
            name="evaluation",
            description="evaluation",
            embedding_model_id=existing.embedding_model_id,
            summary_model_id=existing.summary_model_id,
        )

    async def _default_kb_models(self) -> tuple[str, str]:
        """Pick the first embedding + knowledge-QA model for the tenant."""
        rows = await self._model_service.list_models(
            tenant_id=self._tenant_id,
            include_builtin=True,
        )
        embedding_id = ""
        llm_id = ""
        for row in rows:
            type_name = str(getattr(row, "type", ""))
            if type_name == _EMBEDDING_MODEL_TYPE and not embedding_id:
                embedding_id = str(getattr(row, "id", ""))
            elif type_name == _KNOWLEDGE_QA_MODEL_TYPE and not llm_id:
                llm_id = str(getattr(row, "id", ""))
        if not embedding_id or not llm_id:
            raise ValidationError(
                code="evaluation.no_default_models",
                message="no default embedding / knowledge-qa model for the tenant",
            )
        return embedding_id, llm_id

    async def _pick_default_model(
        self,
        *,
        model_type: str,
    ) -> str:
        """Return the first tenant-visible model of ``model_type``.

        Returns ``""`` when no such model exists; the caller treats
        this as "no default available" (matches the upstream fallback
        that logs a warning and continues).
        """
        rows = await self._model_service.list_models(
            tenant_id=self._tenant_id,
            include_builtin=True,
        )
        for row in rows:
            if str(getattr(row, "type", "")) == model_type:
                return str(getattr(row, "id", ""))
        return ""

    # ── Status / persistence helpers ────────────────────────────────

    def _set_status(self, task_id: str, status: int) -> None:
        """Update the in-memory status; persistence happens at completion."""
        self._memory.set_status(task_id, status=status)

    async def _persist_initial(
        self,
        *,
        task: EvalTask,
        params: EvaluationParams,
        dataset_id: str,
    ) -> None:
        """Insert the durable rows backing a freshly created task."""
        now = _now()
        eval_row = Evaluation(
            id=task.id,
            tenant_id=task.tenant_id,
            dataset_id=dataset_id,
            knowledge_base_id=params.knowledge_base_id,
            chat_model_id=params.chat_model_id,
            rerank_model_id=params.rerank_model_id,
            status=EVAL_STATUS_PENDING,
            total=0,
            finished=0,
            error_msg="",
            params={"rerank_top_k": _DEFAULT_RERANK_TOP_K},
            started_at=task.start_time,
            finished_at=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        await self._evaluation_repo.create(eval_row)
        run_row = EvaluationRun(
            id=task.id,
            evaluation_id=task.id,
            status=EVAL_STATUS_PENDING,
            total=0,
            finished=0,
            error_msg="",
            started_at=task.start_time,
            finished_at=None,
            created_at=now,
            updated_at=now,
            deleted_at=None,
        )
        await self._run_repo.create(run_row)

    def _params_from_row(self, row: Evaluation) -> EvaluationParams:
        """Reconstruct the params payload from the SQL row when memory lacks it."""
        return EvaluationParams(
            knowledge_base_id=row.knowledge_base_id,
            chat_model_id=row.chat_model_id,
            rerank_model_id=row.rerank_model_id,
        )

    def _metric_from_memory(self, task_id: str) -> EvalMetric | None:
        """Return the metric bundle, or ``None`` when none has been recorded."""
        return self._memory.get_metric(task_id)

    @staticmethod
    def _snapshot(
        *,
        task: EvalTask,
        params: EvaluationParams | None,
        metric: EvalMetric | None,
    ) -> EvaluationGetResponseData:
        """Build a :class:`EvaluationGetResponseData` view for the caller."""
        params_view = EvalTaskParams(
            knowledge_base_id=params.knowledge_base_id if params else "",
            chat_model_id=params.chat_model_id if params else "",
            rerank_model_id=params.rerank_model_id if params else "",
            rerank_top_k=_DEFAULT_RERANK_TOP_K,
        )
        return EvaluationGetResponseData(
            task=task,
            params=params_view,
            metric=metric,
        )


# ── Helpers ────────────────────────────────────────────────────────────


def generate_task_id(
    *,
    task_type: str,
    tenant_id: int,
    business_id: str = "",
) -> str:
    """Mint a fresh task id.

    Mirrors the upstream ``utils.GenerateTaskID``: sanitised task type,
    tenant id, current millisecond timestamp, an 8-char uuid fragment,
    and an optional business id.
    """
    safe_type = _sanitize_task_type(task_type)
    short_uuid = uuid.uuid4().hex[:8]
    components = [
        safe_type,
        str(tenant_id),
        str(int(time.time() * 1000)),
        short_uuid,
    ]
    if business_id:
        components.append(_sanitize_business_id(business_id))
    return "_".join(components)


def _sanitize_task_type(task_type: str) -> str:
    """Strip colons / dashes / spaces; lowercase.

    Matches the upstream ``sanitizeTaskType`` helper.
    """
    cleaned = task_type.replace(":", "_").replace("-", "_").replace(" ", "_")
    return cleaned.lower() or "task"


def _sanitize_business_id(business_id: str) -> str:
    """Truncate to 12 chars and drop dashes / underscores / colons.

    Matches the upstream ``sanitizeBusinessID`` helper.
    """
    truncated = business_id[:12]
    for ch in ("-", "_", ":"):
        truncated = truncated.replace(ch, "")
    return truncated


def _status_to_int(status: str) -> int:
    """Translate a SQL status string into the integer the contract uses."""
    mapping = {
        EVAL_STATUS_PENDING: _STATUS_PENDING,
        EVAL_STATUS_RUNNING: _STATUS_RUNNING,
        EVAL_STATUS_SUCCESS: _STATUS_SUCCESS,
        EVAL_STATUS_FAILED: _STATUS_FAILED,
    }
    return mapping.get(status, _STATUS_PENDING)


def _eval_metric_from_result(result: MetricResult) -> EvalMetric:
    """Convert an internal :class:`MetricResult` into the contract DTO."""
    retrieval = EvalRetrievalMetrics(
        precision=result.retrieval.precision,
        recall=result.retrieval.recall,
        ndcg3=result.retrieval.ndcg3,
        ndcg10=result.retrieval.ndcg10,
        mrr=result.retrieval.mrr,
        map=result.retrieval.map_,
    )
    generation = EvalGenerationMetrics(
        bleu1=result.generation.bleu1,
        bleu2=result.generation.bleu2,
        bleu4=result.generation.bleu4,
        rouge1=result.generation.rouge1,
        rouge2=result.generation.rouge2,
        rougel=result.generation.rougel,
    )
    return EvalMetric(
        retrieval_metrics=retrieval,
        generation_metrics=generation,
    )


__all__ = [
    "EvaluationService",
    "generate_task_id",
]

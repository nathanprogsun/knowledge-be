"""Unit tests for the evaluation service orchestration.

Exercises ``create`` / ``get`` / ``run_dataset`` with tiny in-memory
fakes for the repositories and every heavy seam (model service, dataset
service, KB creator/reader/deleter, knowledge factory, QA runner). The
real repositories are not exercised here; their SQL integration tests
live under ``tests/db/``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.common.exception import NotFoundError, ValidationError
from src.core.contracts.evaluation import EvalMetric, EvalTask
from src.core.evaluation.dataset import QAPair
from src.core.evaluation.service.evaluation_service import (
    EvaluationService,
    generate_task_id,
)
from src.core.evaluation.service.seams import (
    EvaluationCreateQuery,
    EvaluationParams,
    QARunner,
)
from src.core.evaluation.service.storage import EvaluationMemoryStorage
from src.db.models.evaluation import (
    Evaluation,
    EvaluationDataset,
    EvaluationMetric,
    EvaluationRun,
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


# ── Value fakes ───────────────────────────────────────────────────────


class _Model:
    """Stand-in for a ``ModelInfo`` row exposing ``id`` / ``type``."""

    def __init__(self, model_id: str, model_type: str) -> None:
        self.id = model_id
        self.type = model_type


class _Knowledge:
    """Stand-in for a ``Knowledge`` domain object carrying ``id``."""

    def __init__(self, knowledge_id: str) -> None:
        self.id = knowledge_id


class _KBInfo:
    """Stand-in for the KB projection exposing the model bindings."""

    def __init__(self, embedding_model_id: str, summary_model_id: str) -> None:
        self.embedding_model_id = embedding_model_id
        self.summary_model_id = summary_model_id


class _Hit:
    """Minimal search-hit stand-in carrying only ``content``."""

    def __init__(self, content: str) -> None:
        self.content = content


class _Chat:
    """Minimal chat-response stand-in carrying only ``content``."""

    def __init__(self, content: str) -> None:
        self.content = content


# ── Repository fakes ──────────────────────────────────────────────────


class _FakeEvaluationRepo:
    def __init__(self) -> None:
        self.store: dict[str, Evaluation] = {}
        self.created: list[Evaluation] = []

    async def create(self, row: Evaluation) -> Evaluation:
        self.store[row.id] = row
        self.created.append(row)
        return row

    async def get_by_id(self, id: str) -> Evaluation:
        row = self.store.get(id)
        if row is None or row.deleted_at is not None:
            raise NotFoundError(
                code="evaluation.not_found",
                message=f"evaluation {id} not found",
            )
        return row


class _FakeDatasetRepo:
    def __init__(self) -> None:
        self.store: dict[str, EvaluationDataset] = {}
        self.created: list[EvaluationDataset] = []

    async def create(self, row: EvaluationDataset) -> EvaluationDataset:
        self.store[row.id] = row
        self.created.append(row)
        return row


class _FakeRunRepo:
    def __init__(self) -> None:
        self.store: dict[str, EvaluationRun] = {}
        self.created: list[EvaluationRun] = []

    async def create(self, row: EvaluationRun) -> EvaluationRun:
        self.store[row.id] = row
        self.created.append(row)
        return row


class _FakeMetricRepo:
    def __init__(self) -> None:
        self.created: list[EvaluationMetric] = []

    async def create(self, row: EvaluationMetric) -> None:
        self.created.append(row)


# ── Seam fakes ────────────────────────────────────────────────────────


class _FakeModelService:
    def __init__(self, models: list[_Model]) -> None:
        self._models = models

    async def list_models(
        self,
        *,
        tenant_id: int,
        model_type: str | None = None,
        source: str | None = None,
        include_builtin: bool = True,
    ) -> list[_Model]:
        rows = self._models
        if model_type:
            rows = [row for row in rows if row.type == model_type]
        return list(rows)


class _FakeDatasetService:
    def __init__(self, pairs: list[QAPair] | None = None) -> None:
        self._pairs = pairs or []

    async def get_dataset_by_id(self, dataset_id: str) -> list[QAPair]:
        return list(self._pairs)


class _FakeKB:
    def __init__(self, *, existing_embedding: str = "emb", existing_summary: str = "sum") -> None:
        self._existing_embedding = existing_embedding
        self._existing_summary = existing_summary
        self.created: list[dict[str, str]] = []
        self.deleted: list[str] = []
        self.reads: list[str] = []

    async def create_knowledge_base(
        self,
        *,
        name: str,
        description: str,
        embedding_model_id: str,
        summary_model_id: str,
    ) -> str:
        kb_id = f"kb-{len(self.created) + 1}"
        self.created.append(
            {
                "name": name,
                "description": description,
                "embedding_model_id": embedding_model_id,
                "summary_model_id": summary_model_id,
            }
        )
        return kb_id

    async def delete_knowledge_base(self, *, knowledge_base_id: str) -> bool:
        self.deleted.append(knowledge_base_id)
        return True

    async def get_knowledge_base_by_id(self, *, knowledge_base_id: str) -> _KBInfo:
        self.reads.append(knowledge_base_id)
        return _KBInfo(self._existing_embedding, self._existing_summary)


class _FakeKnowledgeFactory:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.deleted: list[str] = []
        self._next_id = 0

    async def create_knowledge_from_passages(
        self,
        *,
        tenant_id: int,
        knowledge_base_id: str,
        passages: list[str],
    ) -> _Knowledge:
        self._next_id += 1
        knowledge_id = f"knowledge-{self._next_id}"
        self.created.append(
            {
                "tenant_id": tenant_id,
                "knowledge_base_id": knowledge_base_id,
                "passages": passages,
            }
        )
        return _Knowledge(knowledge_id)

    async def delete_knowledge(self, *, tenant_id: int, knowledge_id: str) -> bool:
        self.deleted.append(knowledge_id)
        return True


class _FakeQARunner:
    """Records each invocation; fills the params with structured artefacts."""

    def __init__(self) -> None:
        self.calls: list[EvaluationParams] = []
        self.search_results: list[list[_Hit]] = []
        self.rerank_results: list[list[_Hit]] = []
        self.chat_responses: list[_Chat | None] = []

    async def knowledge_qa(
        self,
        *,
        tenant_id: int,
        params: EvaluationParams,
    ) -> None:
        self.calls.append(params)
        if self.search_results:
            params.search_result = list(self.search_results.pop(0))
        if self.rerank_results:
            params.rerank_result = list(self.rerank_results.pop(0))
        if self.chat_responses:
            params.chat_response = self.chat_responses.pop(0)


# ── Builder ───────────────────────────────────────────────────────────


def _pairs(n: int = 2) -> list[QAPair]:
    return [
        QAPair(
            qid=i + 1,
            question=f"question {i + 1}",
            pids=[i + 1],
            passages=[f"passage {i + 1}"],
            aid=i + 1,
            answer=f"answer {i + 1}",
        )
        for i in range(n)
    ]


def _build_service(
    *,
    tenant_id: int = 1,
    models: list[_Model] | None = None,
    pairs: list[QAPair] | None = None,
    kb: _FakeKB | None = None,
    runner: QARunner | None = None,
    knowledge_factory: _FakeKnowledgeFactory | None = None,
    evaluation_repo: _FakeEvaluationRepo | None = None,
) -> EvaluationService:
    kb = kb or _FakeKB()
    knowledge_factory = knowledge_factory or _FakeKnowledgeFactory()
    return EvaluationService(
        tenant_id=tenant_id,
        evaluation_repo=evaluation_repo or _FakeEvaluationRepo(),  # type: ignore[arg-type]
        dataset_repo=_FakeDatasetRepo(),  # type: ignore[arg-type]
        run_repo=_FakeRunRepo(),  # type: ignore[arg-type]
        metric_repo=_FakeMetricRepo(),  # type: ignore[arg-type]
        model_service=_FakeModelService(models or []),  # type: ignore[arg-type]
        dataset_service=_FakeDatasetService(pairs or _pairs()),
        kb_creator=kb,
        kb_reader=kb,
        kb_deleter=kb,
        knowledge_factory=knowledge_factory,  # type: ignore[arg-type]
        qa_runner=runner,
        max_workers=2,
    )


# ── create ────────────────────────────────────────────────────────────


class TestCreate:
    async def test_create_persists_rows_and_returns_pending_snapshot(self) -> None:
        service = _build_service()
        response = await service.create(
            EvaluationCreateQuery(
                dataset_id="default",
                knowledge_base_id="kb-1",
                chat_model_id="chat-1",
                rerank_model_id="rerank-1",
            )
        )
        assert response.task.id.startswith("evaluation_")
        assert response.task.tenant_id == 1
        assert response.task.dataset_id == "default"
        assert response.task.status == 0  # pending
        assert response.params.chat_model_id == "chat-1"
        assert response.params.rerank_model_id == "rerank-1"
        assert response.params.knowledge_base_id == "kb-1"
        # Both the evaluation row and its initial run row are persisted.
        assert len(service._evaluation_repo.created) == 1  # type: ignore[attr-defined]
        assert len(service._run_repo.created) == 1  # type: ignore[attr-defined]

    async def test_create_resolves_default_models(self) -> None:
        models = [
            _Model("emb-1", "Embedding"),
            _Model("qa-1", "KnowledgeQA"),
            _Model("rerank-1", "Rerank"),
        ]
        service = _build_service(models=models)
        response = await service.create(
            EvaluationCreateQuery(
                dataset_id="default",
                knowledge_base_id="kb-1",
            )
        )
        assert response.params.chat_model_id == "qa-1"
        assert response.params.rerank_model_id == "rerank-1"

    async def test_create_rejects_missing_dataset_id(self) -> None:
        service = _build_service()
        with pytest.raises(ValidationError) as exc:
            await service.create(EvaluationCreateQuery())
        assert exc.value.code == "evaluation.dataset_id_required"

    async def test_create_rejects_missing_chat_model(self) -> None:
        service = _build_service(models=[_Model("emb-1", "Embedding")])
        with pytest.raises(ValidationError) as exc:
            await service.create(EvaluationCreateQuery(dataset_id="default"))
        assert exc.value.code == "evaluation.no_chat_model"

    async def test_create_returns_default_rerank_when_absent(self) -> None:
        models = [_Model("emb-1", "Embedding"), _Model("qa-1", "KnowledgeQA")]
        service = _build_service(models=models)
        response = await service.create(EvaluationCreateQuery(dataset_id="default"))
        assert response.params.chat_model_id == "qa-1"
        assert response.params.rerank_model_id == ""


# ── get ───────────────────────────────────────────────────────────────


class TestGet:
    async def test_get_returns_snapshot_from_repo(self) -> None:
        service = _build_service()
        response = await service.create(
            EvaluationCreateQuery(
                dataset_id="default",
                knowledge_base_id="kb-1",
                chat_model_id="chat-1",
                rerank_model_id="rerank-1",
            )
        )
        fetched = await service.get(response.task.id)
        assert fetched.task.id == response.task.id
        assert fetched.task.dataset_id == "default"
        assert fetched.params.chat_model_id == "chat-1"

    async def test_get_raises_tenant_mismatch(self) -> None:
        shared_repo = _FakeEvaluationRepo()
        service = _build_service(tenant_id=1, evaluation_repo=shared_repo)
        other = _build_service(tenant_id=2, evaluation_repo=shared_repo)
        response = await service.create(
            EvaluationCreateQuery(
                dataset_id="default",
                knowledge_base_id="kb-1",
                chat_model_id="chat-1",
            )
        )
        with pytest.raises(ValidationError) as exc:
            await other.get(response.task.id)
        assert exc.value.code == "evaluation.tenant_mismatch"

    async def test_get_raises_not_found(self) -> None:
        service = _build_service()
        with pytest.raises(NotFoundError):
            await service.get("missing-task")

    async def test_get_requires_task_id(self) -> None:
        service = _build_service()
        with pytest.raises(ValidationError) as exc:
            await service.get("  ")
        assert exc.value.code == "evaluation.task_id_required"


# ── run_dataset ───────────────────────────────────────────────────────


class TestRunDataset:
    async def test_run_dataset_orchestrates_pipeline(self) -> None:
        runner = _FakeQARunner()
        runner.search_results = [[_Hit("passage 1")], [_Hit("passage 2")]]
        runner.chat_responses = [_Chat("answer 1"), _Chat("answer 2")]
        kb = _FakeKB()
        knowledge_factory = _FakeKnowledgeFactory()
        service = _build_service(
            pairs=_pairs(2),
            kb=kb,
            runner=runner,
            knowledge_factory=knowledge_factory,
        )
        params = EvaluationParams(
            knowledge_base_id="",
            chat_model_id="chat-1",
            rerank_model_id="rerank-1",
        )
        task = EvalTask(
            id="task-1",
            tenant_id=1,
            dataset_id="default",
            start_time=_NOW,
            status=0,
            total=0,
            finished=0,
        )
        memory = EvaluationMemoryStorage()
        memory.register(task=task, params=params)
        service._memory = memory

        metric = await service.run_dataset(
            task_id="task-1",
            dataset_id="default",
            knowledge_base_id="kb-1",
        )
        assert isinstance(metric, EvalMetric)
        assert metric.retrieval_metrics is not None
        assert metric.retrieval_metrics.recall == pytest.approx(1.0)
        assert metric.generation_metrics is not None
        assert metric.generation_metrics.bleu1 == pytest.approx(1.0)
        # The dataset's passages were ingested in one knowledge entry,
        # flattened into a pid-indexed list (pid 0 stays empty).
        assert len(knowledge_factory.created) == 1
        assert knowledge_factory.created[0]["knowledge_base_id"] == "kb-1"
        assert knowledge_factory.created[0]["passages"] == [
            "",
            "passage 1",
            "passage 2",
        ]
        # Cleanup ran for the knowledge entry and the KB.
        assert len(knowledge_factory.deleted) == 1
        assert kb.deleted == ["kb-1"]
        # Every QA pair was run through the runner.
        assert len(runner.calls) == 2
        assert [call.query for call in runner.calls] == ["question 1", "question 2"]
        # The in-memory task now shows total/finished.
        live = memory.get_task("task-1")
        assert live is not None
        assert live.total == 2
        assert live.finished == 2

    async def test_run_dataset_builds_evaluation_kb_when_missing(self) -> None:
        runner = _FakeQARunner()
        runner.search_results = [[_Hit("passage 1")]]
        runner.chat_responses = [_Chat("answer 1")]
        kb = _FakeKB()
        knowledge_factory = _FakeKnowledgeFactory()
        service = _build_service(
            pairs=_pairs(1),
            kb=kb,
            runner=runner,
            knowledge_factory=knowledge_factory,
            models=[
                _Model("emb-1", "Embedding"),
                _Model("qa-1", "KnowledgeQA"),
            ],
        )
        memory = EvaluationMemoryStorage()
        params = EvaluationParams(
            knowledge_base_id="",
            chat_model_id="chat-1",
            rerank_model_id="",
        )
        task = EvalTask(
            id="task-2",
            tenant_id=1,
            dataset_id="default",
            start_time=_NOW,
            status=0,
            total=0,
            finished=0,
        )
        memory.register(task=task, params=params)
        service._memory = memory

        resolved = await service._resolve_knowledge_base_id(knowledge_base_id="")
        assert resolved == "kb-1"
        assert kb.created[0]["embedding_model_id"] == "emb-1"
        assert kb.created[0]["summary_model_id"] == "qa-1"

    async def test_run_dataset_requires_qa_runner(self) -> None:
        service = _build_service(runner=None)
        with pytest.raises(ValidationError) as exc:
            await service.run_dataset(
                task_id="task-1",
                dataset_id="default",
                knowledge_base_id="kb-1",
            )
        assert exc.value.code == "evaluation.qa_runner_required"

    async def test_run_dataset_cleanup_runs_on_failure(self) -> None:
        class _FailingRunner:
            async def knowledge_qa(self, *, tenant_id: int, params: EvaluationParams) -> None:
                raise RuntimeError("boom")

        kb = _FakeKB()
        knowledge_factory = _FakeKnowledgeFactory()
        service = _build_service(
            pairs=_pairs(1),
            kb=kb,
            runner=_FailingRunner(),
            knowledge_factory=knowledge_factory,
        )
        memory = EvaluationMemoryStorage()
        params = EvaluationParams(
            knowledge_base_id="",
            chat_model_id="chat-1",
            rerank_model_id="",
        )
        task = EvalTask(
            id="task-3",
            tenant_id=1,
            dataset_id="default",
            start_time=_NOW,
            status=0,
            total=0,
            finished=0,
        )
        memory.register(task=task, params=params)
        service._memory = memory

        with pytest.raises(RuntimeError, match="boom"):
            await service.run_dataset(
                task_id="task-3",
                dataset_id="default",
                knowledge_base_id="kb-1",
            )
        # Knowledge and KB were still cleaned up.
        assert len(knowledge_factory.deleted) == 1
        assert kb.deleted == ["kb-1"]


# ── Memory storage ────────────────────────────────────────────────────


class TestMemoryStorage:
    def _boom(
        self,
        task: EvalTask | None,
        params: EvaluationParams | None,
        metric: EvalMetric | None,
    ) -> None:
        raise AssertionError("callback must not run for a missing task")

    def test_update_missing_task_is_noop(self) -> None:
        memory = EvaluationMemoryStorage()
        memory.update("nope", self._boom)
        assert memory.get_task("nope") is None

    def test_set_status_finalizes_finished_count(self) -> None:
        memory = EvaluationMemoryStorage()
        task = EvalTask(
            id="t",
            tenant_id=1,
            dataset_id="d",
            start_time=_NOW,
            status=0,
            total=10,
            finished=3,
        )
        params = EvaluationParams(
            knowledge_base_id="",
            chat_model_id="",
            rerank_model_id="",
        )
        memory.register(task=task, params=params)
        memory.set_status("t", status=2)
        live = memory.get_task("t")
        assert live is not None
        assert live.status == 2
        assert live.finished == 10


# ── Task ids ──────────────────────────────────────────────────────────


class TestTaskId:
    def test_format(self) -> None:
        task_id = generate_task_id(
            task_type="Evaluation",
            tenant_id=7,
            business_id="kb-123",
        )
        parts = task_id.split("_")
        assert parts[0] == "evaluation"
        assert parts[1] == "7"
        assert len(parts[2]) == 13  # millisecond timestamp
        assert len(parts[3]) == 8  # uuid fragment
        assert parts[4] == "kb123"  # business id sanitised

    def test_tenant_validation(self) -> None:
        with pytest.raises(ValidationError) as exc:
            EvaluationService(
                tenant_id=0,
                evaluation_repo=_FakeEvaluationRepo(),  # type: ignore[arg-type]
                dataset_repo=_FakeDatasetRepo(),  # type: ignore[arg-type]
                run_repo=_FakeRunRepo(),  # type: ignore[arg-type]
                metric_repo=_FakeMetricRepo(),  # type: ignore[arg-type]
                model_service=_FakeModelService([]),  # type: ignore[arg-type]
                dataset_service=_FakeDatasetService(),
            )
        assert exc.value.code == "evaluation.invalid_tenant_id"


# ── Dataset loader ─────────────────────────────────────────────────────


class TestDatasetService:
    def test_load_csv_flattens_facets(self, tmp_path: Path) -> None:
        (tmp_path / "queries.csv").write_text(
            "id,text\n1,what is qa\n2,second question\n", encoding="utf-8"
        )
        (tmp_path / "corpus.csv").write_text(
            "id,text\n10,passage ten\n20,passage twenty\n", encoding="utf-8"
        )
        (tmp_path / "answers.csv").write_text("id,text\n100,an answer\n", encoding="utf-8")
        (tmp_path / "qrels.csv").write_text("qid,pid\n1,10\n1,20\n2,10\n", encoding="utf-8")
        (tmp_path / "qas.csv").write_text("qid,aid\n1,100\n", encoding="utf-8")
        from src.core.evaluation.dataset import DatasetService

        pairs = DatasetService().load_csv(
            queries_csv=tmp_path / "queries.csv",
            corpus_csv=tmp_path / "corpus.csv",
            qrels_csv=tmp_path / "qrels.csv",
            answers_csv=tmp_path / "answers.csv",
            qas_csv=tmp_path / "qas.csv",
        )
        assert len(pairs) == 2
        first = next(p for p in pairs if p.qid == 1)
        assert first.question == "what is qa"
        assert first.pids == [10, 20]
        assert first.passages == ["passage ten", "passage twenty"]
        assert first.aid == 100
        assert first.answer == "an answer"
        second = next(p for p in pairs if p.qid == 2)
        assert second.pids == [10]
        assert second.aid == 0  # no qas row → default

    async def test_unknown_dataset_raises_not_found(self) -> None:
        from src.core.evaluation.dataset import DatasetService

        with pytest.raises(NotFoundError) as exc:
            await DatasetService().get_dataset_by_id("missing")
        assert exc.value.code == "evaluation.dataset_not_found"

    async def test_default_dataset_falls_back_when_samples_absent(self, tmp_path: Path) -> None:
        from src.core.evaluation.dataset import DatasetService

        service = DatasetService(dataset_dir=tmp_path / "does-not-exist")
        pairs = await service.get_dataset_by_id("default")
        assert len(pairs) == 1
        assert pairs[0].qid == 1

    def test_from_pairs_validates_and_defaults(self) -> None:
        from src.core.evaluation.dataset import DatasetService

        pairs = DatasetService.from_pairs(
            [{"question": "q?", "pids": [1], "passages": ["p"], "answer": "a"}]
        )
        assert pairs[0].qid == 0
        assert pairs[0].aid == 0
        assert pairs[0].answer == "a"

    def test_from_pairs_rejects_unknown_keys(self) -> None:
        from src.core.evaluation.dataset import DatasetService

        with pytest.raises(ValidationError) as exc:
            DatasetService.from_pairs(
                [{"question": "q?", "bogus": 1}]  # type: ignore[typeddict-unknown-key]
            )
        assert exc.value.code == "evaluation.unknown_pair_key"


class TestPassageList:
    def test_get_passage_list_indexes_by_pid(self) -> None:
        from src.core.evaluation.dataset import get_passage_list

        pairs = [
            QAPair(qid=1, question="q1", pids=[1, 3], passages=["a", "c"], aid=0, answer=""),
            QAPair(qid=2, question="q2", pids=[3, 5], passages=["c", "e"], aid=0, answer=""),
        ]
        assert get_passage_list(pairs) == ["", "a", "", "c", "", "e"]


# ── Factory wiring ─────────────────────────────────────────────────────


class TestFactory:
    def test_build_evaluation_service_wires_repositories(self) -> None:
        from unittest.mock import MagicMock

        from src.core.evaluation.factory import build_evaluation_service

        session = MagicMock()
        service = build_evaluation_service(session, tenant_id=1)
        assert service.tenant_id == 1
        assert service._dataset_service is not None
        assert service._qa_runner is None

    def test_build_evaluation_service_with_kb_service(self) -> None:
        from unittest.mock import MagicMock

        from src.core.evaluation.factory import build_evaluation_service

        session = MagicMock()
        kb_service = MagicMock()
        service = build_evaluation_service(session, tenant_id=1, kb_service=kb_service)
        assert service._kb_creator is not None
        assert service._kb_deleter is not None
        assert service._kb_reader is not None

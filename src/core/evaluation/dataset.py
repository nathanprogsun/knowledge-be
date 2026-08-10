"""Dataset loading for offline evaluation — QA pairs + passage extraction.

Mirrors the upstream ``DatasetService`` / ``dataset`` helper pair: the
default dataset is materialised from the sample parquet corpus shipped
with the codebase, and :func:`get_passage_list` flattens a list of QA
pairs into the passage-id-indexed list the knowledge-ingestion step
consumes.

Scope of this module
--------------------

- ``QAPair`` — the domain value carrying question / passages / answer.
- ``DatasetServiceLike`` — the seam the evaluation service depends on
  (structural, so tests can supply a tiny in-memory fake).
- ``DatasetService`` — production implementation backed by the sample
  dataset directory.
- ``get_passage_list`` — reorders corpus passages into ``[pid] → text``
  so ingestion preserves the dataset's ground-truth passage ids.
- ``DEFAULT_DATASET_ID`` — id the evaluation flow falls back to when the
  caller leaves ``dataset_id`` blank.
"""

from __future__ import annotations

import csv
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, TypedDict, runtime_checkable

from src.common.exception import NotFoundError, ValidationError

logger = logging.getLogger(__name__)

#: The built-in dataset id. Matches the upstream default the evaluation
#: flow uses when no explicit dataset is provided.
DEFAULT_DATASET_ID = "default"

#: Sample dataset location relative to the repository root. Mirrors the
#: upstream ``./dataset/samples`` directory convention.
DEFAULT_DATASET_DIR = "dataset/samples"

#: Column names accepted when loading a CSV dataset. Kept as constants so
#: callers can detect a malformed header without a magic string.
COL_QUESTION = "question"
COL_ANSWER = "answer"
COL_PASSAGE = "passage"
COL_PID = "pid"

#: Known QA-pair keys accepted by :meth:`DatasetService.from_pairs`.
KNOWN_PAIR_KEYS: frozenset[str] = frozenset(
    {
        "qid",
        "question",
        "pids",
        "passages",
        "aid",
        "answer",
    }
)


class PairInput(TypedDict, total=False):
    """Raw QA-pair payload accepted by :meth:`DatasetService.from_pairs`.

    ``question`` is the only required key; ``qid`` / ``pids`` /
    ``passages`` / ``aid`` / ``answer`` default when omitted.
    """

    qid: int
    question: str
    pids: list[int]
    passages: list[str]
    aid: int
    answer: str


@dataclass(frozen=True, slots=True)
class QAPair:
    """One evaluation example: question, related passages, answer.

    Mirrors the upstream ``types.QAPair`` shape: ``qid`` / ``pids`` /
    ``aid`` are the integer keys that tie the pair back to the dataset's
    query / passage / answer rows; ``passages`` holds the ground-truth
    passage texts in the same order as ``pids``.
    """

    qid: int
    question: str
    pids: list[int]
    passages: list[str]
    aid: int
    answer: str


@runtime_checkable
class DatasetServiceLike(Protocol):
    """Dataset surface the evaluation service depends on.

    Structural: tests provide a fake with the same method instead of
    standing up the parquet-backed production loader.
    """

    async def get_dataset_by_id(self, dataset_id: str) -> list[QAPair]:
        """Return every QA pair of ``dataset_id``.

        Unknown datasets raise :class:`NotFoundError`.
        """
        ...


class DatasetService:
    """Loads QA pairs for the built-in sample dataset.

    The production loader resolves ``DEFAULT_DATASET_ID`` against the
    sample dataset directory (CSV files; one file per facet — queries,
    corpus, answers, qrels, qas) and flattens them into
    :class:`QAPair` instances in the same order the upstream
    ``dataset.Iterate()`` produces.
    """

    def __init__(self, *, dataset_dir: str | Path = DEFAULT_DATASET_DIR) -> None:
        self._dataset_dir = Path(dataset_dir)

    async def get_dataset_by_id(self, dataset_id: str) -> list[QAPair]:
        """Return every QA pair of ``dataset_id``.

        Only the built-in ``default`` dataset is currently resolvable;
        any other id raises :class:`NotFoundError`.
        """
        if not dataset_id or dataset_id == DEFAULT_DATASET_ID:
            pairs = self._load_default()
            logger.info("loaded %d QA pairs from default dataset", len(pairs))
            return pairs
        raise NotFoundError(
            code="evaluation.dataset_not_found",
            message=f"dataset {dataset_id!r} is not registered",
        )

    def load_csv(
        self,
        *,
        queries_csv: str | Path,
        corpus_csv: str | Path,
        qrels_csv: str | Path,
        answers_csv: str | Path | None = None,
        qas_csv: str | Path | None = None,
    ) -> list[QAPair]:
        """Build QA pairs from five CSV facets.

        Mirrors the upstream parquet layout with CSV as the storage
        format: ``queries`` (``id``, ``text``), ``corpus`` (``id``,
        ``text``), ``answers`` (``id``, ``text``), ``qrels`` (``qid``,
        ``pid``), ``qas`` (``qid``, ``aid``).
        """
        queries = _read_text_rows(queries_csv)
        corpus = _read_text_rows(corpus_csv)
        qrels = _read_int_pair_rows(qrels_csv)
        answers = _read_text_rows(answers_csv) if answers_csv is not None else {}
        qas = _read_int_map(qas_csv) if qas_csv is not None else {}

        pairs: list[QAPair] = []
        for qid, question in queries.items():
            pids = list(qrels.get(qid, ()))
            passages = [corpus[pid] for pid in pids if pid in corpus]
            aid = qas.get(qid, 0)
            answer = answers.get(aid, "")
            pairs.append(
                QAPair(
                    qid=qid,
                    question=question,
                    pids=pids,
                    passages=passages,
                    aid=aid,
                    answer=answer,
                )
            )
        return pairs

    # ── Default dataset ─────────────────────────────────────────────

    def _load_default(self) -> list[QAPair]:
        """Load the sample dataset shipped with the repo.

        When the sample files are absent the loader falls back to a
        tiny built-in pair set so a fresh checkout can still exercise
        the evaluation flow; a missing directory is logged rather than
        raised.
        """
        base = self._dataset_dir
        if not base.is_dir():
            logger.warning(
                "sample dataset dir %s not found — using built-in pair", base,
            )
            return [_fallback_pair()]
        pairs = self.load_csv(
            queries_csv=base / "queries.csv",
            corpus_csv=base / "corpus.csv",
            qrels_csv=base / "qrels.csv",
            answers_csv=base / "answers.csv",
            qas_csv=base / "qas.csv",
        )
        if not pairs:
            logger.warning("sample dataset %s is empty — using built-in pair", base)
            return [_fallback_pair()]
        return pairs

    # ── Alternate input: in-memory pairs ────────────────────────────

    @staticmethod
    def from_pairs(pairs: Iterable[PairInput]) -> list[QAPair]:
        """Build a validated :class:`QAPair` list from raw dictionaries.

        Used for programmatic dataset creation (e.g. the web layer
        building a dataset from user upload). ``question`` is required;
        ``qid`` / ``pids`` / ``passages`` / ``aid`` / ``answer`` are
        optional and default to ``0`` / ``[]`` / ``""``.
        """
        result: list[QAPair] = []
        for raw in pairs:
            if not isinstance(raw, dict):
                raise ValidationError(
                    code="evaluation.invalid_pair",
                    message="each QA pair must be a mapping",
                )
            unknown = set(raw) - KNOWN_PAIR_KEYS
            if unknown:
                raise ValidationError(
                    code="evaluation.unknown_pair_key",
                    message=f"unknown QA pair key(s): {sorted(unknown)}",
                )
            question = str(raw.get("question", "")).strip()
            if not question:
                raise ValidationError(
                    code="evaluation.question_required",
                    message="question is required",
                )
            result.append(
                QAPair(
                    qid=int(raw.get("qid", 0)),
                    question=question,
                    pids=[int(p) for p in raw.get("pids", [])],
                    passages=[str(p) for p in raw.get("passages", [])],
                    aid=int(raw.get("aid", 0)),
                    answer=str(raw.get("answer", "")),
                )
            )
        return result


# ── Passage flattening ────────────────────────────────────────────────


def get_passage_list(qa_pairs: Iterable[QAPair]) -> list[str]:
    """Flatten the dataset's passages into a ``pid``-indexed list.

    Mirrors the upstream ``getPassageList``: builds ``pid → text`` from
    every pair, then returns a list where ``result[pid] == text`` (empty
    strings for gaps). The ingestion step feeds this list into the
    knowledge-entry builder so ground-truth passage ids align with the
    indexed chunk positions.
    """
    pid_map: dict[int, str] = {}
    max_pid = 0
    for qa_pair in qa_pairs:
        for pid, passage in zip(qa_pair.pids, qa_pair.passages, strict=False):
            pid_map[pid] = passage
            max_pid = max(max_pid, pid)
    passages: list[str] = [""] * (max_pid + 1)
    for pid, text in pid_map.items():
        passages[pid] = text
    return passages


# ── CSV facet readers ─────────────────────────────────────────────────


def _read_text_rows(path: str | Path) -> dict[int, str]:
    """Read a ``id,text`` CSV into ``{id: text}``.

    Headers are optional; the first column is the id, the second the
    text. Malformed rows are logged and skipped so one bad line does
    not sink the whole dataset.
    """
    rows: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for line_no, record in enumerate(csv.reader(handle), start=1):
            if not record:
                continue
            if record[0].strip().lower() == "id":
                continue  # header
            if len(record) < 2:
                logger.warning("ignoring short row %d in %s", line_no, path)
                continue
            try:
                rows[int(record[0])] = record[1]
            except ValueError:
                logger.warning("ignoring non-integer id in %s row %d", path, line_no)
    return rows


def _read_int_pair_rows(path: str | Path) -> dict[int, tuple[int, ...]]:
    """Read a ``left,right`` CSV into ``{left: (right, ...)}``.

    Used for ``qrels`` (``qid,pid``): a question may relate to several
    passages, so repeated left keys accumulate their right values.
    """
    result: dict[int, list[int]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for line_no, record in enumerate(csv.reader(handle), start=1):
            if not record:
                continue
            if record[0].strip().lower() == "qid":
                continue  # header
            if len(record) < 2:
                logger.warning("ignoring short row %d in %s", line_no, path)
                continue
            try:
                result.setdefault(int(record[0]), []).append(int(record[1]))
            except ValueError:
                logger.warning("ignoring non-integer ids in %s row %d", path, line_no)
    return {k: tuple(v) for k, v in result.items()}


def _read_int_map(path: str | Path) -> dict[int, int]:
    """Read a ``left,right`` CSV into ``{left: right}``.

    Used for ``qas`` (``qid,aid``): each question has exactly one
    answer id; later rows for the same key win (matching the upstream
    map assignment semantics).
    """
    result: dict[int, int] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for line_no, record in enumerate(csv.reader(handle), start=1):
            if not record:
                continue
            if record[0].strip().lower() == "qid":
                continue  # header
            if len(record) < 2:
                logger.warning("ignoring short row %d in %s", line_no, path)
                continue
            try:
                result[int(record[0])] = int(record[1])
            except ValueError:
                logger.warning("ignoring non-integer ids in %s row %d", path, line_no)
    return result


def _fallback_pair() -> QAPair:
    """Return a single deterministic pair used when samples are missing."""
    return QAPair(
        qid=1,
        question="什么是知识管理？",  # noqa: RUF001 - intentional CJK content
        pids=[1],
        passages=["知识管理是组织收集、整理、共享和利用知识的过程。"],
        aid=1,
        answer="知识管理是组织收集、整理、共享和利用知识的过程。",
    )


# Re-export the seam so callers can depend on the interface only.
__all__ = [
    "COL_ANSWER",
    "COL_PASSAGE",
    "COL_PID",
    "COL_QUESTION",
    "DEFAULT_DATASET_DIR",
    "DEFAULT_DATASET_ID",
    "KNOWN_PAIR_KEYS",
    "DatasetService",
    "DatasetServiceLike",
    "PairInput",
    "QAPair",
    "get_passage_list",
]

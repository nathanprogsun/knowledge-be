"""Search-target model for the agent retrieval scope.

A search target declares what one mention authorizes: either a whole
knowledge base, specific documents inside it, or a tag-constrained
subset. ``SearchTargets`` is the immutable, pre-computed scope attached
to a session so every tool call resolves against the same boundary, and
the tenant ids carried here allow cross-tenant shared knowledge bases to
be queried under their owning tenant.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum


class SearchTargetType(StrEnum):
    """What a search target selects."""

    KNOWLEDGE_BASE = "knowledge_base"
    KNOWLEDGE = "knowledge"


@dataclass(frozen=True, slots=True)
class SearchTarget:
    """One retrieval target: whole KB, explicit documents, or a tag scope.

    ``tenant_id`` is the id of the tenant that owns the knowledge base;
    it is required for cross-tenant shared-KB queries and defaults to the
    request principal otherwise.
    """

    type: SearchTargetType = SearchTargetType.KNOWLEDGE_BASE
    knowledge_base_id: str = ""
    tenant_id: int = 0
    knowledge_ids: tuple[str, ...] = ()
    tag_ids: tuple[str, ...] = ()
    scope_tag_ids: tuple[str, ...] = ()
    disable_recall_thresholds: bool = False

    def recall_thresholds(
        self,
        vector_threshold: float,
        keyword_threshold: float,
    ) -> tuple[float, float]:
        """Return the effective recall thresholds for this target.

        A target with recall thresholds disabled keeps recall broad inside
        an already constrained, user-selected scope: both thresholds become
        zero so the reranker still orders candidates but the vector and
        keyword gates cannot erase the explicit scope first.
        """
        if self.disable_recall_thresholds:
            return 0.0, 0.0
        return vector_threshold, keyword_threshold


@dataclass(frozen=True, slots=True)
class SearchTargets:
    """Immutable set of search targets plus the scope helper queries."""

    targets: tuple[SearchTarget, ...] = ()

    def __iter__(self) -> Iterator[SearchTarget]:
        return iter(self.targets)

    def get_all_knowledge_base_ids(self) -> list[str]:
        """Return unique knowledge-base ids in first-appearance order."""
        seen: set[str] = set()
        result: list[str] = []
        for target in self.targets:
            if not target.knowledge_base_id or target.knowledge_base_id in seen:
                continue
            seen.add(target.knowledge_base_id)
            result.append(target.knowledge_base_id)
        return result

    def get_kb_tenant_map(self) -> dict[str, int]:
        """Map each knowledge-base id to its owning tenant id."""
        result: dict[str, int] = {}
        for target in self.targets:
            if target.knowledge_base_id:
                result[target.knowledge_base_id] = target.tenant_id
        return result

    def get_tenant_id_for_kb(self, kb_id: str) -> int:
        """Return the tenant id owning ``kb_id`` (0 when absent)."""
        for target in self.targets:
            if target.knowledge_base_id == kb_id:
                return target.tenant_id
        return 0

    def contains_kb(self, kb_id: str) -> bool:
        """Whether ``kb_id`` appears in any target."""
        return any(target.knowledge_base_id == kb_id for target in self.targets)

    def has_recall_threshold_override(self) -> bool:
        """Whether any target disables recall thresholds (explicit scope)."""
        return any(target.disable_recall_thresholds for target in self.targets)

    def filter_by_kb_ids(self, kb_ids: tuple[str, ...]) -> SearchTargets:
        """Return the targets whose knowledge base is present in ``kb_ids``."""
        allowed = frozenset(kb_ids)
        return SearchTargets(
            targets=tuple(
                target for target in self.targets if target.knowledge_base_id in allowed
            )
        )


__all__ = ["SearchTarget", "SearchTargetType", "SearchTargets"]

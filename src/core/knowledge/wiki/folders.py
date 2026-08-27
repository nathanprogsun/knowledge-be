"""Wiki folder service — directory-tree management over ``wiki_folders``.

Request-scoped service over the already-merged wiki repositories: folder
create / rename-or-move / delete, path resolution (find-or-create along a
category path), recursive child listings with page counts, and empty-folder
chain pruning. Every page placement write that references a folder goes
through the page service (:meth:`WikiPageService.move_page`); this service
owns the folder rows themselves plus the page-cache recompute that a
subtree move triggers.

The web layer consumes the service through :func:`src.core.knowledge.wiki.factory.build_wiki_folder_service`.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy.exc import IntegrityError

from src.common.exception import ConflictError, NotFoundError, ValidationError
from src.core.knowledge.wiki.hierarchy import (
    apply_folder_to_page,
    normalize_wiki_hierarchy,
    wiki_folder_segments,
)
from src.core.knowledge.wiki.types import (
    WIKI_FOLDER_ROOT_ID,
    WikiFolderNode,
    clean_category_path,
)
from src.db.dao.wiki_page_repository import WikiFolderRepository, WikiPageRepository
from src.db.models.wiki_page import WikiFolder

logger = logging.getLogger(__name__)

# A folder name may not carry a directory separator: a folder name is a
# single tree level (including fullwidth and pipe variants).
_FOLDER_NAME_SEPARATORS = "/｜|／"


def validate_folder_name(name: str) -> str:
    """Trim and validate a single-level folder name; return the cleaned name.

    Raises ``ValidationError`` for blank names or names carrying a path
    separator.
    """
    clean = name.strip()
    if not clean:
        raise ValidationError(
            code="wiki.folder_name_required",
            message="folder name is required",
        )
    if any(separator in clean for separator in _FOLDER_NAME_SEPARATORS):
        raise ValidationError(
            code="wiki.folder_name_separator",
            message=f"folder name {name} must not contain a path separator",
        )
    return clean


def recursive_folder_counts(folders: list[WikiFolder], direct: dict[str, int]) -> dict[str, int]:
    """Map each folder id to the sum of ``direct`` counts over its whole subtree.

    Uses the materialized path so a single pass over the (navigation-sized)
    folder set suffices.
    """
    result: dict[str, int] = {}
    for folder in folders:
        total = direct.get(folder.id, 0)
        prefix = folder.path + "/"
        for other in folders:
            if other.id != folder.id and other.path.startswith(prefix):
                total += direct.get(other.id, 0)
        result[folder.id] = total
    return result


class WikiFolderService:
    """Folder-tree wiki operations; constructed per request."""

    def __init__(
        self,
        *,
        folder_repo: WikiFolderRepository,
        page_repo: WikiPageRepository,
    ) -> None:
        self._folder_repo = folder_repo
        self._page_repo = page_repo

    # ── Read ────────────────────────────────────────────────────────

    async def get_folder(self, *, knowledge_base_id: str, id: str) -> WikiFolder:
        """Return one folder; raise ``wiki.folder_not_found`` when absent."""
        return await self._require_folder(knowledge_base_id, id)

    async def list_child_folders(
        self,
        *,
        knowledge_base_id: str,
        parent_id: str,
        page_types: list[str] | None = None,
    ) -> list[WikiFolderNode]:
        """Return the direct children of ``parent_id`` for a tree view.

        ``page_count`` is the folder's whole subtree (so a parent reflects
        everything filed beneath it). A folder is shown when its subtree
        holds a page matching ``page_types``; wholly-empty folders are only
        listed when multiple types are requested (the merged knowledge
        view).
        """
        all_folders = await self._folder_repo.list_all(knowledge_base_id=knowledge_base_id)
        scoped_direct = await self._page_repo.count_pages_by_folder(
            knowledge_base_id=knowledge_base_id, page_types=page_types
        )
        all_direct = scoped_direct
        if page_types:
            all_direct = await self._page_repo.count_pages_by_folder(
                knowledge_base_id=knowledge_base_id, page_types=None
            )
        rec_scoped = recursive_folder_counts(all_folders, scoped_direct)
        rec_all = recursive_folder_counts(all_folders, all_direct)
        show_empty_folders = len(page_types or []) > 1

        def _relevant(folder_id: str) -> bool:
            if rec_scoped.get(folder_id, 0) > 0:
                return True
            if show_empty_folders:
                return rec_all.get(folder_id, 0) == 0
            return False

        nodes: list[WikiFolderNode] = []
        for folder in all_folders:
            if folder.parent_id != parent_id or not _relevant(folder.id):
                continue
            has_children = any(
                other.parent_id == folder.id and _relevant(other.id) for other in all_folders
            )
            nodes.append(
                WikiFolderNode(
                    folder=folder,
                    page_count=rec_scoped.get(folder.id, 0),
                    has_children=has_children,
                )
            )
        return nodes

    async def list_distinct_category_paths(
        self, *, knowledge_base_id: str, max_paths: int = 150
    ) -> list[list[str]]:
        """Return the existing folder paths, each cleaned into segments."""
        raw_paths = await self._folder_repo.list_distinct_category_paths(
            knowledge_base_id=knowledge_base_id, max_paths=max_paths
        )
        return [wiki_folder_segments(path) for path in raw_paths]

    # ── Create ──────────────────────────────────────────────────────

    async def create_folder(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        parent_id: str,
        name: str,
    ) -> WikiFolder:
        """Create a new empty folder under ``parent_id`` (``""`` = root).

        Raises ``wiki.folder_conflict`` when a live sibling already carries
        the name.
        """
        clean_name = validate_folder_name(name)

        parent_path = ""
        depth = 1
        if parent_id != WIKI_FOLDER_ROOT_ID:
            parent = await self._require_folder(knowledge_base_id, parent_id)
            parent_path = parent.path
            depth = parent.depth + 1

        sibling = await self._folder_repo.get_child_by_name_or_none(
            knowledge_base_id=knowledge_base_id, parent_id=parent_id, name=clean_name
        )
        if sibling is not None:
            raise ConflictError(
                code="wiki.folder_conflict",
                message=f"folder {clean_name} already exists under parent {parent_id}",
            )

        path = clean_name if not parent_path else f"{parent_path}/{clean_name}"
        now = datetime.now(UTC)
        row = WikiFolder(
            id=str(uuid4()),
            tenant_id=tenant_id,
            knowledge_base_id=knowledge_base_id,
            parent_id=parent_id,
            name=clean_name,
            path=path,
            depth=depth,
            created_at=now,
            updated_at=now,
        )
        return await self._folder_repo.create(row)

    async def find_or_create_folder_path(
        self,
        *,
        knowledge_base_id: str,
        tenant_id: int,
        path: list[str],
    ) -> tuple[str, list[str]]:
        """Resolve a category path to a leaf folder id, creating missing folders.

        Concurrency-safe against the unique (KB, parent, name) constraint:
        a create race re-fetches the sibling instead of failing the plan.
        Returns ``(folder_id, cleaned_path)``; an empty path maps to the
        root id ``""``.
        """
        clean = clean_category_path(path)
        if not clean:
            return WIKI_FOLDER_ROOT_ID, []

        parent_id = WIKI_FOLDER_ROOT_ID
        parent_path = ""
        now = datetime.now(UTC)
        for depth, name in enumerate(clean):
            child = await self._folder_repo.get_child_by_name_or_none(
                knowledge_base_id=knowledge_base_id, parent_id=parent_id, name=name
            )
            if child is None:
                full_path = name if not parent_path else f"{parent_path}/{name}"
                child = WikiFolder(
                    id=str(uuid4()),
                    tenant_id=tenant_id,
                    knowledge_base_id=knowledge_base_id,
                    parent_id=parent_id,
                    name=name,
                    path=full_path,
                    depth=depth + 1,
                    created_at=now,
                    updated_at=now,
                )
                try:
                    child = await self._folder_repo.create(child)
                except IntegrityError:
                    # Lost a create race: the sibling must now exist.
                    child = await self._folder_repo.get_child_by_name_or_none(
                        knowledge_base_id=knowledge_base_id, parent_id=parent_id, name=name
                    )
                    if child is None:
                        raise
            parent_id = child.id
            parent_path = child.path
        return parent_id, clean

    # ── Update ──────────────────────────────────────────────────────

    async def rename_or_move_folder(
        self,
        *,
        knowledge_base_id: str,
        id: str,
        new_name: str,
        new_parent_id: str,
        move_parent: bool,
    ) -> WikiFolder:
        """Rename and/or reparent a folder, recomputing its subtree caches.

        Guards against cycles (moving a folder into itself or one of its
        descendants) and sibling name collisions. A no-op (same name and
        parent) returns the folder unchanged.
        """
        folder = await self._require_folder(knowledge_base_id, id)

        name = folder.name
        if new_name.strip():
            name = validate_folder_name(new_name)

        target_parent = folder.parent_id
        if move_parent:
            target_parent = new_parent_id

        parent_path = ""
        depth_base = 0
        if target_parent != WIKI_FOLDER_ROOT_ID:
            if target_parent == folder.id:
                raise ValidationError(
                    code="wiki.folder_move_self",
                    message="cannot move a folder into itself",
                )
            parent = await self._require_folder(knowledge_base_id, target_parent)
            if parent.path == folder.path or parent.path.startswith(folder.path + "/"):
                raise ValidationError(
                    code="wiki.folder_move_descendant",
                    message="cannot move a folder into its own descendant",
                )
            parent_path = parent.path
            depth_base = parent.depth

        sibling = await self._folder_repo.get_child_by_name_or_none(
            knowledge_base_id=knowledge_base_id, parent_id=target_parent, name=name
        )
        if sibling is not None and sibling.id != folder.id:
            raise ConflictError(
                code="wiki.folder_conflict",
                message=f"folder {name} already exists under parent {target_parent}",
            )

        old_path = folder.path
        new_path = name if not parent_path else f"{parent_path}/{name}"
        if new_path == old_path and target_parent == folder.parent_id:
            return folder

        all_folders = await self._folder_repo.list_all(knowledge_base_id=knowledge_base_id)
        now = datetime.now(UTC)
        affected: list[str] = []
        updated: WikiFolder | None = None
        for candidate in all_folders:
            if candidate.id == folder.id:
                next_row = candidate.model_copy(
                    update={
                        "parent_id": target_parent,
                        "name": name,
                        "path": new_path,
                        "depth": depth_base + 1,
                        "updated_at": now,
                    }
                )
            elif candidate.path.startswith(old_path + "/"):
                next_row = candidate.model_copy(
                    update={
                        "path": new_path + candidate.path[len(old_path) :],
                        "depth": len(
                            wiki_folder_segments(new_path + candidate.path[len(old_path) :])
                        ),
                        "updated_at": now,
                    }
                )
            else:
                continue
            persisted = await self._folder_repo.update(row=next_row, now=now)
            affected.append(persisted.id)
            if persisted.id == folder.id:
                updated = persisted

        await self._recompute_pages_for_folders(knowledge_base_id, affected)
        if updated is None:
            updated = folder
        return updated

    # ── Delete ──────────────────────────────────────────────────────

    async def delete_folder(self, *, knowledge_base_id: str, id: str) -> None:
        """Soft-delete an empty folder atomically.

        The emptiness test lives in the same SQL statement as the delete,
        so a concurrent page move or child-folder create cannot leave a
        dangling ``folder_id``. Raises ``wiki.folder_not_empty`` when the
        folder still holds a page or child folder.
        """
        await self._folder_repo.delete(
            knowledge_base_id=knowledge_base_id, id=id, now=datetime.now(UTC)
        )

    async def prune_empty_folder_chains(
        self, *, knowledge_base_id: str, folder_ids: list[str]
    ) -> list[str]:
        """Remove folders that became empty after a retract, plus emptied ancestors.

        Only the supplied folder chains are considered, so intentionally
        empty folders elsewhere in the wiki are preserved. Returns the ids
        of the folders actually deleted (deepest first).
        """
        if not folder_ids:
            return []
        all_folders = await self._folder_repo.list_all(knowledge_base_id=knowledge_base_id)
        by_id = {folder.id: folder for folder in all_folders}

        candidates: dict[str, WikiFolder] = {}
        for folder_id in folder_ids:
            seen: set[str] = set()
            current = folder_id
            while current != WIKI_FOLDER_ROOT_ID:
                if current in seen:
                    break
                seen.add(current)
                folder = by_id.get(current)
                if folder is None:
                    break
                candidates[current] = folder
                current = folder.parent_id

        ordered = sorted(
            candidates.values(),
            key=lambda folder: (folder.depth, folder.path),
            reverse=True,
        )

        deleted: list[str] = []
        for folder in ordered:
            children = await self._folder_repo.list_children(
                knowledge_base_id=knowledge_base_id, parent_id=folder.id
            )
            if children:
                continue
            pages = await self._page_repo.list_pages_by_folder_ids(
                knowledge_base_id=knowledge_base_id, folder_ids=[folder.id]
            )
            if pages:
                continue
            try:
                await self._folder_repo.delete(
                    knowledge_base_id=knowledge_base_id, id=folder.id, now=datetime.now(UTC)
                )
            except (NotFoundError, ConflictError):
                continue
            deleted.append(folder.id)
        return deleted

    # ── Internal helpers ────────────────────────────────────────────

    async def _require_folder(self, knowledge_base_id: str, id: str) -> WikiFolder:
        """Return one live folder or raise ``wiki.folder_not_found``."""
        row = await self._folder_repo.get_by_id_or_none(knowledge_base_id=knowledge_base_id, id=id)
        if row is None:
            raise NotFoundError(
                code="wiki.folder_not_found",
                message=f"wiki folder {id} not found in knowledge base {knowledge_base_id}",
            )
        return row

    async def _recompute_pages_for_folders(
        self, knowledge_base_id: str, folder_ids: list[str]
    ) -> None:
        """Refresh the cached category path of every page under the given folders.

        Bookkeeping-only writes — no version bump. A page deleted
        mid-recompute is skipped rather than aborting the pass.
        """
        if not folder_ids:
            return
        pages = await self._page_repo.list_pages_by_folder_ids(
            knowledge_base_id=knowledge_base_id, folder_ids=folder_ids
        )
        now = datetime.now(UTC)
        for page in pages:
            row = await apply_folder_to_page(page, folder_repo=self._folder_repo)
            row = normalize_wiki_hierarchy(row.model_copy(update={"updated_at": now}))
            try:
                await self._page_repo.update_meta(row=row, now=now)
            except NotFoundError:
                logger.warning("wiki: recompute folder path: page %s gone mid-recompute", page.slug)


__all__ = ["WikiFolderService", "recursive_folder_counts", "validate_folder_name"]

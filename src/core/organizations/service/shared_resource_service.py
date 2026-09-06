"""Cross-tenant shared-resource reads and the per-tenant hide preference.

Request-scoped: constructed per request by
``factory.build_shared_resource_service`` with fresh repositories on the
shared ``AsyncSession``; the web layer never imports ``db`` directly.

The two list operations project the caller's tenant-scoped view of
resources shared into the organizations it participates in:

- ``list_shared_knowledge_bases`` / ``list_shared_agents`` skip rows the
  tenant itself shared (own resources are served by the owning domain's
  own endpoints) and apply the three-way permission cap: share grant,
  tenant's role inside the organization, and the caller's tenant-role
  ceiling (a tenant Viewer cannot exceed viewer on any shared resource).
- Rows are deduplicated per resource, keeping the strongest effective
  permission.

``set_shared_agent_disabled_by_me`` records a tenant-local opt-out that
only affects the tenant's own conversation dropdown; the underlying
share and the owning tenant are untouched.
"""

from __future__ import annotations

from src.common.exception import NotFoundError, PermissionDeniedError
from src.core.agents.types import CustomAgentInfo
from src.core.auth.permissions import TenantRole
from src.core.knowledge.knowledge_bases.types import (
    KNOWLEDGE_BASE_TYPE_DOCUMENT,
    KNOWLEDGE_BASE_TYPE_FAQ,
    KnowledgeBaseInfo,
)
from src.core.organizations.types import (
    ORG_ROLE_EDITOR,
    ORG_ROLE_VIEWER,
    SharedAgentInfo,
    SharedKnowledgeBaseInfo,
)
from src.db.dao.agent_share_repository import AgentShareRepository
from src.db.dao.custom_agent_repository import CustomAgentRepository
from src.db.dao.kb_share_repository import KBShareRepository
from src.db.dao.knowledge_base_repository import KnowledgeBaseRepository
from src.db.dao.organization_repository import (
    OrganizationMemberRepository,
    OrganizationRepository,
)
from src.db.dao.tenant_disabled_shared_agent_repository import (
    TenantDisabledSharedAgentRepository,
)
from src.db.dao.users_repository import UserRepository
from src.db.dao.web_search_provider_repository import WebSearchProviderRepository
from src.db.models.custom_agent import CustomAgent
from src.db.models.kb_share import KnowledgeBaseShare
from src.db.models.organization import has_org_permission

# Error code reused for the "no share row" rejection on the hide toggle.
_NO_ACCESS_CODE = "shared_agent.no_access"


def _min_org_role(left: str, right: str) -> str:
    """Return the lower role on the admin > editor > viewer ladder.

    An empty role is treated as "less than viewer" so it short-circuits
    to whatever the other argument is (mirrors the upstream cap helper).
    """
    if not left:
        return right
    if not right:
        return left
    if has_org_permission(left, right):
        return right
    return left


def _apply_tenant_role_cap(permission: str, caller_tenant_role: str) -> str:
    """Cap a shared-resource grant by the caller's tenant-role ceiling.

    A caller whose own tenant role is Viewer cannot exceed viewer on any
    shared resource, regardless of the org-level grant. Higher roles
    pass through unchanged.
    """
    if caller_tenant_role == TenantRole.VIEWER and has_org_permission(permission, ORG_ROLE_EDITOR):
        return ORG_ROLE_VIEWER
    return permission


class SharedResourceService:
    """Stateless cross-tenant shared-resource service, per request."""

    def __init__(
        self,
        *,
        org_repo: OrganizationRepository,
        member_repo: OrganizationMemberRepository,
        kb_share_repo: KBShareRepository,
        agent_share_repo: AgentShareRepository,
        disabled_repo: TenantDisabledSharedAgentRepository,
        kb_repo: KnowledgeBaseRepository,
        agent_repo: CustomAgentRepository,
        user_repo: UserRepository,
        web_search_provider_repo: WebSearchProviderRepository,
    ) -> None:
        self._org_repo = org_repo
        self._member_repo = member_repo
        self._kb_share_repo = kb_share_repo
        self._agent_share_repo = agent_share_repo
        self._disabled_repo = disabled_repo
        self._kb_repo = kb_repo
        self._agent_repo = agent_repo
        self._user_repo = user_repo
        self._web_search_provider_repo = web_search_provider_repo

    # ── Shared knowledge bases ─────────────────────────────────────

    async def list_shared_knowledge_bases(
        self,
        *,
        tenant_id: int,
        caller_tenant_role: str,
    ) -> list[SharedKnowledgeBaseInfo]:
        """Knowledge bases shared into the tenant's organizations.

        Only cross-tenant rows appear: a knowledge base the tenant owns
        is served by its own list endpoint, never here. The effective
        permission per row applies the three-way cap; rows are
        deduplicated by knowledge base keeping the strongest grant.
        """
        shares = await self._kb_share_repo.list_shared_for_tenant(tenant_id)
        kb_ids = {share.knowledge_base_id for share in shares}
        kb_by_id = await self._load_knowledge_bases(kb_ids)

        member_roles: dict[str, str | None] = {}
        org_names: dict[str, str] = {}
        by_kb: dict[str, SharedKnowledgeBaseInfo] = {}
        for share in shares:
            if share.source_tenant_id == tenant_id:
                continue
            kb = kb_by_id.get(share.knowledge_base_id)
            if kb is None:
                continue
            role = await self._member_role(
                org_id=share.organization_id,
                tenant_id=tenant_id,
                cache=member_roles,
            )
            if role is None:
                continue
            effective = _apply_tenant_role_cap(
                _min_org_role(share.permission, role), caller_tenant_role
            )
            info = SharedKnowledgeBaseInfo(
                knowledge_base=await self._fill_kb_counts(
                    kb, source_tenant_id=share.source_tenant_id
                ),
                share_id=share.id,
                organization_id=share.organization_id,
                org_name=await self._org_name(share.organization_id, cache=org_names),
                permission=effective,
                source_tenant_id=share.source_tenant_id,
                shared_at=share.created_at,
            )
            existing = by_kb.get(kb.id)
            if existing is None or (
                has_org_permission(effective, existing.permission)
                and effective != existing.permission
            ):
                by_kb[kb.id] = info
        return list(by_kb.values())

    async def list_organization_shared_knowledge_bases(
        self,
        *,
        organization_id: str,
        tenant_id: int,
        caller_tenant_role: str,
    ) -> list[SharedKnowledgeBaseInfo]:
        """Knowledge bases shared into one organization.

        Includes grants the caller tenant made (``is_mine``). Agent-carried
        knowledge bases stay out of this list.
        """
        org = await self._org_repo.get_by_id_or_none(organization_id)
        if org is None:
            raise NotFoundError(
                code="organization.not_found",
                message=f"organization {organization_id} not found",
            )
        member = await self._member_repo.get_member(
            organization_id=organization_id,
            tenant_id=tenant_id,
        )
        if member is None:
            raise NotFoundError(
                code="organization.tenant_not_member",
                message=f"tenant {tenant_id} is not a member of this organization",
            )
        shares = await self._kb_share_repo.list_by_organization(organization_id)
        kb_by_id = await self._load_knowledge_bases({share.knowledge_base_id for share in shares})
        items: list[SharedKnowledgeBaseInfo] = []
        for share in shares:
            kb = kb_by_id.get(share.knowledge_base_id)
            if kb is None:
                continue
            items.append(
                await self._organization_shared_kb_row(
                    share,
                    kb=kb,
                    org_name=org.name,
                    tenant_id=tenant_id,
                    member_role=member.role,
                    caller_tenant_role=caller_tenant_role,
                )
            )
        return items

    async def _organization_shared_kb_row(
        self,
        share: KnowledgeBaseShare,
        *,
        kb: KnowledgeBaseInfo,
        org_name: str,
        tenant_id: int,
        member_role: str,
        caller_tenant_role: str,
    ) -> SharedKnowledgeBaseInfo:
        is_mine = share.source_tenant_id == tenant_id
        permission = share.permission
        if not is_mine:
            permission = _apply_tenant_role_cap(
                _min_org_role(permission, member_role),
                caller_tenant_role,
            )
        return SharedKnowledgeBaseInfo(
            knowledge_base=await self._fill_kb_counts(kb, source_tenant_id=share.source_tenant_id),
            share_id=share.id,
            organization_id=share.organization_id,
            org_name=org_name,
            permission=permission,
            source_tenant_id=share.source_tenant_id,
            shared_at=share.created_at,
            is_mine=is_mine,
        )

    # ── Shared agents ──────────────────────────────────────────────

    async def list_shared_agents(
        self,
        *,
        tenant_id: int,
        caller_tenant_role: str,
    ) -> list[SharedAgentInfo]:
        """Agents shared into the tenant's organizations.

        Same cap and dedup rules as the knowledge-base list; each row
        carries the source-tenant web-search availability bit and the
        caller's own hide preference.
        """
        shares = await self._agent_share_repo.list_shared_for_tenant(tenant_id)

        member_roles: dict[str, str | None] = {}
        org_names: dict[str, str] = {}
        web_search_cache: dict[tuple[int, bool, str], bool] = {}
        by_agent: dict[tuple[str, int], SharedAgentInfo] = {}
        for share in shares:
            if share.source_tenant_id == tenant_id:
                continue
            agent = await self._agent_repo.get_by_id_and_tenant(
                id=share.agent_id, tenant_id=share.source_tenant_id
            )
            if agent is None:
                continue
            role = await self._member_role(
                org_id=share.organization_id,
                tenant_id=tenant_id,
                cache=member_roles,
            )
            if role is None:
                continue
            effective = _apply_tenant_role_cap(
                _min_org_role(share.permission, role), caller_tenant_role
            )
            info = SharedAgentInfo(
                agent=CustomAgentInfo.from_row(agent),
                share_id=share.id,
                organization_id=share.organization_id,
                org_name=await self._org_name(share.organization_id, cache=org_names),
                permission=effective,
                source_tenant_id=share.source_tenant_id,
                shared_at=share.created_at,
                shared_by_user_id=share.shared_by_user_id,
                shared_by_username=await self._shared_by_username(share.shared_by_user_id),
                web_search_ready=await self._agent_web_search_ready(
                    agent=agent,
                    source_tenant_id=share.source_tenant_id,
                    cache=web_search_cache,
                ),
            )
            key = (share.agent_id, share.source_tenant_id)
            existing = by_agent.get(key)
            if existing is None or (
                has_org_permission(effective, existing.permission)
                and effective != existing.permission
            ):
                by_agent[key] = info

        disabled_keys = await self._disabled_keys(tenant_id)
        return [
            info.model_copy(
                update={
                    "disabled_by_me": (
                        info.agent.id,
                        info.source_tenant_id,
                    )
                    in disabled_keys
                }
            )
            for info in by_agent.values()
        ]

    # ── Hide preference ────────────────────────────────────────────

    async def set_shared_agent_disabled_by_me(
        self,
        *,
        tenant_id: int,
        agent_id: str,
        disabled: bool,
    ) -> None:
        """Record or clear the tenant's hide preference for one agent.

        The source tenant is resolved from the caller's own agent row
        when the agent belongs to the caller, otherwise from a share the
        caller's tenant can reach. An unresolvable agent is rejected
        with 403 (mirrors the upstream "no access" guard).
        """
        source_tenant_id = await self._resolve_source_tenant(tenant_id=tenant_id, agent_id=agent_id)
        if disabled:
            await self._disabled_repo.add(
                tenant_id=tenant_id,
                agent_id=agent_id,
                source_tenant_id=source_tenant_id,
            )
        else:
            await self._disabled_repo.remove(
                tenant_id=tenant_id,
                agent_id=agent_id,
                source_tenant_id=source_tenant_id,
            )

    # ── Internal helpers ───────────────────────────────────────────

    async def _load_knowledge_bases(self, ids: set[str]) -> dict[str, KnowledgeBaseInfo]:
        """Batch-load knowledge bases by id, projected onto the DTO."""
        if not ids:
            return {}
        rows = await self._kb_repo.get_by_ids(sorted(ids))
        return {row.id: KnowledgeBaseInfo.map_from_db(row) for row in rows}

    async def _fill_kb_counts(
        self,
        info: KnowledgeBaseInfo,
        *,
        source_tenant_id: int,
    ) -> KnowledgeBaseInfo:
        """Best-effort count enrichment for one shared knowledge base.

        Document rows get the document count, FAQ rows the chunk count.
        A failing sibling-table query is ignored (counts are enrichment,
        not the listing itself), leaving the field at its zero default.
        """
        updates: dict[str, int] = {}
        try:
            if info.type == KNOWLEDGE_BASE_TYPE_DOCUMENT:
                updates["knowledge_count"] = await self._kb_repo.count_documents(
                    tenant_id=source_tenant_id, knowledge_base_id=info.id
                )
            elif info.type == KNOWLEDGE_BASE_TYPE_FAQ:
                updates["chunk_count"] = await self._kb_repo.count_chunks(
                    tenant_id=source_tenant_id, knowledge_base_id=info.id
                )
        except Exception:
            return info
        return info.model_copy(update=updates)

    async def _member_role(
        self,
        *,
        org_id: str,
        tenant_id: int,
        cache: dict[str, str | None],
    ) -> str | None:
        """Return the tenant's role in an organization, memoized per org."""
        if org_id not in cache:
            member = await self._member_repo.get_member(organization_id=org_id, tenant_id=tenant_id)
            cache[org_id] = member.role if member is not None else None
        return cache[org_id]

    async def _org_name(
        self,
        org_id: str,
        *,
        cache: dict[str, str],
    ) -> str:
        """Resolve an organization's display name, memoized per org."""
        if org_id not in cache:
            org = await self._org_repo.get_by_id_or_none(org_id)
            cache[org_id] = org.name if org is not None else ""
        return cache[org_id]

    async def _shared_by_username(self, user_id: str) -> str:
        """Resolve the sharing user's username, empty when unresolvable."""
        if not user_id:
            return ""
        user = await self._user_repo.find_by_id(user_id)
        if user is None:
            return ""
        return user.username

    async def _agent_web_search_ready(
        self,
        *,
        agent: CustomAgent,
        source_tenant_id: int,
        cache: dict[tuple[int, bool, str], bool],
    ) -> bool:
        """Whether the agent's web-search provider resolves in its source tenant.

        Only an availability bit is returned; the source tenant's
        provider configuration is never exposed to the caller. The
        result is memoized per (tenant, enabled, provider) tuple.
        """
        config = agent.config or {}
        enabled = bool(config.get("web_search_enabled", False))
        provider_id = str(config.get("web_search_provider_id") or "")
        cache_key = (source_tenant_id, enabled, provider_id)
        if cache_key in cache:
            return cache[cache_key]
        ready = False
        if enabled:
            provider = None
            if provider_id:
                provider = await self._web_search_provider_repo.get_by_id(
                    source_tenant_id, provider_id
                )
            else:
                provider = await self._web_search_provider_repo.get_default(source_tenant_id)
            ready = provider is not None
        cache[cache_key] = ready
        return ready

    async def _disabled_keys(self, tenant_id: int) -> set[tuple[str, int]]:
        """The (agent_id, source_tenant_id) tuples the tenant has hidden."""
        rows = await self._disabled_repo.list_by_tenant(tenant_id)
        return {(row.agent_id, row.source_tenant_id) for row in rows}

    async def _resolve_source_tenant(
        self,
        *,
        tenant_id: int,
        agent_id: str,
    ) -> int:
        """Resolve the owning tenant of ``agent_id`` for the hide toggle.

        Own agents resolve to the caller's tenant; anything else must be
        reachable through a share, otherwise the caller has no access.
        """
        own = await self._agent_repo.get_by_id_and_tenant(id=agent_id, tenant_id=tenant_id)
        if own is not None:
            return tenant_id
        share = await self._agent_share_repo.get_share_for_tenant(
            tenant_id=tenant_id,
            agent_id=agent_id,
            exclude_source_tenant_id=tenant_id,
        )
        if share is None:
            raise PermissionDeniedError(
                code=_NO_ACCESS_CODE,
                message="No access to this agent",
            )
        return share.source_tenant_id


__all__ = [
    "SharedResourceService",
    "_apply_tenant_role_cap",
    "_min_org_role",
]

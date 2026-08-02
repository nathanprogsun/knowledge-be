from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class Organization(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    owner_id: str | None = Field(default=None)
    invite_code: str | None = Field(default=None)
    invite_code_expires_at: datetime | None = Field(default=None)
    invite_code_validity_days: int | None = Field(default=7)
    require_approval: bool | None = Field(default=False)
    searchable: bool | None = Field(default=False)
    member_limit: int | None = Field(default=50)
    member_count: int | None = Field(default=0)
    share_count: int | None = Field(default=0)
    agent_share_count: int | None = Field(default=0)
    pending_join_request_count: int | None = Field(default=0)
    is_owner: bool | None = Field(default=False)
    my_role: str | None = Field(default=None)
    has_pending_upgrade: bool | None = Field(default=False)
    created_at: datetime
    updated_at: datetime


class CreateOrganizationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    invite_code_validity_days: int | None = Field(default=7)
    member_limit: int | None = Field(default=50)


class UpdateOrganizationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    require_approval: bool | None = Field(default=None)
    searchable: bool | None = Field(default=None)
    invite_code_validity_days: int | None = Field(default=None)
    member_limit: int | None = Field(default=None)


class OrganizationPreview(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    description: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    member_count: int | None = Field(default=0)
    share_count: int | None = Field(default=0)
    agent_share_count: int | None = Field(default=0)
    is_already_member: bool | None = Field(default=False)
    require_approval: bool | None = Field(default=False)
    created_at: datetime


class OrganizationList(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[Organization]
    total: int | None = Field(default=0)
    resource_counts: dict[str, dict[str, object]] | None = Field(default=None)


class SearchOrganizationsQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    q: str | None = Field(default=None)
    limit: int = 20


class SearchOrganizationsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[OrganizationPreview]
    total: int


class JoinOrganizationByCodeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    invite_code: str


class JoinRequestRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    invite_code: str
    message: str | None = Field(default=None)
    role: str | None = Field(default=None)


class JoinRequestByIDRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    organization_id: str
    message: str | None = Field(default=None)
    role: str | None = Field(default=None)


class JoinRequestRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str | None = Field(default=None)
    username: str | None = Field(default=None)
    email: str | None = Field(default=None)
    message: str | None = Field(default=None)
    request_type: str
    prev_role: str | None = Field(default=None)
    requested_role: str
    status: str
    created_at: datetime


class JoinRequestListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: list[JoinRequestRecord]
    total: int


class ReviewJoinRequestRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    message: str | None = Field(default=None)
    role: str | None = Field(default=None)


class RequestRoleUpgradeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    requested_role: str
    message: str | None = Field(default=None)


class RegenerateInviteCodeResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    invite_code: str


class OrgMember(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    user_id: str
    username: str | None = Field(default=None)
    email: str | None = Field(default=None)
    avatar: str | None = Field(default=None)
    role: str
    tenant_id: int
    joined_at: datetime


class OrgMemberListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    members: list[OrgMember]
    total: int


class UpdateMemberRoleRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: str


class InviteUserRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_id: str
    role: str


class SearchUsersQuery(BaseModel):
    model_config = ConfigDict(frozen=True)

    q: str
    limit: int = 10


class SearchUsersResponseItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    username: str
    email: str
    avatar: str | None = Field(default=None)


class SearchUsersResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: list[SearchUsersResponseItem]


class KnowledgeBaseShare(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    knowledge_base_id: str | None = Field(default=None)
    knowledge_base_name: str | None = Field(default=None)
    knowledge_base_type: str | None = Field(default=None)
    knowledge_count: int | None = Field(default=None)
    chunk_count: int | None = Field(default=None)
    organization_id: str
    organization_name: str | None = Field(default=None)
    shared_by_user_id: str | None = Field(default=None)
    shared_by_username: str | None = Field(default=None)
    source_tenant_id: int
    permission: str
    my_role_in_org: str | None = Field(default=None)
    my_permission: str | None = Field(default=None)
    created_at: datetime


class KnowledgeBaseShareListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    shares: list[KnowledgeBaseShare]
    total: int


class CreateKnowledgeBaseShareRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    organization_id: str
    permission: str


class UpdateKnowledgeBaseShareRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    permission: str


class SharedKnowledgeBaseListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    knowledge_base: dict[str, object]
    share_id: str
    organization_id: str
    org_name: str | None = Field(default=None)
    permission: str
    source_tenant_id: int
    shared_at: datetime
    shared_by_user_id: str | None = Field(default=None)
    shared_by_username: str | None = Field(default=None)
    is_mine: bool | None = Field(default=False)
    source_from_agent: dict[str, object] | None = Field(default=None)


class SharedKnowledgeBaseListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[SharedKnowledgeBaseListItem]
    total: int


class AgentShare(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    agent_id: str
    agent_name: str | None = Field(default=None)
    agent_avatar: str | None = Field(default=None)
    organization_id: str
    organization_name: str | None = Field(default=None)
    shared_by_user_id: str | None = Field(default=None)
    shared_by_username: str | None = Field(default=None)
    source_tenant_id: int
    permission: str
    my_role_in_org: str | None = Field(default=None)
    my_permission: str | None = Field(default=None)
    scope_kb: str | None = Field(default=None)
    scope_kb_count: int | None = Field(default=None)
    scope_web_search: bool | None = Field(default=None)
    scope_mcp: str | None = Field(default=None)
    created_at: datetime


class AgentShareListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    shares: list[AgentShare]
    total: int


class CreateAgentShareRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    organization_id: str
    permission: str


class SharedAgentListItem(BaseModel):
    model_config = ConfigDict(frozen=True)

    agent: dict[str, object]
    share_id: str
    organization_id: str
    org_name: str | None = Field(default=None)
    permission: str
    source_tenant_id: int
    shared_at: datetime
    shared_by_user_id: str | None = Field(default=None)
    shared_by_username: str | None = Field(default=None)
    disabled_by_me: bool | None = Field(default=False)
    is_mine: bool | None = Field(default=False)


class SharedAgentListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[SharedAgentListItem]
    total: int


__all__ = [
    "AgentShare",
    "AgentShareListResponse",
    "CreateAgentShareRequest",
    "CreateKnowledgeBaseShareRequest",
    "CreateOrganizationRequest",
    "InviteUserRequest",
    "JoinOrganizationByCodeRequest",
    "JoinRequestByIDRequest",
    "JoinRequestListResponse",
    "JoinRequestRecord",
    "JoinRequestRequest",
    "KnowledgeBaseShare",
    "KnowledgeBaseShareListResponse",
    "OrgMember",
    "OrgMemberListResponse",
    "Organization",
    "OrganizationList",
    "OrganizationPreview",
    "RegenerateInviteCodeResponse",
    "RequestRoleUpgradeRequest",
    "ReviewJoinRequestRequest",
    "SearchOrganizationsQuery",
    "SearchOrganizationsResponse",
    "SearchUsersQuery",
    "SearchUsersResponse",
    "SearchUsersResponseItem",
    "SharedAgentListItem",
    "SharedAgentListResponse",
    "SharedKnowledgeBaseListItem",
    "SharedKnowledgeBaseListResponse",
    "UpdateKnowledgeBaseShareRequest",
    "UpdateMemberRoleRequest",
    "UpdateOrganizationRequest",
]

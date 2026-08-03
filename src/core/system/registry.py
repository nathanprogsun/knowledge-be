"""In-code registry of legal system-setting keys.

Mirrors ``internal/application/service/system_setting.go::registry``.
The registry is the **only** authority on which keys are legal, what
type they hold, what their ENV-fallback name is, and what the built-in
default is. Adding a new tunable is a matter of:

1. Adding an entry here.
2. (Optional) adding a SQL seed row in a new migration so the UI shows
   the row even before any operator hits Update.
3. Replacing existing ``os.getenv()`` reads with calls into the
   service.

``SystemSettingService.update`` rejects any key not in this registry —
so the UI cannot inject arbitrary keys into the DB, even with an
attacker-controlled body.

PR-11 scope: the registry is defined but the **side-effect bridges**
(SSRF whitelist application, model-concurrency limiter push, Redis
pubsub invalidation) are stubbed — they land in later PRs alongside
the services they configure.

The ``description`` strings are intentionally Chinese (mirrors the
upstream Go registry verbatim) so the management UI renders the same
copy operators see in the Go deployment. RUF001/002/003 flag the
full-width punctuation as ambiguous; suppressed file-wide because
every description carries it.
"""
# ruff: noqa: RUF001

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class SettingSpec:
    """Registry entry for a known system setting.

    ``value_type`` is one of ``"int"``, ``"string"``, ``"bool"``,
    ``"string_list"``. ``env_name`` is the legacy environment variable
    consulted when the DB row is absent. ``default`` is the built-in
    fallback used when both DB and ENV miss. ``enum`` restricts
    ``update`` to values in this set (only meaningful for
    ``value_type == "string"``).
    """

    value_type: str
    env_name: str
    default: int | str | bool | list[str]
    category: str
    description: str
    requires_restart: bool = False
    enum: tuple[str, ...] = ()


_REGISTRY: Final[dict[str, SettingSpec]] = {
    "ssrf.whitelist": SettingSpec(
        value_type="string_list",
        env_name="SSRF_WHITELIST",
        default=[],
        category="security",
        description=(
            "SSRF 防护白名单。可填入 example.com / *.foo.com / 10.0.0.0/8 / 2001:db8::1。"
            "修改后立即生效。SSRF_WHITELIST_EXTRA 环境变量仍由部署方维护，不在此处覆盖。"
        ),
    ),
    "auth.registration_mode": SettingSpec(
        value_type="string",
        env_name="",
        default="self_serve",
        enum=("self_serve", "invite_only"),
        category="auth",
        description=(
            "自助注册模式。self_serve = 任何人可注册账号；invite_only = 关闭公网注册，"
            "仅 Owner/Admin 可邀请。修改后立即生效，但谨慎对待 self_serve（公网会接受 spam）。"
        ),
    ),
    "auth.default_tenant_mode": SettingSpec(
        value_type="string",
        env_name="WEKNORA_AUTH_DEFAULT_TENANT_MODE",
        default="create_personal",
        enum=("create_personal", "tenantless"),
        category="auth",
        description=(
            "公开注册成功后的默认空间策略。create_personal = 自动创建个人空间并设为 Owner；"
            "tenantless = 仅创建用户，等待接受邀请或主动创建空间。修改后只影响新注册用户。"
        ),
    ),
    "tenant.max_owned_per_user": SettingSpec(
        value_type="int",
        env_name="WEKNORA_TENANT_MAX_OWNED_PER_USER",
        default=10,
        category="tenant",
        description=(
            "每个非超管用户通过自助创建可拥有的最大空间数。每次创建空间时实时读取，"
            "修改后立即生效。0 表示使用内置默认值 10；负数表示完全关闭限制（不建议在公开部署使用）。"
        ),
    ),
    "tenant.self_service_creation_enabled": SettingSpec(
        value_type="bool",
        env_name="WEKNORA_TENANT_SELF_SERVICE_CREATION_ENABLED",
        default=True,
        category="tenant",
        description=(
            "是否允许非超管用户主动创建空间。关闭后，普通用户只能通过邀请加入已有空间；"
            "跨空间超管仍可创建。修改后立即生效。"
        ),
    ),
    "tenant.default_storage_quota_gb": SettingSpec(
        value_type="int",
        env_name="WEKNORA_TENANT_DEFAULT_STORAGE_QUOTA_GB",
        default=10,
        category="tenant",
        description=(
            "新建空间时默认分配的存储配额（GB），包含向量、原文、文本、索引等。"
            "仅在创建时读取，修改后只对之后新建的空间生效，不会回写已存在的空间。"
            "0 或负数表示使用内置默认值 10GB。"
        ),
    ),
    "tenant.auto_create_api_key": SettingSpec(
        value_type="bool",
        env_name="WEKNORA_TENANT_AUTO_CREATE_API_KEY",
        default=False,
        category="tenant",
        description=(
            "创建空间时是否自动生成一个全量权限（full_access）的 API Key，并在创建接口的响应中返回其明文 token。"
            "用于兼容旧版本「创建空间即下发默认 API Key」的行为（属于破坏性变更的回退开关）。"
            "每次创建空间时实时读取，修改后立即生效。默认 false（不自动创建，需通过 API Key 管理显式创建）。"
        ),
    ),
    "asynq.core_concurrency": SettingSpec(
        value_type="int",
        env_name="WEKNORA_ASYNQ_CORE_CONCURRENCY",
        default=4,
        category="worker",
        requires_restart=True,
        description="文档解析、手工重解析等核心任务的每实例保底并发。可额外使用共享弹性池；修改后需重启。",
    ),
    "asynq.postprocess_concurrency": SettingSpec(
        value_type="int",
        env_name="WEKNORA_ASYNQ_POSTPROCESS_CONCURRENCY",
        default=2,
        category="worker",
        requires_restart=True,
        description="解析完成后的轻量编排与富化扇出专用并发，避免被长时间文档解析阻塞；修改后需重启。",
    ),
    "asynq.enrichment_concurrency": SettingSpec(
        value_type="int",
        env_name="WEKNORA_ASYNQ_ENRICHMENT_CONCURRENCY",
        default=2,
        category="worker",
        requires_restart=True,
        description="摘要、图片、图谱和问题生成的每实例保底并发。可额外使用共享弹性池；修改后需重启。",
    ),
    "asynq.maintenance_concurrency": SettingSpec(
        value_type="int",
        env_name="WEKNORA_ASYNQ_MAINTENANCE_CONCURRENCY",
        default=1,
        category="worker",
        requires_restart=True,
        description="数据源同步、批处理、移动和删除清理的每实例保底并发，与用户面流水线硬隔离；修改后需重启。",
    ),
    "asynq.shared_concurrency": SettingSpec(
        value_type="int",
        env_name="WEKNORA_ASYNQ_SHARED_CONCURRENCY",
        default=4,
        category="worker",
        requires_restart=True,
        description="核心解析与内容富化共用的每实例弹性并发。空闲容量由有积压的一侧借用；修改后需重启。",
    ),
    "asynq.wiki_concurrency": SettingSpec(
        value_type="int",
        env_name="WEKNORA_WIKI_ASYNQ_CONCURRENCY",
        default=8,
        category="worker",
        requires_restart=True,
        description=(
            "Wiki 生成专用池的 worker 并发数（与文档解析池相互隔离）。"
            "Wiki 生成以合成大模型调用为主，独立并发预算可避免上传高峰期被解析任务饿死，"
            "同时不会因 Wiki 洪峰拖慢用户面解析。修改后需重启服务进程方可生效。"
        ),
    ),
    "model.max_concurrency": SettingSpec(
        value_type="int",
        env_name="WEKNORA_MODEL_MAX_CONCURRENCY",
        default=32,
        category="worker",
        description=(
            "后台任务（文档入库/富化）对单个模型的默认并发上限，按模型 ID 全副本共享。"
            "每次调用实时读取，修改后立即生效、无需重启。0 或负数表示关闭默认限制"
            "（各模型仍会尊重自身在模型管理里配置的上限）。仅影响后台任务，不影响交互式对话。"
        ),
    ),
}


def get_spec(key: str) -> SettingSpec | None:
    """Return the registry entry for ``key``, or ``None`` if unknown."""
    return _REGISTRY.get(key)


def all_keys() -> list[str]:
    """Return every registered key, sorted for deterministic UI order."""
    return sorted(_REGISTRY.keys())


def all_specs() -> dict[str, SettingSpec]:
    """Return the full registry (read-only view)."""
    return dict(_REGISTRY)


__all__ = ["SettingSpec", "all_keys", "all_specs", "get_spec"]

# core/tenants

工作区（租户）生命周期、成员资格、API 密钥管理。HTTP 入口在
`web/api/tenants/router.py`。

- **入口**：`TenantService.create_tenant` / `get_tenant` /
  `search_tenants`；成员资格 `TenantMemberService.ensure_owner`。
- **禁区**：业务模块不应直接修改 `tenants.credentials`（encryption
  schema 属于这里），也不要自行填充 `last_active_tenant_id`——交由
  auth middleware 兜底。

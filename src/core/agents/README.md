# core/agents

Agent 注册、运行时与内置 Agent 仓库。运行时引擎在 `engine/`，内置清单
在 `builtin_registry.py` / `builtin/`，tool 在 `tools/`，skills 在
`skills/`。

- **入口**：`CustomAgentService.list_agents` / `invoke`。
- **禁区**：业务代码不得绕过 `engine/` 直接拼 tool 调用；内置 Agent
  由 `BUILTIN_AGENT_ORDER` 单一来源控制显示顺序。

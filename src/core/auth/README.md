# core/auth

身份与租户成员资格。HTTP 层在 `web/api/auth/router.py`，DI 在
`web/deps/auth.py`。

- **入口**：`AuthService.login` / `get_me` / `update_my_preferences` —
  所有路由都经过这两个方法。
- **禁区**：不要在这里读 `Request` 或 HTTP 状态；那属于 `web/middleware`。
  异常只抛 `src.common.exception` 的 sanctioned 子类。

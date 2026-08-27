# web

HTTP 层。FastAPI 路由（`web/api/<domain>/router.py` + `views.py`）、
中间件（`middleware/`）、DI 工厂（`deps/`）。

- **入口**：`src.app_context.lifespan.app` 注册了全部路由。
- **禁区**：非 sanctioned 异常（`check_exception_types` 红线）；web 层
  不得导入 `src.db.dao`（应通过 service）；`map_from_db` 必须有 sibling
  `_<NAME>_EXCLUDE_COLUMNS` frozenset。

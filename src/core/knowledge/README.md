# core/knowledge

知识库领域：`knowledge_bases/` 主体，`documents/` 内容，`chunks/` 切片，
`faq/` 问答，`tags/` 标签，`wiki/` 与 `graph/` 图谱衍生。

- **入口**：`KnowledgeBaseService.create_kb` / `ingest` /
  `delete_kb`。其他模块（chat、agents）只能调 service，不得直接
  访问 `db.dao.*` 或 `db.models.*`。
- **禁区**：路由层禁止写业务规则——所有跨文档/切片的协调必须在
  service 中完成；DTO 字段用 `map_from_db` + `from_json`。

# `knowledge-bases` endpoints

Routes registered under `/api/v1/knowledge-bases`. Path parameters use FastAPI `{name}` notation.

| Method | Path |
| --- | --- |
| GET | `/knowledge-bases` |
| POST | `/knowledge-bases` |
| POST | `/knowledge-bases/copy` |
| DELETE | `/knowledge-bases/{id}` |
| GET | `/knowledge-bases/{id}` |
| PUT | `/knowledge-bases/{id}` |
| GET | `/knowledge-bases/{id}/shares` |
| POST | `/knowledge-bases/{id}/shares` |
| DELETE | `/knowledge-bases/{id}/shares/{share_id}` |
| PUT | `/knowledge-bases/{id}/shares/{share_id}` |
| POST | `/knowledge-bases/{id}/duplicate` |
| DELETE | `/knowledge-bases/{id}/faq/entries` |
| GET | `/knowledge-bases/{id}/faq/entries` |
| POST | `/knowledge-bases/{id}/faq/entries` |
| PUT | `/knowledge-bases/{id}/faq/entries/fields` |
| GET | `/knowledge-bases/{id}/faq/entries/export` |
| PUT | `/knowledge-bases/{id}/faq/entries/tags` |
| GET | `/knowledge-bases/{id}/faq/entries/{entry_id}` |
| PUT | `/knowledge-bases/{id}/faq/entries/{entry_id}` |
| POST | `/knowledge-bases/{id}/faq/entry` |
| PUT | `/knowledge-bases/{id}/faq/import/last-result/display` |
| POST | `/knowledge-bases/{id}/faq/search` |
| GET | `/knowledge-bases/{id}/hybrid-search` |
| POST | `/knowledge-bases/{id}/hybrid-search` |
| GET | `/knowledge-bases/{id}/knowledge` |
| POST | `/knowledge-bases/{id}/knowledge/file` |
| POST | `/knowledge-bases/{id}/knowledge/manual` |
| POST | `/knowledge-bases/{id}/knowledge/passage` |
| POST | `/knowledge-bases/{id}/knowledge/url` |
| GET | `/knowledge-bases/{id}/move-targets` |
| PUT | `/knowledge-bases/{id}/pin` |
| GET | `/knowledge-bases/{id}/tags` |
| POST | `/knowledge-bases/{id}/tags` |
| DELETE | `/knowledge-bases/{id}/tags/{tag_id}` |
| PUT | `/knowledge-bases/{id}/tags/{tag_id}` |
| GET | `/knowledge-bases/{kb_id}/files` |

# kb-verify-flow last run

At: 2026-09-03T17:07:54.586030+00:00

## Launch

- started_servers: False
- api reachable: False
- web reachable: False

## Doctor

- env file present: True
- frontend/node_modules: True
- feature headings: [{'file': '.agents/skills/kb-verify-flow/features/knowledge-base-list.md', 'missing': [], 'ok': True}, {'file': '.agents/skills/kb-verify-flow/features/login.md', 'missing': [], 'ok': True}]

## Drive

- login probes: {'knowledge-base-list': {'probes': {'knowledge_bases': {'ok': False, 'reason': 'KB_VERIFY_TOKEN unset; authenticated list not driven', 'skipped': True}}}, 'login': {'probes': {'auth_config': {'error': '<urlopen error [Errno 61] Connection refused>', 'ok': False, 'status': None, 'url': 'http://127.0.0.1:8000/api/v1/auth/config'}, 'web_login': {'error': '<urlopen error [Errno 61] Connection refused>', 'ok': False, 'status': None, 'url': 'http://127.0.0.1:5173/login'}}}}

## Cleanup

Deletes `evidence/tmp/` only. This file stays.

"""Per-router APITestClient wrappers under ``tests/integration/web/api/util/apis/``.

Each module exposes one sync function per FastAPI endpoint in the
corresponding router. Endpoint names match the handler function
names in the source tree so ``APITestClient.post/get/...`` resolves
them through ``app.url_path_for(...)``.

Auth and tenant endpoints intentionally have no wrapper file: per the
alignment doc §7.2 those routers wrap their call sites inline rather
than building per-endpoint wrappers.
"""

from __future__ import annotations

__all__: list[str] = []

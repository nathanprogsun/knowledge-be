#!/usr/bin/env python3
"""Generate ``docs/api/*.md`` endpoint tables from the FastAPI app.

The reference docs tables live one file per top-level path segment
(``docs/api/tenants.md`` for everything under ``/api/v1/tenants``,
``docs/api/agents.md`` for ``/api/v1/agents``, …). The check
``scripts/check_endpoint_coverage.py`` reads these markdown tables to
verify every registered FastAPI route is documented and every
documentation row matches a real route.

This scaffold is a one-shot generator: re-run it whenever a new
domain appears in ``src/web/api/`` and it will create ``docs/api/<domain>.md``
with the full route table pre-filled; existing files are skipped
(deleted-by-hand the file first to overwrite a stub).

Usage::

    python scripts/scaffold_api_docs.py [--api-root /api/v1]
    python scripts/scaffold_api_docs.py --only tenants,agents

Exit codes: 0 = written, 1 = no routes / app import failed.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


def _routes_by_domain(routes: list[dict[str, str]], prefix: str) -> dict[str, list[dict[str, str]]]:
    """Group routes by the first path segment under ``prefix``."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for route in routes:
        path = route["path"].removeprefix(prefix).lstrip("/")
        if not path:
            continue
        domain = path.split("/", 1)[0]
        grouped[domain].append(route)
    return grouped


def _render_table(domain: str, prefix: str, routes: list[dict[str, str]]) -> str:
    full_prefix = prefix.rstrip("/")
    lines = [
        f"# `{domain}` endpoints",
        "",
        "Routes registered under `"
        + f"{full_prefix}/{domain}"
        + "`. Path parameters use FastAPI `{name}` notation.",
        "",
        "| Method | Path |",
        "| --- | --- |",
    ]
    for route in sorted(routes, key=lambda r: (r["path"], r["method"])):
        # The static AST-based checker parses router decorators directly
        # and only knows the per-router ``prefix=`` kwarg, not the runtime
        # ``app.include_router(..., prefix="/api/v1")`` calls in
        # ``src/app_context/lifespan.py``. Strip the API root prefix from
        # the docs path so both sides render the same shape.
        rendered = route["path"]
        if full_prefix and rendered.startswith(full_prefix + "/"):
            rendered = rendered[len(full_prefix):]
        lines.append(f"| {route['method']} | `{rendered}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-root", default="/api/v1", help="API prefix to strip (default /api/v1)")
    parser.add_argument("--docs-root", default="docs/api", help="Output markdown directory")
    parser.add_argument("--only", default="", help="comma-separated domain filter")
    args = parser.parse_args()

    from src.app_context.lifespan import app  # noqa: PLC0415 — local import to avoid DB touch

    routes: list[dict[str, str]] = []

    def _walk(route_list: list, prefix: str = "") -> None:
        for r in route_list:
            path = getattr(r, "path", None)
            methods = getattr(r, "methods", None)
            if type(r).__name__ == "_IncludedRouter":
                inner = getattr(r, "original_router", None)
                # ``include_context.prefix`` carries the path-prefix the
                # parent app passed to ``include_router``; merge it so
                # child routes render fully qualified.
                ic_prefix = ""
                ic = getattr(r, "include_context", None)
                if ic is not None:
                    ic_prefix = getattr(ic, "prefix", "") or ""
                if inner is not None:
                    _walk(list(getattr(inner, "routes", [])), prefix + ic_prefix)
                continue
            child_routes = getattr(r, "routes", None)
            if child_routes and not methods and isinstance(child_routes, list):
                _walk(child_routes, prefix + (path or ""))
                continue
            if not methods or not path or not isinstance(methods, set):
                continue
            cleaned = methods - {"HEAD", "OPTIONS"}
            if not cleaned:
                continue
            for method in sorted(cleaned):
                routes.append({"method": method, "path": prefix + path})

    _walk(app.routes)

    grouped = _routes_by_domain(routes, args.api_root)
    # Drop the FastAPI infra endpoints (OpenAPI / Swagger / Redoc / health /
    # meta) — they live at the app root, not under any domain prefix, and
    # would land in a single misleading ``docs/api/<infra>.md`` file.
    for infra in ("docs", "redoc", "openapi.json", "health", "meta"):
        grouped.pop(infra, None)
    if args.only:
        keep = {d.strip() for d in args.only.split(",") if d.strip()}
        grouped = {k: v for k, v in grouped.items() if k in keep}

    if not grouped:
        print("[FAIL] no routes matched; check --api-root / app loading")
        return 1

    out_dir = Path(args.docs_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for domain, domain_routes in sorted(grouped.items()):
        target = out_dir / f"{domain}.md"
        if target.exists():
            # Skip pre-existing docs — manual edits outrank the scaffold.
            continue
        target.write_text(_render_table(domain, args.api_root, domain_routes), encoding="utf-8")
        written.append(str(target))
    print(f"wrote {len(written)} doc file(s):")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
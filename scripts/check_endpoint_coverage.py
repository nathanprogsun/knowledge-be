#!/usr/bin/env python3
"""Diff FastAPI routes in web/api/*/views.py (and router.py) against the
endpoint tables documented in ``docs/api/*.md``.

Rules:

- For every ``(METHOD, path)`` row in the markdown endpoint tables of the
  reference docs, there MUST be a matching FastAPI route registered under
  ``src/web/api/``. Path parameter placeholders are normalized between the
  two notations: ``:foo`` (Express-style in docs) <-> ``{foo}``
  (FastAPI-style in routes).

- For every FastAPI ``@router.<method>(...)`` route registered under
  ``src/web/api/``, there MUST be a matching markdown row in ``docs/api/*.md``.

Usage::

    python check_endpoint_coverage.py
        [--src-root PATH]
        [--docs-root PATH]

Exit codes:
    0 = full coverage in both directions
    1 = at least one mismatched route or orphaned docs entry
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# Path resolution
# ─────────────────────────────────────────────────────────────────────────


def _resolve(explicit: str | None, default: Path) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.exists() else None
    return default if default.exists() else None


def resolve_src_root(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.is_dir() else None
    env = os.environ.get("KNOWLEDGE_BE_SRC")
    if env:
        p = Path(env).resolve()
        if p.is_dir():
            return p
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        candidate = cur / "src"
        if candidate.is_dir():
            return candidate
        cur = cur.parent
    return None


# Default docs root — the upstream source repo (location of the
# frozen API markdown used by the coverage check).
DEFAULT_DOCS_ROOT = Path("/Users/jung/pro/WeKnora/docs")


# ─────────────────────────────────────────────────────────────────────────
# Docs parsing — extract (method, path) tuples from markdown tables
# ─────────────────────────────────────────────────────────────────────────


_ENDPOINT_TABLE_HEADER = re.compile(
    r"^\s*\|\s*(?:\u65b9\u6cd5|Method|method|Verb|verb|HTTP)\s*\|", re.IGNORECASE
)
_METHOD_RE = re.compile(r"\b(GET|POST|PUT|PATCH|DELETE)\b", re.IGNORECASE)
_PATH_RE = re.compile(r"`(/[^`\s|]*)`")

# Common column-name aliases for the second column ("endpoint path")
_PATH_COLUMN_HINTS = ("path", "\u8def\u5f84", "\u7aef\u70b9", "endpoint", "url")


def parse_docs_endpoints(
    docs_root: Path, domains: set[str] | None = None
) -> list[tuple[str, str, Path, int]]:
    """Return ``[(METHOD, fastapi_path, doc_file, lineno), ...]``."""
    out: list[tuple[str, str, Path, int]] = []
    api_dir = docs_root / "api"
    if not api_dir.is_dir():
        return out
    for md in sorted(api_dir.glob("*.md")):
        if md.name.lower() == "readme.md":
            continue
        if domains:
            stem = md.stem.lower()
            # Docs file names may be singular (``tenant.md``) or hyphenated
            # (``mcp-service.md``) while the route domain is plural or
            # underscored (``tenants`` / ``mcp_services``).
            alias = {
                "model": "models",
                "tenant": "tenants",
                "mcp-service": "mcp_services",
                "vector-store": "vector_stores",
                "storage-backend": "storage_backends",
                "web-search": "web_search",
            }
            if stem not in domains and alias.get(stem) not in domains:
                continue
        try:
            lines = md.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        in_endpoint_table = False
        header_columns: list[str] = []
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            if _ENDPOINT_TABLE_HEADER.match(line):
                in_endpoint_table = True
                header_columns = [c.strip().lower() for c in stripped.strip("|").split("|")]
                continue
            if in_endpoint_table and stripped.startswith("|") and stripped.endswith("|"):
                if "---" in stripped:
                    continue
                if header_columns and len(header_columns) >= 2:
                    cells = [c.strip() for c in stripped.strip("|").split("|")]
                    method_cell = cells[0]
                    # locate the path column from the header
                    path_idx = next(
                        (
                            i
                            for i, col in enumerate(header_columns)
                            if any(hint in col for hint in _PATH_COLUMN_HINTS)
                        ),
                        1,
                    )
                    path_cell = cells[path_idx] if path_idx < len(cells) else ""
                else:
                    cells = [c.strip() for c in stripped.strip("|").split("|")]
                    if len(cells) < 2:
                        continue
                    method_cell = cells[0]
                    path_cell = cells[1]
                m = _METHOD_RE.search(method_cell)
                if m is None:
                    continue
                mp = _PATH_RE.search(path_cell)
                if mp is None:
                    continue
                method = m.group(1).upper()
                path = _express_to_fastapi(mp.group(1))
                out.append((method, path, md, lineno))
            elif in_endpoint_table and stripped and not stripped.startswith("|"):
                in_endpoint_table = False
                header_columns = []
    return out


def _express_to_fastapi(path: str) -> str:
    """Convert ``/sessions/:id/foo`` -> ``/sessions/{id}/foo``."""
    return re.sub(r":([A-Za-z_][A-Za-z_0-9]*)", r"{\1}", path)


def _normalize_path_params(path: str) -> str:
    """Collapse every path-param name to ``{id}``.

    Docs may write ``/tenants/:id`` while routes use ``/tenants/{tenant_id}``.
    Parameter names carry no coverage meaning, so they are normalised away
    before comparison.
    """
    return re.sub(r"\{[A-Za-z_][A-Za-z_0-9]*\}", "{id}", path)


# ─────────────────────────────────────────────────────────────────────────
# FastAPI route parsing — extract (method, path) tuples from AST
# ─────────────────────────────────────────────────────────────────────────


_HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head"}


def _call_kw(call: ast.Call, key: str) -> str | None:
    """Return the string value of keyword argument ``key`` in ``call``, or None."""
    for kw in call.keywords:
        if kw.arg == key and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str):
                return kw.value.value
    return None


def _collect_router_prefixes(tree: ast.Module) -> dict[str, str]:
    """Map router local names -> their declared URL prefix.

    Captures patterns like::

        router = APIRouter(prefix="/auth")
        v1 = APIRouter(prefix="/v1", tags=["v1"])
    """
    prefixes: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        if not (
            (isinstance(func, ast.Name) and func.id == "APIRouter")
            or (isinstance(func, ast.Attribute) and func.attr == "APIRouter")
        ):
            continue
        prefix = _call_kw(node.value, "prefix")
        if prefix is None:
            continue
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                prefixes[tgt.id] = prefix
    return prefixes


def parse_fastapi_routes(
    src_root: Path, domains: set[str] | None = None
) -> list[tuple[str, str, Path, int]]:
    """Return ``[(METHOD, full_path, file, lineno), ...]`` for router decorators.

    The path includes any prefix declared via ``APIRouter(prefix="...")`` on
    the same module. When ``domains`` is given, only routes under
    ``web/api/<domain>/`` are returned. Infrastructure-domain routes
    live one level deeper (``web/api/infra/<domain>/``) and are matched
    on that second segment.
    """
    out: list[tuple[str, str, Path, int]] = []
    web_api_root = src_root / "web" / "api"
    if not web_api_root.is_dir():
        return out
    for file in sorted(web_api_root.rglob("*.py")):
        rel = file.relative_to(web_api_root)
        if domains and rel.parts:
            head = rel.parts[0]
            domain = rel.parts[1] if head == "infra" and len(rel.parts) > 1 else head
            if domain not in domains:
                continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        prefixes = _collect_router_prefixes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if not isinstance(func, ast.Attribute):
                continue
            method = func.attr.lower()
            if method not in _HTTP_METHODS:
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if not isinstance(arg0, ast.Constant) or not isinstance(arg0.value, str):
                continue
            carrier = func.value
            if not (isinstance(carrier, ast.Name) or isinstance(carrier, ast.Attribute)):
                continue
            # Only treat calls on a declared router object (e.g. ``router.get``)
            # as routes. A bare ``config.get(...)`` or ``obj.post(...)`` is not
            # an endpoint even though the attribute name is an HTTP method.
            router_name: str | None = None
            if isinstance(carrier, ast.Name):
                router_name = carrier.id
            if router_name not in prefixes:
                continue
            prefix = prefixes[router_name] or ""
            full_path = prefix.rstrip("/") + "/" + arg0.value.lstrip("/") if arg0.value else prefix
            full_path = full_path.replace("//", "/")
            out.append((method.upper(), full_path, file, node.lineno))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff FastAPI routes against documented endpoints.",
    )
    parser.add_argument("--src-root", default=None)
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated domain names to scope the check to (e.g. "
        "auth,tenants,system). Omit to check the whole tree.",
    )
    parser.add_argument(
        "--docs-root",
        default=None,
        help="Root containing api/*.md (defaults to the bundled upstream docs/).",
    )
    args = parser.parse_args()

    domains: set[str] | None = None
    if args.domains:
        domains = {d.strip() for d in args.domains.split(",") if d.strip()}

    docs_root = _resolve(args.docs_root, DEFAULT_DOCS_ROOT)
    if docs_root is None:
        print("[WARN] docs/ directory not found — nothing to check (exit 0).")
        return 0

    src = resolve_src_root(args.src_root)
    if src is None or not (src / "web" / "api").is_dir():
        print("[WARN] src/web/api/ not found — nothing to check (exit 0).")
        return 0

    doc_eps = parse_docs_endpoints(docs_root, domains)
    routes = parse_fastapi_routes(src, domains)

    if not doc_eps and not routes:
        print("[WARN] No endpoints defined in docs or routes — exit 0.")
        return 0

    doc_keys = {(m, _normalize_path_params(p)) for (m, p, _, _) in doc_eps}
    route_keys = {(m, _normalize_path_params(p)) for (m, p, _, _) in routes}

    errors: list[str] = []

    # 1) routes without docs entries
    for m, p, file, lineno in routes:
        if (m, _normalize_path_params(p)) not in doc_keys:
            errors.append(
                f"{file.relative_to(src)}:{lineno}: route {m} {p} has no matching "
                f"entry in docs/api/*.md"
            )

    # 2) docs entries without route implementations
    for m, p, file, lineno in doc_eps:
        if (m, _normalize_path_params(p)) not in route_keys:
            rel = file.relative_to(docs_root) if file.is_relative_to(docs_root) else file
            errors.append(
                f"{rel}:{lineno}: docs endpoint {m} {p} has no corresponding "
                f"FastAPI route under src/web/api/"
            )

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"[FAIL] {len(errors)} endpoint-coverage mismatch(es)")
        return 1

    print(
        f"[PASS] All {len(routes)} FastAPI routes and {len(doc_eps)} doc endpoints match "
        f"(full bidirectional coverage)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

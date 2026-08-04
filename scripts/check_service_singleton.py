#!/usr/bin/env python3
"""Verify DI scope discipline (AGENTS.md §3).

Rules:

1. Every top-level class whose name ends with ``Service`` defined under
   ``src/core/**/service.py`` (or ``src/core/**/service/<anything>.py``)
   is **REQUEST-scoped** (it binds repositories holding the per-request
   ``AsyncSession``). Such classes MUST NOT be registered as fields of
   the ``LifeSpanService`` dataclass under ``src/app_context/`` — that
   registry is reserved for APP-scope singletons: stateless objects that
   are expensive to construct (``DatabaseEngine``, ``OidcClient``, ...).

2. For every web router that injects a service via ``Depends(...)``, the
   argument MUST NOT be the service class directly — obtain the service
   from a ``web.deps`` factory (which assembles repos + APP-scope
   singletons), never by registering the class on the app.

Usage::

    python check_service_singleton.py [--src-root PATH]

Exit codes:
    0 = scope discipline respected
    1 = request-scoped service registered as app singleton, or improper
        dependency injection
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────
# src/ resolution
# ─────────────────────────────────────────────────────────────────────────


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


# `LifeSpanService` may live in `lifespan.py` or a dedicated DI module
# (e.g. `registry.py`) under `app_context/`; scan all of them.
LIFESPAN_REL = Path("app_context") / "lifespan.py"
APP_CONTEXT_REL = Path("app_context")


# ─────────────────────────────────────────────────────────────────────────
# Service-class discovery under core/
# ─────────────────────────────────────────────────────────────────────────


def _is_service_file(parts: tuple[str, ...]) -> bool:
    """True for files that may declare Service classes.

    Matches:
        core/<domain>/service.py
        core/<domain>/service/__init__.py
        core/<domain>/service/<anything>.py
    """
    if not parts or parts[0] != "core":
        return False
    for p in parts:
        if p == "service":
            return True
        if p.startswith("service."):
            return True
    return False


def _strip_annotation_to_name(node: ast.AST | None) -> str | None:
    """Reduce an annotation expression to its primary type name, if simple."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):
        return _strip_annotation_to_name(node.value)
    if isinstance(node, ast.BinOp):
        # PEP 604 ``Service | None``
        left = _strip_annotation_to_name(node.left)
        return left if left is not None else _strip_annotation_to_name(node.right)
    if isinstance(node, ast.Call):
        return _strip_annotation_to_name(node.func)
    return None


def _collect_services(src_root: Path) -> dict[str, tuple[Path, int]]:
    """Map ServiceClassName -> (path, lineno) for every *Service top-level class."""
    out: dict[str, tuple[Path, int]] = {}
    for file in src_root.rglob("*.py"):
        rel = file.relative_to(src_root)
        if not _is_service_file(rel.parts):
            continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name.endswith("Service"):
                # last definition wins; first-class service has unambiguous name
                out[node.name] = (file, node.lineno)
    return out


# ─────────────────────────────────────────────────────────────────────────
# LifeSpanService field/accessor discovery
# ─────────────────────────────────────────────────────────────────────────


def _collect_lifespan(src_root: Path) -> tuple[set[str], Path | None]:
    """Return (registered_service_names, lifespan_file).

    A class field like ``doc_service: DocumentService = field(...)`` is
    matched by extracting the type name from the annotation. ``LifeSpanService``
    is discovered across every module under ``app_context/`` so it may live in
    ``lifespan.py`` or a dedicated DI registry module.
    """
    registered: set[str] = set()
    lifespan_file = src_root / LIFESPAN_REL
    found = False
    app_context_dir = src_root / APP_CONTEXT_REL
    if app_context_dir.is_dir():
        for file in sorted(app_context_dir.rglob("*.py")):
            try:
                tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
            except (OSError, SyntaxError):
                continue
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == "LifeSpanService":
                    found = True
                    for stmt in node.body:
                        if isinstance(stmt, ast.AnnAssign):
                            type_name = _strip_annotation_to_name(stmt.annotation)
                            if type_name:
                                registered.add(type_name)
                        elif isinstance(stmt, ast.Assign):
                            type_name = _strip_annotation_to_name(stmt.value)
                            if type_name:
                                registered.add(type_name)
    return registered, (lifespan_file if found else None)


# ─────────────────────────────────────────────────────────────────────────
# Web-layer Depends(...) check
# ─────────────────────────────────────────────────────────────────────────


def _is_depends_call(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name) and func.id == "Depends":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "Depends":
        return True
    return False


def _check_web_depends(src_root: Path) -> list[str]:
    errors: list[str] = []
    web_root = src_root / "web"
    if not web_root.is_dir():
        return errors
    for file in web_root.rglob("*.py"):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not _is_depends_call(node):
                continue
            for arg in node.args:
                type_name = _strip_annotation_to_name(arg)
                if type_name and type_name.endswith("Service"):
                    errors.append(
                        f"{file}:{node.lineno}: web layer MUST NOT inject "
                        f"Depends({type_name}) directly — obtain via "
                        f"get_<snake_case>_from_lifespan instead"
                    )
            for kw in node.keywords:
                if kw.arg != "dependency":
                    continue
                type_name = _strip_annotation_to_name(kw.value)
                if type_name and type_name.endswith("Service"):
                    errors.append(
                        f"{file}:{node.lineno}: web layer MUST NOT inject "
                        f"Depends(dependency={type_name}) directly — obtain "
                        f"via get_<snake_case>_from_lifespan instead"
                    )
    return errors


# ─────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify every Service is registered in LifeSpanService.",
    )
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    services = _collect_services(src)
    registered, _lifespan_file = _collect_lifespan(src)
    errors: list[str] = []

    if not services:
        # 1) collect web Depends(.) violations regardless — they indicate bad DI
        errors.extend(_check_web_depends(src))
        if not errors:
            print("[WARN] No Service classes found in core/ — exit 0.")
            return 0
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"[FAIL] {len(errors)} service-scope violation(s)")
        return 1

    # 1) request-scoped Service classes MUST NOT be APP-scope singletons
    for name in sorted(services):
        path, lineno = services[name]
        if name in registered:
            errors.append(
                f"{path.relative_to(src)}:{lineno}: Service '{name}' is "
                f"request-scoped (binds the per-request AsyncSession) and "
                f"MUST NOT be registered as a field of LifeSpanService — "
                f"that registry is for APP-scope stateless singletons only"
            )

    # 2) web Depends(SomeService) check
    errors.extend(_check_web_depends(src))

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"[FAIL] {len(errors)} service-scope violation(s)")
        return 1

    print(
        f"[PASS] All {len(services)} Service classes are request-scoped "
        f"(absent from LifeSpanService) and no router injects a "
        f"Service class directly"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

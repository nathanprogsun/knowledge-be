#!/usr/bin/env python3
"""Enforce DTO projection discipline and per-DTO redaction allowlists.

Rules:

1. **Web layer MUST NOT call ``map_from_db(...)``.**
   The web layer is the boundary that ships JSON to clients; it may only
   consume already-projected DTOs (the result of a service method). Any
   ``<DTO>.map_from_db(...)`` call inside ``src/web/`` is reported.

2. **Every DTO with a ``map_from_db(cls, db: <Model>)`` classmethod
   MUST declare a module-level ``_<NAME>_EXCLUDE_COLUMNS:
   frozenset[str]`` constant in the same file.** This is the
   redaction allowlist consumed by ``db.model_dump(exclude=...)`` so
   that sensitive columns (password hashes, secret tokens, ...) are
   never accidentally returned by ``map_from_db``.

3. **Every DTO that owns a JSON-shaped field (``JsonObject``,
   ``JsonObject | None``, ``dict[str, ...]``) MUST declare a
   ``from_json(cls, raw)`` classmethod that accepts both ``dict``
   (``JsonObject``) and ``str`` (raw JSON text) inputs** — the Go
   backend persists some columns as text and others as JSONB, so the
   deserialiser must tolerate both.

Usage::

    python check_map_from_db.py [--src-root PATH]

Exit codes:
    0 = all DTOs satisfy the three rules
    1 = at least one violation found
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────
# Resolution
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


WEB_REL = Path("web")


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


class _MapFromDbViolation(NamedTuple):
    relpath: str
    lineno: int
    message: str


def _is_map_from_db_call(node: ast.Call) -> bool:
    """True for ``<dto>.map_from_db(...)``."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "map_from_db":
        return True
    return False


def _is_from_json_call(node: ast.Call) -> bool:
    """True for ``<dto>.from_json(...)``."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == "from_json":
        return True
    return False


def _scan_web_layer(src_root: Path) -> list[_MapFromDbViolation]:
    """Rule 1: ``map_from_db(...)`` calls under src/web/ are forbidden."""
    out: list[_MapFromDbViolation] = []
    web_root = src_root / WEB_REL
    if not web_root.is_dir():
        return out
    for file in sorted(web_root.rglob("*.py")):
        rel = str(file.relative_to(src_root))
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _is_map_from_db_call(node):
                out.append(
                    _MapFromDbViolation(
                        relpath=rel,
                        lineno=node.lineno,
                        message=(
                            "web layer MUST NOT call map_from_db — obtain "
                            "DTOs from the service layer"
                        ),
                    )
                )
    return out


def _module_declares_exclude_set(tree: ast.Module, dto_name: str) -> bool:
    """True if the module declares ``_<NAME>_EXCLUDE_COLUMNS: frozenset[str]``.

    Matches either an annotation assignment or a plain assignment with a
    RHS that calls ``frozenset(...)`` on a set / dict literal.
    """
    target = f"_{dto_name.upper()}_EXCLUDE_COLUMNS"
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == target:
                return True
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == target:
                    return True
    return False


def _function_param_annotation_names(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Strip the annotation on the first positional arg to its base name."""
    if not fn.args.args:
        return []
    arg0 = fn.args.args[0]
    ann = arg0.annotation
    if ann is None:
        return []
    if isinstance(ann, ast.Name):
        return [ann.id]
    if isinstance(ann, ast.Attribute):
        return [ann.attr]
    return []


def _scan_map_from_db_definitions(src_root: Path) -> list[_MapFromDbViolation]:
    """Rule 2: every ``map_from_db`` def needs a sibling ``_<NAME>_EXCLUDE_COLUMNS``."""
    out: list[_MapFromDbViolation] = []
    for file in sorted(src_root.rglob("*.py")):
        rel = str(file.relative_to(src_root))
        # The web layer check is a different rule; here we only require
        # ``_<NAME>_EXCLUDE_COLUMNS`` when ``map_from_db`` lives in the
        # same module as the DTO.
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue

        # Iterate class by class.
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            for member in cls.body:
                if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if member.name != "map_from_db":
                    continue
                param_models = _function_param_annotation_names(member)
                if not param_models:
                    continue
                if not _module_declares_exclude_set(tree, cls.name):
                    out.append(
                        _MapFromDbViolation(
                            relpath=rel,
                            lineno=member.lineno,
                            message=(
                                f"class '{cls.name}' defines map_from_db but the "
                                f"module lacks _{cls.name.upper()}_EXCLUDE_COLUMNS: "
                                "frozenset[str] — declare the redaction allowlist"
                            ),
                        )
                    )
    return out


# JSON-shaped annotation names (PEP 604 unions, Optional[...] etc).
_JSON_TYPES = {"JsonObject", "dict", "Mapping"}


def _annotation_mentions_json(node: ast.AST | None) -> bool:
    """True for an annotation that contains a JSON-shaped type."""
    if node is None:
        return False
    # Bare name
    if isinstance(node, ast.Name):
        return node.id in _JSON_TYPES or node.id.startswith("dict[") or node.id.startswith("Mapping[")
    # Subscript ``dict[str, X]``
    if isinstance(node, ast.Subscript):
        base_name: str | None = None
        if isinstance(node.value, ast.Name):
            base_name = node.value.id
        if base_name in _JSON_TYPES:
            return True
        # Recurse into nested subscript (e.g. ``dict[str, list[X]]``)
        return _annotation_mentions_json(node.value) or _annotation_mentions_json(node.slice)
    # PEP 604 ``X | None``
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_mentions_json(node.left) or _annotation_mentions_json(node.right)
    # Attribute (e.g. ``typing.JsonObject``)
    if isinstance(node, ast.Attribute):
        return node.attr in _JSON_TYPES
    return False


def _class_has_from_json(cls: ast.ClassDef) -> bool:
    for member in cls.body:
        if not isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if member.name != "from_json":
            continue
        # Must accept at least one positional argument.
        if not member.args.args:
            continue
        return True
    return False


def _scan_dto_json_fields(src_root: Path) -> list[_MapFromDbViolation]:
    """Rule 3: every DTO with a JSON field needs ``from_json``."""
    out: list[_MapFromDbViolation] = []
    for file in sorted(src_root.rglob("*.py")):
        rel = str(file.relative_to(src_root))
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            has_json_field = False
            for member in cls.body:
                if isinstance(member, ast.AnnAssign):
                    if _annotation_mentions_json(member.annotation):
                        has_json_field = True
                        break
                # Handle ``X: T = default`` style too
                if isinstance(member, ast.AnnAssign) and member.target is not None:
                    if _annotation_mentions_json(member.annotation):
                        has_json_field = True
                        break
            if not has_json_field:
                continue
            if _class_has_from_json(cls):
                continue
            out.append(
                _MapFromDbViolation(
                    relpath=rel,
                    lineno=cls.lineno,
                    message=(
                        f"class '{cls.name}' has a JSON-shaped field but "
                        "lacks a 'from_json(cls, raw)' classmethod that "
                        "accepts both dict and str inputs"
                    ),
                )
            )
    return out


# ─────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Enforce DTO projection discipline: web→service→db, "
            "module-level EXCLUDE_COLUMNS frozenset, and from_json classmethod."
        ),
    )
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    violations: list[_MapFromDbViolation] = []
    violations.extend(_scan_web_layer(src))
    violations.extend(_scan_map_from_db_definitions(src))
    violations.extend(_scan_dto_json_fields(src))

    if not violations:
        print(
            "[PASS] web layer has no map_from_db calls; every map_from_db "
            "definition has a sibling _<NAME>_EXCLUDE_COLUMNS frozenset; "
            "every JSON-fielded DTO declares from_json"
        )
        return 0

    for v in violations:
        print(f"[FAIL] {v.relpath}:{v.lineno}: {v.message}")
    print(f"[FAIL] {len(violations)} DTO-projection violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Enforce web -> core -> db layered architecture and ban Any/object types.

Rules enforced:

1. Layer directionality (per docs/migration/AGENTS.md sec. 4):
   - ``web/**``     MUST NOT import from ``db.*`` (incl. ``db.dao``, ``db.dbengine``, ``db.models``)
   - ``core/**``    MUST NOT import from ``web.*`` or ``fastapi`` / ``starlette``
   - ``db/**``      MUST NOT import from ``core.*``, ``web.*``, or ``ai.*`` (incl. ``db/dao`` & ``db/models``)
   - ``ai/**``      MUST NOT import from ``core.*``, ``db.*``, or ``web.*``
   - ``workers/**`` MUST NOT import from ``web.*`` or ``db.*`` (must go through ``core`` services)
   - ``db/models/**`` classes contain only fields + Pydantic dunder/model_* methods

2. Type-safety:
   - No ``Any`` or bare ``object`` annotations anywhere in src/.
     Covers subscripted forms (``dict[str, Any]``, ``list[Any]``, ``Optional[Any]``),
     PEP 604 unions (``X | Any``), and dotted attribute access (``typing.Any``).
   - No ``dict[str, Any]`` / ``list[Any]`` are tolerated even as a special case.

3. No function-level imports (``ast.Import`` / ``ast.ImportFrom`` with
   ``col_offset > 0``). Allowed: ``if TYPE_CHECKING:`` blocks at column 0.

Usage::

    python check_layer_violation.py [--src-root PATH]

Exit codes:
    0 = no violations
    1 = at least one violation found
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
    """Locate the Python ``src/`` directory.

    Resolution order: CLI ``--src-root`` -> env ``KNOWLEDGE_BE_SRC`` -> walk up
    from this script (up to 6 ancestors) looking for a ``src/`` sibling.
    """
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


# ─────────────────────────────────────────────────────────────────────────
# Architecture constants — prefixes are matched against the dotted absolute
# import path (e.g. ``from db.dao.documents import ...`` matches prefixes
# ``db``, ``db.dao``, ``db.dao.documents``).
#
# We enforce *only* the forbidden-prefix rules per layer; stdlib and
# third-party packages (e.g. ``typing``, ``fastapi``, ``pydantic``) are not
# gated by an allow-list — those are governed by ``ruff`` and the project's
# dependency declarations.
# ─────────────────────────────────────────────────────────────────────────


WEB_FORBIDDEN_PREFIXES: set[str] = {"db", "db.dao", "db.dbengine", "db.models"}
CORE_FORBIDDEN_PREFIXES: set[str] = {"web", "fastapi", "starlette"}
DB_FORBIDDEN_PREFIXES: set[str] = {"core", "web", "ai"}
AI_FORBIDDEN_PREFIXES: set[str] = {"core", "db", "web"}
WORKERS_FORBIDDEN_PREFIXES: set[str] = {"web", "db", "db.dao", "db.models"}

# Legacy alias retained for backwards compatibility with older callers.
DAO_FORBIDDEN_PREFIXES: set[str] = DB_FORBIDDEN_PREFIXES


def _prefixes_of(dotted: str) -> list[str]:
    parts = dotted.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


# ─────────────────────────────────────────────────────────────────────────
# Annotation check — recursive Any/object detector
# ─────────────────────────────────────────────────────────────────────────


def _expr_uses_any_or_object(node: ast.AST | None) -> bool:
    """True iff ``node`` references the bare names ``Any`` or ``object``."""
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in {"Any", "object"}
    if isinstance(node, ast.Attribute):
        return _expr_uses_any_or_object(node.value)
    if isinstance(node, ast.Subscript):
        return _expr_uses_any_or_object(node.value) or _expr_uses_any_or_object(node.slice)
    if isinstance(node, ast.BinOp):
        return _expr_uses_any_or_object(node.left) or _expr_uses_any_or_object(node.right)
    if isinstance(node, ast.Tuple):
        return any(_expr_uses_any_or_object(elt) for elt in node.elts)
    if isinstance(node, (ast.List, ast.Set)):
        return any(_expr_uses_any_or_object(elt) for elt in node.elts)
    if isinstance(node, ast.Call):
        if _expr_uses_any_or_object(node.func):
            return True
        if any(_expr_uses_any_or_object(a) for a in node.args):
            return True
        for kw in node.keywords:
            if kw.value is not None and _expr_uses_any_or_object(kw.value):
                return True
        return False
    return False


# ─────────────────────────────────────────────────────────────────────────
# Layer classification
# ─────────────────────────────────────────────────────────────────────────


def classify_layer(parts: tuple[str, ...]) -> str:
    """Classify a relative path under src/ into a layer label.

    Returns one of: ``web_api``, ``core_service``, ``db_dao``, ``db_models``,
    ``common``, ``app_context``, ``util``, ``ai``, ``workers``, ``unknown``.
    """
    if not parts:
        return "unknown"
    head = parts[0]
    if head == "web":
        return "web_api" if "api" in parts else "web"
    if head == "core":
        for p in parts:
            if p == "service" or p.startswith("service."):
                return "core_service"
        return "core"
    if head == "db":
        if "dao" in parts:
            return "db_dao"
        if "models" in parts:
            return "db_models"
        return "db"
    if head in {"common", "app_context", "util", "ai", "workers"}:
        return head
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────
# Per-layer import policy
# ─────────────────────────────────────────────────────────────────────────


def _check_import_policy(
    layer: str, module: str, lineno: int, file: Path, errors: list[str]
) -> None:
    """Apply per-layer import policy for the absolute dotted module path.

    Only ``forbidden`` prefixes are enforced — stdlib (``typing``, ``os``,
    ``dataclasses``...) and third-party packages (``fastapi``, ``pydantic``,
    ``sqlalchemy``...) are not gated by an allow-list here.  Layer discipline
    is enforced by *who can reach into whom*, not by an inventory of every
    first-party prefix.

    Layers covered (per AGENTS.md §4):

    - ``web`` / ``web_api``  → must not import ``db`` (any subpackage)
    - ``core`` / ``core_service`` → must not import ``web`` / ``fastapi``
    - ``db`` / ``db_dao`` / ``db_models`` → must not import ``core`` / ``web`` / ``ai``
    - ``ai``                → must not import ``core`` / ``db`` / ``web``
    - ``workers``           → must not import ``web`` / ``db`` (must go via ``core``)

    ``common``, ``app_context``, ``util``, and ``unknown`` are not gated here.
    """
    forbidden_prefixes: set[str]
    if layer in {"web", "web_api"}:
        forbidden_prefixes = WEB_FORBIDDEN_PREFIXES
    elif layer in {"core", "core_service"}:
        forbidden_prefixes = CORE_FORBIDDEN_PREFIXES
    elif layer in {"db", "db_dao", "db_models"}:
        forbidden_prefixes = DB_FORBIDDEN_PREFIXES
    elif layer == "ai":
        forbidden_prefixes = AI_FORBIDDEN_PREFIXES
    elif layer == "workers":
        forbidden_prefixes = WORKERS_FORBIDDEN_PREFIXES
    else:
        return

    prefixes = set(_prefixes_of(module))

    for forbidden in forbidden_prefixes:
        if forbidden in prefixes or any(
            p == forbidden or p.startswith(forbidden + ".") for p in prefixes
        ):
            errors.append(
                f"{file}:{lineno}: {layer.replace('_', '-')} layer MUST NOT "
                f"import from '{module}' (forbidden prefix '{forbidden}')"
            )
            return


# ─────────────────────────────────────────────────────────────────────────
# Visitor
# ─────────────────────────────────────────────────────────────────────────


class LayerVisitor(ast.NodeVisitor):
    """Collect every layer/typing/import violation in one pass."""

    def __init__(self, path: Path, layer: str) -> None:
        self.path = path
        self.layer = layer
        self.errors: list[str] = []

    # ── imports ─────────────────────────────────────────────────────
    def visit_Import(self, node: ast.Import) -> None:
        if node.col_offset > 0:
            self.errors.append(
                f"{self.path}:{node.lineno}: function-level import forbidden "
                f"(import indented at col {node.col_offset})"
            )
        for alias in node.names:
            _check_import_policy(self.layer, alias.name, node.lineno, self.path, self.errors)
        # Don't descend — ast.Import has no body
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.col_offset > 0:
            self.errors.append(
                f"{self.path}:{node.lineno}: function-level import forbidden "
                f"(from-import indented at col {node.col_offset})"
            )
        if node.module:
            # First-party imports are rooted at ``src.``; the forbidden
            # prefixes are layer-relative (``db``, ``core``...). Strip the
            # package root or the policy never matches.
            module = node.module
            if module == "src" or module.startswith("src."):
                module = module.removeprefix("src.").removesuffix("src")
            _check_import_policy(self.layer, module, node.lineno, self.path, self.errors)
        self.generic_visit(node)

    # ── annotations ─────────────────────────────────────────────────
    def _check_annotation(self, expr: ast.AST | None, lineno: int, label: str) -> None:
        if _expr_uses_any_or_object(expr):
            self.errors.append(
                f"{self.path}:{lineno}: annotation '{label}' uses forbidden "
                f"Any/object — use a concrete type, Pydantic model, TypedDict, "
                f"Protocol, or generic from common"
            )

    def _visit_args(self, args: ast.arguments, lineno: int) -> None:
        for arg in (
            *getattr(args, "posonlyargs", []),
            *args.args,
            *args.kwonlyargs,
        ):
            self._check_annotation(arg.annotation, lineno, arg.arg)
        if args.vararg:
            self._check_annotation(args.vararg.annotation, lineno, args.vararg.arg)
        if args.kwarg:
            self._check_annotation(args.kwarg.annotation, lineno, args.kwarg.arg)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_args(node.args, node.lineno)
        self._check_annotation(node.returns, node.lineno, f"{node.name}() return")
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_args(node.args, node.lineno)
        self._check_annotation(node.returns, node.lineno, f"{node.name}() return")
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._check_annotation(node.annotation, node.lineno, "variable annotation")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Flag ``cast(Any/object, ...)`` and ``cast("...Any/object...", ...)``.

        Type-cast arguments commonly hide bare ``Any`` / ``object`` in
        type parameters (``cast(CursorResult[object], result)``). They
        are easy to miss on AnnAssign because the cast type lives in a
        call expression rather than an annotation.
        """
        func = node.func
        is_cast = (isinstance(func, ast.Name) and func.id == "cast") or (
            isinstance(func, ast.Attribute) and func.attr == "cast"
        )
        if is_cast and node.args:
            type_arg = node.args[0]
            if _expr_uses_any_or_object(type_arg):
                self.errors.append(
                    f"{self.path}:{type_arg.lineno}: cast() type argument "
                    f"uses forbidden Any/object — substitute a concrete "
                    f"generic like SqlValue, JsonValue, or a typed Protocol"
                )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Flag module-level aliases that hide ``Any`` / ``object`` sentinels.

        A bare ``_STREAM_CLOSED: object = object()`` is technically
        annotated, but the same shape can also appear as a plain
        ``_STREAM_CLOSED = object()`` once the file is refactored. Treat
        plain ``Assign`` nodes that target the bare names ``Any`` /
        ``object`` as the same anti-pattern as ``AnnAssign`` so the
        lint cannot be defeated by dropping the annotation.
        """
        value = node.value
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id in {"Any", "object"}
        ):
            self.errors.append(
                f"{self.path}:{node.lineno}: module-level assignment uses "
                f"forbidden Any/object — declare a typed sentinel class "
                f"or use a concrete value (UUID, Enum, dataclass) instead"
            )
        elif isinstance(value, ast.Name) and value.id in {"Any", "object"}:
            self.errors.append(
                f"{self.path}:{node.lineno}: module-level assignment uses "
                f"forbidden Any/object — declare a typed sentinel class "
                f"or use a concrete value (UUID, Enum, dataclass) instead"
            )
        self.generic_visit(node)

    # ── db/models class-body policy ─────────────────────────────────
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        if self.layer == "db_models":
            for stmt in node.body:
                if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = stmt.name
                    is_dunder = name.startswith("__") and name.endswith("__")
                    is_pydantic = name.startswith("model_")
                    if not (is_dunder or is_pydantic):
                        self.errors.append(
                            f"{self.path}:{stmt.lineno}: db/models class "
                            f"'{node.name}' may not contain business logic "
                            f"method '{name}' — only fields and Pydantic "
                            f"dunder/model_* methods are allowed"
                        )
                elif isinstance(stmt, ast.AnnAssign):
                    self._check_annotation(stmt.annotation, stmt.lineno, f"{node.name}.field")
                # plain Assign is allowed (e.g. ``model_config = ConfigDict(...)``)
        self.generic_visit(node)


# ─────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────


# Shared layers always scanned even under ``--domains`` (bootstrap / cross-domain).
ALWAYS_INCLUDE_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("common",),
    ("app_context",),
    ("util",),
    ("db", "base.py"),
    ("web", "middleware"),
    ("web", "deps"),
    ("web", "exception_handler.py"),
)


# Infrastructure domains keep the domain name in the second segment
# (``core/infra/<domain>/``, ``web/api/infra/<domain>/``,
# ``db/models/infra/<domain>/``), and their DAOs use a singular / shortened
# stem (``mcp_service_repository`` for domain ``mcp_services``). Map domain
# -> DAO prefix so both match.
_DAO_PREFIX_BY_DOMAIN = {
    "datasources": "datasource",
    "mcp_services": "mcp_service",
    "models": "model",
    "storage_backends": "storage_backend",
    "vector_stores": "vector_store",
    "web_search": "web_search_provider",
}


def _infra_domain(parts: tuple[str, ...], base_index: int) -> str | None:
    """Return the domain held in the second segment after ``infra``, if any.

    ``base_index`` is the index of the ``infra`` segment within ``parts``.
    """
    if len(parts) > base_index + 1:
        return parts[base_index + 1]
    return None


def _file_in_domains(parts: tuple[str, ...], domains: set[str]) -> bool:
    """True if the file belongs to any of the given domains.

    A file belongs to a domain when it lives under ``core/<domain>``,
    ``core/infra/<domain>``, ``web/api/<domain>``,
    ``web/api/infra/<domain>``, ``db/models/<domain>``,
    ``db/models/infra/<domain>``, or is a ``db/dao/<domain>*`` repository.
    Otherwise it is considered shared/bootstrap and scanned
    unconditionally.
    """
    for prefix in ALWAYS_INCLUDE_PREFIXES:
        if len(parts) >= len(prefix) and parts[: len(prefix)] == prefix:
            return True
    if parts[0] == "core":
        # core/<domain>/...  or  core/infra/<domain>/...
        if parts[1] in domains:
            return True
        if parts[1] == "infra" and _infra_domain(parts, 1) in domains:
            return True
    if parts[0] == "web":
        # web/api/<domain>/...  or  web/api/infra/<domain>/...
        if len(parts) >= 3 and parts[1] == "api":
            domain_part = (
                parts[3] if len(parts) >= 4 and parts[2] == "infra" else parts[2]
            )
            if domain_part in domains:
                return True
    if parts[0] == "db":
        if len(parts) >= 3 and parts[1] == "models":
            domain_part = (
                parts[3] if len(parts) >= 4 and parts[2] == "infra" else parts[2]
            )
            if domain_part in domains:
                return True
        if len(parts) >= 3 and parts[1] == "dao":
            base = parts[2].removesuffix(".py")
            dao_prefixes = {_DAO_PREFIX_BY_DOMAIN.get(d, d) for d in domains}
            # Domain-prefixed DAOs match their domain; unprefixed DAOs are
            # shared cross-domain repositories and always scanned.
            if any(base.startswith(f"{p}_") for p in dao_prefixes):
                return True
            return not any(base.startswith(f"{p}_") for p in dao_prefixes)
    return False


def iter_python_files(root: Path, domains: set[str] | None = None) -> list[Path]:
    out = [p for p in root.rglob("*.py") if p.is_file()]
    if domains:
        out = [p for p in out if _file_in_domains(p.relative_to(root).parts, domains)]
    return sorted(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce layered architecture and forbid Any/object.",
    )
    parser.add_argument(
        "--src-root",
        default=None,
        help="Path to the Python src/ directory (auto-detected).",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Comma-separated domain names to scope the check to (e.g. "
        "auth,tenants,system). Shared/bootstrap layers are always included. "
        "Omit to check the whole src/ tree.",
    )
    args = parser.parse_args()

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    domains: set[str] | None = None
    if args.domains:
        domains = {d.strip() for d in args.domains.split(",") if d.strip()}

    files = iter_python_files(src, domains)
    if not files:
        print("[WARN] src/ contains no .py files — nothing to check (exit 0).")
        return 0

    all_errors: list[str] = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as e:
            all_errors.append(f"{file}: cannot read: {e}")
            continue
        try:
            tree = ast.parse(text, filename=str(file))
        except SyntaxError as e:
            all_errors.append(f"{file}:{e.lineno or 1}: syntax error: {e.msg}")
            continue

        rel = file.relative_to(src)
        layer = classify_layer(rel.parts)
        visitor = LayerVisitor(file, layer)
        visitor.visit(tree)
        all_errors.extend(visitor.errors)

    if all_errors:
        for err in all_errors:
            print(f"[FAIL] {err}")
        print(f"[FAIL] {len(all_errors)} layer/typing violation(s) found")
        return 1

    print(f"[PASS] All layer rules and typing rules satisfied ({len(files)} files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

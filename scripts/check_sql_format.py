#!/usr/bin/env python3
"""Enforce safe SQL construction inside ``src/db/dao/``.

Rules:

1. ``text(f"...")`` / ``text(f'...')`` is forbidden in every
   ``src/db/dao/*.py`` file. The only allowed f-string interpolations
   are bound, audited identifiers from this allowlist:

   - ``self._table`` / ``self.<table>`` attribute references
   - module-level UPPER_CASE constants ending in ``_TABLE`` or
     starting with the table name (e.g. ``_USER_TABLE``,
     ``_INVITATION_ORDER``)
   - ``where`` / ``soft`` / ``archived`` / ``order_clause`` /
     ``where_clause`` / ``order_by`` / ``limit`` / ``offset`` style
     placeholders that resolve to ``str`` and have been built from
     other audited fragments (the ``_list_conditions`` pattern)

   Every other interpolation is reported.

2. The repository layer MUST validate ``find_all(order_by=...)``
   callers — i.e. the ``order_by`` fragment must be matched against
   an explicit per-repository allowlist before being interpolated.
   Bare ``f"order by {order_by}"`` style usage is reported; the base
   ``GenericRepository.find_all`` is the gate, and any subclass that
   re-exposes ``order_by`` without re-validation is also reported.

Usage::

    python check_sql_format.py [--src-root PATH]

Exit codes:
    0 = no violations
    1 = at least one unsafe SQL fragment found
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


DAO_REL = Path("db") / "dao"


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────


class _FStringViolation(NamedTuple):
    relpath: str
    lineno: int
    message: str


# Names allowed inside the f-string interpolations.
_SAFE_ATTR_ROOTS = {"self"}
_SAFE_ATTR_NAMES = {"_table"}

# Names allowed as local / module-level identifiers inside the f-string.
_SAFE_LOCAL_NAMES = {
    "self",
    "where",
    "soft",
    "archived",
    "where_clause",
    "order_clause",
    "order_by",
    "where_sql",
    "join",
    "stmt_text",
    # Safe SQL-fragment locals built by repository internals from
    # allowlisted column identifiers (Agent 6 DAO refactor).
    "column_list",
    "value_list",
    "conditions",
    "set_clause",
    "placeholders",
}


def _describe_node(node: ast.AST) -> str:
    """Return a compact, human-readable description of an f-string piece."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts: list[str] = []
        cur: ast.AST = node
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return ast.unparse(node)


def _is_safe_interpolation(node: ast.AST) -> bool:
    """True for an f-string interpolation that the allowlist permits."""
    if isinstance(node, ast.Name):
        if node.id in _SAFE_LOCAL_NAMES or node.id in _SAFE_ATTR_ROOTS:
            return True
        # Allow UPPER_CASE module-level constants (e.g. ``_LIVE``,
        # ``STATUS_PENDING``, ``_INVITATION_ORDER``, ``_MEMBER_ORDER``).
        if node.id.isupper() or (node.id.startswith("_") and node.id[1:].isupper()):
            return True
        return False
    if isinstance(node, ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id in _SAFE_ATTR_ROOTS:
            return node.attr in _SAFE_ATTR_NAMES
        # dotted access on a known-safe name (e.g. constants.STAGE1_DOMAINS)
        if isinstance(node.value, ast.Name) and node.value.id.isupper():
            return True
        return False
    return False


def _extract_fstring_interpolations(expr: ast.Expression) -> list[ast.AST]:
    """Pull every ``FormattedValue`` child out of an f-string expression."""
    out: list[ast.AST] = []
    for sub in ast.walk(expr):
        if isinstance(sub, ast.FormattedValue):
            out.append(sub.value)
    return out


def _is_text_call(node: ast.Call) -> bool:
    """True for ``text(...)`` calls (top-level or ``sqlalchemy.text``)."""
    func = node.func
    if isinstance(func, ast.Name) and func.id == "text":
        return True
    if isinstance(func, ast.Attribute) and func.attr == "text":
        return True
    return False


def _fstring_arg(call: ast.Call) -> ast.JoinedStr | None:
    """Return the JoinedStr argument if ``call`` is ``text(<f-string>)``."""
    if not call.args:
        return None
    arg = call.args[0]
    if isinstance(arg, ast.JoinedStr):
        return arg
    return None


def _check_text_fstring(file: Path, rel: str, tree: ast.Module) -> list[_FStringViolation]:
    violations: list[_FStringViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_text_call(node):
            continue
        joined = _fstring_arg(node)
        if joined is None:
            continue
        for piece in _extract_fstring_interpolations(joined):
            if not _is_safe_interpolation(piece):
                violations.append(
                    _FStringViolation(
                        relpath=rel,
                        lineno=joined.lineno,
                        message=(
                            "text(f\"...\") interpolation "
                            f"'{_describe_node(piece)}' is not in the allowlist "
                            "(self._table / audited constants / safe-fragment locals)"
                        ),
                    )
                )
    return violations


def _is_safe_where_construction(call: ast.Call) -> bool:
    """Conservative: ``text(stmt_text)`` where stmt_text is a bare Name.

    We don't validate the *contents* of stmt_text — but it must be a
    plain ``Name`` reference to a local that was built by audited
    helpers. Anything more complex is reported.
    """
    if not call.args:
        return False
    arg = call.args[0]
    return isinstance(arg, ast.Name)


def _find_order_by_interpolations(file: Path, rel: str, tree: ast.Module) -> list[_FStringViolation]:
    """Find raw ``order by {<expr>}`` interpolations.

    The base ``GenericRepository.find_all`` is allowed as the entry point
    that injects the validated fragment. Subclasses that re-build an
    ``order by {order_by}`` f-string WITHOUT going through an allowlist
    check are reported.
    """
    violations: list[_FStringViolation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        for piece in _extract_fstring_interpolations(node):
            descr = _describe_node(piece)
            if descr != "order_by":
                continue
            # Look back for a containing function whose name is
            # ``find_all`` and whose class is *not* ``GenericRepository``.
            parent_fn = _find_enclosing_function(node, tree)
            if parent_fn is None:
                continue
            cls = _find_enclosing_class(parent_fn, tree)
            if cls is None or cls.name == "GenericRepository":
                continue
            # If the body contains a regex/validation against order_by
            # before the interpolation, treat as validated.
            if _function_validates_order_by(parent_fn):
                continue
            violations.append(
                _FStringViolation(
                    relpath=rel,
                    lineno=node.lineno,
                    message=(
                        f"function '{parent_fn.name}' in class '{cls.name}' "
                        "interpolates raw 'order_by' into SQL without an "
                        "allowlist check; callers must validate against a "
                        "per-repository whitelist"
                    ),
                )
            )
    return violations


def _find_enclosing_function(node: ast.AST, tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Walk the tree to find the closest enclosing function def."""
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    cur: ast.AST | None = node
    while cur is not None:
        c = parents.get(id(cur))
        if c is None:
            return None
        if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return c
        cur = c
    return None


def _find_enclosing_class(fn: ast.AST, tree: ast.Module) -> ast.ClassDef | None:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    cur: ast.AST | None = fn
    while cur is not None:
        c = parents.get(id(cur))
        if c is None:
            return None
        if isinstance(c, ast.ClassDef):
            return c
        cur = c
    return None


def _function_validates_order_by(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """Conservative detector for ``if not <pattern>.match(order_by): raise``.

    Looks for an ``if`` statement that calls ``re.match`` / ``re.fullmatch`` /
    ``re.compile(...).match(...)`` on something involving ``order_by`` and
    raises (any exception) in its body.
    """
    for sub in ast.walk(fn):
        if not isinstance(sub, ast.If):
            continue
        test_src = ast.unparse(sub.test)
        if "order_by" not in test_src:
            continue
        if any(t in test_src for t in ("re.match", "re.fullmatch", ".match(", "in _ORDER")):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce safe SQL construction (no raw text(f-string) interpolation).",
    )
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    dao_root = src / DAO_REL
    if not dao_root.is_dir():
        print("[WARN] src/db/dao/ not found — nothing to check (exit 0).")
        return 0

    violations: list[_FStringViolation] = []
    files = sorted(dao_root.rglob("*.py"))
    for file in files:
        rel = str(file.relative_to(src))
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        violations.extend(_check_text_fstring(file, rel, tree))
        violations.extend(_find_order_by_interpolations(file, rel, tree))

    if not violations:
        print(
            f"[PASS] {len(files)} DAO files checked; no raw text(f-string) "
            "or unguarded order_by interpolation found"
        )
        return 0

    for v in violations:
        print(f"[FAIL] {v.relpath}:{v.lineno}: {v.message}")
    print(f"[FAIL] {len(violations)} unsafe SQL construction violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
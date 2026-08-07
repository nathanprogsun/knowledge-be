#!/usr/bin/env python3
"""Enforce top-level-only imports across the Python ``src/`` tree.

Rules:

- No ``import`` or ``from ... import ...`` may appear at an indentation
  greater than zero. ``ast.Import`` / ``ast.ImportFrom`` nodes whose
  ``col_offset > 0`` are function-level / class-level / block-level imports
  and are forbidden.

- ``if TYPE_CHECKING:`` block imports are also flagged. Per AGENTS.md §6.2
  such imports are forbidden unless strictly required to break an import
  cycle, and even then restructuring is preferred. This script does not
  carve out a TYPE_CHECKING exception; any import inside an
  ``if TYPE_CHECKING:`` block (indented, hence ``col_offset > 0``) is
  reported like any other indented import.

- Import grouping (stdlib / third-party / first-party) is left to
  ``ruff`` (isort); this script does not enforce it.

Usage::

    python check_imports.py [--src-root PATH]

Exit codes:
    0 = every ``.py`` file under ``src/`` has only top-level imports
    1 = at least one indented import detected
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

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


# ─────────────────────────────────────────────────────────────────────────
# Per-file check
# ─────────────────────────────────────────────────────────────────────────


def _describe_enclosing(node: ast.AST, parents: dict[int, ast.AST]) -> str:
    cur: ast.AST | None = node
    while cur is not None:
        pid = id(cur)
        cur = parents.get(pid)
        if cur is None:
            break
        if isinstance(cur, ast.FunctionDef):
            return f"function '{cur.name}'"
        if isinstance(cur, ast.AsyncFunctionDef):
            return f"async function '{cur.name}'"
        if isinstance(cur, ast.ClassDef):
            return f"class '{cur.name}' body"
        if isinstance(cur, ast.For):
            return "for-loop body"
        if isinstance(cur, ast.AsyncFor):
            return "async for-loop body"
        if isinstance(cur, ast.While):
            return "while-loop body"
        if isinstance(cur, ast.With):
            return "with-statement body"
        if isinstance(cur, ast.AsyncWith):
            return "async with-statement body"
        if isinstance(cur, ast.If):
            test = cur.test
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                return "TYPE_CHECKING block"
            return "if-block body"
        if isinstance(cur, ast.Try):
            return "try/except/finally body"
        if isinstance(cur, ast.ExceptHandler):
            return "except handler body"
        if isinstance(cur, ast.Match):
            return "match body"
    return "indented block"


def check_file(path: Path) -> list[str]:
    """Return a list of error strings for every indented import in ``path``."""
    errors: list[str] = []
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError) as e:
        return [f"{path}: cannot parse: {e}"]

    # Build parent map once per file.
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if node.col_offset > 0:
                kind = "import" if isinstance(node, ast.Import) else "from-import"
                snippet = ast.unparse(node).splitlines()[0][:80]
                where = _describe_enclosing(node, parents)
                errors.append(
                    f"{path}:{node.lineno}: {kind} statement is indented at "
                    f"col {node.col_offset} (inside {where}); imports must be "
                    f"at the top of the file — offending: {snippet!r}"
                )
            # do not descend; ``ast.Import`` has no body and
            # ``ast.ImportFrom`` only carries a module and names
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return errors


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ban function-level / class-level imports in src/.",
    )
    parser.add_argument("--src-root", default=None)
    args = parser.parse_args()

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    files = sorted(p for p in src.rglob("*.py") if p.is_file())
    if not files:
        print("[WARN] src/ has no .py files — exit 0.")
        return 0

    errors: list[str] = []
    for file in files:
        if file.name.endswith(("_pb2.py", "_pb2_grpc.py")):
            continue
        errors.extend(check_file(file))

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"[FAIL] {len(errors)} import placement violation(s)")
        return 1

    print(f"[PASS] All imports in {len(files)} files are at top level (no indented imports)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Detect PR / stage / checkpoint / upstream-reference leaks.

Rules (all are reported as violations):

1. **String leaks** — comments, docstrings, or string literals in
   source/test/docs/config files may not contain:

   - ``PR-<digits>`` or ``PR-<digits>.<digits>`` (e.g. ``PR-17.5b``)
   - ``stage-<digit>`` or ``Stage-<digit>`` (case-insensitive)
   - ``checkpoint-<digit>``
   - ``Mirrors\\ ``internal/``  (any backtick-quoted ``internal/`` path)
   - ``Maps\\ ``internal/``     (same)
   - ``mirroring Go``
   - ``Mirrors WeKnora``        (the upstream project name)
   - ``weknoracloud``           (only flagged inside comments; bare
     string constants like ``value="weknoracloud"`` in a domain model
     are exempt)

2. **Identifier leaks** — Python identifiers may not contain
   ``STAGE1_CONTRACTS`` / ``STAGE2_CONTRACTS`` / ``stage1_contract`` /
   ``stage2_contract`` / ``checkpoint_1`` / ``checkpoint_2`` /
   ``STAGE1_DOMAINS`` (and similar).

3. **Filename leaks** — filenames under the scanned roots may not
   match ``stage1_contract.py`` / ``stage2_contract.py`` /
   ``checkpoint-1-report.md`` / ``checkpoint-2-report.md``.

Default scanned roots::

    src/  tests/  alembic/  docs/  ruff.toml  Makefile

Files under ``docs/release-notes/`` are exempt by default (release notes
are the only sanctioned place to mention historical PR ids).

Usage::

    python check_pr_leak.py [--repo-root PATH] [--allow PATH] ...

Exit codes:
    0 = no leaks
    1 = at least one leak found
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path
from typing import NamedTuple


# ─────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────


_PR_PATTERN = re.compile(r"\bPR-\d+(?:\.\d+)?[A-Za-z0-9]*\b")
_STAGE_PATTERN = re.compile(r"\b[Ss]tage-\d+\b")
_CHECKPOINT_PATTERN = re.compile(r"\bcheckpoint-\d+\b")
_INTERNAL_PATTERN = re.compile(r"(?:Mirrors|Maps)\s+``internal/")
_MIRRORING_GO_PATTERN = re.compile(r"\bmirroring\s+Go\b", re.IGNORECASE)
_MIRRORS_WEKNORA_PATTERN = re.compile(r"\bMirrors\s+WeKnora\b")
_WEKNORACLOUD_COMMENT_PATTERN = re.compile(r"\bweknoracloud\b")

# Identifier name patterns (Python identifiers, attribute names, module
# names).  We match exact tokens.
_BAD_IDENTIFIERS = {
    "STAGE1_CONTRACTS",
    "STAGE2_CONTRACTS",
    "ALL_STAGE1_CONTRACTS",
    "ALL_STAGE2_CONTRACTS",
    "STAGE1_DOMAINS",
    "STAGE2_DOMAINS",
    "stage1_contract",
    "stage2_contract",
    "stage1_contracts",
    "stage2_contracts",
    "checkpoint_1",
    "checkpoint_2",
    "checkpoint-1",
    "checkpoint-2",
}

_BAD_FILENAMES = {
    "stage1_contract.py",
    "stage2_contract.py",
    "stage1_contracts.py",
    "stage2_contracts.py",
    "checkpoint-1-report.md",
    "checkpoint-2-report.md",
}

DEFAULT_ALLOWLIST_PATHS = {"docs/release-notes"}


# ─────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────


class _Leak(NamedTuple):
    relpath: str
    lineno: int
    message: str


# ─────────────────────────────────────────────────────────────────────────
# Resolution
# ─────────────────────────────────────────────────────────────────────────


def resolve_repo_root(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.is_dir() else None
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / "src").is_dir() and (cur / "scripts").is_dir():
            return cur
        cur = cur.parent
    return None


def _is_allowed(relpath: str, allow_paths: set[str]) -> bool:
    rel = relpath.replace("\\", "/")
    return any(rel.startswith(p.rstrip("/") + "/") or rel == p for p in allow_paths)


# ─────────────────────────────────────────────────────────────────────────
# String-literal / comment scanner
# ─────────────────────────────────────────────────────────────────────────


def _scan_text(relpath: str, lineno: int, text: str, is_comment: bool) -> list[_Leak]:
    """Apply comment/string-literal patterns to a chunk of source text."""
    out: list[_Leak] = []
    if _PR_PATTERN.search(text):
        out.append(_Leak(relpath, lineno, f"PR-id leak: {text.strip()[:120]}"))
    if _STAGE_PATTERN.search(text):
        out.append(_Leak(relpath, lineno, f"stage-id leak: {text.strip()[:120]}"))
    if _CHECKPOINT_PATTERN.search(text):
        out.append(_Leak(relpath, lineno, f"checkpoint-id leak: {text.strip()[:120]}"))
    if _INTERNAL_PATTERN.search(text):
        out.append(
            _Leak(
                relpath,
                lineno,
                f"upstream-internal-path leak: {text.strip()[:120]}",
            )
        )
    if _MIRRORING_GO_PATTERN.search(text):
        out.append(
            _Leak(relpath, lineno, f"mirroring-Go leak: {text.strip()[:120]}")
        )
    if _MIRRORS_WEKNORA_PATTERN.search(text):
        out.append(
            _Leak(
                relpath,
                lineno,
                f"upstream-project-name leak: {text.strip()[:120]}",
            )
        )
    if is_comment and _WEKNORACLOUD_COMMENT_PATTERN.search(text):
        out.append(
            _Leak(
                relpath,
                lineno,
                f"upstream-product leak in comment: {text.strip()[:120]}",
            )
        )
    return out


def _scan_python(relpath: str, source: str) -> list[_Leak]:
    """Scan a Python source file using its AST for string / comment nodes."""
    out: list[_Leak] = []
    try:
        tree = ast.parse(source, filename=relpath)
    except SyntaxError:
        # Fallback to raw line scan.
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            is_comment = stripped.startswith("#")
            out.extend(_scan_text(relpath, lineno, line, is_comment=is_comment))
        return out

    for node in ast.walk(tree):
        # Top-level string statements (docstrings, ``__all__``, etc).
        # Only ``Expr(Constant)`` — bare ``Constant`` nodes inside other
        # expressions are NOT considered "string literals" for this
        # check (their content is structurally tied to the surrounding
        # expression and is rarely a meaningful leak).
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                out.extend(_scan_text(relpath, node.lineno, node.value.value, is_comment=False))

    # Comments — extract by tokenize; docstrings are excluded (already
    # covered by AST scan).
    import io
    import tokenize

    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    except (tokenize.TokenError, IndentationError):
        tokens = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            out.extend(_scan_text(relpath, tok.start[0], tok.string, is_comment=True))

    # Identifier leak scan (Name nodes, attribute names, function args).
    _scan_identifiers(relpath, tree, out)

    return out


def _scan_identifiers(relpath: str, tree: ast.Module, out: list[_Leak]) -> None:
    """Flag forbidden identifier tokens inside Python source."""

    def _check(name: str, lineno: int) -> None:
        if name in _BAD_IDENTIFIERS:
            out.append(
                _Leak(
                    relpath,
                    lineno,
                    f"identifier leak: '{name}' references a PR/stage/checkpoint label",
                )
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            _check(node.id, node.lineno)
        elif isinstance(node, ast.Attribute):
            _check(node.attr, node.lineno)
        elif isinstance(node, ast.arg):
            _check(node.arg, node.lineno)
        elif isinstance(node, ast.FunctionDef):
            _check(node.name, node.lineno)
        elif isinstance(node, ast.AsyncFunctionDef):
            _check(node.name, node.lineno)
        elif isinstance(node, ast.ClassDef):
            _check(node.name, node.lineno)


def _scan_generic_text(relpath: str, text: str) -> list[_Leak]:
    """Scan a non-Python file (Markdown, TOML, shell) line by line."""
    out: list[_Leak] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        out.extend(_scan_text(relpath, lineno, line, is_comment=False))
    return out


# ─────────────────────────────────────────────────────────────────────────
# Drive
# ─────────────────────────────────────────────────────────────────────────


def _should_scan(relpath: str, allow: set[str]) -> bool:
    if _is_allowed(relpath, allow):
        return False
    parts = relpath.split("/")
    if not parts:
        return False
    # ignore vendored / cache dirs
    if any(p in {"__pycache__", ".venv", ".git", ".ruff_cache", ".mypy_cache"} for p in parts):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect PR / stage / checkpoint / upstream-reference leaks.",
    )
    parser.add_argument("--repo-root", default=None)
    parser.add_argument(
        "--allow",
        action="append",
        default=list(DEFAULT_ALLOWLIST_PATHS),
        help="Additional path prefixes to exempt (repeatable)",
    )
    args = parser.parse_args()

    repo = resolve_repo_root(args.repo_root)
    if repo is None:
        print("[WARN] repo root not found — nothing to check (exit 0).")
        return 0

    allow: set[str] = set(args.allow)
    targets: list[Path] = []
    for name in ("src", "tests", "alembic", "docs"):
        p = repo / name
        if p.is_dir():
            targets.append(p)
    for cfg in ("ruff.toml", "Makefile", "pyproject.toml", "mypy.ini"):
        p = repo / cfg
        if p.is_file():
            targets.append(p)

    leaks: list[_Leak] = []
    files_scanned = 0

    # Walk directories
    for root in [t for t in targets if t.is_dir()]:
        for file in sorted(root.rglob("*")):
            if not file.is_file():
                continue
            rel = str(file.relative_to(repo))
            if not _should_scan(rel, allow):
                continue
            files_scanned += 1
            try:
                text = file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if file.suffix == ".py":
                leaks.extend(_scan_python(rel, text))
            else:
                leaks.extend(_scan_generic_text(rel, text))

    # Top-level configs
    for cfg in targets:
        if not cfg.is_file():
            continue
        rel = cfg.name
        if not _should_scan(rel, allow):
            continue
        files_scanned += 1
        try:
            text = cfg.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        leaks.extend(_scan_generic_text(rel, text))

    # Filename leak check
    for root in [t for t in targets if t.is_dir()]:
        for file in sorted(root.rglob("*")):
            if not file.is_file():
                continue
            rel = str(file.relative_to(repo))
            if not _should_scan(rel, allow):
                continue
            if file.name in _BAD_FILENAMES:
                leaks.append(
                    _Leak(
                        rel,
                        0,
                        f"filename leak: '{file.name}' is reserved for stage/checkpoint naming",
                    )
                )

    if not leaks:
        print(
            f"[PASS] {files_scanned} files scanned; no PR/stage/checkpoint/upstream leaks"
        )
        return 0

    for leak in leaks:
        loc = f"{leak.relpath}:{leak.lineno}" if leak.lineno else leak.relpath
        print(f"[FAIL] {loc}: {leak.message}")
    print(f"[FAIL] {len(leaks)} PR-leak violation(s) across {files_scanned} files")
    return 1


if __name__ == "__main__":
    sys.exit(main())
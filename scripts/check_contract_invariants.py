#!/usr/bin/env python3
"""Enforce immutability and baseline invariants for ``src/core/contracts/``.

Rules:

1. Frozen models. Every Pydantic ``BaseModel`` (or any class intended as a
   contract DTO) declared in ``src/core/contracts/`` MUST set
   ``model_config = ConfigDict(frozen=True)`` (or the v1 ``{"frozen": True}``
   syntax).

2. No business logic. Method bodies must contain only ``pass`` or
   ``raise NotImplementedError``. Dunders are exempt; everything else is
   checked.

3. Field-signature immutability. A baseline JSON file
   (``docs/migration/baselines/contracts_pr1.json``) records the
   snapshot as::

       {
         "ModuleName": {
           "field_name": "annotation source",
           ...
         },
         ...
       }

   Any missing field, extra field, or annotation difference is reported.

Usage::

    python check_contract_invariants.py [--src-root PATH] [--baseline PATH]

Exit codes:
    0 = contracts pass
    1 = any missing frozen, illegal method, or field drift
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from typing import NamedTuple

# ─────────────────────────────────────────────────────────────────────────
# Resolutions
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


DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[1] / "docs/migration/baselines/contracts_pr1.json"
)


# ─────────────────────────────────────────────────────────────────────────
# Discovery + invariant checks
# ─────────────────────────────────────────────────────────────────────────


class ScannedClass(NamedTuple):
    node: ast.ClassDef
    file: Path


def scan_contract_classes(src_root: Path) -> list[ScannedClass]:
    contracts_dir = src_root / "core" / "contracts"
    if not contracts_dir.is_dir():
        return []
    out: list[ScannedClass] = []
    for file in sorted(contracts_dir.rglob("*.py")):
        if file.name == "__init__.py":
            continue
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                out.append(ScannedClass(node=node, file=file))
    return out


def _is_frozen_class(node: ast.ClassDef) -> bool:
    for stmt in node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        targets = [t for t in stmt.targets if isinstance(t, ast.Name)]
        if not any(t.id == "model_config" for t in targets):
            continue
        if _frozen_value(stmt.value):
            return True
    return False


def _frozen_value(node: ast.AST) -> bool:
    # ConfigDict(frozen=True, ...)
    if isinstance(node, ast.Call):
        for kw in node.keywords:
            if kw.arg == "frozen" and isinstance(kw.value, ast.Constant):
                if kw.value.value is True:
                    return True
        if node.args:
            first = node.args[0]
            if isinstance(first, ast.Constant) and first.value is True:
                return True
    # {"frozen": True}
    if isinstance(node, ast.Dict):
        for k, v in zip(node.keys, node.values):
            if (
                isinstance(k, ast.Constant)
                and k.value == "frozen"
                and isinstance(v, ast.Constant)
                and v.value is True
            ):
                return True
    return False


def _body_is_pass_or_notimpl(body: list[ast.stmt]) -> bool:
    """True iff every statement in ``body`` is ``pass`` or
    ``raise NotImplementedError``."""
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Raise):
            exc = stmt.exc
            if isinstance(exc, ast.Call):
                f = exc.func
                if (isinstance(f, ast.Name) and f.id == "NotImplementedError") or (
                    isinstance(f, ast.Attribute) and f.attr == "NotImplementedError"
                ):
                    continue
        return False
    return True


def illegal_methods(node: ast.ClassDef) -> list[tuple[str, int]]:
    bad: list[tuple[str, int]] = []
    for stmt in node.body:
        if not isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        is_dunder = stmt.name.startswith("__") and stmt.name.endswith("__")
        if is_dunder:
            continue
        if not _body_is_pass_or_notimpl(stmt.body):
            bad.append((stmt.name, stmt.lineno))
    return bad


def _is_classvar_annotation(node: ast.AST) -> bool:
    """True if the annotation is ``ClassVar[...]`` or ``typing.ClassVar[...]``."""
    if isinstance(node, ast.Name) and node.id == "ClassVar":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "ClassVar":
        return True
    if isinstance(node, ast.Subscript):
        return _is_classvar_annotation(node.value)
    return False


_PYDANTIC_INTERNAL_ASSIGN_TARGETS = frozenset(
    {"model_config", "model_fields", "model_computed_fields"}
)


def _is_exempt_assign_target(name: str) -> bool:
    """True if ``name`` is a Pydantic-internal assignment target.

    Covers the literal names in ``_PYDANTIC_INTERNAL_ASSIGN_TARGETS`` plus
    the ``__pydantic_*__`` dunder family. ``frozenset`` does exact matching,
    so the wildcard must be handled separately (the previous frozenset entry
    ``"__pydantic_*__"`` never matched anything).
    """
    if name in _PYDANTIC_INTERNAL_ASSIGN_TARGETS:
        return True
    return name.startswith("__pydantic_") and name.endswith("__")


def collect_field_signatures(
    classes: list[ScannedClass],
) -> dict[str, dict[str, str]]:
    """``{ClassName: {field_name: annotation_source}}``.

    Excludes Pydantic-internal names (``model_config``, etc.) and ``ClassVar``
    annotations, which are not part of the field-shape contract.
    """
    out: dict[str, dict[str, str]] = {}
    for sc in classes:
        fields: dict[str, str] = {}
        for stmt in sc.node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if _is_classvar_annotation(stmt.annotation):
                    continue
                fields[stmt.target.id] = ast.unparse(stmt.annotation)
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if not isinstance(tgt, ast.Name):
                        continue
                    if _is_exempt_assign_target(tgt.id):
                        continue
                    fields[tgt.id] = ast.unparse(stmt.value)
        out[sc.node.name] = fields
    return out


def class_file_map(classes: list[ScannedClass]) -> dict[str, Path]:
    """``{ClassName: source file (relative to src if you like)}``."""
    return {sc.node.name: sc.file for sc in classes}


def relpath(file: Path, src_root: Path) -> str:
    try:
        return str(file.relative_to(src_root))
    except ValueError:
        return str(file)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Enforce contract invariants (frozen, no logic, no drift).",
    )
    parser.add_argument("--src-root", default=None)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Path to baseline JSON (defaults to "
        "docs/migration/baselines/contracts_pr1.json).",
    )
    args = parser.parse_args()

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    classes = scan_contract_classes(src)
    baseline_path = Path(args.baseline).resolve() if args.baseline else DEFAULT_BASELINE
    baseline: dict[str, dict[str, str]] = {}
    baseline_loaded = False
    if baseline_path.is_file():
        baseline_loaded = True
        try:
            loaded = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"[WARN] Baseline {baseline_path} could not be parsed: {e} — skipping drift check"
            )
            loaded = None
        if isinstance(loaded, dict):
            baseline = {
                str(k): {str(fname): str(ftype) for fname, ftype in v.items()}
                for k, v in loaded.items()
                if isinstance(v, dict)
            }

    if not classes and not baseline:
        print("[WARN] No Pydantic models in src/core/contracts/ — exit 0.")
        return 0

    if not baseline and classes and not baseline_loaded:
        print(
            f"[FAIL] No baseline at {baseline_path} — field-drift check "
            f"required; refusing to skip (exit 1)."
        )
        sys.exit(1)

    errors: list[str] = []

    # 1) frozen + 2) illegal methods
    for sc in classes:
        rel = relpath(sc.file, src)
        if not _is_frozen_class(sc.node):
            errors.append(
                f"{rel}:{sc.node.lineno}: contract model '{sc.node.name}' is "
                f"missing `model_config = ConfigDict(frozen=True)`"
            )
        for name, lineno in illegal_methods(sc.node):
            errors.append(
                f"{rel}:{lineno}: contract model '{sc.node.name}' has illegal "
                f"method '{name}' — body must contain only `pass` or "
                f"`raise NotImplementedError`"
            )

    # 3) field-signature drift (only when a baseline is actually present)
    if baseline:
        live = collect_field_signatures(classes)
        file_for = class_file_map(classes)
        for cname in sorted(set(live) | set(baseline)):
            rel = relpath(file_for.get(cname, src / "core" / "contracts"), src)
            live_fields = live.get(cname, {})
            base_fields = baseline.get(cname, {})
            for fname, ftype in sorted(base_fields.items()):
                if fname not in live_fields:
                    errors.append(
                        f"{rel}: contract model '{cname}' is missing baseline "
                        f"field '{fname}: {ftype}' (baseline snapshot)"
                    )
            for fname in sorted(set(live_fields) - set(base_fields)):
                errors.append(
                    f"{rel}: contract model '{cname}' has new field '{fname}: "
                    f"{live_fields[fname]}' not present in the baseline "
                    f"(field additions are forbidden after the baseline)"
                )
            for fname in sorted(set(live_fields) & set(base_fields)):
                if live_fields[fname] != base_fields[fname]:
                    errors.append(
                        f"{rel}: contract model '{cname}' field '{fname}' type "
                        f"changed: '{base_fields[fname]}' -> '{live_fields[fname]}'"
                    )

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"[FAIL] {len(errors)} contract invariant violation(s)")
        return 1

    detail = (
        f"{len(classes)} model(s) frozen, "
        f"{sum(len(c.node.body) for c in classes)} body stmts scanned"
    )
    print(f"[PASS] All contract invariants satisfied ({detail})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

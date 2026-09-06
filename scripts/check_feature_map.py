#!/usr/bin/env python3
"""Fail when the committed feature map drifts from a fresh generation."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_builder() -> ModuleType:
    builder_path = Path(__file__).resolve().parent / "build_feature_map.py"
    spec = importlib.util.spec_from_file_location("build_feature_map", builder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {builder_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args(argv)
    builder = _load_builder()
    root = builder.repo_root_from(args.repo_root)
    dest = builder.map_path(root)
    if not dest.is_file():
        print(f"[FAIL] missing {dest.relative_to(root)}; run scripts/build_feature_map.py --write")
        return 1
    expected = builder.dump_feature_map(builder.build_feature_map(root))
    actual = dest.read_text(encoding="utf-8")
    if actual != expected:
        print(f"[FAIL] {dest.relative_to(root)} is stale; regenerate with:")
        print("  python scripts/build_feature_map.py --repo-root . --write")
        return 1
    print(f"[PASS] {dest.relative_to(root)} matches a fresh generation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Mypy ratchet gate — the error count may only ever go down.

``mypy`` runs strict over ``src/`` + ``tests/`` and currently reports a
large pre-existing error backlog. Fixing it wholesale blocks delivery;
letting it grow silently loses the signal. The ratchet records a
per-file error-count baseline in
``docs/migration/baselines/mypy_baseline.json`` and fails when:

- any file has MORE errors than its baseline entry, or
- a file absent from the baseline reports errors at all (new code must
  be clean), or
- a brand-new error *code* appears anywhere (new failure classes are
  never ratcheted in).

Improvements (fewer errors, cleaned files) are reported and the
baseline SHOULD be refreshed with ``--update`` in the same change.

Machine-readable output: ``--format json`` prints a single JSON object
for agent consumption.

Usage::

    python scripts/check_mypy_baseline.py            # verify
    python scripts/check_mypy_baseline.py --update   # shrink baseline
    python scripts/check_mypy_baseline.py --format json

Exit codes:
    0 = no regression (or baseline updated)
    1 = regression detected / mypy itself failed
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

BASELINE = Path("docs/migration/baselines/mypy_baseline.json")

ERROR_PREFIX = " error: "


def run_mypy() -> list[str]:
    """Run mypy via the project venv; return raw output lines."""
    proc = subprocess.run(
        [sys.executable, "-m", "mypy"],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [
        line
        for line in (proc.stdout + proc.stderr).splitlines()
        if ERROR_PREFIX in line
    ]
    if proc.returncode not in (0, 1):
        print(proc.stdout + proc.stderr, file=sys.stderr)
        raise SystemExit("mypy crashed (exit %d)" % proc.returncode)
    return lines


def parse(lines: list[str]) -> tuple[Counter[str], Counter[str]]:
    """Split concise output into per-file and per-code counts."""
    per_file: Counter[str] = Counter()
    per_code: Counter[str] = Counter()
    for line in lines:
        path = line.split(":", 1)[0]
        per_file[path] += 1
        if "[" in line and line.rstrip().endswith("]"):
            per_code[line.rsplit("[", 1)[1].rstrip("]")] += 1
        else:
            per_code["uncategorized"] += 1
    return per_file, per_code


def load_baseline() -> dict[str, object]:
    if not BASELINE.exists():
        return {"files": {}, "codes": {}}
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="rewrite the baseline from current output")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    per_file, per_code = parse(run_mypy())
    baseline = load_baseline()
    base_files: dict[str, int] = dict(baseline.get("files", {}))  # type: ignore[arg-type]
    base_codes: dict[str, int] = dict(baseline.get("codes", {}))  # type: ignore[arg-type]

    if args.update:
        BASELINE.parent.mkdir(parents=True, exist_ok=True)
        BASELINE.write_text(
            json.dumps(
                {
                    "files": dict(sorted(per_file.items())),
                    "codes": dict(sorted(per_code.items())),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"baseline updated: {sum(per_file.values())} errors across {len(per_file)} files")
        return 0

    regressions: list[str] = []
    improvements: list[str] = []
    for path, count in sorted(per_file.items()):
        base = base_files.get(path, 0)
        if count > base:
            regressions.append(f"{path}: {base} -> {count} (+{count - base})")
        elif count < base:
            improvements.append(f"{path}: {base} -> {count}")
    for path, base in sorted(base_files.items()):
        if path not in per_file:
            improvements.append(f"{path}: {base} -> 0 (clean)")
    new_codes = sorted(set(per_code) - set(base_codes))

    report = {
        "status": "fail" if (regressions or new_codes) else "pass",
        "total_errors": sum(per_file.values()),
        "baseline_errors": sum(base_files.values()),
        "regressions": regressions,
        "new_error_codes": new_codes,
        "improvements": improvements,
    }

    if args.format == "json":
        print(json.dumps(report, indent=2))
    else:
        print(f"mypy errors: {report['total_errors']} (baseline {report['baseline_errors']})")
        for line in regressions:
            print(f"[REGRESSION] {line}")
        for code in new_codes:
            print(f"[REGRESSION] new error code: {code}")
        for line in improvements:
            print(f"[improved] {line}")
        if not regressions and not new_codes:
            print("[PASS] no mypy regression")

    return 1 if (regressions or new_codes) else 0


if __name__ == "__main__":
    sys.exit(main())

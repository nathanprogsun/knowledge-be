#!/usr/bin/env python3
"""Verify the Agent Notes directory structure and file format.

Agent Notes live under ``.agents/notes/{lifecycle}/{class}/yyyy-mm-dd-topic.md``
(see ``.agents/notes/README.md``). This check enforces that scheme:

- path-encoded lifecycle: proposed / implemented / rejected / archived
- closed class set: feature, bug-fix, simplification, architecture,
  process, testing
- every active note starts with ``# Agent Note: `` and carries a
  ``Status:`` line matching its lifecycle folder
- implemented / rejected notes carry a mandatory
  ``## Alternatives considered`` section
- archived notes are frozen: ``Status: archived`` is required and no
  alternatives section is required (frozen records are historical)

Usage::

    python scripts/verify_agent_notes.py [--repo-root PATH]

Exit codes:
    0 = valid
    1 = at least one violation
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

LIFECYCLES = {"proposed", "implemented", "rejected", "archived"}
CLASSES = {
    "feature",
    "bug-fix",
    "simplification",
    "architecture",
    "process",
    "testing",
}
_NON_NOTE_NAMES = {".gitkeep", "README.md", "_template.md"}

_TITLE_RE = re.compile(r"^# Agent Note: ")
_STATUS_RE = re.compile(r"^Status:\s*(\S+)\s*$")
_ALTERNATIVES_RE = re.compile(r"^##\s+Alternatives considered")
_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-")


def resolve_repo_root(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.is_dir() else None
    cur = Path(__file__).resolve().parent
    for _ in range(6):
        if (cur / "scripts").is_dir() and (cur / ".agents").is_dir():
            return cur
        cur = cur.parent
    return None


def _check_note(rel: str, text: str) -> list[str]:
    """Validate one note file; returns violation messages."""
    out: list[str] = []
    lines = text.splitlines()
    if not lines or not _TITLE_RE.match(lines[0]):
        out.append(f"{rel}: first line must be '# Agent Note: <title>'")
    status = ""
    for line in lines:
        m = _STATUS_RE.match(line)
        if m:
            status = m.group(1)
            break
    if not status:
        out.append(f"{rel}: missing 'Status: <lifecycle>' line")
    parts = Path(rel).parts
    lifecycle = parts[2]
    if status and status != lifecycle:
        out.append(
            f"{rel}: Status: '{status}' does not match lifecycle folder '{lifecycle}'"
        )
    if lifecycle in {"implemented", "rejected"} and not any(
        _ALTERNATIVES_RE.match(line) for line in lines
    ):
        out.append(
            f"{rel}: implemented/rejected notes require '## Alternatives considered'"
        )
    if lifecycle == "archived" and status and status != "archived":
        out.append(f"{rel}: archived notes must carry 'Status: archived'")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify the Agent Notes directory scheme and format.",
    )
    parser.add_argument("--repo-root", default=None)
    args = parser.parse_args()

    repo = resolve_repo_root(args.repo_root)
    if repo is None:
        print("[WARN] repo root not found — nothing to check (exit 0).")
        return 0

    notes_root = repo / ".agents" / "notes"
    violations: list[str] = []
    files_scanned = 0

    if not notes_root.is_dir():
        print("[PASS] .agents/notes/ does not exist — nothing to check.")
        return 0

    for file in sorted(notes_root.rglob("*")):
        if not file.is_file():
            continue
        rel = str(file.relative_to(repo))
        parts = Path(rel).parts
        # Expect: .agents/notes/<lifecycle>/<class>/<file>
        if len(parts) != 5 or parts[:2] != (".agents", "notes"):
            continue  # not a note file (e.g. README.md at notes/ root)
        lifecycle, cls, fname = parts[2], parts[3], parts[4]
        if fname in _NON_NOTE_NAMES or fname.startswith("."):
            continue
        files_scanned += 1
        if lifecycle not in LIFECYCLES:
            violations.append(
                f"{rel}: unknown lifecycle '{lifecycle}' (expected one of "
                + ", ".join(sorted(LIFECYCLES))
                + ")"
            )
            continue
        if cls not in CLASSES:
            violations.append(
                f"{rel}: unknown class '{cls}' (expected one of "
                + ", ".join(sorted(CLASSES))
                + ")"
            )
            continue
        if not _DATE_PREFIX_RE.match(fname):
            violations.append(
                f"{rel}: filename must start with yyyy-mm-dd-"
            )
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            violations.append(f"{rel}: unreadable as UTF-8")
            continue
        violations.extend(_check_note(rel, text))

    if not violations:
        print(
            f"[PASS] {files_scanned} agent note(s) verified; "
            "scheme and format valid"
        )
        return 0

    for msg in violations:
        print(f"[FAIL] {msg}")
    print(f"[FAIL] {len(violations)} agent-note violation(s)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Search Agent Notes by tags and reverse-lookup by file path.

Each note under ``.agents/notes/implemented/`` may carry advisory
metadata lines (anywhere in the header block):

- ``Tags: a, b, c`` — short classification tokens
- ``Related files: path1, path2`` — repository paths the note discusses

``search_notes.py --file PATH`` matches these so an agent editing a
file can pull every relevant decision without guessing keywords. Notes
without metadata are still indexed by title + path.

The script never rewrites notes; it only reads.

Usage::

    python scripts/search_notes.py --tag integration
    python scripts/search_notes.py --file src/web/api/auth/router.py
    python scripts/search_notes.py --query "tenant fallback"
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(".agents/notes")
TAG_RE = re.compile(r"^Tags:\s*(.+?)\s*$", re.MULTILINE)
RELATED_RE = re.compile(r"^Related files?:\s*(.+?)\s*$", re.MULTILINE)


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_metadata(text: str) -> tuple[list[str], list[str]]:
    """Extract ``Tags:`` and ``Related files:`` lines (case-insensitive)."""
    tags = _split_csv(TAG_RE.search(text).group(1)) if TAG_RE.search(text) else []
    rel_match = RELATED_RE.search(text)
    related: list[str] = []
    if rel_match:
        seen: set[str] = set()
        for item in _split_csv(rel_match.group(1)):
            if item not in seen:
                seen.add(item)
                related.append(item)
    return tags, related


def _load_notes() -> list[dict[str, object]]:
    notes: list[dict[str, object]] = []
    if not ROOT.exists():
        return notes
    for path in sorted(ROOT.rglob("*.md")):
        rel = str(path.relative_to(ROOT))
        text = path.read_text(encoding="utf-8")
        tags, related = _parse_metadata(text)
        title = ""
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                title = stripped[2:].strip()
                break
        notes.append(
            {
                "path": str(path),
                "rel": rel,
                "title": title,
                "tags": tags,
                "related_files": related,
            }
        )
    return notes


def _by_tag(notes: list[dict[str, object]], tag: str) -> list[dict[str, object]]:
    return [n for n in notes if tag in (n["tags"] or [])]


def _by_file(notes: list[dict[str, object]], target: str) -> list[dict[str, object]]:
    needle = target.lstrip("./")
    out: list[dict[str, object]] = []
    for n in notes:
        for rel in n["related_files"]:
            rel_str = str(rel).lstrip("./")
            if rel_str == needle or needle.endswith(rel_str) or rel_str.endswith(needle):
                out.append(n)
                break
    return out


def _by_query(notes: list[dict[str, object]], query: str) -> list[dict[str, object]]:
    q = query.lower()
    return [n for n in notes if q in (n["title"] + " " + str(n["rel"])).lower()]


def _print(matches: list[dict[str, object]], fmt: str) -> None:
    if fmt == "json":
        import json

        print(json.dumps([{"path": m["path"], "title": m["title"]} for m in matches], indent=2))
        return
    if not matches:
        print("[no matches]")
        return
    for m in matches:
        print(f"- {m['rel']}: {m['title']}")
        if m["tags"]:
            print(f"    tags: {', '.join(str(t) for t in m['tags'])}")
        if m["related_files"]:
            print(f"    files: {', '.join(str(f) for f in m['related_files'])}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="filter by tag")
    parser.add_argument("--file", help="reverse-lookup notes whose related_files match this path")
    parser.add_argument("--query", help="substring match on title + path")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    parser.add_argument("--repo-root", default=".", help="repo root (notes live at <root>/.agents/notes)")
    args = parser.parse_args()

    global ROOT
    ROOT = Path(args.repo_root) / ".agents/notes"

    notes = _load_notes()
    by_tag: dict[str, list[dict[str, object]]] = defaultdict(list)
    reverse: dict[str, list[dict[str, object]]] = defaultdict(list)
    for n in notes:
        for t in n["tags"]:
            by_tag[str(t)].append(n)
        for f in n["related_files"]:
            reverse[str(f)].append(n)

    if args.tag:
        _print(_by_tag(notes, args.tag), args.format)
    elif args.file:
        _print(_by_file(notes, args.file), args.format)
    elif args.query:
        _print(_by_query(notes, args.query), args.format)
    else:
        print(f"loaded {len(notes)} notes from {ROOT}")
        tag_counts = sorted(((t, len(v)) for t, v in by_tag.items()), key=lambda kv: (-kv[1], kv[0]))
        for tag, count in tag_counts[:20]:
            print(f"  tag[{tag}]: {count}")
        file_counts = sorted(((f, len(v)) for f, v in reverse.items()), key=lambda kv: (-kv[1], kv[0]))
        if file_counts:
            print("  most-referenced files:")
            for f, count in file_counts[:10]:
                print(f"    {f}: {count}")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
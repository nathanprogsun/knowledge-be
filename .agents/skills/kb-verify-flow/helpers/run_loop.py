#!/usr/bin/env python3
"""Launch → Doctor → Drive → Evidence → Cleanup for kb-verify-flow."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from http_probe import get_json_or_text

REQUIRED_HEADINGS = (
    "Sub-features",
    "How to get to it",
    "Driving it with",
    "Gotchas",
)
HEADING_RE = re.compile(r"^## (Sub-features|How to get to it|Driving it with|Gotchas)\s*$", re.M)


def skill_root() -> Path:
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    return skill_root().parents[2]


def evidence_dir() -> Path:
    path = skill_root() / "evidence"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(name: str, payload: dict[str, object]) -> Path:
    dest = evidence_dir() / name
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dest


def api_base() -> str:
    return os.environ.get("KB_API_BASE", "http://127.0.0.1:8000").rstrip("/")


def web_base() -> str:
    return os.environ.get("KB_WEB_BASE", "http://127.0.0.1:5173").rstrip("/")


def cmd_launch() -> dict[str, object]:
    api = get_json_or_text(f"{api_base()}/api/v1/auth/config")
    web = get_json_or_text(f"{web_base()}/login")
    payload: dict[str, object] = {
        "step": "launch",
        "at": datetime.now(timezone.utc).isoformat(),
        "started_servers": False,
        "api": api,
        "web": web,
    }
    tmp = evidence_dir() / "tmp"
    tmp.mkdir(exist_ok=True)
    (tmp / "scratch.txt").write_text("ephemeral\n", encoding="utf-8")
    write_json("launch.json", payload)
    return payload


def _feature_heading_report() -> list[dict[str, object]]:
    features = skill_root() / "features"
    rows: list[dict[str, object]] = []
    for path in sorted(features.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8")
        found = set(HEADING_RE.findall(text))
        rows.append(
            {
                "file": str(path.relative_to(repo_root())),
                "ok": set(REQUIRED_HEADINGS) <= found,
                "missing": sorted(set(REQUIRED_HEADINGS) - found),
            }
        )
    return rows


def cmd_doctor() -> dict[str, object]:
    root = repo_root()
    payload: dict[str, object] = {
        "step": "doctor",
        "at": datetime.now(timezone.utc).isoformat(),
        "env_file": (root / ".env").is_file() or (root / ".env.example").is_file(),
        "frontend_node_modules": (root / "frontend" / "node_modules").is_dir(),
        "feature_headings": _feature_heading_report(),
        "api_auth_config": get_json_or_text(f"{api_base()}/api/v1/auth/config"),
        "web_login": get_json_or_text(f"{web_base()}/login"),
    }
    write_json("doctor.json", payload)
    return payload


def cmd_drive() -> dict[str, object]:
    token = os.environ.get("KB_VERIFY_TOKEN", "").strip()
    login_probe = get_json_or_text(f"{web_base()}/login")
    auth_config = get_json_or_text(f"{api_base()}/api/v1/auth/config")
    kb_probe: dict[str, object]
    if token:
        kb_probe = get_json_or_text(
            f"{api_base()}/api/v1/knowledge-bases",
            headers={"Authorization": f"Bearer {token}"},
        )
    else:
        kb_probe = {
            "ok": False,
            "skipped": True,
            "reason": "KB_VERIFY_TOKEN unset; authenticated list not driven",
        }
    payload: dict[str, object] = {
        "step": "drive",
        "at": datetime.now(timezone.utc).isoformat(),
        "features": {
            "login": {"probes": {"web_login": login_probe, "auth_config": auth_config}},
            "knowledge-base-list": {"probes": {"knowledge_bases": kb_probe}},
        },
    }
    write_json("drive.json", payload)
    return payload


def cmd_evidence() -> dict[str, object]:
    ev = evidence_dir()
    launch = _read_json(ev / "launch.json")
    doctor = _read_json(ev / "doctor.json")
    drive = _read_json(ev / "drive.json")
    lines = [
        "# kb-verify-flow last run",
        "",
        f"At: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Launch",
        "",
        f"- started_servers: {launch.get('started_servers')}",
        f"- api reachable: {_probe_ok(launch.get('api'))}",
        f"- web reachable: {_probe_ok(launch.get('web'))}",
        "",
        "## Doctor",
        "",
        f"- env file present: {doctor.get('env_file')}",
        f"- frontend/node_modules: {doctor.get('frontend_node_modules')}",
        f"- feature headings: {doctor.get('feature_headings')}",
        "",
        "## Drive",
        "",
        f"- login probes: {drive.get('features', {})}",
        "",
        "## Cleanup",
        "",
        "Deletes `evidence/tmp/` only. This file stays.",
        "",
    ]
    dest = ev / "last-run.md"
    dest.write_text("\n".join(lines), encoding="utf-8")
    return {"step": "evidence", "path": str(dest.relative_to(repo_root()))}


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _probe_ok(value: object) -> bool:
    return isinstance(value, dict) and value.get("ok") is True


def cmd_cleanup() -> dict[str, object]:
    tmp = evidence_dir() / "tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    kept = sorted(p.name for p in evidence_dir().iterdir() if p.is_file())
    return {"step": "cleanup", "removed": "evidence/tmp", "kept": kept}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "step",
        choices=("launch", "doctor", "drive", "evidence", "cleanup", "all"),
    )
    args = parser.parse_args()
    steps = {
        "launch": cmd_launch,
        "doctor": cmd_doctor,
        "drive": cmd_drive,
        "evidence": cmd_evidence,
        "cleanup": cmd_cleanup,
    }
    if args.step == "all":
        last_cleanup: dict[str, object] = {}
        for name in ("launch", "doctor", "drive", "evidence", "cleanup"):
            result = steps[name]()
            if name == "cleanup":
                last_cleanup = result
            print(json.dumps({"step": name, "result": result}, default=str)[:1000])
        kept = last_cleanup.get("kept", [])
        if "last-run.md" not in kept:
            print("cleanup dropped last-run.md", file=sys.stderr)
            return 1
        return 0
    print(json.dumps(steps[args.step](), indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

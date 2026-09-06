#!/usr/bin/env python3
"""Build a static feature map from routers, factories, and worker tasks.

The map is agent navigation, not a runtime registry. It is regenerated
from source so a new endpoint or ``build_*_service`` cannot land
without a matching map update (see ``check_feature_map.py``).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import TypedDict

API_V1_PREFIX = "/api/v1"
HTTP_METHODS = frozenset({"get", "post", "put", "patch", "delete", "options", "head"})

# HTTP surfaces whose factory lives under ``core/system`` still belong
# to the favorites product domain on the map.
ENDPOINT_DOMAIN_ALIASES: dict[str, str] = {
    "favorites": "favorites",
}


class EndpointEntry(TypedDict):
    method: str
    path: str
    file: str
    domain: str


class ServiceEntry(TypedDict):
    name: str
    file: str
    domain: str


class TaskEntry(TypedDict):
    name: str
    file: str


class FeatureMap(TypedDict):
    generated_by: str
    endpoints: list[EndpointEntry]
    services: list[ServiceEntry]
    tasks: list[TaskEntry]


def repo_root_from(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return Path(__file__).resolve().parent.parent


def _rel(repo_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(repo_root)).replace("\\", "/")


def _str_const(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _join_path(prefix: str, path: str) -> str:
    if not path or path == "/":
        return prefix or "/"
    if not prefix:
        return path if path.startswith("/") else f"/{path}"
    return f"{prefix.rstrip('/')}/{path.lstrip('/')}"


def _qualified_http_path(router_prefix: str, route_path: str) -> str:
    combined = _join_path(router_prefix, route_path)
    if combined == "/files" or combined.startswith("/files/"):
        return combined
    if combined.startswith(API_V1_PREFIX):
        return combined
    return _join_path(API_V1_PREFIX, combined)


def _endpoint_domain(rel_file: str) -> str:
    parts = Path(rel_file).parts
    try:
        api_idx = parts.index("api")
    except ValueError:
        return "unknown"
    rest = parts[api_idx + 1 :]
    if not rest:
        return "unknown"
    if rest[0] == "infra" and len(rest) > 1:
        return rest[1]
    domain = rest[0]
    return ENDPOINT_DOMAIN_ALIASES.get(domain, domain)


def _service_domain(rel_file: str) -> str:
    parts = Path(rel_file).parts
    try:
        core_idx = parts.index("core")
    except ValueError:
        return "unknown"
    rest = parts[core_idx + 1 :]
    if not rest:
        return "unknown"
    if rest[0] == "infra" and len(rest) > 1:
        return rest[1]
    if rest[0] == "system":
        return "system+favorites"
    return rest[0]


def _router_prefixes(tree: ast.Module) -> dict[str, str]:
    prefixes: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        is_router = (isinstance(func, ast.Name) and func.id == "APIRouter") or (
            isinstance(func, ast.Attribute) and func.attr == "APIRouter"
        )
        if not is_router:
            continue
        prefix = ""
        for kw in node.value.keywords:
            if kw.arg == "prefix":
                value = _str_const(kw.value)
                if value is not None:
                    prefix = value
        for target in node.targets:
            if isinstance(target, ast.Name):
                prefixes[target.id] = prefix
    return prefixes


def collect_endpoints(repo_root: Path) -> list[EndpointEntry]:
    entries: list[EndpointEntry] = []
    api_root = repo_root / "src" / "web" / "api"
    for file in sorted(api_root.rglob("router.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        prefixes = _router_prefixes(tree)
        rel = _rel(repo_root, file)
        domain = _endpoint_domain(rel)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            method = node.func.attr.lower()
            if method not in HTTP_METHODS:
                continue
            if not isinstance(node.func.value, ast.Name):
                continue
            router_name = node.func.value.id
            if router_name not in prefixes or not node.args:
                continue
            route_path = _str_const(node.args[0])
            if route_path is None:
                continue
            entries.append(
                {
                    "method": method.upper(),
                    "path": _qualified_http_path(prefixes[router_name], route_path),
                    "file": rel,
                    "domain": domain,
                }
            )
    entries.sort(key=lambda item: (item["path"], item["method"], item["file"]))
    return entries


def collect_services(repo_root: Path) -> list[ServiceEntry]:
    entries: list[ServiceEntry] = []
    core_root = repo_root / "src" / "core"
    for file in sorted(core_root.rglob("factory.py")):
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        rel = _rel(repo_root, file)
        domain = _service_domain(rel)
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if not node.name.startswith("build_") or not node.name.endswith("_service"):
                continue
            entries.append({"name": node.name, "file": rel, "domain": domain})
    entries.sort(key=lambda item: (item["name"], item["file"]))
    return entries


def _module_str_constants(tree: ast.Module) -> dict[str, str]:
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        value = _str_const(node.value)
        if value is not None:
            constants[target.id] = value
    return constants


def _task_name(call: ast.Call, constants: dict[str, str]) -> str | None:
    if not call.args:
        return None
    arg0 = call.args[0]
    literal = _str_const(arg0)
    if literal is not None:
        return literal
    if isinstance(arg0, ast.Name):
        return constants.get(arg0.id)
    return None


def collect_tasks(repo_root: Path) -> list[TaskEntry]:
    entries: list[TaskEntry] = []
    tasks_root = repo_root / "src" / "workers" / "tasks"
    for file in sorted(tasks_root.glob("*.py")):
        if file.name == "__init__.py":
            continue
        tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        constants = _module_str_constants(tree)
        rel = _rel(repo_root, file)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            is_register = (isinstance(func, ast.Name) and func.id == "register_task") or (
                isinstance(func, ast.Attribute) and func.attr == "register_task"
            )
            if not is_register:
                continue
            name = _task_name(node, constants)
            if name is None:
                continue
            entries.append({"name": name, "file": rel})
    entries.sort(key=lambda item: (item["name"], item["file"]))
    return entries


def build_feature_map(repo_root: Path) -> FeatureMap:
    return {
        "generated_by": "scripts/build_feature_map.py",
        "endpoints": collect_endpoints(repo_root),
        "services": collect_services(repo_root),
        "tasks": collect_tasks(repo_root),
    }


def map_path(repo_root: Path) -> Path:
    return repo_root / ".agents" / "feature-map" / "generated.json"


def dump_feature_map(feature_map: FeatureMap) -> str:
    return json.dumps(feature_map, indent=2, sort_keys=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    root = repo_root_from(args.repo_root)
    feature_map = build_feature_map(root)
    rendered = dump_feature_map(feature_map)
    if args.write:
        dest = map_path(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered, encoding="utf-8")
        print(f"wrote {dest.relative_to(root)}")
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

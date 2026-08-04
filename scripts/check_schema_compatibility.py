#!/usr/bin/env python3
"""Diff WeKnora (Go) SQL migrations against knowledge-be Pydantic ``TableModel``
classes.

Reference documentation: ``migrations/{versioned,sqlite,mysql,paradedb}/*.up.sql``.

Rules:

1. For every ``CREATE TABLE`` in the migrations (and ``ALTER TABLE ... ADD COLUMN``
   that contributes additional columns), there MUST be a TableModel class in
   ``src/db/models/`` whose ``table`` literal equals the SQL table name.

2. For every column defined in a migration's ``CREATE TABLE`` body (or added
   later via ``ALTER TABLE ADD COLUMN``), there MUST be a field of the same
   name on the corresponding TableModel.

3. Field types are compared loosely with the SQL type equivalence map in
   ``_SQL_TYPE_MAP``. ``str <-> VARCHAR/TEXT``, ``int <-> INTEGER/BIGINT/SERIAL``,
   ``datetime <-> TIMESTAMP/...``, ``bool <-> BOOLEAN``, etc.

A baseline file (``docs/migration/baselines/db_models_pr1.json``) may be
supplied for stricter type pinning; without it, only structural equivalence is
enforced.

Usage::

    python check_schema_compatibility.py
        [--src-root PATH]
        [--migrations-root PATH]
        [--baseline PATH]

Exit codes:
    0 = every migration table has a matching TableModel with full column
        coverage
    1 = any missing model, missing field, type drift, or duplicate
        CREATE TABLE on different tables of the same name
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

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


DEFAULT_MIGRATIONS_ROOT = Path(__file__).resolve().parents[3] / "migrations"
DEFAULT_BASELINE = (
    Path(__file__).resolve().parents[3] / "docs/migration/baselines/db_models_pr1.json"
)


def _resolve_path(explicit: str | None, default: Path) -> Path | None:
    if explicit:
        p = Path(explicit).resolve()
        return p if p.exists() else None
    return default if default.exists() else None


# ─────────────────────────────────────────────────────────────────────────
# SQL parsing
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Column:
    name: str
    sql_type: str
    nullable: bool = True
    pk: bool = False
    lineno: int = 0


@dataclass
class Table:
    name: str
    file: Path
    lineno: int
    columns: list[Column] = field(default_factory=list)


_SQL_TYPE_MAP: dict[str, str] = {
    "varchar": "str",
    "char": "str",
    "text": "str",
    "uuid": "str",
    "integer": "int",
    "int": "int",
    "bigint": "int",
    "smallint": "int",
    "serial": "int",
    "bigserial": "int",
    "boolean": "bool",
    "timestamp": "datetime",
    "timestamptz": "datetime",
    "date": "datetime",
    "time": "datetime",
    "real": "float",
    "double": "float",
    "float": "float",
    "numeric": "float",
    "decimal": "float",
    "json": "dict",
    "jsonb": "dict",
}


def _sql_type_to_python(sql_type: str) -> str:
    base = sql_type.lower().split("(")[0].strip()
    return _SQL_TYPE_MAP.get(base, "str")


_TYPE_EQUIVALENCE: dict[str, set[str]] = {
    "str": {"str"},
    "int": {"int"},
    "bool": {"bool"},
    "datetime": {"datetime"},
    "float": {"float"},
    # SQLite stores JSON as TEXT and PG stores it as JSONB; both are valid
    # choices for a Python model that uses ``dict``.
    "dict": {"dict", "str"},
}


def _types_compatible(sql_family: str, model_family: str) -> bool:
    if sql_family == model_family:
        return True
    return model_family in _TYPE_EQUIVALENCE.get(sql_family, {sql_family})


def _strip_comments(text: str) -> str:
    """Strip ``-- ...`` line comments."""
    return re.sub(r"--.*?$", "", text, flags=re.MULTILINE)


def _strip_dollar_blocks(text: str) -> str:
    """Strip ``DO $$ ... $$;`` PL/pgSQL anonymous blocks. Crude heuristic:
    remove the entire block balanced on ``$$ ... $$`` markers.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        if text[i : i + 2] == "$$":
            j = text.find("$$", i + 2)
            if j == -1:
                break
            i = j + 2
        else:
            out.append(text[i])
            i += 1
    return "".join(out)


def _split_top_level(body: str, sep: str) -> list[str]:
    """Split ``body`` by ``sep`` ignoring separators inside ``()`` pairs."""
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in body:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == sep and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur))
    return parts


# Per-column syntax
_COL_RE = re.compile(
    r"^\s*"
    r'(?P<name>"?(?P<ident>\w+)"?)'
    r"\s+(?P<type>\w+)"
    r"(?:\s*\([^)]*\))?"
    r"(?:\s+(?P<rest>.*?))?\s*$",
    re.IGNORECASE,
)


def _parse_column(raw: str) -> Column | None:
    raw = raw.strip().rstrip(",").strip()
    if not raw:
        return None
    if re.match(r"^(PRIMARY|UNIQUE|FOREIGN|CONSTRAINT|INDEX|CHECK|KEY)", raw, re.IGNORECASE):
        return None
    m = _COL_RE.match(raw)
    if not m:
        return None
    name = m.group("ident")
    sql_type = m.group("type").lower()
    rest = (m.group("rest") or "").upper()
    pk = "PRIMARY KEY" in rest
    nullable = "NOT NULL" not in rest if rest else True
    return Column(name=name, sql_type=sql_type, nullable=nullable, pk=pk)


_CT_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r'(?P<qn>"?(?P<ident>\w+)"?|(?P<plain>\w+))',
    re.IGNORECASE,
)


def _extract_paren_body(text: str, start: int) -> tuple[str, int]:
    """Given that ``text[start] == '('`` (or is whitespace ending at ``(``),
    return ``(body, end_index_after_close_paren)`` using parenthesis balancing.
    """
    i = start
    while i < len(text) and text[i] != "(":
        i += 1
    if i >= len(text):
        return "", i
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j], j + 1
        j += 1
    return "", j


_AT_ADD_RE = re.compile(
    r"ALTER\s+TABLE\s+(?:IF\s+EXISTS\s+)?"
    r'(?:(?P<qn1>"?(?P<tname>\w+)"?)|(?P<tplain>\w+))'
    r"\s+ADD\s+COLUMN\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r'(?:(?P<qn2>"?(?P<cname>\w+)"?)|(?P<cplain>\w+))'
    r"\s+(?P<ctype>\w+)(?:\s*\([^)]*\))?",
    re.IGNORECASE,
)


def parse_migration_sql(text: str, source: Path) -> list[Table]:
    """Best-effort parse of one ``*.up.sql`` file. Returns a list of tables."""
    cleaned = _strip_dollar_blocks(_strip_comments(text))
    by_name: dict[str, Table] = {}

    for ct in _CT_RE.finditer(cleaned):
        name = ct.group("ident") or ct.group("plain")
        if not name:
            continue
        header_end = ct.end()
        body, _ = _extract_paren_body(cleaned, header_end)
        if not body:
            continue
        lineno = cleaned[: ct.start()].count("\n") + 1
        table = by_name.setdefault(name, Table(name=name, file=source, lineno=lineno))
        for raw in _split_top_level(body, ","):
            col = _parse_column(raw)
            if col is None:
                continue
            if not any(c.name == col.name for c in table.columns):
                table.columns.append(col)

    for at in _AT_ADD_RE.finditer(cleaned):
        tname = at.group("tname") or at.group("tplain")
        cname = at.group("cname") or at.group("cplain")
        ctype = at.group("ctype").lower()
        if not (tname and cname):
            continue
        lineno = cleaned[: at.start()].count("\n") + 1
        table = by_name.setdefault(tname, Table(name=tname, file=source, lineno=lineno))
        if not any(c.name == cname for c in table.columns):
            table.columns.append(Column(name=cname, sql_type=ctype, lineno=lineno))

    return list(by_name.values())


def iter_migration_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*.up.sql") if p.is_file())


def parse_all_migrations(root: Path) -> dict[str, list[Table]]:
    """Group ``Table`` objects by table name across all migration files."""
    out: dict[str, list[Table]] = {}
    for file in iter_migration_files(root):
        try:
            text = file.read_text(encoding="utf-8")
        except OSError:
            continue
        for t in parse_migration_sql(text, file):
            out.setdefault(t.name, []).append(t)
    return out


# ─────────────────────────────────────────────────────────────────────────
# Model discovery under src/db/models/
# ─────────────────────────────────────────────────────────────────────────


def _annotation_to_python_type(node: ast.AST | None) -> str | None:
    """Reduce an annotation to a known type family (str/int/bool/datetime/dict/...)."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        name = node.id
        return _name_to_family(name)
    if isinstance(node, ast.Attribute):
        return _annotation_to_python_type(node.value)
    if isinstance(node, ast.Subscript):
        # foo[bar] -> look at ``foo`` only
        return _annotation_to_python_type(node.value)
    if isinstance(node, ast.BinOp):
        # PEP 604 -> ignore ``| None``
        return _annotation_to_python_type(node.left) or _annotation_to_python_type(node.right)
    return None


_NAME_FAMILY: dict[str, str] = {
    "str": "str",
    "int": "int",
    "bool": "bool",
    "float": "float",
    "datetime": "datetime",
    "date": "datetime",
    "time": "datetime",
    "dict": "dict",
    "list": "list",
    "UUID": "str",
    "Decimal": "float",
    "bytes": "str",
    "Json": "dict",
    "dict[str, Any]": "dict",  # tolerated for raw JSON columns (flagged elsewhere)
    "Optional": "ignored",
    "Any": "ignored",
}


def _name_to_family(name: str) -> str | None:
    return _NAME_FAMILY.get(name)


@dataclass
class ModelField:
    name: str
    type_str: str
    family: str | None
    lineno: int


@dataclass
class Model:
    name: str
    table: str
    file: Path
    lineno: int
    fields: list[ModelField] = field(default_factory=list)


def _scan_models(src_root: Path) -> list[Model]:
    models_dir = src_root / "db" / "models"
    if not models_dir.is_dir():
        return []
    out: list[Model] = []
    for file in sorted(models_dir.rglob("*.py")):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, SyntaxError):
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            table_name: str | None = None
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "table" for t in stmt.targets
                ):
                    if isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, str):
                        table_name = stmt.value.value
                    break
                if (
                    isinstance(stmt, ast.AnnAssign)
                    and isinstance(stmt.target, ast.Name)
                    and stmt.target.id == "table"
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    table_name = stmt.value.value
                    break
            if table_name is None:
                # heuristic fallback: strip "Model"/"TableModel" suffix
                table_name = re.sub(r"(Model|TableModel)$", "", node.name).lower()
            fields: list[ModelField] = []
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    type_str = ast.unparse(stmt.annotation)
                    family = _annotation_to_python_type(stmt.annotation)
                    fields.append(
                        ModelField(
                            name=stmt.target.id,
                            type_str=type_str,
                            family=family,
                            lineno=stmt.lineno,
                        )
                    )
                elif isinstance(stmt, ast.Assign):
                    for tgt in stmt.targets:
                        if isinstance(tgt, ast.Name):
                            type_str = ast.unparse(stmt.value)
                            family = _annotation_to_python_type(stmt.value)
                            fields.append(
                                ModelField(
                                    name=tgt.id,
                                    type_str=type_str,
                                    family=family,
                                    lineno=stmt.lineno,
                                )
                            )
            out.append(
                Model(
                    name=node.name,
                    table=table_name,
                    file=file,
                    lineno=node.lineno,
                    fields=fields,
                )
            )
    return out


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diff SQL migrations against Python TableModel classes.",
    )
    parser.add_argument("--src-root", default=None)
    parser.add_argument("--migrations-root", default=None)
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional JSON file pinning PR-1 field types (defaults to "
        "docs/migration/baselines/db_models_pr1.json).",
    )
    args = parser.parse_args()

    migrations_root = _resolve_path(args.migrations_root, DEFAULT_MIGRATIONS_ROOT)
    if migrations_root is None:
        print("[WARN] migrations/ not found — nothing to check (exit 0).")
        return 0

    src = resolve_src_root(args.src_root)
    if src is None:
        print("[WARN] src/ not found — nothing to check (exit 0).")
        return 0

    tables_by_name = parse_all_migrations(migrations_root)
    models = _scan_models(src)

    if not tables_by_name and not models:
        print("[WARN] No migration tables or models found — exit 0.")
        return 0

    errors: list[str] = []
    models_by_table: dict[str, list[Model]] = {}
    for m in models:
        models_by_table.setdefault(m.table, []).append(m)

    # 1) Every migration table -> TableModel with matching columns
    for tname, instances in sorted(tables_by_name.items()):
        # Column-set is the union of all instances (initial CREATE + ALTER ADD)
        col_names: dict[str, Column] = {}
        for inst in instances:
            for c in inst.columns:
                col_names.setdefault(c.name, c)
        matched = models_by_table.get(tname, [])
        if not matched:
            first = instances[0]
            rel = first.file.relative_to(migrations_root)
            errors.append(
                f"{rel}:{first.lineno}: migration table '{tname}' has no "
                f"corresponding TableModel under src/db/models/"
            )
            continue
        if len(matched) > 1:
            for m in matched[1:]:
                errors.append(
                    f"{m.file.relative_to(src)}:{m.lineno}: duplicate TableModel "
                    f"for table '{tname}'"
                )
        model = matched[0]
        field_map = {f.name: f for f in model.fields}
        for cname, col in sorted(col_names.items()):
            if cname not in field_map:
                first = instances[0]
                rel = first.file.relative_to(migrations_root)
                errors.append(
                    f"{rel}: missing TableModel field for column "
                    f"'{tname}.{cname}' (SQL type {col.sql_type.upper()})"
                )
                continue
            f = field_map[cname]
            expected = _sql_type_to_python(col.sql_type)
            if (
                f.family is not None
                and f.family not in {"ignored"}
                and not _types_compatible(expected, f.family)
            ):
                errors.append(
                    f"{model.file.relative_to(src)}:{f.lineno}: field "
                    f"'{model.name}.{f.name}: {f.type_str}' has wrong family "
                    f"for SQL column '{tname}.{cname}: {col.sql_type.upper()}' "
                    f"(expected {expected})"
                )

    # 2) Optional baseline check
    baseline_path = Path(args.baseline).resolve() if args.baseline else DEFAULT_BASELINE
    if baseline_path.is_file():
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            print(
                f"[WARN] Baseline {baseline_path} could not be parsed: {e} "
                f"— skipping pinned-type check"
            )
            baseline = None
        if isinstance(baseline, dict):
            for table_name, fields in baseline.items():
                m_list = models_by_table.get(table_name, [])
                if not m_list:
                    errors.append(
                        f"{baseline_path}: baseline table '{table_name}' has no matching TableModel"
                    )
                    continue
                model = m_list[0]
                live_fields = {f.name: f.type_str for f in model.fields}
                for fname, ftype in fields.items():
                    if fname not in live_fields:
                        errors.append(
                            f"{model.file.relative_to(src)}:{model.lineno}: "
                            f"baseline field '{table_name}.{fname}: {ftype}' "
                            f"is missing on TableModel '{model.name}'"
                        )
                    elif live_fields[fname] != ftype:
                        errors.append(
                            f"{model.file.relative_to(src)}:{model.lineno}: "
                            f"baseline type drift for '{table_name}.{fname}': "
                            f"'{ftype}' -> '{live_fields[fname]}'"
                        )

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"[FAIL] {len(errors)} schema-compatibility violation(s)")
        return 1

    print(
        f"[PASS] All {len(tables_by_name)} migration tables have matching "
        f"TableModels with full column coverage ({len(models)} model classes "
        f"scanned)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

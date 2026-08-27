"""SQL validation and secure rewriting for agent database tools.

This module ports the upstream SQL security helpers used by the agent
database-query and data-analysis tools. It validates a model-visible
statement against a conservative allow-list policy and, when asked,
rewrites the query to enforce the agent's server-owned scope:

- tenant isolation (``tenant_id`` injection);
- soft-delete filtering (``deleted_at IS NULL``);
- hidden knowledge-base filtering (``is_temporary = false``);
- enabled-chunk filtering (``is_enabled = true``);
- search-scope injection (knowledge-base / document / tag scopes).

Validation is best-effort without a PostgreSQL parser library: a light
tokenizer walks the top-level statement structure and every unhandled or
suspicious construct is rejected by default, so the policy can never be
loosened by a parse miss.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

#: Column that owns a row in the multi-tenant tables below.
_TENANT_COLUMN = "tenant_id"

#: Tables that carry a ``tenant_id`` column (the only ones the agent SQL
#: tools may read).
_DEFAULT_TENANT_TABLES = frozenset({"knowledge_bases", "knowledges", "chunks"})

#: Soft-delete capable tables injected with ``deleted_at IS NULL``.
_DEFAULT_SOFT_DELETE_TABLES = frozenset({"knowledge_bases", "knowledges", "chunks"})

#: Clauses that may follow a FROM / WHERE expression.
_TAIL_CLAUSES = ("GROUP BY", "ORDER BY", "LIMIT", "OFFSET", "HAVING", "FETCH")

#: Clauses that terminate the FROM clause during table extraction.
_FROM_END_KEYWORDS = frozenset(
    {
        "WHERE",
        "GROUP",
        "ORDER",
        "LIMIT",
        "OFFSET",
        "HAVING",
        "FETCH",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "RETURNING",
        "WINDOW",
        "FOR",
        ";",
    }
)

#: Keywords that may precede a JOIN keyword.
_JOIN_MODIFIERS = frozenset({"INNER", "LEFT", "RIGHT", "FULL", "CROSS", "OUTER", "NATURAL"})

#: Keywords that never name a function (parsed by the engine as their own
#: node kinds, so a trailing ``(`` is not a function call).
_NON_FUNCTION_KEYWORDS = frozenset(
    {
        "SELECT",
        "FROM",
        "WHERE",
        "GROUP",
        "ORDER",
        "BY",
        "HAVING",
        "LIMIT",
        "OFFSET",
        "FETCH",
        "UNION",
        "INTERSECT",
        "EXCEPT",
        "AND",
        "OR",
        "NOT",
        "IN",
        "IS",
        "NULL",
        "TRUE",
        "FALSE",
        "AS",
        "ON",
        "USING",
        "JOIN",
        "INNER",
        "LEFT",
        "RIGHT",
        "FULL",
        "CROSS",
        "OUTER",
        "NATURAL",
        "CASE",
        "WHEN",
        "THEN",
        "ELSE",
        "END",
        "DISTINCT",
        "BETWEEN",
        "LIKE",
        "ILIKE",
        "OVER",
        "PARTITION",
        "WINDOW",
        "EXISTS",
        "CAST",
        "ARRAY",
        "ROW",
        "SOME",
        "ANY",
        "ALL",
        "AT",
        "COLLATE",
        "VALUES",
        "WITH",
        "RETURNING",
        "FOR",
        "INTO",
        "LATERAL",
        "TABLESAMPLE",
        "PRIMARY",
        "KEY",
    }
)

#: Function name prefixes that are always dangerous.
_DANGEROUS_FUNCTION_PREFIXES = ("pg_", "lo_", "dblink", "file_", "copy_", "binary_")

#: Explicitly dangerous functions (read/write/execute surfaces).
_DANGEROUS_FUNCTIONS = frozenset(
    {
        # Configuration and settings.
        "current_setting",
        "set_config",
        # XML / XPath (XXE surface).
        "query_to_xml",
        "xpath",
        "xmlparse",
        "xmlroot",
        "xmlelement",
        "xmlforest",
        "xmlconcat",
        "xmlagg",
        "xmlpi",
        "xmlcomment",
        "xmlexists",
        "xml_is_well_formed",
        "xpath_exists",
        "table_to_xml",
        "cursor_to_xml",
        "database_to_xml",
        "schema_to_xml",
        # Transaction / system introspection.
        "txid_current",
        "txid_current_snapshot",
        "txid_snapshot_xmin",
        "txid_snapshot_xmax",
        # Encoding helpers used in attack payloads.
        "encode",
        "decode",
        # Extension management.
        "create_extension",
        # Copy / dump / restore.
        "copy",
        "copy_to",
        "copy_from",
        "pg_copy_to",
        "pg_dump",
        "pg_dumpall",
        "pg_restore",
        "pg_basebackup",
        # Process / system controls.
        "pg_terminate_backend",
        "pg_cancel_backend",
        "pg_rotate_logfile",
        # Advisory locks (denial-of-service surface).
        "pg_advisory_lock",
        "pg_advisory_unlock",
        "pg_advisory_lock_shared",
        "pg_advisory_unlock_shared",
        "pg_try_advisory_lock",
        "pg_try_advisory_lock_shared",
        # Backup / replication.
        "pg_start_backup",
        "pg_stop_backup",
        "pg_switch_wal",
        "pg_create_restore_point",
        # Foreign data wrappers.
        "postgres_fdw_handler",
        "file_fdw_handler",
        # Procedural languages (code execution).
        "plpgsql_call_handler",
        "plpython_call_handler",
        "plperl_call_handler",
        # System catalogs.
        "pg_catalog",
        "information_schema",
        # DuckDB file readers (data-analysis engine); the data is already
        # loaded into a session table so arbitrary-path readers are never
        # legitimately needed.
        "read_text",
        "read_blob",
        "read_csv",
        "read_csv_auto",
        "read_parquet",
        "read_json",
        "read_json_auto",
        "read_ndjson",
        "read_ndjson_auto",
        "read_json_objects",
        "read_xlsx",
        "sniff_csv",
        "glob",
        "st_read",
        "st_read_meta",
    }
)

#: Whitelisted aggregate / scalar functions (``WithDefaultSafeFunctions``).
_DEFAULT_SAFE_FUNCTIONS = frozenset(
    {
        # Aggregates.
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "array_agg",
        "string_agg",
        "bool_and",
        "bool_or",
        "json_agg",
        "jsonb_agg",
        "json_object_agg",
        "jsonb_object_agg",
        # Safe scalars.
        "coalesce",
        "nullif",
        "greatest",
        "least",
        "abs",
        "ceil",
        "floor",
        "round",
        "length",
        "lower",
        "upper",
        "trim",
        "ltrim",
        "rtrim",
        "substring",
        "concat",
        "concat_ws",
        "replace",
        "left",
        "right",
        "now",
        "current_date",
        "current_timestamp",
        "date_trunc",
        "extract",
        "to_char",
        "to_date",
        "to_timestamp",
        "date_part",
        "age",
    }
)

#: PostgreSQL system columns that must never be referenced.
_SYSTEM_COLUMNS = frozenset({"xmin", "xmax", "cmin", "cmax", "ctid", "tableoid"})

_RE_WHITESPACE = re.compile(r"\s+")

_SQL_ALWAYS_TRUE_PATTERNS = (
    re.compile(r"(^|\s|\()(1\s*=\s*1|'1'\s*=\s*'1'|\"1\"\s*=\s*\"1\")(\s|\)|$|and|or)"),
    re.compile(r"(^|\s|\()(0\s*=\s*0|'0'\s*=\s*'0'|\"0\"\s*=\s*\"0\")(\s|\)|$|and|or)"),
    re.compile(r"(^|\s|\()(true)(\s|\)|$|and|or)"),
    re.compile(r"(^|\s|\()('\s*'\s*=\s*'\s*'|\"\s*\"\s*=\s*\"\s*\")(\s|\)|$|and|or)"),
)

_SQL_ALWAYS_FALSE_PATTERNS = (
    re.compile(r"(^|\s|\()(1\s*=\s*0|0\s*=\s*1|'1'\s*=\s*'0'|\"1\"\s*=\s*\"0\")(\s|\)|$|and|or)"),
    re.compile(r"(^|\s|\()(false)(\s|\)|$|and|or)"),
)

_RE_OR_ALWAYS_TRUE = re.compile(r"or\s+(1\s*=\s*1|'1'\s*=\s*'1'|true)")

_RE_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RE_DOLLAR_QUOTED = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$")


@dataclass(frozen=True, slots=True)
class SQLValidationError:
    """One validation error (type + message + details)."""

    type: str
    message: str
    details: str = ""


@dataclass(frozen=True, slots=True)
class SQLValidationResult:
    """Outcome of a ``validate_sql`` run."""

    valid: bool
    errors: tuple[SQLValidationError, ...] = ()


@dataclass(frozen=True, slots=True)
class SQLParseResult:
    """Parsed components of the leading statement of a SQL string."""

    is_select: bool = False
    table_names: tuple[str, ...] = ()
    table_aliases: Mapping[str, str] = field(default_factory=dict)
    functions: tuple[str, ...] = ()
    where_clause: str = ""
    original_sql: str = ""
    parse_error: str = ""
    statements: tuple[str, ...] = ()
    has_cte: bool = False
    has_compound: bool = False
    has_subquery: bool = False
    has_from_function: bool = False
    select_into: bool = False
    locking_clause: bool = False


@dataclass(frozen=True, slots=True)
class SearchScope:
    """One allowed knowledge scope for search-scope injection.

    Empty ``knowledge_ids`` / ``tag_ids`` means the whole knowledge base is
    in scope.
    """

    knowledge_base_id: str
    knowledge_ids: tuple[str, ...] = ()
    tag_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SQLValidationConfig:
    """Configuration flags controlling ``validate_sql``."""

    min_length: int = 6
    max_length: int = 4096
    check_input: bool = False
    check_select_only: bool = False
    check_single_statement: bool = False
    check_table_names: bool = False
    allowed_tables: frozenset[str] = frozenset()
    check_function_names: bool = False
    allowed_functions: frozenset[str] = frozenset()
    check_injection_risk: bool = False
    check_subqueries: bool = False
    check_ctes: bool = False
    check_system_columns: bool = False
    check_schema_access: bool = False
    check_dangerous_functions: bool = False
    enable_tenant_injection: bool = False
    tenant_id: int = 0
    tables_with_tenant_id: frozenset[str] = frozenset()
    enable_soft_delete: bool = False
    tables_with_deleted_at: frozenset[str] = frozenset()
    enable_hidden_kb_filter: bool = False
    enable_chunk_enabled_filter: bool = False
    enable_search_scope: bool = False
    search_scope_kb_ids: tuple[str, ...] = ()
    search_scope_knowledge_ids: tuple[str, ...] = ()
    search_scopes: tuple[SearchScope, ...] = ()


# ── Validation options ────────────────────────────────────────────────


def with_input_validation(
    cfg: SQLValidationConfig, min_length: int, max_length: int
) -> SQLValidationConfig:
    """Enable basic input validation (null bytes, length bounds)."""
    return replace(
        cfg,
        check_input=True,
        min_length=min_length,
        max_length=max_length,
    )


def with_select_only(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Ensure only SELECT statements are allowed."""
    return replace(cfg, check_select_only=True)


def with_single_statement(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Ensure the input holds exactly one statement."""
    return replace(cfg, check_single_statement=True)


def with_allowed_tables(cfg: SQLValidationConfig, *tables: str) -> SQLValidationConfig:
    """Restrict the query to the given table names."""
    return replace(
        cfg,
        check_table_names=True,
        allowed_tables=frozenset(_lower(t) for t in tables),
    )


def with_allowed_functions(cfg: SQLValidationConfig, *functions: str) -> SQLValidationConfig:
    """Restrict function calls to the given names."""
    return replace(
        cfg,
        check_function_names=True,
        allowed_functions=frozenset(_lower(f) for f in functions),
    )


def with_default_safe_functions(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Enable the default safe-function whitelist."""
    return replace(cfg, check_function_names=True, allowed_functions=_DEFAULT_SAFE_FUNCTIONS)


def with_no_subqueries(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Block all subqueries."""
    return replace(cfg, check_subqueries=True)


def with_no_ctes(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Block Common Table Expressions (WITH clauses)."""
    return replace(cfg, check_ctes=True)


def with_no_system_columns(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Block PostgreSQL system columns."""
    return replace(cfg, check_system_columns=True)


def with_no_schema_access(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Block schema-qualified access except the public schema."""
    return replace(cfg, check_schema_access=True)


def with_no_dangerous_functions(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Block dangerous / file-reading functions."""
    return replace(cfg, check_dangerous_functions=True)


def with_tenant_isolation(
    cfg: SQLValidationConfig, tenant_id: int, *tables: str
) -> SQLValidationConfig:
    """Enable automatic tenant_id injection for the given tables."""
    tenant_tables = frozenset(_lower(t) for t in tables) if tables else _DEFAULT_TENANT_TABLES
    return replace(
        cfg,
        enable_tenant_injection=True,
        tenant_id=tenant_id,
        tables_with_tenant_id=tenant_tables,
    )


def with_soft_delete_filter(cfg: SQLValidationConfig, *tables: str) -> SQLValidationConfig:
    """Enable automatic ``deleted_at IS NULL`` injection for the given tables."""
    delete_tables = frozenset(_lower(t) for t in tables) if tables else _DEFAULT_SOFT_DELETE_TABLES
    return replace(cfg, enable_soft_delete=True, tables_with_deleted_at=delete_tables)


def with_hidden_kb_filter(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Exclude internal / temporary knowledge bases (``is_temporary = false``)."""
    return replace(cfg, enable_hidden_kb_filter=True)


def with_chunk_enabled_filter(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Exclude disabled chunks (``is_enabled = true``)."""
    return replace(cfg, enable_chunk_enabled_filter=True)


def with_injection_risk_check(cfg: SQLValidationConfig) -> SQLValidationConfig:
    """Check the WHERE clause for classic injection patterns."""
    return replace(cfg, check_injection_risk=True)


def with_search_scope_filter(
    cfg: SQLValidationConfig,
    kb_ids: Sequence[str],
    knowledge_ids: Sequence[str],
) -> SQLValidationConfig:
    """Restrict the query to the given knowledge bases / documents."""
    if not kb_ids:
        return cfg
    return replace(
        cfg,
        enable_search_scope=True,
        search_scope_kb_ids=tuple(dict.fromkeys(kb_ids)),
        search_scope_knowledge_ids=tuple(dict.fromkeys(knowledge_ids)),
    )


def with_search_scopes(
    cfg: SQLValidationConfig, scopes: Sequence[SearchScope]
) -> SQLValidationConfig:
    """Restrict the query with structured OR'd search scopes."""
    if not scopes:
        return cfg
    return replace(
        cfg,
        enable_search_scope=True,
        search_scopes=tuple(scopes),
    )


def with_security_defaults(cfg: SQLValidationConfig, tenant_id: int) -> SQLValidationConfig:
    """Apply the comprehensive security defaults for the database-query tool."""
    cfg = with_input_validation(cfg, 6, 4096)
    cfg = with_select_only(cfg)
    cfg = with_single_statement(cfg)
    cfg = with_no_subqueries(cfg)
    cfg = with_no_ctes(cfg)
    cfg = with_no_system_columns(cfg)
    cfg = with_no_schema_access(cfg)
    cfg = with_no_dangerous_functions(cfg)
    cfg = with_default_safe_functions(cfg)
    cfg = with_tenant_isolation(cfg, tenant_id)
    return with_allowed_tables(cfg, "knowledge_bases", "knowledges", "chunks")


# ── Tokenizer / parser helpers ────────────────────────────────────────


def _lower(value: str) -> str:
    return value.lower()


def _skip_quoted(sql: str, index: int, quote: str) -> int:
    """Return the index just past the quoted literal starting at ``index``."""
    n = len(sql)
    i = index + 1
    while i < n:
        char = sql[i]
        if char == quote:
            if i + 1 < n and sql[i + 1] == quote:
                i += 2
                continue
            return i + 1
        if char == "\\":
            i += 2
            continue
        i += 1
    return n


def _skip_dollar_quoted(sql: str, index: int) -> int | None:
    """Return the index just past a dollar-quoted string, or ``None``."""
    match = _RE_DOLLAR_QUOTED.match(sql, index)
    if match is None:
        return None
    tag = match.group(0)
    closing = sql.find(tag, index + len(tag))
    if closing == -1:
        return len(sql)
    return closing + len(tag)


def _scan_top_level_tokens(
    sql: str,
    include_parens: bool = False,
) -> list[tuple[int, str]]:
    """Return ``(offset, token)`` pairs of the statement.

    Tokens inside quoted strings or comments are always skipped. By default
    only tokens at parenthesis depth zero are returned so a subquery can
    never smuggle a table or function past the statement-level checks; with
    ``include_parens`` every depth is emitted so a full function scan can
    see (and validate) calls nested inside subqueries.
    """
    tokens: list[tuple[int, str]] = []
    i = 0
    n = len(sql)
    depth = 0
    while i < n:
        char = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if char == "-" and nxt == "-":
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
            continue
        if char == "/" and nxt == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if char == "'":
            i = _skip_quoted(sql, i, "'")
            continue
        if char == '"':
            end = _skip_quoted(sql, i, '"')
            if depth == 0 or include_parens:
                tokens.append((i, sql[i:end]))
            i = end
            continue
        if char == "$":
            dollar_end = _skip_dollar_quoted(sql, i)
            if dollar_end is not None:
                i = dollar_end
                continue
        if char.isspace():
            i += 1
            continue
        if char in "();":
            if char == "(":
                if depth == 0 or include_parens:
                    tokens.append((i, char))
                depth += 1
            elif char == ")":
                depth = max(depth - 1, 0)
                if depth == 0 or include_parens:
                    tokens.append((i, char))
            elif depth == 0 or include_parens:
                tokens.append((i, char))
            i += 1
            continue
        if depth == 0 or include_parens:
            end = i + 1
            while end < n:
                cc = sql[end]
                if cc.isspace() or cc in "()'\"`;":
                    break
                if cc == "-" and end + 1 < n and sql[end + 1] == "-":
                    break
                if cc == "/" and end + 1 < n and sql[end + 1] == "*":
                    break
                end += 1
            tokens.append((i, sql[i:end]))
            i = end
            continue
        i += 1
    return tokens


def _token_words(tokens: Sequence[tuple[int, str]]) -> list[str]:
    return [token.upper() for _pos, token in tokens]


def _split_top_level_statements(sql: str, tokens: Sequence[tuple[int, str]]) -> list[str]:
    """Split ``sql`` into statements on top-level semicolons."""
    statements: list[str] = []
    start = 0
    for pos, token in tokens:
        if token != ";":
            continue
        chunk = sql[start:pos].strip()
        if chunk:
            statements.append(chunk)
        start = pos + 1
    tail = sql[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def _unquote_identifier(token: str) -> str:
    """Strip double quotes from an identifier token, unescaping doubled quotes."""
    if token.startswith('"') and token.endswith('"') and len(token) >= 2:
        return token[1:-1].replace('""', '"')
    return token


def _base_identifier(ref: str) -> str:
    """Return the last dotted segment of a (possibly quoted) table reference."""
    parts = re.split(r"\.", _unquote_identifier(ref))
    return parts[-1]


def _schema_of(ref: str) -> str:
    """Return the leading schema segment of a qualified reference, if any."""
    parts = re.split(r"\.", _unquote_identifier(ref))
    if len(parts) > 1:
        return parts[0].lower()
    return ""


def _matching_paren(sql: str, open_index: int) -> int:
    """Return the index just past the ``)`` matching ``open_index``."""
    n = len(sql)
    depth = 0
    i = open_index
    while i < n:
        char = sql[i]
        nxt = sql[i + 1] if i + 1 < n else ""
        if char == "-" and nxt == "-":
            end = sql.find("\n", i)
            i = n if end == -1 else end + 1
            continue
        if char == "/" and nxt == "*":
            end = sql.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        if char == "'":
            i = _skip_quoted(sql, i, "'")
            continue
        if char == '"':
            i = _skip_quoted(sql, i, '"')
            continue
        if char == "$":
            dollar_end = _skip_dollar_quoted(sql, i)
            if dollar_end is not None:
                i = dollar_end
                continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return n


def _contains_select_word(sql: str, start: int, end: int) -> bool:
    """Whether the region contains a SELECT keyword (i.e. is a subquery)."""
    return re.search(r"\bselect\b", sql[start:end], re.IGNORECASE) is not None


def _extract_from_clause(
    sql: str,
    tokens: Sequence[tuple[int, str]],
) -> tuple[dict[str, str], list[str], bool, bool]:
    """Parse the FROM clause into ``(aliases, table_names, has_subquery, has_function)``.

    ``aliases`` maps table name -> alias (the table name itself when no alias
    is present). ``table_names`` keeps the referenced table names in order. A
    parenthesized FROM item only counts as a subquery when its balanced region
    contains a SELECT (so ``IN (1, 2)``-style lists are not flagged). A group's
    trailing ``AS alias`` / ``alias`` is consumed as the group's alias (never
    registered as a table), and tables referenced inside an allowed subquery
    are collected recursively.
    """
    aliases: dict[str, str] = {}
    table_names: list[str] = []
    has_subquery = False
    has_function = False

    from_index = -1
    for index, (_pos, token) in enumerate(tokens):
        if token.upper() == "FROM":
            from_index = index
            break
    if from_index == -1:
        return aliases, table_names, has_subquery, has_function

    spec: list[str] = []
    mode = "spec"
    skip_until = -1
    #: True while the next spec token is the alias of a closed group.
    pending_alias = False

    def _finalize() -> None:
        nonlocal spec
        if not spec:
            return
        if spec[-1].upper() == "AS":
            spec = spec[:-1]
        if not spec:
            return
        if len(spec) == 1:
            ref = spec[0]
            alias = ""
        else:
            ref = spec[0]
            alias = _unquote_identifier(spec[-1]).lower()
        name = _base_identifier(ref).lower()
        if name:
            table_names.append(name)
            aliases.setdefault(name, alias or name)
        spec = []

    def _register_group(open_index: int, end: int) -> None:
        """Collect the tables/functions of one parenthesized FROM group."""
        nonlocal has_subquery, has_function
        if not _contains_select_word(sql, open_index, end):
            return
        has_subquery = True
        inner_sql = sql[open_index + 1 : max(end - 1, open_index + 1)]
        inner_tokens = _scan_top_level_tokens(inner_sql)
        _inner_aliases, inner_tables, inner_sub, inner_func = _extract_from_clause(
            inner_sql, inner_tokens
        )
        has_subquery = has_subquery or inner_sub
        has_function = has_function or inner_func
        for table, alias in _inner_aliases.items():
            aliases.setdefault(table, alias)
        for table in inner_tables:
            if table not in table_names:
                table_names.append(table)

    for index in range(from_index + 1, len(tokens)):
        pos, token = tokens[index]
        if pos < skip_until:
            continue
        upper = token.upper()

        if token == ")":
            # Closing parenthesis of a FROM group; the alias (if any) follows.
            mode = "spec"
            continue

        if token == "(":
            end = _matching_paren(sql, pos)
            _register_group(pos, end)
            skip_until = end
            _finalize()
            mode = "spec"
            pending_alias = True
            continue

        if upper in _FROM_END_KEYWORDS:
            _finalize()
            break

        if upper == "ON" or upper == "USING":
            _finalize()
            mode = "on"
            pending_alias = False
            continue

        if upper in _JOIN_MODIFIERS or upper == "JOIN":
            _finalize()
            mode = "spec"
            pending_alias = False
            continue

        if token == ",":
            _finalize()
            mode = "spec"
            pending_alias = False
            continue

        if mode == "spec":
            if pending_alias:
                # ``AS alias`` or bare ``alias`` of the group that just closed.
                if upper == "AS":
                    continue
                pending_alias = False
                continue
            spec.append(token)
            if index + 1 < len(tokens) and tokens[index + 1][1] == "(":
                has_function = True
                _finalize()
                mode = "on"

    _finalize()
    return aliases, table_names, has_subquery, has_function


def _extract_functions(tokens: Sequence[tuple[int, str]]) -> list[str]:
    """Return candidate function names (word token directly before ``(``)."""
    names: list[str] = []
    upper = _token_words(tokens)
    for index in range(len(tokens) - 1):
        token = tokens[index][1]
        if tokens[index + 1][1] != "(":
            continue
        name = _base_identifier(token)
        if not name:
            continue
        if name.upper() in _NON_FUNCTION_KEYWORDS:
            continue
        if upper[index] in _NON_FUNCTION_KEYWORDS:
            continue
        names.append(name.lower())
    return names


def _extract_where_clause(sql: str, tokens: Sequence[tuple[int, str]]) -> str:
    """Return the raw WHERE-clause text of the leading statement."""
    where_index = -1
    for index, (_pos, token) in enumerate(tokens):
        if token.upper() == "WHERE":
            where_index = index
            break
    if where_index == -1:
        return ""

    where_pos = tokens[where_index][0] + len("WHERE")
    end = len(sql)
    lower = sql[where_pos:].lower()
    for clause in (
        "group by",
        "order by",
        "limit",
        "having",
        "union",
        "intersect",
        "except",
        "offset",
        "fetch",
    ):
        pos = lower.find(clause)
        if pos != -1 and where_pos + pos < end:
            end = where_pos + pos
    return sql[where_pos:end].strip()


# ── Parsing ───────────────────────────────────────────────────────────


def parse_sql(sql: str) -> SQLParseResult:
    """Parse the leading statement of ``sql`` into its security-relevant parts."""
    tokens = _scan_top_level_tokens(sql)
    statements = _split_top_level_statements(sql, tokens)

    if not statements:
        return SQLParseResult(
            original_sql=sql,
            parse_error="empty query",
            statements=(),
        )

    first = statements[0]
    first_tokens = _scan_top_level_tokens(first)
    words = _token_words(first_tokens)

    if not words:
        return SQLParseResult(
            original_sql=sql, parse_error="empty query", statements=tuple(statements)
        )

    first_keyword = words[0]
    is_select = first_keyword == "SELECT"
    has_cte = first_keyword == "WITH"

    aliases, table_names, has_subquery, has_from_function = _extract_from_clause(
        first, first_tokens
    )
    # Scan for function calls at every nesting depth (not just the top
    # level) so dangerous functions hidden inside an allowed subquery are
    # still caught — mirroring the upstream recursive statement inspection.
    functions = _extract_functions(_scan_top_level_tokens(first, include_parens=True))

    has_compound = any(upper in {"UNION", "INTERSECT", "EXCEPT"} for upper in words[1:])
    select_into = "INTO" in words[1:]
    locking_clause = any(
        (
            words[i] == "FOR"
            and i + 1 < len(words)
            and words[i + 1] in {"UPDATE", "SHARE", "NO", "KEY"}
        )
        for i in range(len(words))
    )

    where_clause = _extract_where_clause(first, first_tokens)

    return SQLParseResult(
        is_select=is_select,
        table_names=tuple(dict.fromkeys(table_names)),
        table_aliases=aliases,
        functions=tuple(dict.fromkeys(functions)),
        where_clause=where_clause,
        original_sql=sql,
        statements=tuple(statements),
        has_cte=has_cte,
        has_compound=has_compound,
        has_subquery=has_subquery,
        has_from_function=has_from_function,
        select_into=select_into,
        locking_clause=locking_clause,
    )


# ── Validation ────────────────────────────────────────────────────────


def _check_input(sql: str, config: SQLValidationConfig, errors: list[SQLValidationError]) -> None:
    if "\x00" in sql:
        errors.append(
            SQLValidationError(
                type="input_validation_error",
                message="Input validation failed",
                details="invalid character in SQL query",
            )
        )
        return
    if len(sql) < config.min_length:
        errors.append(
            SQLValidationError(
                type="input_validation_error",
                message="Input validation failed",
                details=f"SQL query too short (min {config.min_length} characters)",
            )
        )
    elif len(sql) > config.max_length:
        errors.append(
            SQLValidationError(
                type="input_validation_error",
                message="Input validation failed",
                details=f"SQL query too long (max {config.max_length} characters)",
            )
        )


def _validate_statement(
    parsed: SQLParseResult, config: SQLValidationConfig, errors: list[SQLValidationError]
) -> None:
    if parsed.has_compound:
        errors.append(
            SQLValidationError(
                type="statement_validation_error",
                message="Statement validation failed",
                details="compound queries (UNION/INTERSECT/EXCEPT) are not allowed",
            )
        )
    if config.check_ctes and parsed.has_cte:
        errors.append(
            SQLValidationError(
                type="statement_validation_error",
                message="Statement validation failed",
                details="WITH clause (CTEs) is not allowed",
            )
        )
    if parsed.select_into:
        errors.append(
            SQLValidationError(
                type="statement_validation_error",
                message="Statement validation failed",
                details="SELECT INTO is not allowed",
            )
        )
    if parsed.locking_clause:
        errors.append(
            SQLValidationError(
                type="statement_validation_error",
                message="Statement validation failed",
                details="locking clauses (FOR UPDATE, etc.) are not allowed",
            )
        )
    if config.check_subqueries and parsed.has_subquery:
        errors.append(
            SQLValidationError(
                type="statement_validation_error",
                message="Statement validation failed",
                details="subqueries are not allowed",
            )
        )
    if parsed.has_from_function:
        errors.append(
            SQLValidationError(
                type="statement_validation_error",
                message="Statement validation failed",
                details="functions in FROM clause are not allowed",
            )
        )
    if config.check_schema_access:
        for ref in _schema_qualified_refs(parsed.table_aliases):
            schema = _schema_of(ref)
            if schema and schema != "public":
                errors.append(
                    SQLValidationError(
                        type="statement_validation_error",
                        message="Statement validation failed",
                        details=f"access to schema '{schema}' is not allowed",
                    )
                )
    if config.check_table_names:
        for table in parsed.table_names:
            if table not in config.allowed_tables:
                errors.append(
                    SQLValidationError(
                        type="table_not_allowed",
                        message=f"Table '{table}' is not in the allowed list",
                        details=f"Allowed tables: {sorted(config.allowed_tables)}",
                    )
                )
    if config.check_system_columns:
        for name in _column_references(parsed.original_sql):
            if name in _SYSTEM_COLUMNS:
                errors.append(
                    SQLValidationError(
                        type="statement_validation_error",
                        message="Statement validation failed",
                        details=f"access to system column '{name}' is not allowed",
                    )
                )
            elif name.startswith("pg_"):
                errors.append(
                    SQLValidationError(
                        type="statement_validation_error",
                        message="Statement validation failed",
                        details=f"access to '{name}' is not allowed",
                    )
                )

    for name in parsed.functions:
        if config.check_schema_access:
            schema = _schema_of(name)
            if schema and schema not in {"", "pg_catalog", "public"}:
                errors.append(
                    SQLValidationError(
                        type="statement_validation_error",
                        message="Statement validation failed",
                        details=f"schema-qualified function calls are not allowed: {schema}",
                    )
                )
        if config.check_dangerous_functions:
            func_name = _base_identifier(name)
            if any(func_name.startswith(prefix) for prefix in _DANGEROUS_FUNCTION_PREFIXES):
                errors.append(
                    SQLValidationError(
                        type="statement_validation_error",
                        message="Statement validation failed",
                        details=f"function '{func_name}' is not allowed (dangerous prefix)",
                    )
                )
            elif func_name in _DANGEROUS_FUNCTIONS:
                errors.append(
                    SQLValidationError(
                        type="statement_validation_error",
                        message="Statement validation failed",
                        details=f"function '{func_name}' is not allowed",
                    )
                )
        if config.check_function_names and _base_identifier(name) not in config.allowed_functions:
            errors.append(
                SQLValidationError(
                    type="statement_validation_error",
                    message="Statement validation failed",
                    details=f"function not allowed: {_base_identifier(name)}",
                )
            )


def _schema_qualified_refs(aliases: Mapping[str, str]) -> list[str]:
    """Return the raw qualified references (schema.table) of the FROM clause."""
    refs: list[str] = []
    for _table, alias in aliases.items():
        if "." in alias:
            refs.append(alias)
    return refs


def _column_references(sql: str) -> list[str]:
    """Return candidate column names from the statement (best-effort)."""
    names: list[str] = []
    for match in _RE_WORD.finditer(sql):
        name = match.group(0).lower()
        if name.startswith("pg_") or name in _SYSTEM_COLUMNS:
            names.append(name)
    return names


def _check_injection_risks(where_clause: str, errors: list[SQLValidationError]) -> None:
    if not where_clause:
        return
    normalized = _RE_WHITESPACE.sub(" ", where_clause.strip().lower())
    for pattern in _SQL_ALWAYS_TRUE_PATTERNS:
        if pattern.search(normalized):
            errors.append(
                SQLValidationError(
                    type="sql_injection_risk",
                    message="Potential SQL injection risk detected",
                    details=f"Always-true condition found in WHERE clause: {where_clause}",
                )
            )
    for pattern in _SQL_ALWAYS_FALSE_PATTERNS:
        if pattern.search(normalized):
            errors.append(
                SQLValidationError(
                    type="sql_injection_risk",
                    message="Suspicious SQL pattern detected",
                    details=f"Always-false condition found in WHERE clause: {where_clause}",
                )
            )
    if _RE_OR_ALWAYS_TRUE.search(normalized):
        errors.append(
            SQLValidationError(
                type="sql_injection_risk",
                message="High-risk SQL injection pattern detected",
                details=f"OR with always-true condition found in WHERE clause: {where_clause}",
            )
        )


def validate_sql(sql: str, config: SQLValidationConfig) -> SQLValidationResult:
    """Validate ``sql`` against ``config``, returning a result value."""
    errors: list[SQLValidationError] = []

    if config.check_input:
        _check_input(sql, config, errors)
        if errors:
            return SQLValidationResult(valid=False, errors=tuple(errors))

    parsed = parse_sql(sql)
    if parsed.parse_error:
        errors.append(
            SQLValidationError(
                type="parse_error",
                message="Failed to parse SQL",
                details=f"SQL parse error: {parsed.parse_error}",
            )
        )
        return SQLValidationResult(valid=False, errors=tuple(errors))

    if not parsed.statements:
        errors.append(
            SQLValidationError(
                type="empty_query",
                message="Empty query",
                details="No statements found in SQL",
            )
        )
        return SQLValidationResult(valid=False, errors=tuple(errors))

    if config.check_single_statement and len(parsed.statements) > 1:
        errors.append(
            SQLValidationError(
                type="multiple_statements",
                message="Multiple statements are not allowed",
                details=f"Found {len(parsed.statements)} statements, only 1 is allowed",
            )
        )
        return SQLValidationResult(valid=False, errors=tuple(errors))

    if config.check_select_only and not parsed.is_select:
        errors.append(
            SQLValidationError(
                type="not_select_statement",
                message="Only SELECT queries are allowed",
                details="Statement is not a SELECT query",
            )
        )
        return SQLValidationResult(valid=False, errors=tuple(errors))

    if parsed.is_select:
        _validate_statement(parsed, config, errors)

    if config.check_injection_risk and parsed.is_select:
        _check_injection_risks(parsed.where_clause, errors)

    return SQLValidationResult(valid=not errors, errors=tuple(errors))


# ── Secure rewriting ──────────────────────────────────────────────────


_RE_WHERE_KEYWORD = re.compile(r"\bWHERE\b")
_RE_TAIL_CLAUSE = re.compile(r"\b(GROUP BY|ORDER BY|LIMIT|OFFSET|HAVING|FETCH)\b", re.IGNORECASE)


def inject_and_conditions(sql: str, filter_clause: str) -> str:
    """Inject ``filter_clause`` into ``sql`` using AND semantics.

    An existing WHERE expression is wrapped in parentheses to keep OR
    precedence from widening the filter. Trailing clauses (ORDER BY, LIMIT,
    ...) are left intact.
    """
    filter_clause = filter_clause.strip()
    if not filter_clause:
        return sql

    where_match = _RE_WHERE_KEYWORD.search(sql)
    if where_match is not None:
        where_expr_start = where_match.end()
        tail_match = _RE_TAIL_CLAUSE.search(sql[where_expr_start:])
        if tail_match is None:
            original = sql[where_expr_start:].strip()
            return f"{sql[: where_match.start()]}WHERE {filter_clause} AND ({original})"
        where_expr_end = where_expr_start + tail_match.start()
        original = sql[where_expr_start:where_expr_end].strip()
        tail_clause = sql[where_expr_end:].lstrip(" \t\r\n")
        return f"{sql[: where_match.start()]}WHERE {filter_clause} AND ({original}) {tail_clause}"

    tail_match = _RE_TAIL_CLAUSE.search(sql)
    if tail_match is not None:
        prefix = sql[: tail_match.start()].rstrip(" \t\r\n")
        suffix = sql[tail_match.start() :].lstrip(" \t\r\n")
        return f"{prefix} WHERE {filter_clause} {suffix}"

    return f"{sql} WHERE {filter_clause}"


def _quote_sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _table_alias_map(sql: str) -> dict[str, str]:
    """Return the table -> alias map of the leading statement."""
    parsed = parse_sql(sql)
    return dict(parsed.table_aliases)


def _inject_tenant_conditions(
    sql: str, config: SQLValidationConfig, aliases: Mapping[str, str]
) -> str:
    if not config.enable_tenant_injection:
        return sql
    conditions: list[str] = []
    for table, alias in aliases.items():
        if table not in config.tables_with_tenant_id:
            continue
        if table == "tenants":
            conditions.append(f"{alias}.id = {config.tenant_id}")
        else:
            conditions.append(f"{alias}.{_TENANT_COLUMN} = {config.tenant_id}")
    if not conditions:
        return sql
    return inject_and_conditions(sql, " AND ".join(conditions))


def _inject_soft_delete_conditions(
    sql: str, config: SQLValidationConfig, aliases: Mapping[str, str]
) -> str:
    if not config.enable_soft_delete:
        return sql
    conditions = [
        f"{alias}.deleted_at IS NULL"
        for table, alias in aliases.items()
        if table in config.tables_with_deleted_at
    ]
    if not conditions:
        return sql
    return inject_and_conditions(sql, " AND ".join(conditions))


def _inject_hidden_kb_filter(
    sql: str, config: SQLValidationConfig, aliases: Mapping[str, str]
) -> str:
    if not config.enable_hidden_kb_filter:
        return sql
    alias = aliases.get("knowledge_bases")
    if not alias:
        return sql
    return inject_and_conditions(sql, f"{alias}.is_temporary = false")


def _inject_chunk_enabled_filter(
    sql: str, config: SQLValidationConfig, aliases: Mapping[str, str]
) -> str:
    if not config.enable_chunk_enabled_filter:
        return sql
    alias = aliases.get("chunks")
    if not alias:
        return sql
    return inject_and_conditions(sql, f"{alias}.is_enabled = true")


def _inject_search_scope_conditions(
    sql: str, config: SQLValidationConfig, aliases: Mapping[str, str]
) -> str:
    if not config.enable_search_scope:
        return sql
    if config.search_scopes:
        return _inject_structured_search_scopes(sql, config, aliases)
    if not config.search_scope_kb_ids:
        return sql

    quoted_kb_ids = ", ".join(_quote_sql_string(value) for value in config.search_scope_kb_ids)
    conditions: list[str] = []
    alias = aliases.get("knowledge_bases")
    if alias:
        conditions.append(f"{alias}.id IN ({quoted_kb_ids})")
    alias = aliases.get("knowledges")
    if alias:
        conditions.append(f"{alias}.knowledge_base_id IN ({quoted_kb_ids})")
        if config.search_scope_knowledge_ids:
            quoted_kids = ", ".join(
                _quote_sql_string(value) for value in config.search_scope_knowledge_ids
            )
            conditions.append(f"{alias}.id IN ({quoted_kids})")
    alias = aliases.get("chunks")
    if alias:
        conditions.append(f"{alias}.knowledge_base_id IN ({quoted_kb_ids})")
        if config.search_scope_knowledge_ids:
            quoted_kids = ", ".join(
                _quote_sql_string(value) for value in config.search_scope_knowledge_ids
            )
            conditions.append(f"{alias}.knowledge_id IN ({quoted_kids})")
    if not conditions:
        return sql
    return inject_and_conditions(sql, " AND ".join(conditions))


def _inject_structured_search_scopes(
    sql: str, config: SQLValidationConfig, aliases: Mapping[str, str]
) -> str:
    conditions: list[str] = []
    alias = aliases.get("knowledge_bases")
    if alias:
        condition = _build_knowledge_base_scope_condition(alias, config.search_scopes)
        if condition:
            conditions.append(condition)
    alias = aliases.get("knowledges")
    if alias:
        condition = _build_document_scope_condition(alias, "id", config.search_scopes)
        if condition:
            conditions.append(condition)
    alias = aliases.get("chunks")
    if alias:
        condition = _build_document_scope_condition(alias, "knowledge_id", config.search_scopes)
        if condition:
            conditions.append(condition)
    if not conditions:
        return sql
    return inject_and_conditions(sql, " AND ".join(conditions))


def _build_knowledge_base_scope_condition(alias: str, scopes: Sequence[SearchScope]) -> str:
    kb_ids = _unique_preserving_order(
        scope.knowledge_base_id for scope in scopes if scope.knowledge_base_id
    )
    if not kb_ids:
        return ""
    quoted = ", ".join(_quote_sql_string(value) for value in kb_ids)
    return f"{alias}.id IN ({quoted})"


def _build_scope_clause(alias: str, knowledge_id_column: str, scope: SearchScope) -> str:
    if not scope.knowledge_base_id:
        return ""
    conditions = [f"{alias}.knowledge_base_id = {_quote_sql_string(scope.knowledge_base_id)}"]
    if scope.knowledge_ids:
        quoted = ", ".join(_quote_sql_string(value) for value in scope.knowledge_ids)
        conditions.append(f"{alias}.{knowledge_id_column} IN ({quoted})")
    if scope.tag_ids:
        quoted_tags = ", ".join(_quote_sql_string(value) for value in scope.tag_ids)
        conditions.append(
            f"EXISTS (SELECT 1 FROM knowledge_tag_relations ktr WHERE ktr.knowledge_id = {alias}.{knowledge_id_column} "
            f"AND ktr.tag_id IN ({quoted_tags}))"
        )
    if len(conditions) == 1:
        return conditions[0]
    return "(" + " AND ".join(conditions) + ")"


def _build_document_scope_condition(
    alias: str, knowledge_id_column: str, scopes: Sequence[SearchScope]
) -> str:
    clauses = [
        clause
        for scope in scopes
        if (clause := _build_scope_clause(alias, knowledge_id_column, scope)) != ""
    ]
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    return "(" + " OR ".join(clauses) + ")"


def validate_and_secure_sql(
    sql: str, config: SQLValidationConfig
) -> tuple[str, SQLValidationResult]:
    """Validate ``sql`` and return ``(secured_sql, result)``.

    ``secured_sql`` is empty when validation fails; otherwise it carries the
    injected tenant / soft-delete / scope filters.
    """
    validation = validate_sql(sql, config)
    if not validation.valid:
        return "", validation

    rewriting_needed = (
        config.enable_tenant_injection
        or config.enable_soft_delete
        or config.enable_hidden_kb_filter
        or config.enable_chunk_enabled_filter
        or config.enable_search_scope
    )
    if not rewriting_needed:
        return sql, validation

    aliases = _table_alias_map(sql)
    secured = sql
    secured = _inject_tenant_conditions(secured, config, aliases)
    secured = _inject_soft_delete_conditions(secured, config, aliases)
    secured = _inject_hidden_kb_filter(secured, config, aliases)
    secured = _inject_chunk_enabled_filter(secured, config, aliases)
    secured = _inject_search_scope_conditions(secured, config, aliases)
    return secured, validation


__all__ = [
    "SQLParseResult",
    "SQLValidationConfig",
    "SQLValidationError",
    "SQLValidationResult",
    "SearchScope",
    "inject_and_conditions",
    "parse_sql",
    "validate_and_secure_sql",
    "validate_sql",
    "with_allowed_functions",
    "with_allowed_tables",
    "with_chunk_enabled_filter",
    "with_default_safe_functions",
    "with_hidden_kb_filter",
    "with_injection_risk_check",
    "with_input_validation",
    "with_no_ctes",
    "with_no_dangerous_functions",
    "with_no_schema_access",
    "with_no_subqueries",
    "with_no_system_columns",
    "with_search_scope_filter",
    "with_search_scopes",
    "with_security_defaults",
    "with_select_only",
    "with_single_statement",
    "with_soft_delete_filter",
    "with_tenant_isolation",
]

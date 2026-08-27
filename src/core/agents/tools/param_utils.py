"""Schema-driven tool parameter casting and validation.

LLMs sometimes return arguments with the wrong JSON type (e.g. ``"true"``
instead of ``true``, ``"123"`` instead of ``123``). ``cast_params``
attempts safe, schema-driven conversions before execution;
``validate_params`` checks the arguments against the tool's JSON Schema
(required, type, enum, numeric and string-length bounds) so invalid calls
fail fast instead of wasting a tool execution and an LLM round.

Both functions operate on raw JSON strings and return the input unchanged
when the schema is absent, unparseable, or nothing changed, matching the
upstream semantics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from src.common.json import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class ValidationError:
    """A single parameter validation failure."""

    param: str = ""
    message: str = ""


def cast_params(args: str, schema: str) -> str:
    """Return ``args`` with values cast toward the schema's declared types.

    Returns the original ``args`` string when no cast was applied or the
    inputs cannot be parsed.
    """
    if not schema or not args:
        return args
    schema_def = _parse_object(schema)
    args_map = _parse_object(args)
    if schema_def is None or args_map is None:
        return args
    properties = schema_def.get("properties")
    if not isinstance(properties, dict) or not properties:
        return args

    changed = False
    for key, value in list(args_map.items()):
        prop_def = properties.get(key)
        if not isinstance(prop_def, dict):
            continue
        target_type = prop_def.get("type")
        if not isinstance(target_type, str) or not target_type:
            continue
        new_value, did_cast = cast_value(value, target_type)
        if did_cast:
            args_map[key] = new_value
            changed = True
    if not changed:
        return args
    return json.dumps(args_map, ensure_ascii=False)


def _parse_object(raw: str) -> JsonObject | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _format_float(value: float) -> str:
    """Render a float the way Go's ``FormatFloat(..., 'f', -1, 64)`` does."""
    if value.is_integer():
        return str(int(value))
    return repr(value)


def cast_value(value: JsonValue, target_type: str) -> tuple[JsonValue, bool]:
    """Attempt to convert ``value`` to the expected ``target_type``.

    Returns ``(new_value, True)`` when a conversion was applied and
    ``(value, False)`` otherwise.
    """
    if target_type == "array" and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed, True
        return [value], True

    if target_type == "boolean":
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes"}:
                return True, True
            if lowered in {"false", "0", "no"}:
                return False, True
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            if value == 0:
                return False, True
            if value == 1:
                return True, True

    if target_type == "integer":
        if isinstance(value, str):
            try:
                return int(value), True
            except ValueError:
                pass
        if isinstance(value, bool):
            return value, False
        if isinstance(value, int):
            return value, False
        if isinstance(value, float) and value.is_integer():
            return int(value), True

    if target_type == "number":
        if isinstance(value, str):
            try:
                return float(value), True
            except ValueError:
                pass
        if isinstance(value, bool):
            return value, False
        if isinstance(value, (int, float)):
            return value, False

    if target_type == "string":
        if isinstance(value, bool):
            return ("true" if value else "false"), True
        if isinstance(value, float):
            return _format_float(value), True
        if isinstance(value, int):
            return str(value), True

    return value, False


def validate_params(args: str, schema: str) -> list[ValidationError]:
    """Check ``args`` against the schema and return the failures.

    Extra parameters are allowed (LLMs sometimes add them). ``None``
    values are only caught by the required check.
    """
    if not schema or not args:
        return []
    schema_def = _parse_object(schema)
    args_map = _parse_object(args)
    if schema_def is None or args_map is None:
        return []
    properties = schema_def.get("properties")
    if not isinstance(properties, dict) or not properties:
        return []

    errors: list[ValidationError] = []

    required = schema_def.get("required")
    if isinstance(required, list):
        for field_name in required:
            if not isinstance(field_name, str):
                continue
            if field_name not in args_map or args_map[field_name] is None:
                errors.append(
                    ValidationError(
                        param=field_name,
                        message=f"required parameter '{field_name}' is missing",
                    )
                )

    for key, value in args_map.items():
        prop_def = properties.get(key)
        if not isinstance(prop_def, dict):
            continue
        errors.extend(validate_property(key, value, prop_def))

    return errors


def validate_property(
    name: str,
    value: JsonValue,
    prop: dict[str, JsonValue],
) -> list[ValidationError]:
    """Validate one parameter value against its property schema."""
    if value is None:
        return []

    errors: list[ValidationError] = []

    target_type = prop.get("type")
    target_type = target_type if isinstance(target_type, str) else ""

    # Type check (skip further checks when the type is wrong).
    if target_type and not check_type(value, target_type):
        errors.append(
            ValidationError(
                param=name,
                message=f"parameter '{name}' should be type '{target_type}'",
            )
        )
        return errors

    # Enum check.
    enum_raw = prop.get("enum")
    if isinstance(enum_raw, list) and enum_raw and not is_in_enum(value, enum_raw):
        allowed = format_enum(enum_raw)
        errors.append(
            ValidationError(
                param=name,
                message=f"parameter '{name}' must be one of [{allowed}]",
            )
        )

    # Numeric bounds.
    if target_type in {"number", "integer"}:
        num_value = to_float(value)
        minimum = get_float(prop, "minimum")
        if minimum is not None and num_value < minimum:
            errors.append(
                ValidationError(
                    param=name,
                    message=f"parameter '{name}' must be >= {minimum}",
                )
            )
        maximum = get_float(prop, "maximum")
        if maximum is not None and num_value > maximum:
            errors.append(
                ValidationError(
                    param=name,
                    message=f"parameter '{name}' must be <= {maximum}",
                )
            )

    # String length bounds.
    if target_type == "string" and isinstance(value, str):
        min_length = get_float(prop, "minLength")
        if min_length is not None and len(value) < min_length:
            errors.append(
                ValidationError(
                    param=name,
                    message=(f"parameter '{name}' must have at least {int(min_length)} characters"),
                )
            )
        max_length = get_float(prop, "maxLength")
        if max_length is not None and len(value) > max_length:
            errors.append(
                ValidationError(
                    param=name,
                    message=(f"parameter '{name}' must have at most {int(max_length)} characters"),
                )
            )

    return errors


def check_type(value: JsonValue, target_type: str) -> bool:
    """Whether ``value`` matches the expected JSON Schema type."""
    if target_type == "string":
        return isinstance(value, str)
    if target_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if target_type == "integer":
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        if isinstance(value, float):
            return value.is_integer()
        return False
    if target_type == "boolean":
        return isinstance(value, bool)
    if target_type == "array":
        return isinstance(value, list)
    if target_type == "object":
        return isinstance(value, dict)
    return True


def _enum_repr(value: JsonValue) -> str:
    """Render a JSON value the way ``fmt.Sprintf("%v", …)`` does."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def is_in_enum(value: JsonValue, enum_list: list[JsonValue]) -> bool:
    """Whether ``value`` matches any value in ``enum_list``."""
    rendered = _enum_repr(value)
    return any(rendered == _enum_repr(entry) for entry in enum_list)


def format_enum(enum_list: list[JsonValue]) -> str:
    """Render the enum values for error messages."""
    return ", ".join(_enum_repr(entry) for entry in enum_list)


def get_float(prop: dict[str, JsonValue], key: str) -> float | None:
    """Extract a numeric bound from a property definition."""
    value = prop.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return None


def to_float(value: JsonValue) -> float:
    """Convert a JSON numeric value to ``float`` (0 for non-numbers)."""
    if isinstance(value, bool):
        return 0.0
    if isinstance(value, int):
        return float(value)
    if isinstance(value, float):
        return value
    return 0.0


def format_validation_errors(errors: list[ValidationError]) -> str:
    """Format a list of validation failures into a human-readable string."""
    if not errors:
        return ""
    messages = "; ".join(error.message for error in errors)
    return f"Parameter validation failed: {messages}"


__all__ = [
    "ValidationError",
    "cast_params",
    "cast_value",
    "check_type",
    "format_validation_errors",
    "validate_params",
    "validate_property",
]

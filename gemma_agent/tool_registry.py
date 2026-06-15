from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .safety import neutralize_model_facing_metadata
from .safety import neutralize_model_facing_metadata_key

TOOL_EXECUTION_MODES = {"process", "thread"}
SUPPORTED_SCHEMA_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_TRUSTED_TERMINAL_RESULT_MARKER = object()
UNSUPPORTED_SCHEMA_KEYS = {
    "$ref",
    "allOf",
    "anyOf",
    "additionalItems",
    "const",
    "contains",
    "dependentRequired",
    "dependentSchemas",
    "else",
    "maxContains",
    "maxProperties",
    "minContains",
    "minProperties",
    "multipleOf",
    "if",
    "not",
    "oneOf",
    "patternProperties",
    "prefixItems",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
    "uniqueItems",
}
THREAD_UNSAFE_ACTIONS = {
    "append",
    "chmod",
    "chown",
    "copy",
    "create",
    "del",
    "delete",
    "edit",
    "mkdir",
    "modify",
    "move",
    "overwrite",
    "patch",
    "remove",
    "rename",
    "replace",
    "rm",
    "rmdir",
    "save",
    "symlink",
    "touch",
    "truncate",
    "unlink",
    "update",
    "write",
}
THREAD_UNSAFE_DIRECT_ACTIONS = THREAD_UNSAFE_ACTIONS - {"create"}
FILE_RESOURCE_TERMS = {"file", "files", "path", "dir", "directory", "workspace"}
PATH_SCHEMA_FORMATS = {"path", "file-path", "directory-path", "workspace-path"}
THREAD_POLICY_PATH_NAMES = {
    "path",
    "file",
    "files",
    "file_path",
    "filepath",
    "paths",
    "dir",
    "directory",
    "directory_path",
    "source",
    "target",
    "destination",
}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    callable: Callable[..., Any]
    schema: dict[str, Any]
    description: str = ""
    timeout_sec: float | None = None
    execution_mode: str = "process"
    _trusted_terminal_result_marker: object | None = field(default=None, repr=False, compare=False)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}
        self.safety_guard: Any | None = None

    def register(
        self,
        name: str,
        callable_: Callable[..., Any] | None = None,
        schema: dict[str, Any] | None = None,
        *,
        description: str = "",
        timeout_sec: float | None = None,
        execution_mode: str = "process",
    ) -> ToolDefinition:
        if callable_ is None:
            raise ValueError("callable is required")
        if not name or not isinstance(name, str):
            raise ValueError("tool name must be a non-empty string")
        if execution_mode not in TOOL_EXECUTION_MODES:
            raise ValueError(f"tool execution_mode must be one of {sorted(TOOL_EXECUTION_MODES)!r}")
        strict_schema = _copy_and_validate_strict_schema(schema)
        _validate_model_facing_call_contract(name, strict_schema)
        policy_error = validate_thread_mode_policy(name, strict_schema, execution_mode)
        if policy_error:
            raise ValueError(policy_error)
        definition = ToolDefinition(
            name=name,
            callable=callable_,
            schema=strict_schema,
            description=description,
            timeout_sec=timeout_sec,
            execution_mode=execution_mode,
        )
        self._tools[name] = definition
        return definition

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def validate_args(self, name: str, args: Any) -> list[str]:
        definition = self.get(name)
        if definition is None:
            return [f"tool {name!r} is not allowlisted"]
        return _validate_value(args, definition.schema, "args")

    def list_tool_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "name": neutralize_model_facing_metadata(definition.name),
                "description": neutralize_model_facing_metadata(definition.description),
                "schema": _sanitize_model_facing_schema(definition.schema),
            }
            for definition in self._tools.values()
            if _is_model_facing_callable_definition(definition)
        ]


def _copy_and_validate_strict_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValueError("tool schema must be a JSON-schema object")
    copied = copy.deepcopy(schema)
    _validate_strict_schema(copied)
    return copied


def _validate_strict_schema(schema: Any) -> None:
    if not isinstance(schema, dict):
        raise ValueError("tool schema must be a JSON-schema object")
    _validate_schema_node(schema, "schema")
    if schema.get("type") != "object":
        raise ValueError("tool schema must have type 'object'")


def validate_thread_mode_policy(name: str, schema: Any, execution_mode: str) -> str | None:
    if execution_mode != "thread":
        return None
    if not _has_thread_unsafe_action_name(name):
        return None
    if not _schema_has_thread_policy_path(schema):
        return None
    return (
        f"tool {name!r} is edit-capable with path-like arguments and cannot run in "
        "thread execution_mode; use process execution_mode"
    )


def _validate_schema_node(schema: Any, path: str) -> None:
    if not isinstance(schema, dict):
        raise ValueError(f"{path} must be a JSON-schema object")

    unsupported = sorted(key for key in UNSUPPORTED_SCHEMA_KEYS if key in schema)
    if unsupported:
        raise ValueError(f"{path} uses unsupported JSON Schema constructs: {unsupported}")

    declared_type = schema.get("type")
    if isinstance(declared_type, list):
        raise ValueError(f"{path}.type must not be an array")
    if declared_type not in SUPPORTED_SCHEMA_TYPES:
        raise ValueError(f"{path}.type must be one of {sorted(SUPPORTED_SCHEMA_TYPES)!r}")

    if declared_type == "object":
        _validate_object_schema_node(schema, path)
    elif declared_type == "array":
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            raise ValueError(f"{path}.items must be a JSON-schema object")
        _validate_schema_node(item_schema, f"{path}.items")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise ValueError(f"{path}.enum must be a list")
    if "pattern" in schema and not isinstance(schema["pattern"], str):
        raise ValueError(f"{path}.pattern must be a string")


def _validate_object_schema_node(schema: dict[str, Any], path: str) -> None:
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        raise ValueError(f"{path} must define properties")
    if schema.get("additionalProperties") is not False:
        raise ValueError(f"{path} must set additionalProperties to False")
    required = schema.get("required", [])
    if not isinstance(required, list):
        raise ValueError(f"{path}.required must be a list")
    missing_properties = [name for name in required if name not in properties]
    if missing_properties:
        raise ValueError(f"{path} required properties are not defined: {missing_properties}")

    for property_name, property_schema in properties.items():
        if not isinstance(property_name, str):
            raise ValueError(f"{path}.properties keys must be strings")
        if not isinstance(property_schema, dict):
            raise ValueError(f"{path}.properties.{property_name} must be a JSON-schema object")
        _validate_schema_node(property_schema, f"{path}.properties.{property_name}")


def _validate_model_facing_call_contract(name: str, schema: dict[str, Any]) -> None:
    if neutralize_model_facing_metadata(name) != name:
        raise ValueError("tool name would be redacted in model-facing schema")
    _validate_schema_model_facing_call_contract(schema, "schema")


def _is_model_facing_callable_definition(definition: ToolDefinition) -> bool:
    try:
        _validate_strict_schema(definition.schema)
        _validate_model_facing_call_contract(definition.name, definition.schema)
    except ValueError:
        return False
    return validate_thread_mode_policy(
        definition.name,
        definition.schema,
        definition.execution_mode,
    ) is None


def _validate_schema_model_facing_call_contract(schema: dict[str, Any], path: str) -> None:
    if "enum" in schema:
        for index, value in enumerate(schema["enum"]):
            if neutralize_model_facing_metadata(value) != value:
                raise ValueError(f"{path}.enum[{index}] would be redacted in model-facing schema")
    if "pattern" in schema and neutralize_model_facing_metadata(schema["pattern"]) != schema["pattern"]:
        raise ValueError(f"{path}.pattern would be redacted in model-facing schema")

    declared_type = schema.get("type")
    if declared_type == "object":
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for property_name, property_schema in properties.items():
            if neutralize_model_facing_metadata_key(property_name) != property_name:
                raise ValueError(f"{path}.properties.{property_name} would be redacted in model-facing schema")
            _validate_schema_model_facing_call_contract(property_schema, f"{path}.properties.{property_name}")
        for required_name in required:
            if not isinstance(required_name, str):
                raise ValueError(f"{path}.required entries must be strings")
            if neutralize_model_facing_metadata_key(required_name) != required_name:
                raise ValueError(f"{path}.required entry would be redacted in model-facing schema")
        return
    if declared_type == "array":
        _validate_schema_model_facing_call_contract(schema["items"], f"{path}.items")


def validate_legacy_definition_contract(definition: ToolDefinition) -> str | None:
    try:
        _validate_strict_schema(definition.schema)
        _validate_model_facing_call_contract(definition.name, definition.schema)
    except ValueError as exc:
        return str(exc)
    return validate_thread_mode_policy(
        definition.name,
        definition.schema,
        definition.execution_mode,
    )


def mark_trusted_terminal_result(definition: ToolDefinition) -> ToolDefinition:
    object.__setattr__(definition, "_trusted_terminal_result_marker", _TRUSTED_TERMINAL_RESULT_MARKER)
    return definition


def has_trusted_terminal_result(definition: ToolDefinition) -> bool:
    return getattr(definition, "_trusted_terminal_result_marker", None) is _TRUSTED_TERMINAL_RESULT_MARKER


def _has_thread_unsafe_action_name(name: str) -> bool:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", camel_split)
    normalized = re.sub(r"[^a-z0-9]+", "_", camel_split.lower())
    parts = {part for part in normalized.split("_") if part}
    if parts == {"create"}:
        return True
    for part in parts:
        if part in THREAD_UNSAFE_DIRECT_ACTIONS:
            return True
        for action in THREAD_UNSAFE_ACTIONS:
            suffix = part.removeprefix(action) if part.startswith(action) else ""
            if suffix in FILE_RESOURCE_TERMS:
                return True
    if "create" in parts and parts & FILE_RESOURCE_TERMS:
        return True
    return False


def _schema_has_thread_policy_path(
    schema: Any,
    property_name: str = "",
    inherited_path_name: str = "",
) -> bool:
    if not isinstance(schema, dict):
        return False
    path_name = property_name if _is_thread_policy_path_name(property_name) else inherited_path_name
    declared_type = schema.get("type")
    if declared_type == "object":
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            return False
        return any(
            _schema_has_thread_policy_path(item_schema, name, path_name)
            for name, item_schema in properties.items()
        )
    if declared_type == "array":
        return _schema_has_thread_policy_path(schema.get("items"), property_name, path_name)
    if declared_type == "string":
        return _is_thread_policy_path_schema(path_name or property_name, schema)
    return False


def _is_thread_policy_path_schema(property_name: str, schema: dict[str, Any]) -> bool:
    if schema.get("format") in PATH_SCHEMA_FORMATS:
        return True
    return _is_thread_policy_path_name(property_name)


def _is_thread_policy_path_name(property_name: str) -> bool:
    normalized = _normalize_schema_name(property_name)
    return (
        normalized in THREAD_POLICY_PATH_NAMES
        or normalized.endswith("_path")
        or normalized.endswith("_paths")
    )


def _normalize_schema_name(name: str) -> str:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", camel_split)
    return camel_split.lower().replace("-", "_")


def _sanitize_model_facing_schema(value: Any) -> Any:
    if isinstance(value, dict):
        property_name_map: dict[Any, Any] = {}
        properties = value.get("properties")
        if isinstance(properties, dict):
            used_property_names: dict[Any, Any] = {}
            for property_name in properties:
                safe_property_name = neutralize_model_facing_metadata_key(property_name)
                safe_property_name = _dedupe_model_facing_key(used_property_names, safe_property_name)
                used_property_names[safe_property_name] = True
                property_name_map[property_name] = safe_property_name

        sanitized: dict[Any, Any] = {}
        for key, item in value.items():
            safe_key = neutralize_model_facing_metadata_key(key)
            safe_key = _dedupe_model_facing_key(sanitized, safe_key)
            if key == "properties" and isinstance(item, dict):
                sanitized[safe_key] = {
                    property_name_map[property_name]: _sanitize_model_facing_schema(property_schema)
                    for property_name, property_schema in item.items()
                }
                continue
            if key == "required" and isinstance(item, list):
                sanitized[safe_key] = [
                    property_name_map.get(required_name, neutralize_model_facing_metadata_key(required_name))
                    if isinstance(required_name, str)
                    else neutralize_model_facing_metadata(required_name)
                    for required_name in item
                ]
                continue
            sanitized[safe_key] = _sanitize_model_facing_schema(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_model_facing_schema(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_model_facing_schema(item) for item in value)
    return neutralize_model_facing_metadata(value)


def _dedupe_model_facing_key(existing: dict[Any, Any], key: Any) -> Any:
    if key not in existing:
        return key
    base = str(key) or "key"
    index = 2
    candidate = f"{base}_{index}"
    while candidate in existing:
        index += 1
        candidate = f"{base}_{index}"
    return candidate


def _validate_value(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']!r}")
        return errors

    expected = schema.get("type")
    if expected == "object":
        if not isinstance(value, dict):
            return [f"{path} must be an object"]
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                errors.append(f"{path}.{name} is required")
        if schema.get("additionalProperties") is False:
            for name in value:
                if name not in properties:
                    errors.append(f"{path}.{name} is not allowed")
        for name, item in value.items():
            if name in properties:
                errors.extend(_validate_value(item, properties[name], f"{path}.{name}"))
        return errors
    if expected == "array":
        if not isinstance(value, list):
            return [f"{path} must be an array"]
        min_items = _number_constraint(schema, "minItems")
        max_items = _number_constraint(schema, "maxItems")
        if min_items is not None and len(value) < min_items:
            errors.append(f"{path} must contain >= {int(min_items)} items")
        if max_items is not None and len(value) > max_items:
            errors.append(f"{path} must contain <= {int(max_items)} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                errors.extend(_validate_value(item, item_schema, f"{path}[{index}]"))
        return errors
    if expected == "string":
        if not isinstance(value, str):
            errors.append(f"{path} must be a string")
        else:
            errors.extend(_validate_string_constraints(value, schema, path))
    elif expected == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            errors.append(f"{path} must be an integer")
        else:
            errors.extend(_validate_number_constraints(value, schema, path))
    elif expected == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"{path} must be a number")
        else:
            errors.extend(_validate_number_constraints(value, schema, path))
    elif expected == "boolean" and not isinstance(value, bool):
        errors.append(f"{path} must be a boolean")
    elif expected == "null" and value is not None:
        errors.append(f"{path} must be null")
    return errors


def _validate_string_constraints(value: str, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    min_length = _number_constraint(schema, "minLength")
    max_length = _number_constraint(schema, "maxLength")
    if min_length is not None and len(value) < min_length:
        errors.append(f"{path} length must be >= {int(min_length)}")
    if max_length is not None and len(value) > max_length:
        errors.append(f"{path} length must be <= {int(max_length)}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str):
        try:
            if re.search(pattern, value) is None:
                errors.append(f"{path} must match pattern {pattern!r}")
        except re.error:
            errors.append(f"{path} has invalid schema pattern {pattern!r}")
    return errors


def _validate_number_constraints(value: int | float, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    minimum = _number_constraint(schema, "minimum")
    maximum = _number_constraint(schema, "maximum")
    exclusive_minimum = _number_constraint(schema, "exclusiveMinimum")
    exclusive_maximum = _number_constraint(schema, "exclusiveMaximum")
    if minimum is not None and value < minimum:
        errors.append(f"{path} must be >= {_format_number(minimum)}")
    if maximum is not None and value > maximum:
        errors.append(f"{path} must be <= {_format_number(maximum)}")
    if exclusive_minimum is not None and value <= exclusive_minimum:
        errors.append(f"{path} must be > {_format_number(exclusive_minimum)}")
    if exclusive_maximum is not None and value >= exclusive_maximum:
        errors.append(f"{path} must be < {_format_number(exclusive_maximum)}")
    return errors


def _number_constraint(schema: dict[str, Any], key: str) -> int | float | None:
    value = schema.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _format_number(value: int | float) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)

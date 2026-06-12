from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any, Callable

TOOL_EXECUTION_MODES = {"process", "thread"}


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    callable: Callable[..., Any]
    schema: dict[str, Any]
    description: str = ""
    timeout_sec: float | None = None
    execution_mode: str = "process"


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

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
                "name": definition.name,
                "description": definition.description,
                "schema": copy.deepcopy(definition.schema),
            }
            for definition in self._tools.values()
        ]


def _copy_and_validate_strict_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValueError("tool schema must be a JSON-schema object")
    copied = copy.deepcopy(schema)
    if copied.get("type") != "object":
        raise ValueError("tool schema must have type 'object'")
    if not isinstance(copied.get("properties"), dict):
        raise ValueError("tool schema must define properties")
    if copied.get("additionalProperties") is not False:
        raise ValueError("strict tool schema must set additionalProperties to False")
    required = copied.get("required", [])
    if not isinstance(required, list):
        raise ValueError("tool schema required field must be a list")
    missing_properties = [name for name in required if name not in copied["properties"]]
    if missing_properties:
        raise ValueError(f"required properties are not defined: {missing_properties}")
    return copied


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

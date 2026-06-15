from __future__ import annotations

import json
import multiprocessing
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from queue import Empty
from typing import Any

from .processes import prepare_current_process_group, terminate_process_tree
from .safety import PathValidationError, SafetyGuard
from .schemas import ToolResult
from .tool_registry import ToolDefinition
from .tool_registry import ToolRegistry
from .tool_registry import has_trusted_terminal_result
from .tool_registry import validate_legacy_definition_contract

TERMINAL_STRUCTURED_FAILURE_CODES = {"stalled", "timeout"}


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        safety_guard: SafetyGuard | None = None,
        default_timeout_sec: float = 30,
    ) -> None:
        self.registry = registry
        registry_guard = getattr(registry, "safety_guard", None)
        self.safety_guard = safety_guard or registry_guard or SafetyGuard()
        self.default_timeout_sec = default_timeout_sec

    def execute(self, tool_name: str, raw_args: Any) -> ToolResult:
        started = time.perf_counter()
        args, parse_error = _coerce_args(raw_args)
        if parse_error:
            return self._failed(tool_name, {}, parse_error, "invalid_json", started, executed=False)

        definition = self.registry.get(tool_name)
        if definition is None:
            return self._failed(
                tool_name,
                args,
                f"tool {tool_name!r} is not allowlisted",
                "tool_not_allowed",
                started,
                executed=False,
            )

        contract_error = validate_legacy_definition_contract(definition)
        if contract_error:
            return self._failed(
                tool_name,
                args,
                contract_error,
                "validation_error",
                started,
                executed=False,
            )

        validation_errors = self.registry.validate_args(tool_name, args)
        if validation_errors:
            return self._failed(
                tool_name,
                args,
                "; ".join(validation_errors),
                "validation_error",
                started,
                executed=False,
            )

        path_error = self._validate_path_args(definition, args)
        if path_error:
            return self._failed(
                tool_name,
                args,
                path_error,
                "path_validation_error",
                started,
                executed=False,
            )

        timeout = definition.timeout_sec or self.default_timeout_sec
        if definition.execution_mode == "thread":
            output, error_code, error, executed = _run_thread_tool(definition.callable, args, timeout)
        else:
            output, error_code, error, executed = _run_process_tool(definition.callable, args, timeout)

        if error_code:
            return self._failed(tool_name, args, error or "", error_code, started, executed=executed)

        structured_failure = _structured_failure(definition, output)
        if structured_failure is not None:
            structured_error_code, structured_error = structured_failure
            return self._failed(
                tool_name,
                args,
                structured_error,
                structured_error_code,
                started,
                executed=executed,
                output=output,
            )

        return ToolResult(
            tool_name=tool_name,
            args=args,
            ok=True,
            output=self.safety_guard.wrap_untrusted_output(tool_name, output),
            elapsed_ms=_elapsed_ms(started),
            executed=True,
        )

    def _failed(
        self,
        tool_name: str,
        args: dict[str, Any],
        error: str,
        error_code: str,
        started: float,
        *,
        executed: bool,
        output: Any = None,
    ) -> ToolResult:
        return ToolResult(
            tool_name=tool_name,
            args=args,
            ok=False,
            output=self.safety_guard.wrap_untrusted_output(tool_name, output) if output is not None else None,
            error=error,
            error_code=error_code,
            elapsed_ms=_elapsed_ms(started),
            executed=executed,
        )

    def _validate_path_args(self, definition: ToolDefinition, args: dict[str, Any]) -> str | None:
        for value in _iter_path_arg_values(args, definition.schema):
            try:
                self.safety_guard.validate_path(value)
            except PathValidationError as exc:
                return str(exc)
        return None


def _coerce_args(raw_args: Any) -> tuple[dict[str, Any], str | None]:
    if isinstance(raw_args, dict):
        return raw_args, None
    if isinstance(raw_args, str):
        try:
            loaded = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            return {}, f"tool args are not valid JSON: {exc.msg}"
        if not isinstance(loaded, dict):
            return {}, "tool args JSON must decode to an object"
        return loaded, None
    return {}, "tool args must be an object or JSON object string"


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _structured_failure(definition: ToolDefinition, output: Any) -> tuple[str, str] | None:
    if not isinstance(output, dict) or output.get("ok") is not False:
        return None
    status = _safe_failure_code(output.get("status") or output.get("error_code"), definition, output)
    stderr_tail = str(output.get("stderr_tail") or "").strip()
    if stderr_tail:
        return status, stderr_tail
    recovery = output.get("recovery")
    if isinstance(recovery, dict) and recovery.get("reason"):
        return status, f"tool returned {status}: {recovery['reason']}"
    return status, f"tool returned {status}"


def _safe_failure_code(value: Any, definition: ToolDefinition, output: Any) -> str:
    if not isinstance(value, str):
        return "tool_failed"
    normalized = value.strip().lower().replace("-", "_")
    if normalized == "failed":
        return "tool_failed"
    if normalized == "tool_failed":
        return normalized
    if normalized in TERMINAL_STRUCTURED_FAILURE_CODES and _is_trusted_terminal_failure(
        definition,
        output,
        normalized,
    ):
        return normalized
    return "tool_failed"


def _is_trusted_terminal_failure(definition: ToolDefinition, output: Any, status: str) -> bool:
    if not has_trusted_terminal_result(definition) or not isinstance(output, dict):
        return False
    if output.get("status") != status or output.get("ok") is not False:
        return False
    command = output.get("command")
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        return False
    return (
        _is_non_bool_int(output.get("stdout_length"))
        and _is_non_bool_int(output.get("stderr_length"))
        and _is_non_bool_number(output.get("elapsed_ms"))
        and _is_sha256(output.get("stdout_sha256"))
        and _is_sha256(output.get("stderr_sha256"))
    )


def _is_non_bool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_non_bool_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _run_thread_tool(callable_, args: dict[str, Any], timeout: float) -> tuple[Any, str | None, str | None, bool]:
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(callable_, **args)
    try:
        return future.result(timeout=timeout), None, None, True
    except TimeoutError:
        future.cancel()
        return None, "timeout", f"tool timed out after {timeout} seconds", True
    except Exception as exc:
        return None, "tool_exception", str(exc), True
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def _run_process_tool(callable_, args: dict[str, Any], timeout: float) -> tuple[Any, str | None, str | None, bool]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=_process_tool_worker, args=(callable_, args, result_queue))

    try:
        process.start()
    except Exception as exc:
        return (
            None,
            "tool_spawn_error",
            f"tool cannot run in hard-timeout process mode: {exc}",
            False,
        )

    process.join(timeout)
    if process.is_alive():
        terminate_process_tree(process.pid or -1, wait_process=process.join, timeout=1)
        process.join(1)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(1)
        result_queue.close()
        return None, "timeout", f"tool timed out after {timeout} seconds", True

    try:
        status, payload = result_queue.get_nowait()
    except Empty:
        exit_code = process.exitcode
        return None, "tool_exception", f"tool process exited without a result; exit code {exit_code}", True
    finally:
        result_queue.close()

    if status == "ok":
        return payload, None, None, True
    return None, "tool_exception", str(payload), True


def _process_tool_worker(callable_, args: dict[str, Any], result_queue) -> None:
    prepare_current_process_group()
    try:
        result_queue.put(("ok", callable_(**args)))
    except Exception as exc:
        result_queue.put(("error", f"{exc.__class__.__name__}: {exc}"))


def _iter_path_arg_values(
    value: Any,
    schema: dict[str, Any],
    property_name: str = "",
    inherited_path_name: str = "",
):
    expected = schema.get("type")
    path_name = property_name if _is_path_property_name(property_name) else inherited_path_name
    if expected == "object" and isinstance(value, dict):
        properties = schema.get("properties", {})
        for name, item in value.items():
            item_schema = properties.get(name)
            if isinstance(item_schema, dict):
                yield from _iter_path_arg_values(item, item_schema, name, path_name)
        return
    if expected == "array" and isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for item in value:
                yield from _iter_path_arg_values(item, item_schema, property_name, path_name)
        return
    if expected == "string" and isinstance(value, str) and _is_path_schema(path_name or property_name, schema):
        yield Path(value)


def _is_path_schema(property_name: str, schema: dict[str, Any]) -> bool:
    if schema.get("format") in {"path", "file-path", "directory-path", "workspace-path"}:
        return True
    return _is_path_property_name(property_name)


def _is_path_property_name(property_name: str) -> bool:
    normalized = _normalize_schema_name(property_name)
    return normalized in {
        "path",
        "file",
        "files",
        "file_path",
        "filepath",
        "paths",
        "dir",
        "directory",
        "directory_path",
        "target",
        "destination",
        "source",
    } or normalized.endswith("_path") or normalized.endswith("_paths")


def _normalize_schema_name(name: str) -> str:
    camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    camel_split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", camel_split)
    return camel_split.lower().replace("-", "_")

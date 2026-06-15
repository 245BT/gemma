from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import RuntimeConfig
from .safety import SafetyGuard, neutralize_model_facing_metadata
from .terminal import TerminalCommandRunner
from .tool_registry import mark_trusted_terminal_result
from .tool_registry import ToolRegistry


TERMINAL_COMMAND_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ["run", "status", "kill"]},
        "command": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "minItems": 1,
            "maxItems": 64,
        },
        "cwd": {"type": "string", "format": "directory-path"},
        "timeout_sec": {"type": "number", "minimum": 0.1, "maximum": 1800},
        "idle_timeout_sec": {"type": "number", "minimum": 0.1, "maximum": 600},
        "background_after_sec": {"type": "number", "minimum": 0.1, "maximum": 300},
        "job_id": {"type": "string", "minLength": 1, "maxLength": 100},
        "kill_reason": {"type": "string", "minLength": 1, "maxLength": 300},
    },
    "required": [],
    "additionalProperties": False,
}

DUCKDUCKGO_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 500},
        "max_results": {"type": "integer", "minimum": 1, "maximum": 10},
        "recency_days": {"type": "integer", "minimum": 1, "maximum": 3650},
    },
    "required": ["query"],
    "additionalProperties": False,
}

CONTEXT7_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "library": {"type": "string", "minLength": 1, "maxLength": 200},
        "query": {"type": "string", "minLength": 1, "maxLength": 500},
        "library_id": {"type": "string", "minLength": 2, "maxLength": 300},
    },
    "required": ["library", "query"],
    "additionalProperties": False,
}


def build_default_tool_registry(
    *,
    workspace_roots: list[str | Path] | None = None,
    runtime_config: RuntimeConfig | None = None,
    terminal_runner: TerminalCommandRunner | None = None,
    duckduckgo_fetcher: Any | None = None,
    context7_runner: Any | None = None,
    context7_cwd: str | Path | None = None,
) -> ToolRegistry:
    config = runtime_config or RuntimeConfig()
    guard_roots = workspace_roots if workspace_roots is not None else list(config.workspace_roots)
    guard = SafetyGuard(guard_roots)
    registry = ToolRegistry()
    registry.safety_guard = guard

    if config.terminal_tools_enabled:
        runner = terminal_runner or TerminalCommandRunner()

        def terminal_command(
            command: list[str] | None = None,
            cwd: str | None = None,
            action: str = "run",
            timeout_sec: float | None = None,
            idle_timeout_sec: float | None = None,
            background_after_sec: float | None = None,
            job_id: str | None = None,
            kill_reason: str | None = None,
        ) -> dict[str, Any]:
            action_name = action or "run"
            if action_name == "status":
                if not job_id:
                    raise ValueError("job_id is required for terminal status")
                result = runner.job_status(job_id)
            elif action_name == "kill":
                if not job_id:
                    raise ValueError("job_id is required for terminal kill")
                result = runner.kill_job(job_id, reason=kill_reason or "agent requested termination")
            elif action_name == "run":
                if not command:
                    raise ValueError("command is required for terminal run")
                resolved_cwd = guard.validate_path(cwd or Path.cwd())
                result = runner.run(
                    command,
                    cwd=resolved_cwd,
                    timeout_sec=timeout_sec,
                    idle_timeout_sec=idle_timeout_sec,
                    background_after_sec=background_after_sec,
                )
            else:
                raise ValueError("action must be run, status, or kill")
            return asdict(result)

        definition = registry.register(
            "terminal_command",
            terminal_command,
            TERMINAL_COMMAND_SCHEMA,
            description=(
                "Run, poll, or kill a terminal command. Commands have a 30 minute hard cap. "
                "Long or idle commands return a live job_id so the agent can inspect status "
                "and decide whether to kill rather than relying on a shell-side 10s timeout."
            ),
            execution_mode="thread",
            timeout_sec=3700,
        )
        mark_trusted_terminal_result(definition)

    def context7_search(
        library: str,
        query: str,
        library_id: str | None = None,
    ) -> dict[str, Any]:
        return _context7_search(
            library,
            query,
            library_id=library_id,
            runner=context7_runner,
            cwd=context7_cwd,
        )

    registry.register(
        "context7_search",
        context7_search,
        CONTEXT7_SEARCH_SCHEMA,
        description=(
            "Search Context7 documentation for software, code, package, dependency, "
            "install, command, API, SDK, and framework questions before retrying commands."
        ),
        execution_mode="thread",
        timeout_sec=75,
    )

    def duckduckgo_search(
        query: str,
        max_results: int = 5,
        recency_days: int | None = None,
    ) -> dict[str, Any]:
        from duckduckgo_mcp import fetch_duckduckgo_html, search_duckduckgo_response

        fetcher = duckduckgo_fetcher or fetch_duckduckgo_html
        return search_duckduckgo_response(
            query,
            max_results=max_results,
            fetcher=fetcher,
            timeout=15,
            recency_days=recency_days,
        )

    registry.register(
        "duckduckgo_search",
        duckduckgo_search,
        DUCKDUCKGO_SEARCH_SCHEMA,
        description=(
            "Search DuckDuckGo for public news, people, and non-code public info; "
            "not for software, code, package, install, command, API, SDK, or framework questions."
        ),
        execution_mode="thread",
        timeout_sec=35,
    )
    return registry


def _context7_search(
    library: str,
    query: str,
    *,
    library_id: str | None,
    runner: Any | None,
    cwd: str | Path | None,
) -> dict[str, Any]:
    library_text = _clean_context7_text(library)
    query_text = _clean_context7_text(query)
    selected_library_id = _clean_context7_text(library_id) if library_id else ""
    if not library_text or not query_text:
        return _context7_response(
            "invalid_input",
            library=library_text,
            query=query_text,
            message="Context7 library and query must be non-empty strings.",
        )

    resolution = None
    if not selected_library_id:
        library_result = _run_context7_cli(
            ["library", library_text, query_text, "--json"],
            runner=runner,
            timeout=30,
            cwd=cwd,
        )
        if not library_result["ok"]:
            return _context7_response(
                "unavailable",
                library=library_text,
                query=query_text,
                message=library_result["message"],
                resolution=library_result,
            )
        resolution = _parse_context7_json(library_result["stdout"])
        selected_library_id = _select_context7_library_id(resolution, library_text)
        if not selected_library_id:
            return _context7_response(
                "no_library",
                library=library_text,
                query=query_text,
                message=f'Context7 returned no usable library id for "{library_text}".',
                resolution=resolution,
            )

    docs_result = _run_context7_cli(
        ["docs", selected_library_id, query_text, "--json"],
        runner=runner,
        timeout=45,
        cwd=cwd,
    )
    if not docs_result["ok"]:
        return _context7_response(
            "unavailable",
            library=library_text,
            query=query_text,
            library_id=selected_library_id,
            message=docs_result["message"],
            resolution=resolution,
            docs=docs_result,
        )

    return _context7_response(
        "ok",
        library=library_text,
        query=query_text,
        library_id=selected_library_id,
        message=f'Context7 returned documentation for "{library_text}".',
        resolution=resolution,
        docs=_parse_context7_json(docs_result["stdout"]),
    )


def _clean_context7_text(value: Any) -> str:
    return str(value or "").strip()


def _run_context7_cli(
    args: list[str],
    *,
    runner: Any | None,
    timeout: float,
    cwd: str | Path | None,
) -> dict[str, Any]:
    cwd_path = Path(cwd).resolve() if cwd is not None else Path.cwd().resolve()
    if runner is None:
        command = _context7_command(args, cwd_path)
        if command is None:
            return {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": "",
                "message": "ctx7 CLI is not installed. Run npm install in the Gemma workspace.",
            }
        completed = subprocess.run(
            command,
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        result = _coerce_context7_completed(completed)
    else:
        result = _coerce_context7_completed(runner(args, timeout=timeout, cwd=str(cwd_path)))

    if result["returncode"] != 0:
        stderr = result["stderr"].strip()
        return {
            **result,
            "ok": False,
            "message": stderr or f"ctx7 exited with code {result['returncode']}.",
        }
    return {**result, "ok": True, "message": "ctx7 command completed."}


def _context7_command(args: list[str], cwd: Path) -> list[str] | None:
    executable = "ctx7.cmd" if os.name == "nt" else "ctx7"
    for local in _context7_local_bin_candidates(cwd, executable):
        if local.exists():
            return [str(local), *args]
    discovered = shutil.which(executable) or shutil.which("ctx7")
    if discovered:
        return [discovered, *args]
    npx = shutil.which("npx.cmd" if os.name == "nt" else "npx") or shutil.which("npx")
    if npx:
        return [npx, "-y", "ctx7@latest", *args]
    return None


def _context7_local_bin_candidates(cwd: Path, executable: str) -> list[Path]:
    module_root = Path(__file__).resolve().parents[1]
    roots = [cwd, *cwd.parents, module_root]
    candidates: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        candidate = root / "node_modules" / ".bin" / executable
        if candidate in seen:
            continue
        seen.add(candidate)
        candidates.append(candidate)
    return candidates


def _coerce_context7_completed(result: Any) -> dict[str, Any]:
    if isinstance(result, subprocess.CompletedProcess):
        return {
            "returncode": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }
    if isinstance(result, dict):
        return {
            "returncode": int(result.get("returncode", result.get("exit_code", 0))),
            "stdout": str(result.get("stdout", "")),
            "stderr": str(result.get("stderr", "")),
        }
    return {"returncode": 0, "stdout": str(result), "stderr": ""}


def _parse_context7_json(stdout: str) -> Any:
    text = stdout.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": _sanitize_context7_raw_text(text)}


def _sanitize_context7_raw_text(text: str) -> str:
    sanitized = text
    for marker in ("<|tool_call|>", "<tool_call|>", "<|channel>", "<channel|>"):
        sanitized = sanitized.replace(marker, "[redacted_context7_marker]")
    return neutralize_model_facing_metadata(sanitized)


def _select_context7_library_id(payload: Any, requested_library: str = "") -> str:
    candidates: list[Any] = []
    if isinstance(payload, list):
        candidates.extend(payload)
    elif isinstance(payload, dict):
        for key in ("results", "libraries", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        candidates.append(payload)

    requested = _normalize_context7_label(requested_library)
    if requested:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = _candidate_context7_library_id(candidate)
            if not candidate_id:
                continue
            labels = [
                candidate.get("title"),
                candidate.get("name"),
                candidate.get("library"),
            ]
            if any(_normalize_context7_label(label) == requested for label in labels):
                return candidate_id

        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_id = _candidate_context7_library_id(candidate)
            if candidate_id and _normalize_context7_label(candidate_id.rsplit("/", 1)[-1]) == requested:
                return candidate_id

    for candidate in candidates:
        if isinstance(candidate, str) and candidate.startswith("/"):
            return candidate
        if not isinstance(candidate, dict):
            continue
        candidate_id = _candidate_context7_library_id(candidate)
        if candidate_id:
            return candidate_id
    return ""


def _candidate_context7_library_id(candidate: dict[str, Any]) -> str:
    for key in ("id", "library_id", "libraryId", "context7CompatibleLibraryID"):
        value = candidate.get(key)
        if isinstance(value, str) and value.startswith("/"):
            return value
    return ""


def _normalize_context7_label(value: Any) -> str:
    return "".join(character for character in str(value or "").lower() if character.isalnum())


def _context7_response(
    status: str,
    *,
    library: str,
    query: str,
    message: str,
    library_id: str = "",
    resolution: Any = None,
    docs: Any = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "source": "Context7",
        "library": library,
        "query": query,
        "library_id": library_id,
        "message": message,
        "resolution": resolution,
        "docs": docs,
        "untrusted": True,
    }

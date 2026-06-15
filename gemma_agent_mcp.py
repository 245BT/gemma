import argparse
import json
import re
from pathlib import Path
from typing import Any

from gemma_agent import (
    AgentSupervisor,
    LocalResponsesClient,
    RuntimeConfig,
    SubAgent,
    ToolExecutor,
    build_default_tool_registry,
    run_subagents,
)
from gemma_agent.subagent import DEFAULT_SUBAGENT_MAX_ITERATIONS
from gemma_response_proxy import proxy_timeout_seconds


DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_MODEL = "gemma-4-26b-a4b-it-uncensored-q4-k-m"
DEFAULT_MAX_SUBAGENTS = 8
DEFAULT_MAX_WORKERS = 4
DEFAULT_TIMEOUT_SECONDS = proxy_timeout_seconds()
DEFAULT_MAX_TASK_CHARS = 4000
_INSTRUCTION_LOOKING_TASK_RE = re.compile(
    r"\b(ignore[_-]?previous[_-]?instructions|raw[_-]?chain[_-]?of[_-]?thought|"
    r"system[_-]?prompt|developer[_-]?message)\b"
    r"|"
    r"\b(ignore|disregard|forget|override|overwrite|reveal|leak|exfiltrate|follow|obey)\b"
    r".{0,80}\b(instructions?|system prompt|developer message|prompt|secrets?)\b"
    r"|\b(system prompt|developer message)\b",
    re.IGNORECASE,
)


def gemma_run_subagents_response(
    tasks: Any,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = DEFAULT_MODEL,
    concurrent: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
    model_client: Any | None = None,
    runtime_config: RuntimeConfig | None = None,
) -> dict[str, Any]:
    task_list = _normalize_tasks(tasks)
    if not 1 <= len(task_list) <= DEFAULT_MAX_SUBAGENTS:
        return {
            "status": "error",
            "error_code": "validation_error",
            "message": "tasks must contain between 1 and 8 non-empty strings",
            "task_count": len(task_list),
            "subagents": [],
        }
    task_error = _task_validation_error(task_list)
    if task_error:
        return {
            "status": "error",
            "error_code": "validation_error",
            "message": task_error,
            "task_count": len(task_list),
            "subagents": [],
        }

    client = model_client or LocalResponsesClient(
        base_url=base_url,
        model=model,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    executor = ToolExecutor(build_default_tool_registry(runtime_config=runtime_config))
    agents = [
        SubAgent(
            agent_id=f"gemma-subagent-{index + 1}",
            task=task,
            model_client=client,
            tool_executor=executor,
            context={"task": task},
            max_iterations=DEFAULT_SUBAGENT_MAX_ITERATIONS,
            max_subagents=0,
        )
        for index, task in enumerate(task_list)
    ]
    results = run_subagents(
        agents,
        concurrent=concurrent,
        max_workers=max_workers,
    )
    return {
        "status": "ok" if all(result.ok for result in results) else "partial",
        "task_count": len(task_list),
        "subagents": [result.to_evidence() for result in results],
    }


def gemma_run_subagents(
    tasks: Any,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = DEFAULT_MODEL,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> str:
    return json.dumps(
        gemma_run_subagents_response(
            tasks,
            base_url=base_url,
            model=model,
            max_workers=max_workers,
        ),
        ensure_ascii=True,
    )


def context7_search_response(
    library: str,
    query: str,
    library_id: str | None = None,
    *,
    context7_runner: Any | None = None,
) -> dict[str, Any]:
    args = {"library": library, "query": query}
    if library_id:
        args["library_id"] = library_id
    executor = ToolExecutor(
        build_default_tool_registry(
            context7_runner=context7_runner,
            context7_cwd=Path(__file__).resolve().parent,
        )
    )
    result = executor.execute("context7_search", args)
    if result.ok and isinstance(result.output, dict):
        content = result.output.get("content")
        if isinstance(content, dict):
            return content
    return {
        "status": "error",
        "error_code": result.error_code or "context7_search_failed",
        "message": result.error or "Context7 search failed.",
        "untrusted": True,
    }


def context7_search(library: str, query: str, library_id: str | None = None) -> str:
    return json.dumps(
        context7_search_response(library, query, library_id=library_id),
        ensure_ascii=True,
    )


def _normalize_tasks(tasks: Any) -> list[str]:
    if isinstance(tasks, str):
        try:
            decoded = json.loads(tasks)
        except json.JSONDecodeError:
            decoded = [tasks]
        tasks = decoded
    if not isinstance(tasks, list):
        return []
    normalized = []
    for item in tasks:
        if not isinstance(item, str):
            return []
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized


def _task_validation_error(tasks: list[str]) -> str | None:
    for index, task in enumerate(tasks, start=1):
        if len(task) > DEFAULT_MAX_TASK_CHARS:
            return f"task length exceeds {DEFAULT_MAX_TASK_CHARS} characters at index {index}"
        if _INSTRUCTION_LOOKING_TASK_RE.search(task):
            return f"task contains instruction-looking control text at index {index}"
    return None


def build_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gemma-agent", json_response=True)

    @server.tool(name="gemma_run_subagents")
    def gemma_run_subagents_tool(tasks: list[str], max_workers: int = DEFAULT_MAX_WORKERS) -> str:
        return gemma_run_subagents(tasks, max_workers=max_workers)

    @server.tool(name="context7_search")
    def context7_search_tool(library: str, query: str, library_id: str | None = None) -> str:
        """Fetch Context7 docs for software/code/package/install/API questions."""
        return context7_search(library, query, library_id=library_id)

    return server


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the local Gemma agent MCP server.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parse_args()
    build_server().run(transport="stdio")

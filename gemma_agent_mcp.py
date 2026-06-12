import argparse
import json
from typing import Any

from gemma_agent import (
    AgentSupervisor,
    LocalResponsesClient,
    SubAgent,
    ToolExecutor,
    ToolRegistry,
    run_subagents,
)


DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_MODEL = "gemma-4-26b-a4b-it-uncensored-q4-k-m"
DEFAULT_MAX_SUBAGENTS = 8
DEFAULT_MAX_WORKERS = 4
DEFAULT_TIMEOUT_SECONDS = 120


def gemma_run_subagents_response(
    tasks: Any,
    *,
    base_url: str = DEFAULT_BASE_URL,
    model: str | None = DEFAULT_MODEL,
    concurrent: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
    model_client: Any | None = None,
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

    client = model_client or LocalResponsesClient(
        base_url=base_url,
        model=model,
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    executor = ToolExecutor(ToolRegistry())
    agents = [
        SubAgent(
            agent_id=f"gemma-subagent-{index + 1}",
            task=task,
            model_client=client,
            tool_executor=executor,
            context={"task": task},
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


def build_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("gemma-agent", json_response=True)

    @server.tool(name="gemma_run_subagents")
    def gemma_run_subagents_tool(tasks: list[str], max_workers: int = DEFAULT_MAX_WORKERS) -> str:
        return gemma_run_subagents(tasks, max_workers=max_workers)

    return server


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the local Gemma agent MCP server.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    parse_args()
    build_server().run(transport="stdio")

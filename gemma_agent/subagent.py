from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

from .schemas import SubAgentResult

DEFAULT_MAX_SUBAGENT_WORKERS = 8


class SubAgent:
    def __init__(
        self,
        *,
        agent_id: str,
        task: str,
        model_client: Any,
        tool_executor: Any,
        context: dict[str, Any] | None = None,
        max_iterations: int = 4,
        max_subagents: int = DEFAULT_MAX_SUBAGENT_WORKERS,
        subagent_budget: Any | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.task = task
        self.model_client = model_client
        self.tool_executor = tool_executor
        self.context = copy.deepcopy(context or {})
        self.max_iterations = max_iterations
        self.max_subagents = max_subagents
        self.subagent_budget = subagent_budget

    def run(self) -> SubAgentResult:
        from .supervisor import AgentSupervisor

        try:
            result = AgentSupervisor(
                self.model_client,
                self.tool_executor,
                max_iterations=self.max_iterations,
                max_subagents=self.max_subagents,
                subagent_budget=self.subagent_budget,
            ).run(self.task, context=copy.deepcopy(self.context))
        except Exception as exc:
            return SubAgentResult(
                agent_id=self.agent_id,
                task=self.task,
                ok=False,
                error=str(exc),
            )
        return SubAgentResult(
            agent_id=self.agent_id,
            task=self.task,
            final=result.final or "",
            ok=result.ok,
            tool_results=list(result.tool_results),
            error=None if result.ok else "subagent did not produce a final action",
        )


def run_subagents(
    subagents: Iterable[Any],
    *,
    concurrent: bool = False,
    max_workers: int | None = None,
) -> list[SubAgentResult]:
    agents = list(subagents)
    if not concurrent or len(agents) <= 1:
        return [_run_one(agent) for agent in agents]
    worker_count = _bounded_worker_count(len(agents), max_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        return list(pool.map(_run_one, agents))


def _run_one(agent: Any) -> SubAgentResult:
    return agent.run()


def _bounded_worker_count(agent_count: int, requested_workers: int | None) -> int:
    if agent_count <= 0:
        return 1
    if isinstance(requested_workers, bool) or not isinstance(requested_workers, int) or requested_workers < 1:
        requested_workers = DEFAULT_MAX_SUBAGENT_WORKERS
    return max(1, min(agent_count, requested_workers, DEFAULT_MAX_SUBAGENT_WORKERS))

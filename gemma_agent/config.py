from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str | None = None
    request_timeout_sec: float = 120
    default_tool_timeout_sec: float = 30
    max_iterations: int = 8
    max_subagents: int = 8
    workspace_roots: tuple[Path, ...] = field(default_factory=lambda: (Path.cwd(),))

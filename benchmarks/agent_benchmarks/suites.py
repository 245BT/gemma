from __future__ import annotations

from pathlib import Path

from setup_local_codex import MODEL_SLUG


def mode_args(mode: str) -> list[str]:
    if mode == "direct":
        return []
    if mode == "reasoning":
        return ["--reasoning"]
    raise ValueError("mode must be 'direct' or 'reasoning'")


def yolo_args(enabled: bool) -> list[str]:
    return ["--yolo"] if enabled else []


def codex_home(root: Path, mode: str) -> Path:
    if mode == "direct":
        return root / ".codex-local"
    if mode == "reasoning":
        return root / ".codex-local-reasoning"
    raise ValueError("mode must be 'direct' or 'reasoning'")


def model_slug() -> str:
    return MODEL_SLUG

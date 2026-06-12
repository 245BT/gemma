from __future__ import annotations

import re
from typing import Any

from .citations import CitationManager
from .memory import MemoryStore

DEFAULT_MAX_ENTRIES_PER_TIER = {
    "session_summary": 4,
    "project_map": 12,
    "task_scratchpad": 12,
    "evidence": 20,
    "long_term_cache": 12,
}
DEFAULT_MAX_CHARS_PER_TIER = {
    "session_summary": 4000,
    "project_map": 8000,
    "task_scratchpad": 6000,
    "evidence": 12000,
    "long_term_cache": 8000,
}
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class ContextBuilder:
    def __init__(
        self,
        memory: MemoryStore | None = None,
        citations: CitationManager | None = None,
        *,
        max_entries_per_tier: dict[str, int] | None = None,
        max_chars_per_tier: dict[str, int] | None = None,
    ) -> None:
        self.memory = MemoryStore() if memory is None else memory
        self.citations = CitationManager() if citations is None else citations
        self.max_entries_per_tier = {
            **DEFAULT_MAX_ENTRIES_PER_TIER,
            **(max_entries_per_tier or {}),
        }
        self.max_chars_per_tier = {
            **DEFAULT_MAX_CHARS_PER_TIER,
            **(max_chars_per_tier or {}),
        }

    def build(self, *, task: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        evidence = _dedupe(
            [
                *_contents(self.memory, "evidence"),
                *_contents(self.memory, "evidence_store"),
            ]
        )
        context: dict[str, Any] = {
            "task": task,
            "session_summary": self._select("session_summary", _contents(self.memory, "session_summary"), task),
            "project_map": self._select("project_map", _contents(self.memory, "project_map"), task),
            "task_scratchpad": self._select("task_scratchpad", _contents(self.memory, "task_scratchpad"), task),
            "evidence": self._select("evidence", evidence, task),
            "long_term_cache": self._select("long_term_cache", _contents(self.memory, "long_term_cache"), task),
            "citations": [citation.to_context() for citation in self.citations.entries()],
        }
        if extra:
            context["extra"] = dict(extra)
        return context

    def _select(self, tier: str, values: list[str], task: str) -> list[str]:
        max_entries = _positive_int(self.max_entries_per_tier.get(tier), len(values))
        max_chars = _positive_int(self.max_chars_per_tier.get(tier), sum(len(value) for value in values))
        ranked = sorted(enumerate(values), key=lambda item: (-_relevance(task, item[1]), item[0]))
        selected: list[str] = []
        used_chars = 0
        for _, value in ranked:
            if len(selected) >= max_entries:
                break
            if used_chars + len(value) > max_chars:
                continue
            selected.append(value)
            used_chars += len(value)
        return selected


def _contents(memory: MemoryStore, tier: str) -> list[str]:
    return [entry.content for entry in memory.entries(tier)]


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _relevance(task: str, value: str) -> int:
    task_tokens = _tokens(task)
    if not task_tokens:
        return 0
    return len(task_tokens.intersection(_tokens(value)))


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text) if len(token) > 2}


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return default
    return value

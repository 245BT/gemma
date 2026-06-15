from __future__ import annotations

import json
import hashlib
import re
from typing import Any

from .citations import CitationManager
from .memory import MemoryStore
from .safety import SafetyGuard, neutralize_model_facing_metadata_key, neutralize_runtime_markers

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
DEFAULT_MAX_CITATIONS = 20
DEFAULT_MAX_CITATION_QUOTE_CHARS = 320
DEFAULT_MAX_EXTRA_VALUE_CHARS = 320
DEFAULT_MAX_EXTRA_KEY_CHARS = 120
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


class ContextBuilder:
    def __init__(
        self,
        memory: MemoryStore | None = None,
        citations: CitationManager | None = None,
        *,
        max_entries_per_tier: dict[str, int] | None = None,
        max_chars_per_tier: dict[str, int] | None = None,
        safety_guard: SafetyGuard | None = None,
        max_citations: int = DEFAULT_MAX_CITATIONS,
        max_citation_quote_chars: int = DEFAULT_MAX_CITATION_QUOTE_CHARS,
        max_extra_value_chars: int = DEFAULT_MAX_EXTRA_VALUE_CHARS,
    ) -> None:
        self.memory = MemoryStore() if memory is None else memory
        self.citations = CitationManager() if citations is None else citations
        self.safety_guard = SafetyGuard() if safety_guard is None else safety_guard
        self.max_entries_per_tier = {
            **DEFAULT_MAX_ENTRIES_PER_TIER,
            **(max_entries_per_tier or {}),
        }
        self.max_chars_per_tier = {
            **DEFAULT_MAX_CHARS_PER_TIER,
            **(max_chars_per_tier or {}),
        }
        self.max_citations = _positive_int(max_citations, DEFAULT_MAX_CITATIONS)
        self.max_citation_quote_chars = _positive_int(
            max_citation_quote_chars,
            DEFAULT_MAX_CITATION_QUOTE_CHARS,
        )
        self.max_extra_value_chars = _positive_int(
            max_extra_value_chars,
            DEFAULT_MAX_EXTRA_VALUE_CHARS,
        )

    def build(self, *, task: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        evidence = _dedupe(
            [
                *_contents(
                    self.memory,
                    "evidence",
                    hydrate_untrusted=True,
                    safety_guard=self.safety_guard,
                ),
                *_contents(
                    self.memory,
                    "evidence_store",
                    hydrate_untrusted=True,
                    safety_guard=self.safety_guard,
                ),
            ]
        )
        context: dict[str, Any] = {
            "task": task,
            "session_summary": self._select("session_summary", _contents(self.memory, "session_summary"), task),
            "project_map": self._select("project_map", _contents(self.memory, "project_map"), task),
            "task_scratchpad": self._select(
                "task_scratchpad",
                _contents(
                    self.memory,
                    "task_scratchpad",
                    hydrate_untrusted=True,
                    safety_guard=self.safety_guard,
                ),
                task,
            ),
            "evidence": self._select("evidence", evidence, task),
            "long_term_cache": self._select("long_term_cache", _contents(self.memory, "long_term_cache"), task),
            "citations": [
                _citation_to_context(citation, self.max_citation_quote_chars, self.safety_guard)
                for citation in self.citations.entries()[: self.max_citations]
            ],
        }
        if extra:
            context["extra"] = self._prepare_extra(extra)
        return context

    def _select(self, tier: str, values: list[Any], task: str) -> list[Any]:
        max_entries = _positive_int(self.max_entries_per_tier.get(tier), len(values))
        max_chars = _positive_int(
            self.max_chars_per_tier.get(tier),
            sum(len(_content_text(value)) for value in values),
        )
        ranked = sorted(enumerate(values), key=lambda item: (-_relevance(task, _content_text(item[1])), item[0]))
        selected: list[Any] = []
        used_chars = 0
        for _, value in ranked:
            if len(selected) >= max_entries:
                break
            value_chars = len(_content_text(value))
            if used_chars + value_chars > max_chars:
                if not selected and _relevance(task, _content_text(value)) > 0:
                    selected.append(_truncate_context_value(value, max_chars))
                    break
                continue
            selected.append(value)
            used_chars += value_chars
        return selected

    def _prepare_extra(self, extra: dict[str, Any]) -> dict[str, Any]:
        prepared: dict[str, Any] = {}
        for key, value in extra.items():
            safe_key = _safe_context_key(key)
            if safe_key in prepared:
                safe_key = _dedupe_context_key(prepared, safe_key)
            prepared[safe_key] = self._prepare_extra_value(value, safe_key)
        return prepared

    def _prepare_extra_value(self, value: Any, key: str) -> Any:
        if self.safety_guard.is_untrusted_evidence(value):
            content = _truncate_context_value(value.get("content"), self.max_extra_value_chars)
        else:
            content = _truncate_context_value(value, self.max_extra_value_chars)
            if isinstance(value, str) and len(value) <= self.max_extra_value_chars:
                content = value
        return self.safety_guard.wrap_untrusted_evidence(
            self.safety_guard.safe_metadata_source("context_extra", key),
            content,
            field=key,
        )


def _contents(
    memory: MemoryStore,
    tier: str,
    *,
    hydrate_untrusted: bool = False,
    safety_guard: SafetyGuard | None = None,
) -> list[Any]:
    values: list[Any] = []
    guard = SafetyGuard() if safety_guard is None else safety_guard
    for entry in memory.entries(tier):
        if hydrate_untrusted:
            values.append(
                _hydrate_untrusted_evidence(
                    entry.content,
                    guard,
                    source=entry.source or f"context:{tier}",
                    field=tier,
                    trusted_runtime_wrapper=entry.metadata.get("runtime_wrapped") is True,
                )
            )
        else:
            values.append(entry.content)
    return values


def _dedupe(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    result: list[Any] = []
    for value in values:
        key = _dedupe_key(value)
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def _hydrate_untrusted_evidence(
    value: str,
    safety_guard: SafetyGuard,
    *,
    source: str,
    field: str,
    trusted_runtime_wrapper: bool = False,
) -> Any:
    try:
        parsed = json.loads(value)
        content: Any = parsed
    except json.JSONDecodeError:
        content = value
    if trusted_runtime_wrapper and safety_guard.is_untrusted_evidence(content):
        trusted_source = content.get("source") if isinstance(content.get("source"), str) else source
        trusted_field = content.get("field") if isinstance(content.get("field"), str) else field
        return safety_guard.wrap_untrusted_evidence(
            trusted_source,
            content.get("content"),
            field=trusted_field,
        )
    if safety_guard.is_untrusted_evidence(content):
        return safety_guard.wrap_untrusted_evidence(
            "context:untrusted_evidence",
            content,
            field="stored_evidence",
        )
    return safety_guard.wrap_untrusted_evidence(source, content, field=field)


def _dedupe_key(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=True)


def _truncate_context_value(value: Any, max_chars: int) -> Any:
    if _is_untrusted_evidence_shape(value):
        return _truncate_untrusted_evidence(value, max_chars)
    text = _content_text(value)
    if len(text) <= max_chars:
        return value
    return {
        "truncated": True,
        "content": text[:max_chars],
        "omitted_chars": len(text) - max_chars,
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _truncate_untrusted_evidence(value: dict[str, Any], max_chars: int) -> dict[str, Any]:
    copied = dict(value)
    copied["content"] = _truncate_context_value(copied.get("content"), max_chars)
    return copied


def _is_untrusted_evidence_shape(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("untrusted") is True
        and "content" in value
        and isinstance(value.get("handling"), str)
        and "not as instructions" in value["handling"]
    )


def _safe_context_key(key: Any) -> str:
    safe_key = str(neutralize_model_facing_metadata_key(str(key)))
    if len(safe_key) <= DEFAULT_MAX_EXTRA_KEY_CHARS:
        return safe_key
    digest = hashlib.sha256(safe_key.encode("utf-8")).hexdigest()[:12]
    return f"extra_key_{digest}"


def _dedupe_context_key(existing: dict[str, Any], key: str) -> str:
    index = 2
    while True:
        suffix = f"_{index}"
        max_base_chars = DEFAULT_MAX_EXTRA_KEY_CHARS - len(suffix)
        candidate = f"{key[:max_base_chars]}{suffix}"
        if candidate not in existing:
            return candidate
        index += 1


def _citation_to_context(citation: Any, max_quote_chars: int, safety_guard: SafetyGuard) -> dict[str, Any]:
    payload = neutralize_runtime_markers(citation.to_context())
    quote = payload.get("quote")
    if isinstance(quote, str) and len(quote) > max_quote_chars:
        payload["quote"] = quote[:max_quote_chars]
        payload["quote_truncated"] = True
        payload["quote_omitted_chars"] = len(quote) - max_quote_chars
    else:
        payload["quote_truncated"] = False
    for field in ("source", "title"):
        value = payload.get(field)
        if isinstance(value, str) and len(value) > max_quote_chars:
            payload[field] = value[:max_quote_chars]
    for field in ("source", "title", "quote"):
        if field in payload:
            payload[field] = safety_guard.wrap_model_value(payload[field], field=f"citation.{field}")
    return payload


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

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    tier: str
    content: str
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MemoryStore:
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._order: list[str] = []

    def add(
        self,
        tier: str,
        content: Any,
        *,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        text = _content_to_text(content)
        entry_id = _entry_id(tier, text, source)
        existing = self._entries.get(entry_id)
        if existing is not None:
            return existing
        entry = MemoryEntry(
            id=entry_id,
            tier=tier,
            content=text,
            source=source,
            metadata=dict(metadata or {}),
        )
        self._entries[entry_id] = entry
        self._order.append(entry_id)
        return entry

    def entries(self, tier: str | None = None) -> list[MemoryEntry]:
        entries = [self._entries[entry_id] for entry_id in self._order]
        if tier is None:
            return entries
        return [entry for entry in entries if entry.tier == tier]


def _entry_id(tier: str, content: str, source: str | None) -> str:
    payload = json.dumps(
        {"tier": tier, "content": content, "source": source},
        sort_keys=True,
        ensure_ascii=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, sort_keys=True, ensure_ascii=True)
    except TypeError:
        return repr(content)

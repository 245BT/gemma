from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    id: str
    source: str
    title: str | None = None
    quote: str | None = None

    def to_context(self) -> dict[str, str | None]:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "quote": self.quote,
        }


class CitationManager:
    def __init__(self) -> None:
        self._by_key: dict[tuple[str, str | None, str | None], Citation] = {}
        self._order: list[tuple[str, str | None, str | None]] = []

    def add(
        self,
        *,
        source: str,
        title: str | None = None,
        quote: str | None = None,
    ) -> Citation:
        key = (source, title, quote)
        existing = self._by_key.get(key)
        if existing is not None:
            return existing
        citation = Citation(id=f"C{len(self._order) + 1}", source=source, title=title, quote=quote)
        self._by_key[key] = citation
        self._order.append(key)
        return citation

    def entries(self) -> list[Citation]:
        return [self._by_key[key] for key in self._order]

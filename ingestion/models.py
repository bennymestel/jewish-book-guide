"""
Pydantic models for Sefaria /api/v2/raw/index/<slug> API responses.
Only the fields we care about are captured; extras are ignored.
"""
from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator, model_validator


class SefariaIndexResponse(BaseModel):
    title: str
    categories: list[str] = Field(default_factory=list)

    # authors is a list of slug strings: ["shneur-zalman-of-liadi"]
    authors: list[str] = Field(default_factory=list)

    # schema contains the heTitle inside its titles list
    schema_: dict = Field(default_factory=dict, alias="schema")

    era: str | None = None
    compDate: list[int] | None = None
    compPlace: str | None = None
    pubDate: list[int] | None = None
    pubPlace: str | None = None

    enDesc: str | None = None
    enShortDesc: str | None = None
    is_cited: bool = False

    model_config = {"extra": "ignore", "populate_by_name": True}

    @field_validator("authors", mode="before")
    @classmethod
    def _coerce_authors(cls, v: Any) -> list[str]:
        if v is None:
            return []
        # Could be list of strings or list of dicts (older API versions)
        result = []
        for item in (v if isinstance(v, list) else [v]):
            if isinstance(item, str):
                result.append(item)
            elif isinstance(item, dict):
                result.append(item.get("slug") or item.get("en") or "")
        return [s for s in result if s]

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def category(self) -> str:
        return self.categories[0] if self.categories else "Unknown"

    @property
    def subcategory(self) -> str | None:
        return self.categories[1] if len(self.categories) > 1 else None

    @property
    def author_slugs(self) -> list[str]:
        return self.authors

    @property
    def comp_date_start(self) -> int | None:
        return self.compDate[0] if self.compDate else None

    @property
    def comp_date_end(self) -> int | None:
        return self.compDate[1] if self.compDate and len(self.compDate) > 1 else None

    @property
    def pub_date(self) -> int | None:
        return self.pubDate[0] if self.pubDate else None

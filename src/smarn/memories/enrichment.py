from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from smarn.memories.categories import MemoryCategory, coerce_memory_category
from smarn.memories.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryEnrichment:
    summary: str | None
    category: MemoryCategory
    tags: list[str]
    importance_score: int


class MemoryEnrichmentService:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def enrich(self, raw_text: str) -> MemoryEnrichment:
        try:
            response = self._llm_provider.complete(
                system_prompt=(
                    "You enrich user memory text for a private memory app. "
                    "Return only a JSON object with keys: summary, category, tags, "
                    "importance_score. category must be one of: personal, work, "
                    "learning, command, idea, reminder_candidate, unknown. "
                    "Use work for job, project, ticket, bug, and engineering "
                    "task memories. Extract concise tags, including ticket IDs, "
                    "tools, bug names, and solution terms. Give completed fixes "
                    "or notable work progress an importance_score from 3 to 5. "
                    "importance_score must be an integer from 1 to 5. "
                    "Do not change the raw memory text."
                ),
                user_prompt=json.dumps({"raw_text": raw_text}, ensure_ascii=False),
            )
            enrichment = _normalize_enrichment(_parse_json_object(response))
            logger.info(
                "memory_enrichment_completed",
                extra={
                    "category": enrichment.category.value,
                    "importance_score": enrichment.importance_score,
                    "tag_count": len(enrichment.tags),
                },
            )
            return enrichment
        except Exception:
            logger.exception("memory_enrichment_failed")
            return fallback_enrichment()


def fallback_enrichment() -> MemoryEnrichment:
    return MemoryEnrichment(
        summary=None,
        category=MemoryCategory.UNKNOWN,
        tags=[],
        importance_score=1,
    )


def _parse_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")

    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON was not an object.")
    return parsed


def _normalize_enrichment(data: dict[str, Any]) -> MemoryEnrichment:
    summary = data.get("summary")
    if not isinstance(summary, str):
        summary = None
    else:
        summary = summary.strip() or None

    try:
        category = coerce_memory_category(data.get("category"))
    except ValueError:
        category = MemoryCategory.UNKNOWN

    raw_tags = data.get("tags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if isinstance(tag, str):
                cleaned = tag.strip()
                if cleaned and cleaned not in tags:
                    tags.append(cleaned)

    importance_score = data.get("importance_score", 1)
    try:
        normalized_importance = int(importance_score)
    except (TypeError, ValueError):
        normalized_importance = 1
    normalized_importance = min(5, max(1, normalized_importance))

    return MemoryEnrichment(
        summary=summary,
        category=category,
        tags=tags,
        importance_score=normalized_importance,
    )

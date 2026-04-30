from __future__ import annotations

from smarn.memories.categories import MemoryCategory
from smarn.memories.enrichment import MemoryEnrichmentService


class RaisingLLMProvider:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        raise RuntimeError("provider failed")


class StaticLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return self.response


def test_memory_enrichment_falls_back_when_llm_fails() -> None:
    service = MemoryEnrichmentService(RaisingLLMProvider())

    enrichment = service.enrich(
        "Fixed TFG-231 duplicate ingestion bug using composite key upsert"
    )

    assert enrichment.summary is None
    assert enrichment.category is MemoryCategory.UNKNOWN
    assert enrichment.tags == []
    assert enrichment.importance_score == 1


def test_memory_enrichment_normalizes_llm_json() -> None:
    service = MemoryEnrichmentService(
        StaticLLMProvider(
            """
            {
              "summary": "Fixed duplicate ingestion for TFG-231.",
              "category": "work",
              "tags": ["TFG-231", "duplicate ingestion", "composite key", "upsert"],
              "importance_score": 4
            }
            """
        )
    )

    enrichment = service.enrich(
        "Fixed TFG-231 duplicate ingestion bug using composite key upsert"
    )

    assert enrichment.summary == "Fixed duplicate ingestion for TFG-231."
    assert enrichment.category is MemoryCategory.WORK
    assert enrichment.tags == [
        "TFG-231",
        "duplicate ingestion",
        "composite key",
        "upsert",
    ]
    assert enrichment.importance_score == 4

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from smarn.memories.answer import (
    NO_MEMORY_ANSWER,
    AnswerSynthesisService,
    RetrievedMemory,
)
from smarn.memories.categories import MemoryCategory


class CapturingLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.user_prompt: str | None = None

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        self.user_prompt = user_prompt
        return self.response


def test_answer_synthesis_returns_no_memory_message_without_memories() -> None:
    provider = CapturingLLMProvider("unused")
    service = AnswerSynthesisService(provider)

    answer = service.synthesize("What did I eat in 2022?", [])

    assert answer == NO_MEMORY_ANSWER
    assert provider.user_prompt is None


def test_answer_synthesis_passes_only_retrieved_memories_to_llm() -> None:
    provider = CapturingLLMProvider(
        "You worked on TFG-231 and fixed a duplicate ingestion bug."
    )
    service = AnswerSynthesisService(provider)
    memory = RetrievedMemory(
        id=uuid.uuid4(),
        raw_text="Fixed TFG-231 duplicate ingestion bug using composite key upsert",
        summary="Fixed duplicate ingestion for TFG-231.",
        category=MemoryCategory.WORK,
        tags=["TFG-231", "duplicate ingestion", "composite key", "upsert"],
        importance_score=4,
        created_at=datetime(2026, 4, 30, tzinfo=timezone.utc),
        score=0.2,
    )

    answer = service.synthesize("What did I do for TFG-231?", [memory])

    assert answer == "You worked on TFG-231 and fixed a duplicate ingestion bug."
    assert provider.user_prompt is not None
    assert "Fixed TFG-231 duplicate ingestion bug" in provider.user_prompt
    assert "Apr 30, 2026" in provider.user_prompt

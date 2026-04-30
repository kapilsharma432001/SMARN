from __future__ import annotations

import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

from smarn.memories.categories import MemoryCategory
from smarn.memories.review import NO_DAILY_REVIEW, ReviewService


class StaticRepository:
    def __init__(self, entries: list[object]) -> None:
        self.entries = entries
        self.calls: list[dict[str, object]] = []

    def list_created_between(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        user_id: str | None = None,
    ) -> list[object]:
        self.calls.append(
            {"start_at": start_at, "end_at": end_at, "user_id": user_id}
        )
        return self.entries


class CapturingLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.user_prompt: str | None = None

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt
        self.user_prompt = user_prompt
        return self.response


def test_daily_review_returns_clear_message_when_no_memories_exist() -> None:
    repository = StaticRepository([])
    provider = CapturingLLMProvider("unused")
    service = ReviewService(repository=repository, llm_provider=provider)

    review = service.daily_review(
        user_id="telegram-user",
        now=datetime(2026, 4, 30, 10, 0, tzinfo=timezone.utc),
    )

    assert review.text == NO_DAILY_REVIEW
    assert review.memories == []
    assert provider.user_prompt is None
    assert repository.calls[0]["user_id"] == "telegram-user"


def test_weekly_review_generation_payload_contains_selected_memories_only() -> None:
    entry = SimpleNamespace(
        id=uuid.uuid4(),
        raw_text="Fixed TFG-231 duplicate ingestion bug using composite key upsert",
        summary="Fixed duplicate ingestion for TFG-231.",
        category=MemoryCategory.WORK,
        tags=["TFG-231", "duplicate ingestion"],
        importance_score=4,
        created_at=datetime(2026, 4, 30, 9, 30, tzinfo=timezone.utc),
    )
    repository = StaticRepository([entry])
    provider = CapturingLLMProvider(
        "This week, you fixed the TFG-231 duplicate ingestion issue."
    )
    service = ReviewService(repository=repository, llm_provider=provider)

    review = service.weekly_review(
        user_id="telegram-user",
        now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )

    assert review.text == "This week, you fixed the TFG-231 duplicate ingestion issue."
    assert provider.user_prompt is not None
    assert "Fixed TFG-231 duplicate ingestion bug" in provider.user_prompt
    assert "TFG-231" in provider.user_prompt
    assert "2026-04-24T12:00:00+00:00" in provider.user_prompt
    assert repository.calls[0]["user_id"] == "telegram-user"

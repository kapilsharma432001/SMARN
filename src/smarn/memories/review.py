from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Sequence
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from smarn.config import Settings, get_settings
from smarn.memories.answer import RetrievedMemory
from smarn.memories.llm import LLMProvider, OpenAILLMProvider, UnavailableLLMProvider
from smarn.memories.repository import MemoryRepository

logger = logging.getLogger(__name__)

NO_DAILY_REVIEW = "I do not have enough memories for today yet."
NO_WEEKLY_REVIEW = "I do not have enough memories for this week yet."


@dataclass(frozen=True)
class MemoryReview:
    text: str
    memories: list[RetrievedMemory]
    start_at: datetime
    end_at: datetime


class ReviewService:
    def __init__(
        self,
        session: Session | None = None,
        *,
        repository: MemoryRepository | None = None,
        llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if repository is None:
            if session is None:
                raise ValueError("session or repository is required.")
            repository = MemoryRepository(session)
        self.repository = repository

        if llm_provider is None:
            api_key = self.settings.openai_api_key
            if api_key is None:
                llm_provider = UnavailableLLMProvider()
            else:
                llm_provider = OpenAILLMProvider(
                    api_key=api_key.get_secret_value(),
                    model=self.settings.openai_llm_model,
                )
        self._llm_provider = llm_provider

    def daily_review(
        self,
        *,
        user_id: str | None = None,
        now: datetime | None = None,
    ) -> MemoryReview:
        current = now or datetime.now(timezone.utc)
        timezone_info = ZoneInfo(self.settings.review_timezone)
        local_now = current.astimezone(timezone_info)
        local_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
        local_end = local_start + timedelta(days=1)
        return self.review_range(
            user_id=user_id,
            start_at=local_start.astimezone(timezone.utc),
            end_at=local_end.astimezone(timezone.utc),
            review_kind="daily",
            empty_message=NO_DAILY_REVIEW,
        )

    def weekly_review(
        self,
        *,
        user_id: str | None = None,
        now: datetime | None = None,
    ) -> MemoryReview:
        current = now or datetime.now(timezone.utc)
        end_at = current.astimezone(timezone.utc)
        start_at = end_at - timedelta(days=7)
        return self.review_range(
            user_id=user_id,
            start_at=start_at,
            end_at=end_at,
            review_kind="weekly",
            empty_message=NO_WEEKLY_REVIEW,
        )

    def review_range(
        self,
        *,
        user_id: str | None,
        start_at: datetime,
        end_at: datetime,
        review_kind: str,
        empty_message: str,
    ) -> MemoryReview:
        memories: list[RetrievedMemory] = []
        for entry in self.repository.list_created_between(
            user_id=user_id,
            start_at=start_at,
            end_at=end_at,
        ):
            if not start_at <= entry.created_at < end_at:
                continue
            memories.append(
                RetrievedMemory(
                    id=entry.id,
                    raw_text=entry.raw_text,
                    summary=entry.summary,
                    category=entry.category,
                    tags=entry.tags,
                    importance_score=entry.importance_score,
                    created_at=entry.created_at,
                    score=0.0,
                )
            )

        if not memories:
            return MemoryReview(
                text=empty_message,
                memories=[],
                start_at=start_at,
                end_at=end_at,
            )

        return MemoryReview(
            text=self._synthesize_review(
                review_kind=review_kind,
                memories=memories,
                start_at=start_at,
                end_at=end_at,
            ),
            memories=memories,
            start_at=start_at,
            end_at=end_at,
        )

    def _synthesize_review(
        self,
        *,
        review_kind: str,
        memories: Sequence[RetrievedMemory],
        start_at: datetime,
        end_at: datetime,
    ) -> str:
        payload = [
            {
                "id": str(memory.id),
                "date": (
                    f"{memory.created_at:%b} "
                    f"{memory.created_at.day}, "
                    f"{memory.created_at:%Y}"
                ),
                "raw_text": memory.raw_text,
                "summary": memory.summary,
                "category": memory.category.value,
                "tags": memory.tags,
                "importance_score": memory.importance_score,
                "source": "memory_entries",
            }
            for memory in memories
        ]

        try:
            review = self._llm_provider.complete(
                system_prompt=(
                    "You write concise SMARN memory reviews. Use only the memories "
                    "provided in the user message. Do not invent details or use "
                    "outside knowledge. Mention important items, work progress, "
                    "learnings, and recurring themes when present."
                ),
                user_prompt=json.dumps(
                    {
                        "review_kind": review_kind,
                        "range": {
                            "start_at": start_at.isoformat(),
                            "end_at": end_at.isoformat(),
                        },
                        "memories": payload,
                    },
                    ensure_ascii=False,
                    default=str,
                ),
            ).strip()
            if not review:
                raise ValueError("LLM returned an empty review.")
            logger.info(
                "review_synthesis_completed",
                extra={"review_kind": review_kind, "memory_count": len(memories)},
            )
            return review
        except Exception:
            logger.exception(
                "review_synthesis_failed",
                extra={"review_kind": review_kind, "memory_count": len(memories)},
            )
            return "SMARN does not have enough memory detail to generate that review yet."

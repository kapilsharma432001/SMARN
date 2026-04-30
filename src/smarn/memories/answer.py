from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from smarn.memories.categories import MemoryCategory
from smarn.memories.llm import LLMProvider

logger = logging.getLogger(__name__)

NO_MEMORY_ANSWER = "I do not have any saved memory about that yet."
WEAK_MEMORY_ANSWER = "SMARN does not have enough memory yet."


@dataclass(frozen=True)
class RetrievedMemory:
    id: uuid.UUID
    raw_text: str
    summary: str | None
    category: MemoryCategory
    tags: list[str]
    importance_score: int
    created_at: datetime
    score: float


class AnswerSynthesisService:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def synthesize(self, question: str, memories: Sequence[RetrievedMemory]) -> str:
        if not memories:
            return NO_MEMORY_ANSWER

        memory_payload = [
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
                "retrieval_score": memory.score,
            }
            for memory in memories
        ]

        try:
            answer = self._llm_provider.complete(
                system_prompt=(
                    "You answer questions for SMARN, a private memory recall app. "
                    "Use only the retrieved memories in the user message. "
                    "Do not use outside knowledge and do not invent details. "
                    "If the retrieved memories do not answer the question, reply exactly: "
                    f"{WEAK_MEMORY_ANSWER} "
                    "Write a concise, natural answer. Include memory dates when useful."
                ),
                user_prompt=json.dumps(
                    {"question": question, "retrieved_memories": memory_payload},
                    ensure_ascii=False,
                    default=str,
                ),
            ).strip()
            if not answer:
                raise ValueError("LLM returned an empty answer.")
            logger.info(
                "answer_synthesis_completed",
                extra={"memory_count": len(memories)},
            )
            return answer
        except Exception:
            logger.exception(
                "answer_synthesis_failed",
                extra={"memory_count": len(memories)},
            )
            return WEAK_MEMORY_ANSWER

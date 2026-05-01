from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.orm import Session

from smarn.config import Settings, get_settings
from smarn.memories.answer import (
    AnswerSynthesisService,
    NO_MEMORY_ANSWER,
    RetrievedMemory,
)
from smarn.memories.categories import MemoryCategory, coerce_memory_category
from smarn.memories.embeddings import EmbeddingProvider, OpenAIEmbeddingProvider
from smarn.memories.enrichment import MemoryEnrichmentService
from smarn.memories.llm import LLMProvider, OpenAILLMProvider, UnavailableLLMProvider
from smarn.memories.observations import ObservationExtractionService
from smarn.memories.repository import MemoryRepository, ObservationRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemorySearchResult:
    id: uuid.UUID
    raw_text: str
    summary: str | None
    category: MemoryCategory
    tags: list[str]
    importance_score: int
    created_at: datetime
    score: float


@dataclass(frozen=True)
class MemoryAnswer:
    text: str
    memories: list[MemorySearchResult]


@dataclass(frozen=True)
class RememberedMemory:
    id: uuid.UUID
    raw_text: str
    summary: str | None
    category: MemoryCategory
    tags: list[str]
    importance_score: int


class MemoryService:
    def __init__(
        self,
        session: Session,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        llm_provider: LLMProvider | None = None,
        enrichment_service: MemoryEnrichmentService | None = None,
        observation_extraction_service: ObservationExtractionService | None = None,
        answer_synthesis_service: AnswerSynthesisService | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.repository = MemoryRepository(session)
        self.observation_repository = ObservationRepository(session)
        if embedding_provider is not None:
            self.embedding_provider = embedding_provider
        else:
            api_key = self.settings.openai_api_key
            if api_key is None:
                raise ValueError("OPENAI_API_KEY must be set in the environment.")
            self.embedding_provider = OpenAIEmbeddingProvider(
                api_key=api_key.get_secret_value(),
                dimensions=self.settings.embedding_dimensions,
            )
        if llm_provider is None:
            api_key = self.settings.openai_api_key
            if api_key is None:
                llm_provider = UnavailableLLMProvider()
            else:
                llm_provider = OpenAILLMProvider(
                    api_key=api_key.get_secret_value(),
                    model=self.settings.openai_llm_model,
                )
        self.enrichment_service = enrichment_service or MemoryEnrichmentService(
            llm_provider
        )
        self.observation_extraction_service = (
            observation_extraction_service
            or ObservationExtractionService(llm_provider)
        )
        self.answer_synthesis_service = (
            answer_synthesis_service or AnswerSynthesisService(llm_provider)
        )

    def remember(
        self,
        raw_text: str,
        *,
        user_id: str | None = None,
        source: str = "telegram",
        summary: str | None = None,
        category: MemoryCategory | str | None = None,
        tags: list[str] | None = None,
        importance_score: int | None = None,
    ) -> uuid.UUID:
        return self.remember_with_details(
            raw_text,
            user_id=user_id,
            source=source,
            summary=summary,
            category=category,
            tags=tags,
            importance_score=importance_score,
        ).id

    def remember_with_details(
        self,
        raw_text: str,
        *,
        user_id: str | None = None,
        source: str = "telegram",
        summary: str | None = None,
        category: MemoryCategory | str | None = None,
        tags: list[str] | None = None,
        importance_score: int | None = None,
    ) -> RememberedMemory:
        cleaned = raw_text.strip()
        if not cleaned:
            raise ValueError("Memory content cannot be empty.")

        enrichment = self.enrichment_service.enrich(cleaned)
        cleaned_summary = summary.strip() if summary else enrichment.summary
        cleaned_category = (
            coerce_memory_category(category)
            if category is not None
            else enrichment.category
        )
        cleaned_tags = tags if tags is not None else enrichment.tags
        cleaned_importance_score = (
            importance_score
            if importance_score is not None
            else enrichment.importance_score
        )
        cleaned_importance_score = _normalize_importance_score(cleaned_importance_score)

        entry = self.repository.create(
            raw_text=cleaned,
            embedding=self.embedding_provider.embed(cleaned),
            source=source,
            user_id=user_id,
            summary=cleaned_summary or None,
            category=cleaned_category,
            tags=cleaned_tags,
            importance_score=cleaned_importance_score,
        )
        logger.info(
            "memory_saved",
            extra={
                "memory_id": str(entry.id),
                "user_id": user_id,
                "category": entry.category.value,
            },
        )
        self._extract_observations(entry_id=entry.id, user_id=user_id, raw_text=cleaned)
        return RememberedMemory(
            id=entry.id,
            raw_text=entry.raw_text,
            summary=entry.summary,
            category=entry.category,
            tags=entry.tags,
            importance_score=entry.importance_score,
        )

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> list[MemorySearchResult]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Search query cannot be empty.")

        rows = self.repository.search(
            embedding=self.embedding_provider.embed(cleaned),
            user_id=user_id,
            limit=limit or self.settings.memory_search_limit,
        )

        return [
            MemorySearchResult(
                id=entry.id,
                raw_text=entry.raw_text,
                summary=entry.summary,
                category=entry.category,
                tags=entry.tags,
                importance_score=entry.importance_score,
                created_at=entry.created_at,
                score=score,
            )
            for entry, score in rows
        ]

    def ask(
        self,
        question: str,
        *,
        user_id: str | None = None,
        limit: int | None = None,
    ) -> MemoryAnswer:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty.")

        results = self.search(cleaned, user_id=user_id, limit=limit)
        relevant_results = [
            result
            for result in results
            if result.score <= self.settings.memory_relevance_max_distance
        ]

        if not relevant_results:
            return MemoryAnswer(text=NO_MEMORY_ANSWER, memories=[])

        answer = self.answer_synthesis_service.synthesize(
            cleaned,
            [
                RetrievedMemory(
                    id=result.id,
                    raw_text=result.raw_text,
                    summary=result.summary,
                    category=result.category,
                    tags=result.tags,
                    importance_score=result.importance_score,
                    created_at=result.created_at,
                    score=result.score,
                )
                for result in relevant_results
            ],
        )
        return MemoryAnswer(text=answer, memories=relevant_results)

    def _extract_observations(
        self,
        *,
        entry_id: uuid.UUID,
        user_id: str | None,
        raw_text: str,
    ) -> None:
        try:
            observations = self.observation_extraction_service.extract(raw_text)
            self.observation_repository.create_many(
                memory_id=entry_id,
                user_id=user_id,
                observations=observations,
            )
            if observations:
                logger.info(
                    "memory_observations_saved",
                    extra={
                        "memory_id": str(entry_id),
                        "user_id": user_id,
                        "observation_count": len(observations),
                    },
                )
        except Exception:
            logger.exception(
                "memory_observation_extraction_ignored",
                extra={"memory_id": str(entry_id), "user_id": user_id},
            )


def _normalize_importance_score(value: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = 1
    return min(5, max(1, normalized))

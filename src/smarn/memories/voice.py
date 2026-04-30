from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from smarn.config import Settings, get_settings
from smarn.memories.categories import MemoryCategory
from smarn.memories.service import MemoryService, RememberedMemory
from smarn.memories.transcription import (
    OpenAITranscriptionProvider,
    TranscriptionProvider,
    UnavailableTranscriptionProvider,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VoiceIngestionResult:
    saved: bool
    error_message: str | None = None
    memory: RememberedMemory | None = None


class VoiceMemoryService:
    def __init__(
        self,
        session: Session | None,
        *,
        memory_service: MemoryService | None = None,
        transcription_provider: TranscriptionProvider | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if memory_service is None:
            if session is None:
                raise ValueError("session or memory_service is required.")
            memory_service = MemoryService(session)
        self.memory_service = memory_service

        if transcription_provider is None:
            api_key = self.settings.openai_api_key
            if api_key is None:
                transcription_provider = UnavailableTranscriptionProvider()
            else:
                transcription_provider = OpenAITranscriptionProvider(
                    api_key=api_key.get_secret_value(),
                    model=self.settings.openai_transcription_model,
                )
        self.transcription_provider = transcription_provider

    def remember_voice(
        self,
        audio_path: Path,
        *,
        user_id: str | None = None,
        source: str = "telegram_voice",
    ) -> VoiceIngestionResult:
        try:
            transcript = self.transcription_provider.transcribe(audio_path)
            memory = self.memory_service.remember_with_details(
                transcript,
                user_id=user_id,
                source=source,
            )
            logger.info(
                "voice_memory_saved",
                extra={
                    "memory_id": str(memory.id),
                    "user_id": user_id,
                    "source": source,
                    "category": memory.category.value,
                    "tag_count": len(memory.tags),
                },
            )
            return VoiceIngestionResult(saved=True, memory=memory)
        except Exception:
            logger.exception(
                "voice_memory_failed",
                extra={"user_id": user_id, "source": source},
            )
            return VoiceIngestionResult(
                saved=False,
                error_message=(
                    "I could not transcribe that voice note. Please try again "
                    "or send it as text."
                ),
            )


def format_voice_confirmation(memory: RememberedMemory) -> str:
    preview = memory.summary or _truncate(memory.raw_text, max_length=140)
    lines = [
        "Voice memory saved.",
        f"Summary: {preview}",
        f"Category: {_format_category(memory.category)}",
    ]
    if memory.tags:
        lines.append(f"Tags: {', '.join(memory.tags)}")
    return "\n".join(lines)


def _format_category(category: MemoryCategory) -> str:
    return category.value.replace("_", " ")


def _truncate(text: str, *, max_length: int) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_length:
        return cleaned
    return f"{cleaned[: max_length - 3].rstrip()}..."

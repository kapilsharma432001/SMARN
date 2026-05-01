from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from smarn.config import Settings, get_settings
from smarn.memories.categories import MemoryCategory
from smarn.memories.service import MemoryAnswer, MemoryService, RememberedMemory
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


@dataclass(frozen=True)
class VoiceQuestionResult:
    answered: bool
    answer: MemoryAnswer | None = None
    transcript: str | None = None
    error_message: str | None = None


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
            logger.info(
                "voice_transcription_started",
                extra={
                    "user_id": user_id,
                    "source": source,
                    "audio_path": str(audio_path),
                    "audio_bytes": audio_path.stat().st_size,
                    "provider": self.transcription_provider.__class__.__name__,
                },
            )
            transcript = self.transcription_provider.transcribe(audio_path)
            logger.info(
                "voice_transcription_completed",
                extra={
                    "user_id": user_id,
                    "source": source,
                    "transcript_length": len(transcript),
                    "transcript_preview": _truncate(transcript, max_length=120),
                },
            )
            logger.info(
                "voice_memory_save_started",
                extra={
                    "user_id": user_id,
                    "source": source,
                    "transcript_length": len(transcript),
                },
            )
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

    def ask_voice(
        self,
        audio_path: Path,
        *,
        user_id: str | None = None,
    ) -> VoiceQuestionResult:
        try:
            logger.info(
                "voice_question_transcription_started",
                extra={
                    "user_id": user_id,
                    "audio_path": str(audio_path),
                    "audio_bytes": audio_path.stat().st_size,
                    "provider": self.transcription_provider.__class__.__name__,
                },
            )
            transcript = self.transcription_provider.transcribe(audio_path)
            logger.info(
                "voice_question_transcription_completed",
                extra={
                    "user_id": user_id,
                    "transcript_length": len(transcript),
                    "transcript_preview": _truncate(transcript, max_length=120),
                },
            )
            answer = self.memory_service.ask(transcript, user_id=user_id)
            logger.info(
                "voice_question_answered",
                extra={
                    "user_id": user_id,
                    "retrieved_memory_count": len(answer.memories),
                },
            )
            return VoiceQuestionResult(
                answered=True,
                answer=answer,
                transcript=transcript,
            )
        except Exception:
            logger.exception("voice_question_failed", extra={"user_id": user_id})
            return VoiceQuestionResult(
                answered=False,
                error_message=(
                    "I could not answer that voice question. Please try again "
                    "or send it as text with /ask."
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

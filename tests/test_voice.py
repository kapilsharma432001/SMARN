from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace

from smarn.memories.categories import MemoryCategory
from smarn.memories.service import MemoryAnswer, RememberedMemory
from smarn.memories.voice import VoiceMemoryService
from smarn.telegram import bot as telegram_bot
from smarn.telegram.voice_questions import PendingVoiceQuestionStore


class RaisingTranscriptionProvider:
    def transcribe(self, audio_path: Path) -> str:
        del audio_path
        raise RuntimeError("transcription failed")


class UnusedMemoryService:
    def remember_with_details(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("memory should not be saved when transcription fails")


def test_voice_ingestion_returns_graceful_failure_when_transcription_fails(
    tmp_path,
) -> None:
    audio_path = tmp_path / "voice.ogg"
    audio_path.write_bytes(b"not real audio")
    service = VoiceMemoryService(
        None,
        memory_service=UnusedMemoryService(),
        transcription_provider=RaisingTranscriptionProvider(),
    )

    result = service.remember_voice(audio_path, user_id="telegram-user")

    assert result.saved is False
    assert result.memory is None
    assert result.error_message is not None
    assert "could not transcribe" in result.error_message


class FakeMessage:
    def __init__(self, *, voice: object | None = None) -> None:
        self.message_id = 123
        self.voice = voice
        self.replies: list[str] = []

    async def reply_text(self, text: str) -> None:
        self.replies.append(text)


class FakeTelegramFile:
    async def download_to_drive(self, *, custom_path: Path) -> None:
        custom_path.write_bytes(b"voice bytes")


class FakeBot:
    async def get_file(self, file_id: str) -> FakeTelegramFile:
        del file_id
        return FakeTelegramFile()


class FakeSessionScope:
    def __enter__(self) -> object:
        return object()

    def __exit__(self, *args: object) -> None:
        return None


def _fake_update(message: FakeMessage) -> SimpleNamespace:
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=42),
    )


def _fake_context(*, bot_data: dict[str, object] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        args=[],
        bot=FakeBot(),
        application=SimpleNamespace(bot_data=bot_data if bot_data is not None else {}),
    )


async def test_ask_voice_command_creates_pending_state() -> None:
    message = FakeMessage()
    context = _fake_context()

    await telegram_bot.ask_voice(_fake_update(message), context)

    store = context.application.bot_data[telegram_bot.VOICE_QUESTION_STORE_KEY]
    assert isinstance(store, PendingVoiceQuestionStore)
    assert store.consume_if_pending("42") is True
    assert message.replies == ["Send your voice question now."]


async def test_next_voice_note_after_ask_voice_is_treated_as_question(monkeypatch) -> None:
    calls: list[str] = []

    class FakeVoiceMemoryService:
        def __init__(self, session: object) -> None:
            del session

        def ask_voice(self, audio_path: Path, *, user_id: str | None = None):
            assert audio_path.exists()
            assert user_id == "42"
            calls.append("ask_voice")
            return SimpleNamespace(
                answered=True,
                answer=MemoryAnswer(text="The answer from memory.", memories=[]),
            )

        def remember_voice(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("pending voice question must not be saved")

    monkeypatch.setattr(telegram_bot, "session_scope", lambda: FakeSessionScope())
    monkeypatch.setattr(telegram_bot, "VoiceMemoryService", FakeVoiceMemoryService)
    store = PendingVoiceQuestionStore()
    store.mark_pending("42")
    context = _fake_context(bot_data={telegram_bot.VOICE_QUESTION_STORE_KEY: store})
    message = FakeMessage(
        voice=SimpleNamespace(
            file_id="file-id",
            file_unique_id="unique-id",
            duration=3,
            file_size=100,
            mime_type="audio/ogg",
        )
    )

    await telegram_bot.voice_note(_fake_update(message), context)

    assert calls == ["ask_voice"]
    assert message.replies == ["The answer from memory."]


async def test_normal_voice_note_is_saved_when_no_ask_voice_state_exists(
    monkeypatch,
) -> None:
    calls: list[str] = []
    memory = RememberedMemory(
        id=uuid.uuid4(),
        raw_text="I ate an apple.",
        summary="Ate an apple.",
        category=MemoryCategory.PERSONAL,
        tags=["food"],
        importance_score=1,
    )

    class FakeVoiceMemoryService:
        def __init__(self, session: object) -> None:
            del session

        def ask_voice(self, *args: object, **kwargs: object) -> object:
            raise AssertionError("normal voice notes must not be answered")

        def remember_voice(self, audio_path: Path, *, user_id: str | None = None):
            assert audio_path.exists()
            assert user_id == "42"
            calls.append("remember_voice")
            return SimpleNamespace(saved=True, memory=memory)

    monkeypatch.setattr(telegram_bot, "session_scope", lambda: FakeSessionScope())
    monkeypatch.setattr(telegram_bot, "VoiceMemoryService", FakeVoiceMemoryService)
    context = _fake_context()
    message = FakeMessage(
        voice=SimpleNamespace(
            file_id="file-id",
            file_unique_id="unique-id",
            duration=3,
            file_size=100,
            mime_type="audio/ogg",
        )
    )

    await telegram_bot.voice_note(_fake_update(message), context)

    assert calls == ["remember_voice"]
    assert message.replies[0].startswith("Voice memory saved.")

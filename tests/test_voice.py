from __future__ import annotations

from pathlib import Path

from smarn.memories.voice import VoiceMemoryService


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

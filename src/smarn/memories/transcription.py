from __future__ import annotations

from pathlib import Path
from typing import Protocol

from openai import OpenAI


class TranscriptionProvider(Protocol):
    def transcribe(self, audio_path: Path) -> str:
        raise NotImplementedError


class OpenAITranscriptionProvider:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def transcribe(self, audio_path: Path) -> str:
        with audio_path.open("rb") as audio_file:
            response = self._client.audio.transcriptions.create(
                file=audio_file,
                model=self._model,
            )
        transcript = response.text.strip()
        if not transcript:
            raise ValueError("Transcription returned an empty transcript.")
        return transcript


class UnavailableTranscriptionProvider:
    def transcribe(self, audio_path: Path) -> str:
        del audio_path
        raise RuntimeError("No transcription provider is configured.")

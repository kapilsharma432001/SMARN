from __future__ import annotations

from typing import Protocol

from openai import OpenAI


class LLMProvider(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


class OpenAILLMProvider:
    def __init__(self, *, api_key: str, model: str) -> None:
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned an empty response.")
        return content


class UnavailableLLMProvider:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        raise RuntimeError("No LLM provider is configured.")

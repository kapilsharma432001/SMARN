from __future__ import annotations

from smarn.memories.observations import ObservationExtractionService


class RaisingLLMProvider:
    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        raise RuntimeError("provider failed")


class StaticLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return self.response


def test_observation_extraction_falls_back_when_llm_fails() -> None:
    service = ObservationExtractionService(RaisingLLMProvider())

    observations = service.extract("I woke up at 7:30 AM and ate cake.")

    assert observations == []


def test_observation_extraction_normalizes_time_and_food_metadata() -> None:
    service = ObservationExtractionService(
        StaticLLMProvider(
            """
            {
              "observations": [
                {
                  "observation_type": "wake_time",
                  "label": null,
                  "value_text": "7:30 AM",
                  "value_number": null,
                  "unit": null,
                  "occurred_at": "2026-04-15",
                  "confidence": 1.2,
                  "metadata": {}
                },
                {
                  "observation_type": "food_intake",
                  "label": "Chocolate cake",
                  "value_text": null,
                  "value_number": null,
                  "unit": null,
                  "occurred_at": null,
                  "confidence": 0.81,
                  "metadata": {
                    "categories": ["sweet", "vegetarian"],
                    "processed_food": "yes"
                  }
                }
              ]
            }
            """
        )
    )

    observations = service.extract("I woke up at 7:30 AM and ate chocolate cake.")

    wake_time = observations[0]
    assert wake_time.observation_type == "wake_time"
    assert wake_time.value_number == 450
    assert wake_time.unit == "minutes_after_midnight"
    assert wake_time.confidence == 1.0

    food = observations[1]
    assert food.observation_type == "food_intake"
    assert food.label == "Chocolate cake"
    assert food.metadata["sweet"] is True
    assert food.metadata["vegetarian"] is True
    assert food.metadata["processed_food"] is True

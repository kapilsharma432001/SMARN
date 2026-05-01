from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

from smarn.memories.analytics import ANALYTICS_CAVEAT, INSUFFICIENT_DATA, AnalyticsService


class StaticObservationRepository:
    def __init__(self, observations: list[object]) -> None:
        self.observations = observations
        self.calls: list[dict[str, object]] = []

    def list_for_analytics(
        self,
        *,
        user_id: str | None,
        start_at: datetime,
        end_at: datetime,
        observation_types: list[str],
    ) -> list[object]:
        self.calls.append(
            {
                "user_id": user_id,
                "start_at": start_at,
                "end_at": end_at,
                "observation_types": observation_types,
            }
        )
        return [
            observation
            for observation in self.observations
            if observation.observation_type in observation_types
        ]


class StaticMemoryRepository:
    def __init__(self, memories: list[object] | None = None) -> None:
        self.memories = memories or []

    def list_created_between(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        user_id: str | None = None,
        category: object | None = None,
    ) -> list[object]:
        del start_at, end_at, user_id, category
        return self.memories


def test_analytics_returns_clear_message_when_no_observations_exist() -> None:
    observation_repository = StaticObservationRepository([])
    service = AnalyticsService(
        observation_repository=observation_repository,
        memory_repository=StaticMemoryRepository(),
    )

    answer = service.analyze(
        "How many times did I wake up before 8 AM last month?",
        user_id="telegram-user",
        now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )

    assert INSUFFICIENT_DATA in answer.text
    assert ANALYTICS_CAVEAT in answer.text
    assert observation_repository.calls[0]["user_id"] == "telegram-user"


def test_analytics_counts_wake_time_before_8_am() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                SimpleNamespace(
                    observation_type="wake_time",
                    value_number=Decimal("450"),
                    label=None,
                    value_text=None,
                    metadata_={},
                ),
                SimpleNamespace(
                    observation_type="wake_time",
                    value_number=Decimal("510"),
                    label=None,
                    value_text=None,
                    metadata_={},
                ),
            ]
        ),
        memory_repository=StaticMemoryRepository(),
    )

    answer = service.analyze(
        "How many times did I wake up before 8 AM last month?",
        user_id="telegram-user",
        now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )

    assert "1 time" in answer.text
    assert ANALYTICS_CAVEAT in answer.text


def test_analytics_counts_food_intake_from_metadata() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                SimpleNamespace(
                    observation_type="food_intake",
                    value_number=None,
                    label="chips",
                    value_text=None,
                    metadata_={"processed_food": True},
                ),
                SimpleNamespace(
                    observation_type="food_intake",
                    value_number=None,
                    label="chocolate cake",
                    value_text=None,
                    metadata_={"sweet": True},
                ),
                SimpleNamespace(
                    observation_type="food_intake",
                    value_number=None,
                    label="apple",
                    value_text=None,
                    metadata_={"healthy_food": True},
                ),
            ]
        ),
        memory_repository=StaticMemoryRepository(),
    )

    processed = service.analyze(
        "How many times did I eat processed food last 7 days?",
        user_id="telegram-user",
        now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )
    sweets = service.analyze(
        "How many times did I eat sweets last 7 days?",
        user_id="telegram-user",
        now=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )

    assert "1 time" in processed.text
    assert "1 time" in sweets.text
    assert ANALYTICS_CAVEAT in processed.text
    assert ANALYTICS_CAVEAT in sweets.text

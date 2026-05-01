from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from smarn.config import Settings, get_settings
from smarn.memories.repository import MemoryRepository, ObservationRepository

ANALYTICS_CAVEAT = "This is based only on your saved memories."
INSUFFICIENT_DATA = "SMARN does not have enough logged memories yet."


@dataclass(frozen=True)
class DateRange:
    start_at: datetime
    end_at: datetime
    label: str


@dataclass(frozen=True)
class AnalyticsAnswer:
    text: str
    observation_count: int
    memory_count: int
    date_range: DateRange


class AnalyticsService:
    def __init__(
        self,
        session: Session | None = None,
        *,
        observation_repository: ObservationRepository | None = None,
        memory_repository: MemoryRepository | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if observation_repository is None or memory_repository is None:
            if session is None:
                raise ValueError(
                    "session or both observation_repository and memory_repository "
                    "are required."
                )
            observation_repository = observation_repository or ObservationRepository(
                session
            )
            memory_repository = memory_repository or MemoryRepository(session)
        self.observation_repository = observation_repository
        self.memory_repository = memory_repository

    def analyze(
        self,
        question: str,
        *,
        user_id: str | None = None,
        now: datetime | None = None,
    ) -> AnalyticsAnswer:
        cleaned = question.strip()
        if not cleaned:
            raise ValueError("Analytics question cannot be empty.")

        current = now or datetime.now(timezone.utc)
        date_range = parse_date_range(cleaned, now=current, settings=self.settings)
        intent = _parse_count_intent(cleaned)
        memories = self.memory_repository.list_created_between(
            user_id=user_id,
            start_at=date_range.start_at,
            end_at=date_range.end_at,
        )

        if intent is None:
            return AnalyticsAnswer(
                text=(
                    f"{INSUFFICIENT_DATA} I can currently answer logged count "
                    f"questions for wake times and food categories. {ANALYTICS_CAVEAT}"
                ),
                observation_count=0,
                memory_count=len(memories),
                date_range=date_range,
            )

        observations = self.observation_repository.list_for_analytics(
            user_id=user_id,
            start_at=date_range.start_at,
            end_at=date_range.end_at,
            observation_types=[intent.observation_type],
        )
        matching = [
            observation for observation in observations if intent.matches(observation)
        ]

        if not observations:
            text = f"{INSUFFICIENT_DATA} {ANALYTICS_CAVEAT}"
        else:
            text = (
                f"{_format_count_question(cleaned)}: {len(matching)} time"
                f"{'' if len(matching) == 1 else 's'} in {date_range.label}. "
                f"{ANALYTICS_CAVEAT}"
            )

        return AnalyticsAnswer(
            text=text,
            observation_count=len(observations),
            memory_count=len(memories),
            date_range=date_range,
        )


@dataclass(frozen=True)
class CountIntent:
    observation_type: str
    metadata_key: str | None = None
    value_less_than: float | None = None
    label_keywords: tuple[str, ...] = ()

    def matches(self, observation: object) -> bool:
        if getattr(observation, "observation_type") != self.observation_type:
            return False
        if self.value_less_than is not None:
            value = _number(getattr(observation, "value_number", None))
            return value is not None and value < self.value_less_than
        if self.metadata_key is not None:
            metadata = _metadata(observation)
            if metadata.get(self.metadata_key) is True:
                return True
            if not self.label_keywords:
                return False
        if self.label_keywords:
            haystack = " ".join(
                value
                for value in [
                    getattr(observation, "label", None),
                    getattr(observation, "value_text", None),
                ]
                if isinstance(value, str)
            ).lower()
            return any(keyword in haystack for keyword in self.label_keywords)
        return True


def parse_date_range(
    question: str,
    *,
    now: datetime,
    settings: Settings | None = None,
) -> DateRange:
    settings = settings or get_settings()
    timezone_info = ZoneInfo(settings.review_timezone)
    local_now = now.astimezone(timezone_info)
    lowered = question.lower()

    if "last month" in lowered:
        local_month_start = local_now.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        if local_month_start.month == 1:
            start = local_month_start.replace(
                year=local_month_start.year - 1,
                month=12,
            )
        else:
            start = local_month_start.replace(month=local_month_start.month - 1)
        return _date_range(start, local_month_start, "last month")

    if "this week" in lowered:
        start = (local_now - timedelta(days=local_now.weekday())).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        return _date_range(start, local_now, "this week")

    if "last 7 days" in lowered:
        return _date_range(local_now - timedelta(days=7), local_now, "the last 7 days")

    if "last 1 month" in lowered or "past month" in lowered:
        return _date_range(local_now - timedelta(days=30), local_now, "the last month")

    return _date_range(local_now - timedelta(days=30), local_now, "the last 30 days")


def _date_range(start: datetime, end: datetime, label: str) -> DateRange:
    return DateRange(
        start_at=start.astimezone(timezone.utc),
        end_at=end.astimezone(timezone.utc),
        label=label,
    )


def _parse_count_intent(question: str) -> CountIntent | None:
    lowered = question.lower()
    if "wake" in lowered and ("before 8" in lowered or "before 8 am" in lowered):
        return CountIntent(
            observation_type="wake_time",
            value_less_than=8 * 60,
        )
    if "processed" in lowered and "food" in lowered:
        return CountIntent(
            observation_type="food_intake",
            metadata_key="processed_food",
        )
    if "sweet" in lowered or "sweets" in lowered:
        return CountIntent(
            observation_type="food_intake",
            metadata_key="sweet",
            label_keywords=("sweet", "dessert", "cake", "candy", "chocolate"),
        )
    return None


def _format_count_question(question: str) -> str:
    cleaned = question.strip().rstrip("?")
    if cleaned.lower().startswith("how many"):
        return "Logged count"
    return cleaned[:1].upper() + cleaned[1:]


def _metadata(observation: object) -> dict[str, object]:
    metadata = getattr(observation, "metadata_", None)
    if metadata is None:
        metadata = getattr(observation, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _number(value: object) -> float | None:
    if isinstance(value, Decimal):
        return float(value)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

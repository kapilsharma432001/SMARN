from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Sequence
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session

from smarn.config import Settings, get_settings
from smarn.memories.categories import MemoryCategory, coerce_memory_category
from smarn.memories.llm import LLMProvider, OpenAILLMProvider, UnavailableLLMProvider
from smarn.memories.repository import MemoryRepository, ObservationRepository

logger = logging.getLogger(__name__)

ANALYTICS_CAVEAT = "This is based only on your saved memories."
ANALYTICS_UNSUPPORTED = "SMARN could not understand this analytics question yet."
INSUFFICIENT_DATA = "SMARN does not have enough logged memories yet."

SUPPORTED_ANSWER_TYPES = {"count", "list", "summary", "trend", "unknown"}
SUPPORTED_OBSERVATION_TYPES = {
    "wake_time",
    "sleep_time",
    "food_intake",
    "exercise",
    "mood",
    "work_activity",
    "learning_activity",
    "health_event",
}
SUPPORTED_FILTER_OPERATORS = {
    "equals",
    "contains",
    "contains_any",
    "less_than",
    "greater_than",
    "greater_than_or_equal",
    "less_than_or_equal",
    "metadata_true",
}
SUPPORTED_PLAN_KEYS = {
    "answer_type",
    "date_range",
    "observation_types",
    "memory_categories",
    "filters",
    "group_by",
    "limit",
    "needs_raw_memories",
}
SUPPORTED_FILTER_KEYS = {"field", "operator", "value"}


@dataclass(frozen=True)
class DateRange:
    start_at: datetime
    end_at: datetime
    label: str


@dataclass(frozen=True)
class AnalyticsFilter:
    field: str
    operator: str
    value: object


@dataclass(frozen=True)
class AnalyticsPlan:
    answer_type: str
    date_range: DateRange
    observation_types: list[str]
    memory_categories: list[MemoryCategory] | None
    filters: list[AnalyticsFilter]
    group_by: str | None
    limit: int | None
    needs_raw_memories: bool


@dataclass(frozen=True)
class AnalyticsAnswer:
    text: str
    observation_count: int
    memory_count: int
    date_range: DateRange


class AnalyticsPlanningError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        planner_response_length: int | None = None,
    ) -> None:
        super().__init__(message)
        self.planner_response_length = planner_response_length


class AnalyticsPlannerService:
    def __init__(
        self,
        llm_provider: LLMProvider,
        *,
        settings: Settings | None = None,
    ) -> None:
        self._llm_provider = llm_provider
        self.settings = settings or get_settings()

    def plan(
        self,
        question: str,
        *,
        now: datetime | None = None,
    ) -> AnalyticsPlan:
        current = now or datetime.now(timezone.utc)
        response: str | None = None
        try:
            response = self._llm_provider.complete(
                system_prompt=(
                    "You are the analytics planner for SMARN, a private memory app. "
                    "Return strict JSON only. Do not answer the user's question. "
                    "Convert the question into a plan with keys: answer_type, "
                    "date_range, observation_types, memory_categories, filters, "
                    "group_by, limit, needs_raw_memories. answer_type must be one of "
                    "count, list, summary, trend, unknown. observation_types may include "
                    "wake_time, sleep_time, food_intake, exercise, mood, work_activity, "
                    "learning_activity, health_event. filter operators may include "
                    "equals, contains, contains_any, less_than, greater_than, "
                    "greater_than_or_equal, less_than_or_equal, metadata_true. "
                    "For wake/sleep times use value_number minutes after midnight. "
                    "For wake after 10 AM use field value_number, operator "
                    "greater_than_or_equal, value 600. For food categories use "
                    "metadata_true with fields like processed_food, sweet, junk_food, "
                    "outside_food, healthy_food. For DSA or system design study "
                    "questions use learning_activity and raw memories when useful."
                ),
                user_prompt=json.dumps(
                    {
                        "question": question,
                        "now": current.isoformat(),
                        "timezone": self.settings.review_timezone,
                    },
                    ensure_ascii=False,
                ),
            )
            return normalize_analytics_plan(
                response,
                now=current,
                settings=self.settings,
            )
        except AnalyticsPlanningError as error:
            if error.planner_response_length is not None or response is None:
                raise
            raise AnalyticsPlanningError(
                str(error),
                planner_response_length=len(response),
            ) from error


class AnalyticsSynthesisService:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def synthesize(
        self,
        *,
        question: str,
        plan: AnalyticsPlan,
        observations: Sequence[object],
        memories: Sequence[object],
    ) -> str:
        payload = {
            "question": question,
            "answer_type": plan.answer_type,
            "date_range": {
                "start_at": plan.date_range.start_at.isoformat(),
                "end_at": plan.date_range.end_at.isoformat(),
                "label": plan.date_range.label,
            },
            "observations": [
                _observation_payload(observation) for observation in observations
            ],
            "memories": [_memory_payload(memory) for memory in memories],
        }
        try:
            answer = self._llm_provider.complete(
                system_prompt=(
                    "You answer SMARN analytics questions using only the selected "
                    "observations and memories in the user message. Do not invent "
                    "facts. If the selected data is insufficient, say that clearly. "
                    "Keep the answer concise."
                ),
                user_prompt=json.dumps(payload, ensure_ascii=False, default=str),
            ).strip()
            if not answer:
                raise ValueError("LLM returned an empty analytics answer.")
            return _with_caveat(answer)
        except Exception:
            logger.exception(
                "analytics_synthesis_failed",
                extra={"observation_count": len(observations)},
            )
            return f"{INSUFFICIENT_DATA} {ANALYTICS_CAVEAT}"


class AnalyticsService:
    def __init__(
        self,
        session: Session | None = None,
        *,
        observation_repository: ObservationRepository | None = None,
        memory_repository: MemoryRepository | None = None,
        planner_service: AnalyticsPlannerService | None = None,
        synthesis_service: AnalyticsSynthesisService | None = None,
        llm_provider: LLMProvider | None = None,
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

        if llm_provider is None:
            api_key = self.settings.openai_api_key
            if api_key is None:
                llm_provider = UnavailableLLMProvider()
            else:
                llm_provider = OpenAILLMProvider(
                    api_key=api_key.get_secret_value(),
                    model=self.settings.openai_llm_model,
                )
        self.planner_service = planner_service or AnalyticsPlannerService(
            llm_provider,
            settings=self.settings,
        )
        self.synthesis_service = synthesis_service or AnalyticsSynthesisService(
            llm_provider
        )

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
        try:
            plan = self.planner_service.plan(cleaned, now=current)
        except Exception as error:
            _log_planning_failure(error, user_id=user_id)
            plan = fallback_analytics_plan(
                cleaned,
                now=current,
                settings=self.settings,
            )
            if plan is None:
                fallback_range = parse_date_range(
                    cleaned,
                    now=current,
                    settings=self.settings,
                )
                return AnalyticsAnswer(
                    text=f"{ANALYTICS_UNSUPPORTED} {ANALYTICS_CAVEAT}",
                    observation_count=0,
                    memory_count=0,
                    date_range=fallback_range,
                )

        if plan.answer_type == "unknown":
            fallback_plan = fallback_analytics_plan(
                cleaned,
                now=current,
                settings=self.settings,
            )
            if fallback_plan is None:
                return AnalyticsAnswer(
                    text=f"{ANALYTICS_UNSUPPORTED} {ANALYTICS_CAVEAT}",
                    observation_count=0,
                    memory_count=0,
                    date_range=plan.date_range,
                )
            plan = fallback_plan

        observations = self.observation_repository.list_for_analytics(
            user_id=user_id,
            start_at=plan.date_range.start_at,
            end_at=plan.date_range.end_at,
            observation_types=plan.observation_types or None,
        )
        matching_observations = [
            observation
            for observation in observations
            if _matches_filters(observation, plan.filters, data_kind="observation")
        ]

        memories = self._retrieve_memories(
            plan=plan,
            user_id=user_id,
        )
        matching_memories = [
            memory
            for memory in memories
            if _matches_filters(memory, plan.filters, data_kind="memory")
        ]

        if plan.limit is not None:
            matching_observations = matching_observations[: plan.limit]
            matching_memories = matching_memories[: plan.limit]

        if plan.answer_type == "count":
            count = (
                len(matching_observations)
                if matching_observations
                else len(matching_memories)
            )
            if count == 0 and not matching_observations and not matching_memories:
                text = f"{INSUFFICIENT_DATA} {ANALYTICS_CAVEAT}"
            else:
                text = (
                    f"You logged this {count} time"
                    f"{'' if count == 1 else 's'} in {plan.date_range.label}. "
                    f"{ANALYTICS_CAVEAT}"
                )
        elif not matching_observations and not matching_memories:
            text = f"{INSUFFICIENT_DATA} {ANALYTICS_CAVEAT}"
        else:
            text = self.synthesis_service.synthesize(
                question=cleaned,
                plan=plan,
                observations=matching_observations,
                memories=matching_memories,
            )

        return AnalyticsAnswer(
            text=text,
            observation_count=len(matching_observations),
            memory_count=len(matching_memories),
            date_range=plan.date_range,
        )

    def _retrieve_memories(
        self,
        *,
        plan: AnalyticsPlan,
        user_id: str | None,
    ) -> list[object]:
        if not plan.needs_raw_memories and not plan.memory_categories:
            return []
        if not plan.memory_categories:
            return self.memory_repository.list_created_between(
                user_id=user_id,
                start_at=plan.date_range.start_at,
                end_at=plan.date_range.end_at,
            )

        memories: list[object] = []
        seen_ids: set[str] = set()
        for category in plan.memory_categories:
            for memory in self.memory_repository.list_created_between(
                user_id=user_id,
                start_at=plan.date_range.start_at,
                end_at=plan.date_range.end_at,
                category=category,
            ):
                memory_id = str(getattr(memory, "id", id(memory)))
                if memory_id not in seen_ids:
                    seen_ids.add(memory_id)
                    memories.append(memory)
        return memories


def normalize_analytics_plan(
    content: str | dict[str, object],
    *,
    now: datetime,
    settings: Settings | None = None,
) -> AnalyticsPlan:
    settings = settings or get_settings()
    data = _parse_plan_payload(content)
    if set(data) - SUPPORTED_PLAN_KEYS:
        raise AnalyticsPlanningError("Planner returned unsupported fields.")
    answer_type = data.get("answer_type")
    if not isinstance(answer_type, str) or answer_type not in SUPPORTED_ANSWER_TYPES:
        raise AnalyticsPlanningError("Unsupported answer_type.")

    date_range = _normalize_date_range(
        data.get("date_range"),
        now=now,
        settings=settings,
    )
    observation_types = _normalize_observation_types(data.get("observation_types"))
    memory_categories = _normalize_memory_categories(data.get("memory_categories"))
    filters = _normalize_filters(data.get("filters"))
    group_by = data.get("group_by")
    if group_by is not None and not isinstance(group_by, str):
        raise AnalyticsPlanningError("group_by must be a string or null.")

    limit = data.get("limit")
    if limit is not None:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            raise AnalyticsPlanningError("limit must be an integer.") from None
        if limit < 1:
            limit = None
        else:
            limit = min(limit, 100)

    needs_raw_memories = data.get("needs_raw_memories", False)
    needs_raw_memories = _coerce_bool(needs_raw_memories)

    return AnalyticsPlan(
        answer_type=answer_type,
        date_range=date_range,
        observation_types=observation_types,
        memory_categories=memory_categories,
        filters=filters,
        group_by=group_by,
        limit=limit,
        needs_raw_memories=needs_raw_memories,
    )


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
    today_start = local_now.replace(hour=0, minute=0, second=0, microsecond=0)

    if "yesterday" in lowered:
        start = today_start - timedelta(days=1)
        return _date_range(start, today_start, "yesterday")

    if "today" in lowered:
        return _date_range(today_start, local_now, "today")

    if "last month" in lowered or "this month" in lowered:
        local_month_start = local_now.replace(
            day=1,
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        if "last month" in lowered:
            if local_month_start.month == 1:
                start = local_month_start.replace(
                    year=local_month_start.year - 1,
                    month=12,
                )
            else:
                start = local_month_start.replace(month=local_month_start.month - 1)
            return _date_range(start, local_month_start, "last month")
        return _date_range(local_month_start, local_now, "this month")

    if "last week" in lowered:
        this_week_start = (local_now - timedelta(days=local_now.weekday())).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        )
        return _date_range(
            this_week_start - timedelta(days=7),
            this_week_start,
            "last week",
        )

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

    if "last 30 days" in lowered or "past 30 days" in lowered:
        return _date_range(local_now - timedelta(days=30), local_now, "the last 30 days")

    if "last 1 month" in lowered or "past month" in lowered:
        return _date_range(local_now - timedelta(days=30), local_now, "the last month")

    return _date_range(local_now - timedelta(days=30), local_now, "the last 30 days")


def _parse_plan_payload(content: str | dict[str, object]) -> dict[str, object]:
    if isinstance(content, dict):
        return content
    cleaned = _strip_json_fence(content.strip())
    parsed = _decode_first_json_object(cleaned)
    if not isinstance(parsed, dict):
        raise AnalyticsPlanningError("Planner JSON must be an object.")
    return parsed


def _normalize_date_range(
    value: object,
    *,
    now: datetime,
    settings: Settings,
) -> DateRange:
    if not isinstance(value, dict):
        return parse_date_range("", now=now, settings=settings)
    label = value.get("label")
    if not isinstance(label, str) or not label.strip():
        label = "the selected range"
    start_at = _parse_datetime(value.get("start_at"))
    end_at = _parse_datetime(value.get("end_at"))
    if start_at is None or end_at is None or start_at >= end_at:
        fallback = parse_date_range(label, now=now, settings=settings)
        start_at = fallback.start_at
        end_at = fallback.end_at
    return DateRange(start_at=start_at, end_at=end_at, label=label.strip())


def _normalize_observation_types(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise AnalyticsPlanningError("observation_types must be a list.")
    observation_types: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in SUPPORTED_OBSERVATION_TYPES:
            raise AnalyticsPlanningError("Unsupported observation_type.")
        if item not in observation_types:
            observation_types.append(item)
    return observation_types


def _normalize_memory_categories(value: object) -> list[MemoryCategory] | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        raise AnalyticsPlanningError("memory_categories must be a list or null.")
    categories: list[MemoryCategory] = []
    for item in value:
        try:
            category = coerce_memory_category(item)
        except ValueError:
            raise AnalyticsPlanningError("Unsupported memory category.") from None
        if category not in categories:
            categories.append(category)
    return categories


def _normalize_filters(value: object) -> list[AnalyticsFilter]:
    if value is None:
        return []
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        raise AnalyticsPlanningError("filters must be a list.")
    filters: list[AnalyticsFilter] = []
    for item in value:
        if not isinstance(item, dict):
            raise AnalyticsPlanningError("filter must be an object.")
        if set(item) - SUPPORTED_FILTER_KEYS:
            raise AnalyticsPlanningError("filter returned unsupported fields.")
        field = item.get("field")
        operator = item.get("operator")
        if not isinstance(field, str) or not field.strip():
            raise AnalyticsPlanningError("filter field must be a string.")
        if (
            not isinstance(operator, str)
            or operator not in SUPPORTED_FILTER_OPERATORS
        ):
            raise AnalyticsPlanningError("Unsupported filter operator.")
        filters.append(
            AnalyticsFilter(
                field=field.strip(),
                operator=operator,
                value=item.get("value"),
            )
        )
    return filters


def fallback_analytics_plan(
    question: str,
    *,
    now: datetime,
    settings: Settings | None = None,
) -> AnalyticsPlan | None:
    lowered = _compact_question(question)
    date_range = parse_date_range(question, now=now, settings=settings)

    if "wake" in lowered and _contains_time_direction(lowered, "before", "8"):
        return _count_plan(
            date_range=date_range,
            observation_types=["wake_time"],
            filters=[
                AnalyticsFilter("value_number", "less_than", 480),
            ],
        )

    if "wake" in lowered and _contains_time_direction(lowered, "after", "10"):
        return _count_plan(
            date_range=date_range,
            observation_types=["wake_time"],
            filters=[
                AnalyticsFilter("value_number", "greater_than_or_equal", 600),
            ],
        )

    if "dsa" in lowered and ("problem" in lowered or "problems" in lowered):
        return AnalyticsPlan(
            answer_type="count",
            date_range=date_range,
            observation_types=["learning_activity"],
            memory_categories=[MemoryCategory.LEARNING],
            filters=[
                AnalyticsFilter(
                    "search_text",
                    "contains_any",
                    ["DSA", "LeetCode", "problem", "problems", "solved"],
                ),
            ],
            group_by=None,
            limit=None,
            needs_raw_memories=True,
        )

    if "system design" in lowered and ("study" in lowered or "studied" in lowered):
        return AnalyticsPlan(
            answer_type="list",
            date_range=date_range,
            observation_types=["learning_activity"],
            memory_categories=[MemoryCategory.LEARNING],
            filters=[
                AnalyticsFilter("search_text", "contains", "system design"),
            ],
            group_by="topic",
            limit=20,
            needs_raw_memories=True,
        )

    if "food" in lowered or "sweet" in lowered or "sweets" in lowered:
        metadata_key = _food_metadata_key(lowered)
        if metadata_key is None:
            return None
        return _count_plan(
            date_range=date_range,
            observation_types=["food_intake"],
            filters=[
                AnalyticsFilter(f"metadata.{metadata_key}", "metadata_true", None),
            ],
        )

    return None


def _strip_json_fence(content: str) -> str:
    cleaned = content.strip()
    if not cleaned.startswith("```"):
        return cleaned

    first_newline = cleaned.find("\n")
    if first_newline == -1:
        return cleaned.removeprefix("```").removesuffix("```").strip()
    body = cleaned[first_newline + 1 :]
    if body.rstrip().endswith("```"):
        body = body.rstrip()[:-3]
    return body.strip()


def _decode_first_json_object(content: str) -> object:
    decoder = json.JSONDecoder()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        return parsed

    for index, char in enumerate(content):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise AnalyticsPlanningError("Planner response did not contain a JSON object.")


def _log_planning_failure(error: Exception, *, user_id: str | None) -> None:
    logger.warning(
        "analytics_planning_failed",
        extra={
            "error_type": error.__class__.__name__,
            "planner_response_length": getattr(
                error,
                "planner_response_length",
                None,
            ),
            "user_id": user_id,
        },
    )


def _count_plan(
    *,
    date_range: DateRange,
    observation_types: list[str],
    filters: list[AnalyticsFilter],
) -> AnalyticsPlan:
    return AnalyticsPlan(
        answer_type="count",
        date_range=date_range,
        observation_types=observation_types,
        memory_categories=None,
        filters=filters,
        group_by=None,
        limit=None,
        needs_raw_memories=False,
    )


def _compact_question(question: str) -> str:
    normalized = question.lower().replace("a.m.", "am").replace("p.m.", "pm")
    return re.sub(r"\s+", " ", normalized).strip()


def _contains_time_direction(question: str, direction: str, hour: str) -> bool:
    return re.search(rf"\b{direction}\s+{hour}\s*(?:am)?\b", question) is not None


def _food_metadata_key(question: str) -> str | None:
    if "processed" in question:
        return "processed_food"
    if "outside" in question:
        return "outside_food"
    if "sweet" in question or "sweets" in question:
        return "sweet"
    if "junk" in question:
        return "junk_food"
    return None


def _date_range(start: datetime, end: datetime, label: str) -> DateRange:
    return DateRange(
        start_at=start.astimezone(timezone.utc),
        end_at=end.astimezone(timezone.utc),
        label=label,
    )


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _matches_filters(
    item: object,
    filters: Sequence[AnalyticsFilter],
    *,
    data_kind: str,
) -> bool:
    return all(
        _matches_filter(item, analytics_filter, data_kind=data_kind)
        for analytics_filter in filters
    )


def _matches_filter(
    item: object,
    analytics_filter: AnalyticsFilter,
    *,
    data_kind: str,
) -> bool:
    field = analytics_filter.field
    if analytics_filter.operator == "metadata_true":
        metadata = _metadata(item)
        key = (
            analytics_filter.value
            if isinstance(analytics_filter.value, str)
            else field
        )
        key = key.removeprefix("metadata.")
        return metadata.get(key) is True

    actual = _field_value(item, field, data_kind=data_kind)
    expected = analytics_filter.value
    operator = analytics_filter.operator

    if operator == "equals":
        return _normalize_text(actual) == _normalize_text(expected)
    if operator == "contains":
        return _contains(actual, expected)
    if operator == "contains_any":
        values = expected if isinstance(expected, list) else [expected]
        return any(_contains(actual, value) for value in values)
    if operator == "less_than":
        actual_number = _number(actual)
        expected_number = _number(expected)
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number < expected_number
        )
    if operator == "greater_than":
        actual_number = _number(actual)
        expected_number = _number(expected)
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number > expected_number
        )
    if operator == "greater_than_or_equal":
        actual_number = _number(actual)
        expected_number = _number(expected)
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number >= expected_number
        )
    if operator == "less_than_or_equal":
        actual_number = _number(actual)
        expected_number = _number(expected)
        return (
            actual_number is not None
            and expected_number is not None
            and actual_number <= expected_number
        )
    return False


def _field_value(item: object, field: str, *, data_kind: str) -> object:
    if field == "search_text":
        return " ".join(
            value
            for value in [
                getattr(item, "label", None),
                getattr(item, "value_text", None),
                getattr(item, "summary", None),
                getattr(item, "raw_text", None),
                " ".join(str(tag) for tag in getattr(item, "tags", []) or []),
            ]
            if isinstance(value, str)
        )
    if field.startswith("metadata."):
        return _metadata(item).get(field.removeprefix("metadata."))
    if data_kind == "memory" and field == "value_text":
        return getattr(item, "raw_text", None)
    if data_kind == "memory" and field == "label":
        return " ".join(
            value
            for value in [
                getattr(item, "summary", None),
                getattr(item, "raw_text", None),
            ]
            if isinstance(value, str)
        )
    return getattr(item, field, None)


def _contains(actual: object, expected: object) -> bool:
    if expected is None:
        return False
    if isinstance(actual, list):
        expected_text = str(expected).lower()
        return any(str(item).lower() == expected_text for item in actual)
    return str(expected).lower() in str(actual or "").lower()


def _metadata(item: object) -> dict[str, object]:
    metadata = getattr(item, "metadata_", None)
    if metadata is None:
        metadata = getattr(item, "metadata", None)
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


def _normalize_text(value: object) -> str:
    if isinstance(value, MemoryCategory):
        value = value.value
    return str(value or "").strip().lower()


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def _observation_payload(observation: object) -> dict[str, object]:
    return {
        "observation_type": getattr(observation, "observation_type", None),
        "label": getattr(observation, "label", None),
        "value_text": getattr(observation, "value_text", None),
        "value_number": getattr(observation, "value_number", None),
        "unit": getattr(observation, "unit", None),
        "occurred_at": getattr(observation, "occurred_at", None),
        "confidence": getattr(observation, "confidence", None),
        "metadata": _metadata(observation),
    }


def _memory_payload(memory: object) -> dict[str, object]:
    category = getattr(memory, "category", None)
    if isinstance(category, MemoryCategory):
        category = category.value
    return {
        "id": str(getattr(memory, "id", "")),
        "raw_text": getattr(memory, "raw_text", None),
        "summary": getattr(memory, "summary", None),
        "category": category,
        "tags": getattr(memory, "tags", []),
        "created_at": getattr(memory, "created_at", None),
    }


def _with_caveat(answer: str) -> str:
    if ANALYTICS_CAVEAT in answer:
        return answer
    return f"{answer.rstrip()} {ANALYTICS_CAVEAT}"

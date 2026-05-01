from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

from smarn.memories.analytics import (
    ANALYTICS_CAVEAT,
    ANALYTICS_UNSUPPORTED,
    INSUFFICIENT_DATA,
    AnalyticsFilter,
    AnalyticsPlan,
    AnalyticsPlannerService,
    AnalyticsService,
    DateRange,
    normalize_analytics_plan,
    parse_date_range,
)
from smarn.memories.categories import MemoryCategory


NOW = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
LAST_MONTH = DateRange(
    start_at=datetime(2026, 3, 31, 18, 30, tzinfo=timezone.utc),
    end_at=datetime(2026, 4, 30, 18, 30, tzinfo=timezone.utc),
    label="last month",
)
THIS_WEEK = DateRange(
    start_at=datetime(2026, 4, 26, 18, 30, tzinfo=timezone.utc),
    end_at=NOW,
    label="this week",
)


class StaticPlanner:
    def __init__(self, plan: AnalyticsPlan | Exception) -> None:
        self.plan_value = plan

    def plan(self, question: str, *, now: datetime | None = None) -> AnalyticsPlan:
        del question, now
        if isinstance(self.plan_value, Exception):
            raise self.plan_value
        return self.plan_value


class StaticSynthesisService:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def synthesize(
        self,
        *,
        question: str,
        plan: AnalyticsPlan,
        observations: list[object],
        memories: list[object],
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "plan": plan,
                "observations": observations,
                "memories": memories,
            }
        )
        return f"{self.response} {ANALYTICS_CAVEAT}"


class StaticLLMProvider:
    def __init__(self, response: str) -> None:
        self.response = response

    def complete(self, *, system_prompt: str, user_prompt: str) -> str:
        del system_prompt, user_prompt
        return self.response


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
        observation_types: list[str] | None,
    ) -> list[object]:
        self.calls.append(
            {
                "user_id": user_id,
                "start_at": start_at,
                "end_at": end_at,
                "observation_types": observation_types,
            }
        )
        if not observation_types:
            return self.observations
        return [
            observation
            for observation in self.observations
            if observation.observation_type in observation_types
        ]


class StaticMemoryRepository:
    def __init__(self, memories: list[object] | None = None) -> None:
        self.memories = memories or []
        self.calls: list[dict[str, object]] = []

    def list_created_between(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        user_id: str | None = None,
        category: object | None = None,
    ) -> list[object]:
        self.calls.append(
            {
                "start_at": start_at,
                "end_at": end_at,
                "user_id": user_id,
                "category": category,
            }
        )
        if category is None:
            return self.memories
        return [memory for memory in self.memories if memory.category == category]


def test_planner_normalizes_wake_time_after_10_am_plan() -> None:
    planner = AnalyticsPlannerService(
        StaticLLMProvider(
            """
            {
              "answer_type": "count",
              "date_range": {
                "start_at": "2026-03-31T18:30:00+00:00",
                "end_at": "2026-04-30T18:30:00+00:00",
                "label": "last month"
              },
              "observation_types": ["wake_time"],
              "memory_categories": null,
              "filters": [
                {
                  "field": "value_number",
                  "operator": "greater_than_or_equal",
                  "value": 600
                }
              ],
              "group_by": null,
              "limit": null,
              "needs_raw_memories": false
            }
            """
        )
    )

    plan = planner.plan(
        "How many times did I wake up after 10 AM last month?",
        now=NOW,
    )

    assert plan.answer_type == "count"
    assert plan.observation_types == ["wake_time"]
    assert plan.filters == [
        AnalyticsFilter(
            field="value_number",
            operator="greater_than_or_equal",
            value=600,
        )
    ]


def test_parse_plan_payload_accepts_json_fences() -> None:
    plan = normalize_analytics_plan(
        """
        ```json
        {
          "answer_type": "count",
          "date_range": {
            "start_at": "2026-04-01T00:00:00+00:00",
            "end_at": "2026-05-01T00:00:00+00:00",
            "label": "last month"
          },
          "observation_types": ["wake_time"],
          "memory_categories": null,
          "filters": {"field": "value_number", "operator": "less_than", "value": 480},
          "group_by": null,
          "limit": 0,
          "needs_raw_memories": false
        }
        ```
        """,
        now=NOW,
    )

    assert plan.answer_type == "count"
    assert plan.filters == [AnalyticsFilter("value_number", "less_than", 480)]
    assert plan.limit is None


def test_parse_plan_payload_accepts_leading_and_trailing_text() -> None:
    plan = normalize_analytics_plan(
        """
        Here is the plan:
        {
          "answer_type": "list",
          "date_range": {
            "start_at": "2026-04-26T18:30:00+00:00",
            "end_at": "2026-05-01T12:00:00+00:00",
            "label": "this week"
          },
          "observation_types": "learning_activity",
          "memory_categories": "learning",
          "filters": [
            {"field": "search_text", "operator": "contains", "value": "system design"}
          ],
          "group_by": "topic",
          "limit": null,
          "needs_raw_memories": "true"
        }
        Done.
        """,
        now=NOW,
    )

    assert plan.answer_type == "list"
    assert plan.observation_types == ["learning_activity"]
    assert plan.memory_categories == [MemoryCategory.LEARNING]
    assert plan.needs_raw_memories is True


def test_analytics_uses_fallback_for_wake_before_8_am_when_planner_fails() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                _observation("wake_time", value_number=Decimal("430")),
                _observation("wake_time", value_number=Decimal("500")),
            ]
        ),
        memory_repository=StaticMemoryRepository(),
        planner_service=StaticPlanner(RuntimeError("planner failed")),
    )

    answer = service.analyze(
        "How many times did I wake up before 8 AM last month?",
        user_id="telegram-user",
        now=NOW,
    )

    assert "You logged this 1 time" in answer.text
    assert ANALYTICS_CAVEAT in answer.text


def test_analytics_uses_fallback_for_wake_after_10_am_when_planner_fails() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                _observation("wake_time", value_number=Decimal("599")),
                _observation("wake_time", value_number=Decimal("600")),
                _observation("wake_time", value_number=Decimal("660")),
            ]
        ),
        memory_repository=StaticMemoryRepository(),
        planner_service=StaticPlanner(RuntimeError("planner failed")),
    )

    answer = service.analyze(
        "how many times did I wake up after 10AM in the last 30 days?",
        user_id="telegram-user",
        now=NOW,
    )

    assert "You logged this 2 times" in answer.text
    assert ANALYTICS_CAVEAT in answer.text


def test_analytics_uses_fallback_for_dsa_problem_counts_when_planner_fails() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository([]),
        memory_repository=StaticMemoryRepository(
            [
                _memory(
                    raw_text="Solved two LeetCode DSA problems.",
                    category=MemoryCategory.LEARNING,
                    tags=["DSA", "LeetCode", "solved"],
                ),
                _memory(
                    raw_text="Read about Kafka partitions.",
                    category=MemoryCategory.LEARNING,
                    tags=["system design"],
                ),
            ]
        ),
        planner_service=StaticPlanner(RuntimeError("planner failed")),
    )

    answer = service.analyze(
        "how many DSA problems did I solve in the last 30 days?",
        user_id="telegram-user",
        now=NOW,
    )

    assert "You logged this 1 time" in answer.text
    assert ANALYTICS_CAVEAT in answer.text


def test_analytics_uses_fallback_for_system_design_topics_when_planner_fails() -> None:
    synthesis = StaticSynthesisService("You studied load balancers and caching.")
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                _observation(
                    "learning_activity",
                    value_text="Studied system design caching.",
                )
            ]
        ),
        memory_repository=StaticMemoryRepository(
            [
                _memory(
                    raw_text="Studied system design load balancers.",
                    category=MemoryCategory.LEARNING,
                    tags=["system design", "load balancers"],
                )
            ]
        ),
        planner_service=StaticPlanner(RuntimeError("planner failed")),
        synthesis_service=synthesis,
    )

    answer = service.analyze(
        "which system design topic did I study this week?",
        user_id="telegram-user",
        now=NOW,
    )

    assert "You studied load balancers and caching." in answer.text
    assert ANALYTICS_CAVEAT in answer.text
    assert len(synthesis.calls[0]["observations"]) == 1
    assert len(synthesis.calls[0]["memories"]) == 1


def test_analytics_returns_clear_message_for_invalid_planner_json() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository([]),
        memory_repository=StaticMemoryRepository(),
        planner_service=AnalyticsPlannerService(StaticLLMProvider("not json")),
    )

    answer = service.analyze(
        "What did I mostly work on in the last 30 days?",
        user_id="telegram-user",
        now=NOW,
    )

    assert ANALYTICS_UNSUPPORTED in answer.text
    assert ANALYTICS_CAVEAT in answer.text


def test_analytics_planning_failure_log_is_debug_safe(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="smarn.memories.analytics")
    private_question = "how many times did I wake up before 8AM in the last 30 days?"
    service = AnalyticsService(
        observation_repository=StaticObservationRepository([]),
        memory_repository=StaticMemoryRepository(),
        planner_service=AnalyticsPlannerService(StaticLLMProvider("not json")),
    )

    service.analyze(private_question, user_id="telegram-user", now=NOW)

    record = next(
        item for item in caplog.records if item.message == "analytics_planning_failed"
    )
    assert record.error_type == "AnalyticsPlanningError"
    assert record.planner_response_length == len("not json")
    assert record.user_id == "telegram-user"
    assert private_question not in caplog.text
    assert "not json" not in caplog.text


def test_parse_date_range_supports_common_relative_ranges() -> None:
    today = parse_date_range("today", now=NOW)
    yesterday = parse_date_range("yesterday", now=NOW)
    last_week = parse_date_range("last week", now=NOW)
    past_30 = parse_date_range("past 30 days", now=NOW)

    assert today.label == "today"
    assert yesterday.label == "yesterday"
    assert last_week.label == "last week"
    assert past_30.label == "the last 30 days"


def test_analytics_returns_clear_message_when_no_observations_exist() -> None:
    observation_repository = StaticObservationRepository([])
    service = AnalyticsService(
        observation_repository=observation_repository,
        memory_repository=StaticMemoryRepository(),
        planner_service=StaticPlanner(
            AnalyticsPlan(
                answer_type="count",
                date_range=LAST_MONTH,
                observation_types=["wake_time"],
                memory_categories=None,
                filters=[
                    AnalyticsFilter(
                        field="value_number",
                        operator="less_than",
                        value=480,
                    )
                ],
                group_by=None,
                limit=None,
                needs_raw_memories=False,
            )
        ),
    )

    answer = service.analyze(
        "How many times did I wake up before 8 AM last month?",
        user_id="telegram-user",
        now=NOW,
    )

    assert INSUFFICIENT_DATA in answer.text
    assert ANALYTICS_CAVEAT in answer.text
    assert observation_repository.calls[0]["user_id"] == "telegram-user"


def test_analytics_counts_wake_time_after_10_am() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                _observation("wake_time", value_number=Decimal("450")),
                _observation("wake_time", value_number=Decimal("600")),
                _observation("wake_time", value_number=Decimal("670")),
            ]
        ),
        memory_repository=StaticMemoryRepository(),
        planner_service=StaticPlanner(
            AnalyticsPlan(
                answer_type="count",
                date_range=LAST_MONTH,
                observation_types=["wake_time"],
                memory_categories=None,
                filters=[
                    AnalyticsFilter(
                        field="value_number",
                        operator="greater_than_or_equal",
                        value=600,
                    )
                ],
                group_by=None,
                limit=None,
                needs_raw_memories=False,
            )
        ),
    )

    answer = service.analyze(
        "How many times did I wake up after 10 AM last month?",
        user_id="telegram-user",
        now=NOW,
    )

    assert "2 times" in answer.text
    assert ANALYTICS_CAVEAT in answer.text


def test_analytics_counts_food_intake_from_metadata() -> None:
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                _observation("food_intake", label="chips", metadata={"processed_food": True}),
                _observation("food_intake", label="chocolate cake", metadata={"sweet": True}),
                _observation("food_intake", label="apple", metadata={"healthy_food": True}),
            ]
        ),
        memory_repository=StaticMemoryRepository(),
        planner_service=StaticPlanner(
            AnalyticsPlan(
                answer_type="count",
                date_range=LAST_MONTH,
                observation_types=["food_intake"],
                memory_categories=None,
                filters=[
                    AnalyticsFilter(
                        field="metadata.processed_food",
                        operator="metadata_true",
                        value=None,
                    )
                ],
                group_by=None,
                limit=None,
                needs_raw_memories=False,
            )
        ),
    )

    answer = service.analyze(
        "How many times did I eat processed food last month?",
        user_id="telegram-user",
        now=NOW,
    )

    assert "1 time" in answer.text
    assert ANALYTICS_CAVEAT in answer.text


def test_analytics_counts_dsa_problem_memories_when_observations_are_absent() -> None:
    memory_repository = StaticMemoryRepository(
        [
            _memory(
                raw_text="Solved 3 LeetCode DSA problems on graphs.",
                category=MemoryCategory.LEARNING,
                tags=["DSA", "LeetCode", "solved"],
            ),
            _memory(
                raw_text="Studied Postgres indexes.",
                category=MemoryCategory.LEARNING,
                tags=["database"],
            ),
        ]
    )
    service = AnalyticsService(
        observation_repository=StaticObservationRepository([]),
        memory_repository=memory_repository,
        planner_service=StaticPlanner(
            AnalyticsPlan(
                answer_type="count",
                date_range=LAST_MONTH,
                observation_types=["learning_activity"],
                memory_categories=[MemoryCategory.LEARNING],
                filters=[
                    AnalyticsFilter(
                        field="tags",
                        operator="contains_any",
                        value=["DSA", "problem", "LeetCode", "solved"],
                    )
                ],
                group_by=None,
                limit=None,
                needs_raw_memories=True,
            )
        ),
    )

    answer = service.analyze(
        "How many DSA problems did I solve last month?",
        user_id="telegram-user",
        now=NOW,
    )

    assert "1 time" in answer.text
    assert ANALYTICS_CAVEAT in answer.text
    assert memory_repository.calls[0]["category"] is MemoryCategory.LEARNING


def test_analytics_summarizes_system_design_topics_from_selected_data() -> None:
    synthesis = StaticSynthesisService("You studied caching, sharding, and queues.")
    service = AnalyticsService(
        observation_repository=StaticObservationRepository(
            [
                    _observation(
                        "learning_activity",
                        label="system design",
                        value_text="Studied system design caching and sharding.",
                    )
            ]
        ),
        memory_repository=StaticMemoryRepository(
            [
                _memory(
                    raw_text="Studied system design queues and backpressure.",
                    category=MemoryCategory.LEARNING,
                    tags=["system design", "queues"],
                )
            ]
        ),
        planner_service=StaticPlanner(
            AnalyticsPlan(
                answer_type="list",
                date_range=THIS_WEEK,
                observation_types=["learning_activity"],
                memory_categories=[MemoryCategory.LEARNING],
                filters=[
                    AnalyticsFilter(
                        field="value_text",
                        operator="contains",
                        value="system design",
                    )
                ],
                group_by="topic",
                limit=10,
                needs_raw_memories=True,
            )
        ),
        synthesis_service=synthesis,
    )

    answer = service.analyze(
        "Which system design topics did I study this week?",
        user_id="telegram-user",
        now=NOW,
    )

    assert answer.text == (
        "You studied caching, sharding, and queues. "
        "This is based only on your saved memories."
    )
    assert len(synthesis.calls) == 1
    assert len(synthesis.calls[0]["observations"]) == 1
    assert len(synthesis.calls[0]["memories"]) == 1


def _observation(
    observation_type: str,
    *,
    label: str | None = None,
    value_text: str | None = None,
    value_number: Decimal | None = None,
    metadata: dict[str, object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        observation_type=observation_type,
        value_number=value_number,
        label=label,
        value_text=value_text,
        metadata_=metadata or {},
        occurred_at=NOW,
        confidence=0.9,
    )


def _memory(
    *,
    raw_text: str,
    category: MemoryCategory,
    tags: list[str],
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        raw_text=raw_text,
        summary=None,
        category=category,
        tags=tags,
        created_at=NOW,
    )

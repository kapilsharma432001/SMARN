from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from typing import Any

from smarn.memories.llm import LLMProvider

logger = logging.getLogger(__name__)

OBSERVATION_TYPES = {
    "wake_time",
    "sleep_time",
    "food_intake",
    "exercise",
    "mood",
    "work_activity",
    "learning_activity",
    "health_event",
}

FOOD_CATEGORY_KEYS = {
    "processed_food",
    "sweet",
    "junk_food",
    "healthy_food",
    "vegetarian",
}


@dataclass(frozen=True)
class ExtractedObservation:
    observation_type: str
    label: str | None
    value_text: str | None
    value_number: float | None
    unit: str | None
    occurred_at: datetime | None
    confidence: float
    metadata: dict[str, object]


class ObservationExtractionService:
    def __init__(self, llm_provider: LLMProvider) -> None:
        self._llm_provider = llm_provider

    def extract(self, raw_text: str) -> list[ExtractedObservation]:
        try:
            response = self._llm_provider.complete(
                system_prompt=(
                    "Extract structured observations from private memory text. "
                    "Return only JSON with an observations array. Extract only clear "
                    "facts for these observation_type values: wake_time, sleep_time, "
                    "food_intake, exercise, mood, work_activity, learning_activity, "
                    "health_event. For wake_time and sleep_time, set value_number to "
                    "minutes after midnight and unit to minutes_after_midnight. For "
                    "food_intake, set label to the food item and metadata booleans "
                    "when applicable: processed_food, sweet, junk_food, healthy_food, "
                    "vegetarian. Include confidence from 0 to 1. Use null when a "
                    "field is unknown. Do not rewrite the raw memory."
                ),
                user_prompt=json.dumps({"raw_text": raw_text}, ensure_ascii=False),
            )
            observations = normalize_observations(_parse_observation_payload(response))
            logger.info(
                "observation_extraction_completed",
                extra={"observation_count": len(observations)},
            )
            return observations
        except Exception:
            logger.exception("observation_extraction_failed")
            return []


def normalize_observations(payload: object) -> list[ExtractedObservation]:
    if isinstance(payload, dict):
        raw_items = payload.get("observations", [])
    else:
        raw_items = payload

    if not isinstance(raw_items, list):
        return []

    observations: list[ExtractedObservation] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_observation(item)
        if normalized is not None:
            observations.append(normalized)
    return observations


def _normalize_observation(data: dict[str, Any]) -> ExtractedObservation | None:
    observation_type = _clean_string(data.get("observation_type"))
    if observation_type not in OBSERVATION_TYPES:
        return None

    label = _clean_string(data.get("label"))
    value_text = _clean_string(data.get("value_text"))
    value_number = _coerce_float(data.get("value_number"))
    unit = _clean_string(data.get("unit"))
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)

    if observation_type in {"wake_time", "sleep_time"}:
        parsed_minutes = _coerce_minutes_after_midnight(value_number, value_text, label)
        value_number = parsed_minutes
        unit = "minutes_after_midnight" if parsed_minutes is not None else unit

    if observation_type == "food_intake":
        metadata = _normalize_food_metadata(metadata)

    return ExtractedObservation(
        observation_type=observation_type,
        label=label,
        value_text=value_text,
        value_number=value_number,
        unit=unit,
        occurred_at=_parse_datetime(data.get("occurred_at")),
        confidence=_coerce_confidence(data.get("confidence")),
        metadata=metadata,
    )


def _parse_observation_payload(content: str) -> object:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        cleaned = cleaned.removesuffix("```").strip()

    object_start = cleaned.find("{")
    array_start = cleaned.find("[")
    starts = [index for index in [object_start, array_start] if index != -1]
    if not starts:
        raise ValueError("LLM response did not contain JSON.")
    start = min(starts)
    end = cleaned.rfind("}" if cleaned[start] == "{" else "]")
    if end == -1 or end < start:
        raise ValueError("LLM response JSON was incomplete.")

    return json.loads(cleaned[start : end + 1])


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_confidence(value: object) -> float:
    number = _coerce_float(value)
    if number is None:
        return 0.0
    return min(1.0, max(0.0, number))


def _coerce_minutes_after_midnight(
    value_number: float | None,
    value_text: str | None,
    label: str | None,
) -> float | None:
    if value_number is not None and 0 <= value_number < 1440:
        return float(int(value_number))
    for candidate in [value_text, label]:
        parsed = _parse_time_to_minutes(candidate)
        if parsed is not None:
            return float(parsed)
    return None


def _parse_time_to_minutes(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(
        r"\b(?P<hour>\d{1,2})(?::(?P<minute>\d{2}))?\s*(?P<ampm>a\.?m\.?|p\.?m\.?)?\b",
        value.lower(),
    )
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    ampm = match.group("ampm")
    if minute > 59 or hour > 23:
        return None
    if ampm:
        if hour < 1 or hour > 12:
            return None
        if ampm.startswith("p") and hour != 12:
            hour += 12
        if ampm.startswith("a") and hour == 12:
            hour = 0
    return hour * 60 + minute


def _parse_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    cleaned = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        try:
            parsed_date = date.fromisoformat(cleaned)
        except ValueError:
            return None
        return datetime.combine(parsed_date, time.min, tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _normalize_food_metadata(metadata: dict[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    raw_categories = metadata.get("categories")
    if isinstance(raw_categories, list):
        for category in raw_categories:
            if isinstance(category, str):
                key = category.strip().lower().replace(" ", "_")
                if key in FOOD_CATEGORY_KEYS:
                    normalized[key] = True

    for key, value in metadata.items():
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key in FOOD_CATEGORY_KEYS:
            normalized[normalized_key] = _coerce_bool(value)
        elif normalized_key != "categories":
            normalized[normalized_key] = value
    return normalized


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "y", "1"}
    if isinstance(value, (int, float)):
        return bool(value)
    return False

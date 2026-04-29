from __future__ import annotations

from enum import Enum


class MemoryCategory(str, Enum):
    PERSONAL = "personal"
    WORK = "work"
    LEARNING = "learning"
    COMMAND = "command"
    IDEA = "idea"
    REMINDER_CANDIDATE = "reminder_candidate"
    UNKNOWN = "unknown"


def memory_category_values() -> list[str]:
    return [category.value for category in MemoryCategory]


def coerce_memory_category(category: MemoryCategory | str | None) -> MemoryCategory:
    if category is None:
        return MemoryCategory.UNKNOWN
    if isinstance(category, MemoryCategory):
        return category
    return MemoryCategory(category)

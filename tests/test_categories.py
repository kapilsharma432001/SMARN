from __future__ import annotations

import pytest

from smarn.memories.categories import (
    MemoryCategory,
    coerce_memory_category,
    memory_category_values,
)


def test_memory_category_values_match_allowed_set() -> None:
    assert memory_category_values() == [
        "personal",
        "work",
        "learning",
        "command",
        "idea",
        "reminder_candidate",
        "unknown",
    ]


def test_coerce_memory_category_defaults_to_unknown() -> None:
    assert coerce_memory_category(None) is MemoryCategory.UNKNOWN
    assert coerce_memory_category("work") is MemoryCategory.WORK


def test_coerce_memory_category_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        coerce_memory_category("invalid")

from __future__ import annotations

import math

from smarn.memories.embeddings import HashEmbeddingProvider


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    provider = HashEmbeddingProvider(dimensions=16)

    first = provider.embed("remember the blue notebook")
    second = provider.embed("remember the blue notebook")

    assert first == second
    assert len(first) == 16
    assert math.isclose(sum(value * value for value in first), 1.0)

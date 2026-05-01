from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone


@dataclass(frozen=True)
class PendingVoiceQuestion:
    user_id: str
    expires_at: datetime


class PendingVoiceQuestionStore:
    def __init__(self, *, ttl: timedelta = timedelta(minutes=2)) -> None:
        self._ttl = ttl
        self._pending: dict[str, PendingVoiceQuestion] = {}

    def mark_pending(
        self,
        user_id: str,
        *,
        now: datetime | None = None,
    ) -> PendingVoiceQuestion:
        current = now or datetime.now(timezone.utc)
        pending = PendingVoiceQuestion(
            user_id=user_id,
            expires_at=current + self._ttl,
        )
        self._pending[user_id] = pending
        return pending

    def consume_if_pending(
        self,
        user_id: str | None,
        *,
        now: datetime | None = None,
    ) -> bool:
        if user_id is None:
            return False
        current = now or datetime.now(timezone.utc)
        pending = self._pending.get(user_id)
        if pending is None:
            return False
        if pending.expires_at <= current:
            self._pending.pop(user_id, None)
            return False
        self._pending.pop(user_id, None)
        return True

    def expire_pending(self, *, now: datetime | None = None) -> None:
        current = now or datetime.now(timezone.utc)
        expired = [
            user_id
            for user_id, pending in self._pending.items()
            if pending.expires_at <= current
        ]
        for user_id in expired:
            self._pending.pop(user_id, None)

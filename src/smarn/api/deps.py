from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from smarn.db.session import get_session


def get_db() -> Generator[Session, None, None]:
    yield from get_session()

from __future__ import annotations

from smarn.logging import _redact_sensitive_values


def test_log_redaction_masks_openai_style_keys() -> None:
    redacted = _redact_sensitive_values(
        {
            "message": "key sk-test_1234567890abcdef in payload",
            "nested": ["https://api.telegram.org/bot123456:ABC/sendMessage"],
        }
    )

    assert redacted["message"] == "key sk-[REDACTED] in payload"
    assert redacted["nested"] == ["https://api.telegram.org/bot[REDACTED]/sendMessage"]

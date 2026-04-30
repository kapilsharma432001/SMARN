from __future__ import annotations

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from smarn.config import get_settings
from smarn.db.session import session_scope
from smarn.logging import configure_logging
from smarn.memories.review import ReviewService
from smarn.memories.service import MemoryService
from smarn.memories.voice import VoiceMemoryService, format_voice_confirmation

logger = logging.getLogger(__name__)


def _telegram_user_id(update: Update) -> str | None:
    if update.effective_user is None:
        return None
    return str(update.effective_user.id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return

    await update.effective_message.reply_text(
        "SMARN is ready. Use /remember <text>, voice notes, /ask <question>, "
        "/daily_review, or /weekly_review."
    )


async def remember(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return

    content = " ".join(context.args).strip()
    if not content:
        await update.effective_message.reply_text("Usage: /remember <text>")
        return

    user_id = _telegram_user_id(update)
    logger.info(
        "telegram_remember_received",
        extra={
            "user_id": user_id,
            "message_id": update.effective_message.message_id,
            "text_length": len(content),
        },
    )

    with session_scope() as session:
        MemoryService(session).remember(
            content,
            user_id=user_id,
            source="telegram",
        )

    logger.info(
        "telegram_remember_completed",
        extra={
            "user_id": user_id,
            "message_id": update.effective_message.message_id,
        },
    )
    await update.effective_message.reply_text("Remembered.")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return

    question = " ".join(context.args).strip()
    if not question:
        await update.effective_message.reply_text("Usage: /ask <question>")
        return

    user_id = _telegram_user_id(update)
    logger.info(
        "telegram_ask_received",
        extra={
            "user_id": user_id,
            "message_id": update.effective_message.message_id,
            "question_length": len(question),
        },
    )

    with session_scope() as session:
        answer = MemoryService(session).ask(
            question,
            user_id=user_id,
        )

    logger.info(
        "telegram_ask_completed",
        extra={
            "user_id": user_id,
            "message_id": update.effective_message.message_id,
            "retrieved_memory_count": len(answer.memories),
        },
    )
    await update.effective_message.reply_text(answer.text)


async def voice_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_message.voice is None:
        return

    user_id = _telegram_user_id(update)
    voice = update.effective_message.voice
    message_id = update.effective_message.message_id

    logger.info(
        "telegram_voice_received",
        extra={
            "user_id": user_id,
            "message_id": message_id,
            "file_id": voice.file_id,
            "file_unique_id": voice.file_unique_id,
            "duration_seconds": voice.duration,
            "file_size": voice.file_size,
            "mime_type": voice.mime_type,
        },
    )

    try:
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "telegram_voice.ogg"
            logger.info(
                "telegram_voice_get_file_started",
                extra={
                    "user_id": user_id,
                    "message_id": message_id,
                    "file_id": voice.file_id,
                },
            )
            telegram_file = await context.bot.get_file(
                voice.file_id
            )
            logger.info(
                "telegram_voice_download_started",
                extra={
                    "user_id": user_id,
                    "message_id": message_id,
                    "file_id": voice.file_id,
                },
            )
            await telegram_file.download_to_drive(custom_path=audio_path)
            logger.info(
                "telegram_voice_downloaded",
                extra={
                    "user_id": user_id,
                    "message_id": message_id,
                    "file_id": voice.file_id,
                    "downloaded_bytes": audio_path.stat().st_size,
                },
            )

            with session_scope() as session:
                result = VoiceMemoryService(session).remember_voice(
                    audio_path,
                    user_id=user_id,
                )
    except Exception:
        logger.exception(
            "telegram_voice_processing_failed",
            extra={"user_id": user_id, "message_id": message_id},
        )
        await update.effective_message.reply_text(
            "I could not process that voice note. Please try again or send it as text."
        )
        return

    if not result.saved or result.memory is None:
        logger.warning(
            "telegram_voice_not_saved",
            extra={
                "user_id": user_id,
                "message_id": message_id,
                "error_message": result.error_message,
            },
        )
        await update.effective_message.reply_text(
            result.error_message
            or "I could not process that voice note. Please try again or send it as text."
        )
        return

    logger.info(
        "telegram_voice_reply_sent",
        extra={
            "user_id": user_id,
            "message_id": message_id,
            "memory_id": str(result.memory.id),
        },
    )
    await update.effective_message.reply_text(format_voice_confirmation(result.memory))


async def daily_review(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return

    user_id = _telegram_user_id(update)
    logger.info(
        "telegram_daily_review_received",
        extra={"user_id": user_id, "message_id": update.effective_message.message_id},
    )
    with session_scope() as session:
        review = ReviewService(session).daily_review(user_id=user_id)

    logger.info(
        "telegram_daily_review_completed",
        extra={
            "user_id": user_id,
            "message_id": update.effective_message.message_id,
            "memory_count": len(review.memories),
        },
    )
    await update.effective_message.reply_text(review.text)


async def weekly_review(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return

    user_id = _telegram_user_id(update)
    logger.info(
        "telegram_weekly_review_received",
        extra={"user_id": user_id, "message_id": update.effective_message.message_id},
    )
    with session_scope() as session:
        review = ReviewService(session).weekly_review(user_id=user_id)

    logger.info(
        "telegram_weekly_review_completed",
        extra={
            "user_id": user_id,
            "message_id": update.effective_message.message_id,
            "memory_count": len(review.memories),
        },
    )
    await update.effective_message.reply_text(review.text)


def build_application(token: str | None = None) -> Application:
    settings = get_settings()
    configure_logging(settings)
    token_value = token

    if token_value is None and settings.telegram_bot_token is not None:
        token_value = settings.telegram_bot_token.get_secret_value()

    if not token_value:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required to run the Telegram bot.")

    application = Application.builder().token(token_value).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("remember", remember))
    application.add_handler(CommandHandler("ask", ask))
    application.add_handler(CommandHandler("daily_review", daily_review))
    application.add_handler(CommandHandler("weekly_review", weekly_review))
    application.add_handler(MessageHandler(filters.VOICE, voice_note))
    return application


def main() -> None:
    configure_logging()
    logger.info("telegram_bot_starting")
    build_application().run_polling()


if __name__ == "__main__":
    main()

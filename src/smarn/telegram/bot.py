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

    with session_scope() as session:
        MemoryService(session).remember(
            content,
            user_id=_telegram_user_id(update),
            source="telegram",
        )

    await update.effective_message.reply_text("Remembered.")


async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None:
        return

    question = " ".join(context.args).strip()
    if not question:
        await update.effective_message.reply_text("Usage: /ask <question>")
        return

    with session_scope() as session:
        answer = MemoryService(session).ask(
            question,
            user_id=_telegram_user_id(update),
        )

    await update.effective_message.reply_text(answer.text)


async def voice_note(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_message is None or update.effective_message.voice is None:
        return

    user_id = _telegram_user_id(update)

    try:
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "telegram_voice.ogg"
            telegram_file = await context.bot.get_file(
                update.effective_message.voice.file_id
            )
            await telegram_file.download_to_drive(custom_path=audio_path)

            with session_scope() as session:
                result = VoiceMemoryService(session).remember_voice(
                    audio_path,
                    user_id=user_id,
                )
    except Exception:
        logger.exception("telegram_voice_processing_failed", extra={"user_id": user_id})
        await update.effective_message.reply_text(
            "I could not process that voice note. Please try again or send it as text."
        )
        return

    if not result.saved or result.memory is None:
        await update.effective_message.reply_text(
            result.error_message
            or "I could not process that voice note. Please try again or send it as text."
        )
        return

    await update.effective_message.reply_text(format_voice_confirmation(result.memory))


async def daily_review(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return

    with session_scope() as session:
        review = ReviewService(session).daily_review(user_id=_telegram_user_id(update))

    await update.effective_message.reply_text(review.text)


async def weekly_review(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return

    with session_scope() as session:
        review = ReviewService(session).weekly_review(user_id=_telegram_user_id(update))

    await update.effective_message.reply_text(review.text)


def build_application(token: str | None = None) -> Application:
    settings = get_settings()
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
    build_application().run_polling()


if __name__ == "__main__":
    main()

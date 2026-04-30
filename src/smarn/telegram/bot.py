from __future__ import annotations

from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

from smarn.config import get_settings
from smarn.db.session import session_scope
from smarn.memories.service import MemoryService


def _telegram_user_id(update: Update) -> str | None:
    if update.effective_user is None:
        return None
    return str(update.effective_user.id)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    del context
    if update.effective_message is None:
        return

    await update.effective_message.reply_text(
        "SMARN is ready. Use /remember <text> to store a memory and /ask <question> to search."
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
    return application


def main() -> None:
    build_application().run_polling()


if __name__ == "__main__":
    main()

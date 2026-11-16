"""
app/bot/__main__.py — Main entry point for the Voucher OCR Telegram Bot.
Run with: python -m app.bot
"""
import asyncio
import logging
import subprocess

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from app.bot.config import BOT_TOKEN
from app.bot.handlers import admin, common, search, upload
from app.bot.middlewares import AdminCheckMiddleware, RateLimitMiddleware

logging.basicConfig(level=logging.INFO)


def _apply_migrations() -> None:
    try:
        logging.info("Applying database migrations…")
        subprocess.run(["alembic", "upgrade", "head"], check=True)
        logging.info("Database migrations applied successfully.")
    except Exception as exc:
        logging.warning("Automatic migration failed or skipped: %s", exc)


async def main() -> None:
    _apply_migrations()

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()

    # Register middlewares (order matters: AdminCheck first so RateLimit can read is_admin)
    dp.message.middleware(AdminCheckMiddleware())
    dp.message.middleware(RateLimitMiddleware())

    # Register routers
    dp.include_router(common.router)
    dp.include_router(search.router)
    dp.include_router(upload.router)
    dp.include_router(admin.router)

    logging.info("Starting bot polling…")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())


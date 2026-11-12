"""
middlewares.py — aiogram 3.x middlewares for Voucher OCR Bot.

- AdminCheckMiddleware : injects data["is_admin"] = True/False
- RateLimitMiddleware  : limits search commands to RATE_LIMIT_SEARCH_PER_MINUTE per user
"""
import time
from collections import defaultdict, deque
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject

from app.bot.config import ADMIN_IDS, RATE_LIMIT_SEARCH_PER_MINUTE


class AdminCheckMiddleware(BaseMiddleware):
    """
    Sets data["is_admin"] = True if the message sender is in ADMIN_IDS.
    Works for all message updates; no-op for non-Message updates.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user = data.get("event_from_user")
        if user is not None:
            data["is_admin"] = user.id in ADMIN_IDS
        else:
            data["is_admin"] = False
        return await handler(event, data)


class RateLimitMiddleware(BaseMiddleware):
    """
    Sliding-window rate limit: at most RATE_LIMIT_SEARCH_PER_MINUTE requests
    per user per 60-second window.

    Only active on Message events; ignores other update types.
    Skipped for admin users (admins are not rate-limited).
    """

    def __init__(self) -> None:
        # user_id -> deque of timestamps (float, seconds since epoch)
        self._windows: dict[int, deque] = defaultdict(deque)

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message):
            return await handler(event, data)

        user = data.get("event_from_user")
        if user is None:
            return await handler(event, data)

        # Admins bypass rate limit
        if data.get("is_admin", False):
            return await handler(event, data)

        # Only apply to search-like text commands
        text = event.text or ""
        if not text.startswith("/cari"):
            return await handler(event, data)

        uid = user.id
        now = time.monotonic()
        window = self._windows[uid]

        # Drop entries older than 60 seconds
        while window and now - window[0] > 60.0:
            window.popleft()

        if len(window) >= RATE_LIMIT_SEARCH_PER_MINUTE:
            oldest = window[0]
            retry_after = int(60 - (now - oldest)) + 1
            await event.answer(
                f"⏳ Terlalu banyak permintaan. Coba lagi dalam {retry_after} detik."
            )
            return  # stop propagation

        window.append(now)
        return await handler(event, data)

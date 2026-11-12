"""
keyboards.py — Inline keyboard stubs for Voucher OCR Bot.
Extend as needed; kept minimal for MVP.
"""
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton


def confirm_upload_keyboard() -> InlineKeyboardMarkup:
    """Example keyboard stub — not wired to any handler yet."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="✅ Konfirmasi", callback_data="confirm_upload")],
            [InlineKeyboardButton(text="❌ Batalkan", callback_data="cancel_upload")],
        ]
    )

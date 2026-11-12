"""
handlers/search.py — /cari and /cek command handlers.
"""
import os

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message

from app.db.base import get_session
from app.db.repositories import get_batch_by_id, get_voucher_by_serial

router = Router()


def _build_caption(voucher, batch_code: str, *, include_detail: bool = False) -> str:
    prefix = "✅" if voucher.status == "verified" else "⚠️"
    lines = [
        f"{prefix} Voucher ditemukan!",
        "",
        f"Serial Number: {voucher.serial_number}",
        f"Kode Voucher: {voucher.voucher_code}",
        f"Status: {voucher.status}",
        f"Batch: {batch_code}",
    ]
    if voucher.status == "review":
        lines.append("ℹ️ Hasil OCR perlu verifikasi")

    if include_detail:
        conf = float(voucher.confidence) if getattr(voucher, "confidence", None) is not None else None
        lines += [
            "",
            f"Confidence: {conf:.2%}" if conf is not None else "Confidence: N/A",
            f"Batch ID: {voucher.batch_id}",
            f"Dibuat: {voucher.created_at.strftime('%Y-%m-%d %H:%M:%S') if voucher.created_at else 'N/A'}",
        ]

    return "\n".join(lines)


async def _handle_search(
    message: Message,
    serial: str,
    *,
    include_detail: bool = False,
    is_admin: bool = False,
) -> None:
    serial = serial.strip()
    if not serial:
        await message.answer("⚠️ Masukkan serial number. Contoh: /cari ABC123")
        return

    # Only admins can use /cek
    if include_detail and not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return

    async with get_session() as session:
        voucher = await get_voucher_by_serial(session, serial)
        if voucher is None:
            await message.answer(
                f"❌ Voucher dengan serial number <b>{serial}</b> tidak ditemukan.",
                parse_mode="HTML",
            )
            return

        batch = await get_batch_by_id(session, voucher.batch_id) if voucher.batch_id else None
        batch_code = batch.batch_code if batch else "N/A"

    caption = _build_caption(voucher, batch_code, include_detail=include_detail)

    crop_path = getattr(voucher, "crop_image", None)
    if crop_path and os.path.exists(crop_path):
        try:
            await message.answer_photo(
                FSInputFile(crop_path),
                caption=caption,
                parse_mode="HTML",
            )
            return
        except Exception:
            pass  # Fall through to text reply if photo fails

    await message.answer(caption, parse_mode="HTML")


@router.message(Command("cari"))
async def cmd_cari(message: Message, is_admin: bool = False) -> None:
    args = (message.text or "").split(maxsplit=1)
    serial = args[1] if len(args) > 1 else ""
    await _handle_search(message, serial, include_detail=False, is_admin=is_admin)


@router.message(Command("cek"))
async def cmd_cek(message: Message, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return
    args = (message.text or "").split(maxsplit=1)
    serial = args[1] if len(args) > 1 else ""
    await _handle_search(message, serial, include_detail=True, is_admin=is_admin)

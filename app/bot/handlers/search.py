"""
handlers/search.py — /cari and /cek command handlers.
"""
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import FSInputFile, Message

from app.db.base import get_session
from app.db.repositories import get_voucher_by_serial

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
        lines += [
            "",
            f"Confidence: {voucher.confidence_score:.2%}" if voucher.confidence_score is not None else "Confidence: N/A",
            f"Batch ID: {voucher.batch_id}",
            f"Dibuat: {voucher.created_at.strftime('%Y-%m-%d %H:%M:%S') if voucher.created_at else 'N/A'}",
        ]
        if voucher.raw_ocr_text:
            # Truncate long raw text
            raw = voucher.raw_ocr_text[:300]
            if len(voucher.raw_ocr_text) > 300:
                raw += "…"
            lines += ["", f"Raw OCR:\n<code>{raw}</code>"]

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
        result = await get_voucher_by_serial(session, serial)

    if result is None:
        await message.answer(
            f"❌ Voucher dengan serial number <b>{serial}</b> tidak ditemukan.",
            parse_mode="HTML",
        )
        return

    voucher, batch_code = result  # repository returns (Voucher, batch_code str)

    caption = _build_caption(voucher, batch_code, include_detail=include_detail)

    crop_path = voucher.crop_image_path
    if crop_path:
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

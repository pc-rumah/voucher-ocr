"""
handlers/search.py — /cari and /cek command handlers.
"""
import io
import logging
import os

import cv2

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, FSInputFile, Message

from app.db.base import get_session
from app.db.repositories import get_batch_by_id, get_voucher_by_serial

router = Router()
logger = logging.getLogger(__name__)


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


def _try_send_crop_from_original(batch) -> bytes | None:
    """
    Fallback: jika crop_image tidak ada di DB atau file hilang, coba crop
    voucher dari gambar original menggunakan serial_number sebagai petunjuk.
    Mengembalikan bytes JPEG, atau None jika tidak bisa.
    """
    if batch is None:
        return None
    original_path = getattr(batch, "original_image", None)
    if not original_path or not os.path.exists(original_path):
        return None

    try:
        from app.ocr.detector import VoucherDetector
        from app.ocr.preprocessor import ImagePreprocessor

        preprocessor = ImagePreprocessor()
        detector = VoucherDetector()

        img = preprocessor.load_image(original_path)
        vouchers_meta = detector.detect_vouchers(img)

        if not vouchers_meta:
            return None

        # Ambil voucher pertama (atau semua jika hanya satu yang ditemukan)
        meta = vouchers_meta[0]
        bbox = meta["bbox"]
        crop = detector.crop_voucher(img, bbox)

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()
    except Exception as exc:
        logger.warning("Gagal crop dari original: %s", exc)
        return None


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

    # --- Prioritas 1: crop_image tersimpan di DB dan file ada ---
    crop_path = getattr(voucher, "crop_image", None)
    if crop_path and os.path.exists(crop_path):
        try:
            await message.answer_photo(
                FSInputFile(crop_path),
                caption=caption,
                parse_mode="HTML",
            )
            logger.info("Mengirim crop dari file: %s", crop_path)
            return
        except Exception as exc:
            logger.warning("Gagal kirim FSInputFile (%s): %s", crop_path, exc)

    # --- Prioritas 2: crop on-the-fly dari gambar original ---
    crop_bytes = _try_send_crop_from_original(batch)
    if crop_bytes:
        try:
            await message.answer_photo(
                BufferedInputFile(crop_bytes, filename=f"voucher-{serial}.jpg"),
                caption=caption,
                parse_mode="HTML",
            )
            logger.info("Mengirim crop on-the-fly untuk serial: %s", serial)
            return
        except Exception as exc:
            logger.warning("Gagal kirim crop on-the-fly: %s", exc)

    # --- Fallback: teks saja ---
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

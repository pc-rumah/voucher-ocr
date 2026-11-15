"""
handlers/search.py — /cari and /cek command handlers.
"""
import io
import logging
import os
import re

import cv2

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, FSInputFile, Message

from app.db.base import get_session
from app.db.repositories import get_batch_by_id, get_voucher_by_serial, get_voucher_by_voucher_code

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


def _is_valid_crop(crop_path: str, batch) -> bool:
    """Reject a stored crop that is effectively the complete source image."""
    original_path = getattr(batch, "original_image", None)
    if not original_path or not os.path.exists(original_path):
        return True

    crop = cv2.imread(crop_path)
    original = cv2.imread(original_path)
    if crop is None or original is None:
        return False

    crop_area = crop.shape[0] * crop.shape[1]
    original_area = original.shape[0] * original.shape[1]
    if crop_area > 0.80 * original_area:
        logger.warning(
            "Mengabaikan crop yang mencakup %.1f%% gambar original.",
            100 * crop_area / original_area,
        )
        return False
    return True


def _try_send_crop_from_original(batch, serial: str) -> bytes | None:
    """
    Fallback: deteksi semua voucher dari gambar original, lalu pilih crop
    yang OCR-nya cocok dengan serial yang sedang dicari.

    Mengembalikan bytes JPEG, atau None jika tidak bisa / tidak aman.
    """
    if batch is None:
        return None
    original_path = getattr(batch, "original_image", None)
    if not original_path or not os.path.exists(original_path):
        return None

    try:
        from app.ocr.detector import VoucherDetector
        from app.ocr.preprocessor import ImagePreprocessor
        from app.ocr.recognizer import VoucherRecognizer

        preprocessor = ImagePreprocessor()
        detector = VoucherDetector()

        img = preprocessor.load_image(original_path)
        vouchers_meta = detector.detect_vouchers(img)
        if not vouchers_meta:
            return None

        recognizer = VoucherRecognizer()
        requested_serial = serial.strip()
        requested_digits = "".join(char for char in requested_serial if char.isdigit())
        for meta in vouchers_meta:
            crop = detector.crop_voucher(img, meta["bbox"])
            ocr_ready = preprocessor.preprocess_for_ocr(crop)

            # Coba baca barcode dari crop terlebih dahulu
            barcode_val: str | None = None
            try:
                from app.ocr.barcode import read_barcode  # noqa: PLC0415
                barcode_val = read_barcode(crop)
            except Exception:
                pass

            ocr_result = recognizer.extract_voucher_data(ocr_ready, barcode_value=barcode_val)
            detected = ocr_result["serial_number"]
            detected_voucher_code = ocr_result.get("voucher_code", "")
            detected_serial = "".join(char for char in (detected or "") if char.isdigit())
            matches = detected == requested_serial
            if requested_digits and requested_digits == detected_serial:
                matches = True
            # Fallback: cek juga cocok di voucher_code (nomor telepon kiri)
            if not matches and detected_voucher_code:
                vc_digits = "".join(char for char in detected_voucher_code if char.isdigit())
                if requested_digits and requested_digits == vc_digits:
                    matches = True
            if matches:
                _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                return buf.tobytes()

        logger.info("Serial %s tidak ditemukan pada crop hasil deteksi.", serial)
        return None
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

        # Fallback 1: normalisasi (hapus spasi/strip) lalu cari lagi
        if voucher is None:
            normalized = re.sub(r"[\s\-]", "", serial)
            if normalized != serial:
                voucher = await get_voucher_by_serial(session, normalized)

        # Fallback 2: cari berdasarkan voucher_code (nomor telepon kiri)
        if voucher is None:
            voucher = await get_voucher_by_voucher_code(session, serial)
        if voucher is None:
            normalized = re.sub(r"[\s\-]", "", serial)
            if normalized != serial:
                voucher = await get_voucher_by_voucher_code(session, normalized)

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
    if crop_path and os.path.exists(crop_path) and _is_valid_crop(crop_path, batch):
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
    target_serial = getattr(voucher, "serial_number", serial)
    crop_bytes = _try_send_crop_from_original(batch, target_serial)
    if crop_bytes:
        try:
            await message.answer_photo(
                BufferedInputFile(crop_bytes, filename=f"voucher-{target_serial}.jpg"),
                caption=caption,
                parse_mode="HTML",
            )
            logger.info("Mengirim crop on-the-fly untuk serial: %s", target_serial)
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

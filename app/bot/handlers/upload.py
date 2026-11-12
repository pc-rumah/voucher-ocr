"""
handlers/upload.py — /upload command and photo handler (admin only).
"""
import os
import secrets
from datetime import datetime, timezone

from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.types import Message

from app.bot.config import MAX_FILE_SIZE_MB, ORIGINALS_DIR
from app.db.base import get_session
from app.db.repositories import create_batch
from app.worker import process_voucher_batch

router = Router()

UPLOAD_INSTRUCTIONS = """\
📋 <b>Cara Mengunggah Voucher</b>

1. Pastikan foto voucher terlihat jelas dan tidak buram.
2. Kirim foto langsung ke chat ini (bukan sebagai dokumen).
3. Bot akan memproses OCR secara otomatis.

Gunakan /riwayat untuk memantau status pemrosesan batch.
"""


@router.message(Command("upload"))
async def cmd_upload(message: Message, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return
    await message.answer(UPLOAD_INSTRUCTIONS, parse_mode="HTML")


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Hanya admin yang dapat mengunggah foto.")
        return

    # Pick highest-resolution photo
    photo = message.photo[-1]

    # File size check (Telegram provides file_size in bytes)
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if photo.file_size and photo.file_size > max_bytes:
        await message.answer(
            f"❌ Ukuran file terlalu besar. Maksimum {MAX_FILE_SIZE_MB} MB."
        )
        return

    # Generate batch code: BATCH-YYYYMMDD-XXXXXX
    today = datetime.now(tz=timezone.utc).strftime("%Y%m%d")
    rand_hex = secrets.token_hex(3).upper()  # 6 hex chars
    batch_code = f"BATCH-{today}-{rand_hex}"

    # Prepare save path
    batch_dir = os.path.join(ORIGINALS_DIR, batch_code)
    os.makedirs(batch_dir, exist_ok=True)
    image_path = os.path.join(batch_dir, "original.jpg")

    # Download file
    file = await bot.get_file(photo.file_id)
    await bot.download_file(file.file_path, destination=image_path)

    # Create batch record in DB
    async with get_session() as session:
        batch = await create_batch(
            session,
            batch_code=batch_code,
            original_image=image_path,
            uploaded_by=message.from_user.id if message.from_user else None,
            status="processing",
        )
        batch_id = batch.id

    # Dispatch Celery task (fire-and-forget)
    process_voucher_batch.delay(batch_id, image_path)

    await message.answer(
        f"📸 Foto diterima! Batch <b>{batch_code}</b> sedang diproses. "
        f"Gunakan /riwayat untuk memeriksa hasilnya.",
        parse_mode="HTML",
    )

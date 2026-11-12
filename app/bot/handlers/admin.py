"""
handlers/admin.py — Admin-only handlers: /riwayat, /statistik, /review
"""
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.db.base import get_session
from app.db.repositories import (
    get_last_batches,
    get_voucher_stats,
    get_vouchers_for_review,
)

router = Router()


@router.message(Command("riwayat"))
async def cmd_riwayat(message: Message, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return

    async with get_session() as session:
        batches = await get_last_batches(session, limit=10)

    if not batches:
        await message.answer("📭 Belum ada batch yang diproses.")
        return

    lines = ["📂 <b>10 Batch Terakhir:</b>", ""]
    for b in batches:
        created = b.created_at.strftime("%Y-%m-%d %H:%M") if b.created_at else "N/A"
        lines.append(
            f"• <b>{b.batch_code}</b>\n"
            f"  Status: {b.status} | ✅ {b.total_success or 0} "
            f"| ⚠️ {b.total_review or 0}\n"
            f"  {created}"
        )

    await message.answer("\n".join(lines), parse_mode="HTML")


@router.message(Command("statistik"))
async def cmd_statistik(message: Message, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return

    async with get_session() as session:
        stats = await get_voucher_stats(session)

    text = (
        "📊 <b>Statistik Voucher</b>\n\n"
        f"Total Voucher  : {stats['total']}\n"
        f"✅ Verified    : {stats['verified']}\n"
        f"⚠️ Review      : {stats['review']}\n"
        f"❌ Failed      : {stats['failed']}\n"
        f"📦 Total Batch : {stats['total_batches']}"
    )
    await message.answer(text, parse_mode="HTML")


@router.message(Command("review"))
async def cmd_review(message: Message, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return

    async with get_session() as session:
        vouchers = await get_vouchers_for_review(session, limit=10)

    if not vouchers:
        await message.answer("✅ Tidak ada voucher yang perlu diverifikasi.")
        return

    lines = ["⚠️ <b>Voucher Perlu Verifikasi (maks. 10):</b>", ""]
    for v, batch_code in vouchers:
        serial_display = v.serial_number[:20] + "…" if len(v.serial_number or "") > 20 else (v.serial_number or "N/A")
        created = v.created_at.strftime("%Y-%m-%d %H:%M") if v.created_at else "N/A"
        lines.append(
            f"• Serial: <code>{serial_display}</code>\n"
            f"  Batch: {batch_code} | {created}"
        )

    await message.answer("\n".join(lines), parse_mode="HTML")

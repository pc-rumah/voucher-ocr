"""
handlers/admin.py — Admin-only handlers: /riwayat, /statistik, /review, /hapus_batch, /hapus
"""
import html

from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

from app.db.base import get_session
from app.db.repositories import (
    delete_batch_by_code,
    delete_voucher_by_serial,
    get_batch_failure_summary,
    get_failed_errors_by_batch,
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
        total_failed = getattr(b, "total_failed", 0) or 0
        failed_part = f" | ❌ {total_failed}" if total_failed > 0 else ""
        lines.append(
            f"• <b>{b.batch_code}</b>\n"
            f"  Status: {b.status} | ✅ {b.total_success or 0} "
            f"| ⚠️ {b.total_review or 0}{failed_part}\n"
            f"  {created}"
        )

        # Tampilkan alasan jika ada voucher review atau failed
        unverified = (b.total_review or 0) + total_failed
        if unverified > 0:
            async with get_session() as err_session:
                failure_summary = await get_batch_failure_summary(err_session, b.id)
            if failure_summary:
                for reason, count in failure_summary.items():
                    safe_reason = html.escape(str(reason))
                    lines.append(f"    ↳ {safe_reason}: <b>{count}</b>")

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


@router.message(Command("hapus_batch"))
async def cmd_hapus_batch(message: Message, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return

    args = (message.text or "").split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        await message.answer(
            "⚠️ Masukkan kode batch yang ingin dihapus.\n"
            "Contoh: <code>/hapus_batch BATCH-20261113-123456</code>",
            parse_mode="HTML",
        )
        return

    batch_code = args[1].strip()
    async with get_session() as session:
        success = await delete_batch_by_code(session, batch_code)

    if success:
        await message.answer(
            f"🗑️ Batch <b>{batch_code}</b> dan semua voucher di dalamnya berhasil dihapus.",
            parse_mode="HTML",
        )
    else:
        await message.answer(
            f"❌ Batch <b>{batch_code}</b> tidak ditemukan.",
            parse_mode="HTML",
        )


@router.message(Command("hapus"))
async def cmd_hapus_voucher(message: Message, is_admin: bool = False) -> None:
    if not is_admin:
        await message.answer("⛔ Perintah ini hanya untuk admin.")
        return

    args = (message.text or "").split(maxsplit=1)
    if len(args) < 2 or not args[1].strip():
        await message.answer(
            "⚠️ Masukkan serial number voucher yang ingin dihapus.\n"
            "Contoh: <code>/hapus 04690566144157</code>",
            parse_mode="HTML",
        )
        return

    serial = args[1].strip()
    async with get_session() as session:
        success = await delete_voucher_by_serial(session, serial)

    if success:
        await message.answer(
            f"🗑️ Voucher dengan serial <b>{serial}</b> berhasil dihapus.",
            parse_mode="HTML",
        )
    else:
        await message.answer(
            f"❌ Voucher dengan serial <b>{serial}</b> tidak ditemukan.",
            parse_mode="HTML",
        )

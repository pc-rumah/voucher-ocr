"""
handlers/common.py — Public command handlers: /start, /help
"""
from aiogram import Router
from aiogram.filters import Command
from aiogram.types import Message

router = Router()

HELP_TEXT = """\
🤖 <b>Voucher OCR Bot — Daftar Perintah</b>

<b>Perintah Publik:</b>
/start — Pesan sambutan
/help — Tampilkan daftar perintah ini
/cari <i>SERIAL</i> — Cari voucher berdasarkan serial number

<b>Perintah Admin:</b>
/upload — Instruksi mengunggah foto voucher
/cek <i>SERIAL</i> — Detail lengkap voucher (termasuk skor OCR)
/riwayat — 10 batch unggah terakhir
/statistik — Statistik total voucher
/review — Daftar voucher yang perlu diverifikasi
/hapus_batch <i>BATCH_CODE</i> — Hapus seluruh data 1 batch
/hapus <i>SERIAL</i> — Hapus 1 voucher satuan
📷 Kirim foto — Mulai proses OCR batch baru
"""


@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Selamat datang di Voucher OCR Bot! "
        "Gunakan /help untuk melihat perintah yang tersedia."
    )


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT, parse_mode="HTML")

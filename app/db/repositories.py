"""
Async repository functions for the Voucher OCR application.

All functions accept an ``AsyncSession`` as the first argument so they can be
composed inside a single transaction.  The caller is responsible for committing
(or rolling back) the session — the helper ``get_session()`` context manager in
``app.db.base`` does that automatically.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import OcrLog, Voucher, VoucherBatch


# ---------------------------------------------------------------------------
# VoucherBatch
# ---------------------------------------------------------------------------


async def create_batch(
    session: AsyncSession,
    batch_code: str,
    original_image: Optional[str] = None,
    uploaded_by: Optional[int] = None,
    status: str = "processing",
) -> VoucherBatch:
    """Insert a new VoucherBatch row and return it (flushed, not yet committed)."""
    batch = VoucherBatch(
        batch_code=batch_code,
        original_image=original_image,
        uploaded_by=uploaded_by,
        status=status,
    )
    session.add(batch)
    await session.flush()
    await session.refresh(batch)
    return batch


async def get_batch_by_id(
    session: AsyncSession,
    batch_id: int,
) -> Optional[VoucherBatch]:
    """Return a VoucherBatch by primary key, or None if not found."""
    result = await session.execute(
        select(VoucherBatch).where(VoucherBatch.id == batch_id)
    )
    return result.scalar_one_or_none()


async def get_batch_by_code(
    session: AsyncSession,
    batch_code: str,
) -> Optional[VoucherBatch]:
    """Return a VoucherBatch by batch_code, or None if not found."""
    result = await session.execute(
        select(VoucherBatch).where(VoucherBatch.batch_code == batch_code)
    )
    return result.scalar_one_or_none()


async def delete_batch_by_code(
    session: AsyncSession,
    batch_code: str,
) -> bool:
    """Delete a VoucherBatch and all associated files and records."""
    batch = await get_batch_by_code(session, batch_code)
    if not batch:
        return False

    # Clean up original image and its directory
    if batch.original_image and os.path.exists(batch.original_image):
        try:
            batch_folder = os.path.dirname(batch.original_image)
            shutil.rmtree(batch_folder, ignore_errors=True)
        except Exception:
            pass

    # Clean up crops directory for this batch
    from app.config import CROPS_DIR

    batch_crops = os.path.join(CROPS_DIR, batch.batch_code)
    if os.path.exists(batch_crops):
        shutil.rmtree(batch_crops, ignore_errors=True)

    await session.delete(batch)
    return True


async def update_batch_stats(
    session: AsyncSession,
    batch_id: int,
    total_detected: int,
    total_success: int,
    total_review: int,
    status: str,
    total_failed: int = 0,
) -> None:
    """Update aggregate counters and status for a VoucherBatch."""
    await session.execute(
        update(VoucherBatch)
        .where(VoucherBatch.id == batch_id)
        .values(
            total_detected=total_detected,
            total_success=total_success,
            total_review=total_review,
            total_failed=total_failed,
            status=status,
        )
    )


async def get_batches_by_uploader(
    session: AsyncSession,
    uploaded_by: int,
    limit: int = 10,
) -> list[VoucherBatch]:
    """Return the most-recent *limit* batches uploaded by a Telegram user."""
    result = await session.execute(
        select(VoucherBatch)
        .where(VoucherBatch.uploaded_by == uploaded_by)
        .order_by(VoucherBatch.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_recent_batches(
    session: AsyncSession,
    limit: int = 10,
) -> list[VoucherBatch]:
    """Return the most recent batches across all uploaders."""
    result = await session.execute(
        select(VoucherBatch)
        .order_by(VoucherBatch.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


get_last_batches = get_recent_batches


# ---------------------------------------------------------------------------
# Voucher
# ---------------------------------------------------------------------------


async def create_voucher(
    session: AsyncSession,
    batch_id: int,
    serial_number: str,
    voucher_code: Optional[str],
    crop_image: Optional[str],
    confidence: Optional[float],
    status: str = "review",
) -> Optional[Voucher]:
    """Insert a Voucher row.

    Returns the created ``Voucher`` instance, or ``None`` when the
    ``serial_number`` already exists (duplicate detected via IntegrityError).
    """
    voucher = Voucher(
        batch_id=batch_id,
        serial_number=serial_number,
        voucher_code=voucher_code,
        crop_image=crop_image,
        confidence=confidence,
        status=status,
    )
    session.add(voucher)
    try:
        await session.flush()
        await session.refresh(voucher)
        return voucher
    except IntegrityError:
        await session.rollback()
        return None


async def get_voucher_by_serial(
    session: AsyncSession,
    serial_number: str,
) -> Optional[Voucher]:
    """Return a Voucher by its serial_number, or None."""
    result = await session.execute(
        select(Voucher).where(Voucher.serial_number == serial_number)
    )
    return result.scalar_one_or_none()


async def get_voucher_by_voucher_code(
    session: AsyncSession,
    voucher_code: str,
) -> Optional[Voucher]:
    """Return a Voucher by its voucher_code (nomor telepon kiri), or None."""
    result = await session.execute(
        select(Voucher).where(Voucher.voucher_code == voucher_code)
    )
    return result.scalar_one_or_none()


async def delete_voucher_by_serial(
    session: AsyncSession,
    serial_number: str,
) -> bool:
    """Delete a Voucher by serial_number, remove its crop image, and adjust batch counters."""
    voucher = await get_voucher_by_serial(session, serial_number)
    if not voucher:
        return False

    # Remove physical crop file if it exists
    if voucher.crop_image and os.path.exists(voucher.crop_image):
        try:
            os.remove(voucher.crop_image)
        except Exception:
            pass

    batch_id = voucher.batch_id
    status = voucher.status

    await session.delete(voucher)

    # Adjust batch counters if batch exists
    if batch_id:
        batch = await get_batch_by_id(session, batch_id)
        if batch:
            batch.total_detected = max(0, (batch.total_detected or 0) - 1)
            if status == "verified":
                batch.total_success = max(0, (batch.total_success or 0) - 1)
            elif status == "review":
                batch.total_review = max(0, (batch.total_review or 0) - 1)
            elif status == "failed":
                batch.total_failed = max(0, (getattr(batch, "total_failed", 0) or 0) - 1)

    return True


async def get_vouchers_by_batch(
    session: AsyncSession,
    batch_id: int,
) -> list[Voucher]:
    """Return all vouchers that belong to a batch, ordered by id."""
    result = await session.execute(
        select(Voucher)
        .where(Voucher.batch_id == batch_id)
        .order_by(Voucher.id)
    )
    return list(result.scalars().all())


async def get_vouchers_by_status(
    session: AsyncSession,
    status: str = "review",
    limit: int = 50,
) -> list[Voucher]:
    """Return vouchers filtered by status."""
    result = await session.execute(
        select(Voucher)
        .where(Voucher.status == status)
        .order_by(Voucher.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_vouchers_for_review(
    session: AsyncSession,
    limit: int = 10,
) -> list[tuple[Voucher, str]]:
    """Return vouchers pending review along with their batch code."""
    result = await session.execute(
        select(Voucher, VoucherBatch.batch_code)
        .join(VoucherBatch, Voucher.batch_id == VoucherBatch.id)
        .where(Voucher.status == "review")
        .order_by(Voucher.created_at.desc())
        .limit(limit)
    )
    return [(row[0], row[1]) for row in result.all()]


# ---------------------------------------------------------------------------
# OcrLog
# ---------------------------------------------------------------------------


async def create_ocr_log(
    session: AsyncSession,
    batch_id: int,
    raw_result: dict,
    voucher_id: Optional[int] = None,
    error_message: Optional[str] = None,
    processing_time: Optional[float] = None,
) -> OcrLog:
    """Insert an OcrLog row and return it."""
    log = OcrLog(
        batch_id=batch_id,
        voucher_id=voucher_id,
        raw_result=raw_result,
        error_message=error_message,
        processing_time=processing_time,
    )
    session.add(log)
    await session.flush()
    await session.refresh(log)
    return log


async def get_batch_failure_summary(
    session: AsyncSession,
    batch_id: int,
) -> dict[str, int]:
    """Return summary dictionary mapping failure reason to count of affected vouchers."""
    # 1. Try to read from ocr_logs
    result = await session.execute(
        select(OcrLog.error_message)
        .where(OcrLog.batch_id == batch_id, OcrLog.error_message.isnot(None))
    )
    counts: dict[str, int] = {}
    for row in result.all():
        err = row[0]
        if not err:
            continue
        for part in err.split("; "):
            part = part.strip()
            if not part:
                continue
            # Normalisasi pengelompokan confidence rendah
            if part.startswith("Confidence rendah"):
                part = "Confidence rendah (< 70%)"
            counts[part] = counts.get(part, 0) + 1

    if counts:
        return counts

    # 2. Fallback for older batches: deduce reasons from vouchers table
    v_result = await session.execute(
        select(Voucher.serial_number, Voucher.voucher_code, Voucher.confidence, Voucher.status)
        .where(Voucher.batch_id == batch_id, Voucher.status != "verified")
    )
    for s_num, v_code, conf, stat in v_result.all():
        conf_val = float(conf) if conf is not None else 0.0
        if not s_num or s_num.startswith("UNKNOWN"):
            counts["Barcode / Serial tidak terbaca"] = counts.get("Barcode / Serial tidak terbaca", 0) + 1
        if not v_code:
            counts["Kode voucher tidak terdeteksi"] = counts.get("Kode voucher tidak terdeteksi", 0) + 1
        if conf_val < 0.70:
            counts["Confidence rendah (< 70%)"] = counts.get("Confidence rendah (< 70%)", 0) + 1
        if stat == "failed" and not counts:
            counts["Gagal diproses"] = counts.get("Gagal diproses", 0) + 1

    return counts


async def get_failed_errors_by_batch(
    session: AsyncSession,
    batch_id: int,
    limit: int = 5,
) -> list[str]:
    """Return distinct non-null error_messages from ocr_logs for a batch (max *limit* entries)."""
    result = await session.execute(
        select(OcrLog.error_message)
        .where(OcrLog.batch_id == batch_id, OcrLog.error_message.isnot(None))
        .order_by(OcrLog.id)
        .limit(limit)
    )
    return [row[0] for row in result.all() if row[0]]


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------


async def get_stats(session: AsyncSession) -> dict:
    """Return high-level aggregate statistics."""
    total_batches_result = await session.execute(
        select(func.count()).select_from(VoucherBatch)
    )
    total_batches: int = total_batches_result.scalar_one() or 0

    total_vouchers_result = await session.execute(
        select(func.count()).select_from(Voucher)
    )
    total_vouchers: int = total_vouchers_result.scalar_one() or 0

    total_verified_result = await session.execute(
        select(func.count()).select_from(Voucher).where(Voucher.status == "verified")
    )
    total_verified: int = total_verified_result.scalar_one() or 0

    total_review_result = await session.execute(
        select(func.count()).select_from(Voucher).where(Voucher.status == "review")
    )
    total_review: int = total_review_result.scalar_one() or 0

    total_failed_result = await session.execute(
        select(func.count()).select_from(Voucher).where(Voucher.status == "failed")
    )
    total_failed: int = total_failed_result.scalar_one() or 0

    return {
        "total": total_vouchers,
        "total_vouchers": total_vouchers,
        "verified": total_verified,
        "total_verified": total_verified,
        "review": total_review,
        "total_review": total_review,
        "failed": total_failed,
        "total_failed": total_failed,
        "total_batches": total_batches,
    }


get_voucher_stats = get_stats
get_overall_stats = get_stats

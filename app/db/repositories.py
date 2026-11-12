"""
Async repository functions for the Voucher OCR application.

All functions accept an ``AsyncSession`` as the first argument so they can be
composed inside a single transaction.  The caller is responsible for committing
(or rolling back) the session — the helper ``get_session()`` context manager in
``app.db.base`` does that automatically.
"""

from __future__ import annotations

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
    original_image: Optional[str],
    uploaded_by: Optional[int],
) -> VoucherBatch:
    """Insert a new VoucherBatch row and return it (flushed, not yet committed)."""
    batch = VoucherBatch(
        batch_code=batch_code,
        original_image=original_image,
        uploaded_by=uploaded_by,
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


async def update_batch_stats(
    session: AsyncSession,
    batch_id: int,
    total_detected: int,
    total_success: int,
    total_review: int,
    status: str,
) -> None:
    """Update aggregate counters and status for a VoucherBatch."""
    await session.execute(
        update(VoucherBatch)
        .where(VoucherBatch.id == batch_id)
        .values(
            total_detected=total_detected,
            total_success=total_success,
            total_review=total_review,
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


# ---------------------------------------------------------------------------
# Aggregate stats
# ---------------------------------------------------------------------------


async def get_stats(session: AsyncSession) -> dict:
    """Return high-level aggregate statistics.

    Note: a session is required so this function follows the same calling
    convention as the rest of the repository.
    """
    total_batches_result = await session.execute(
        select(func.count()).select_from(VoucherBatch)
    )
    total_batches: int = total_batches_result.scalar_one()

    total_vouchers_result = await session.execute(
        select(func.count()).select_from(Voucher)
    )
    total_vouchers: int = total_vouchers_result.scalar_one()

    total_verified_result = await session.execute(
        select(func.count()).select_from(Voucher).where(Voucher.status == "verified")
    )
    total_verified: int = total_verified_result.scalar_one()

    total_review_result = await session.execute(
        select(func.count()).select_from(Voucher).where(Voucher.status == "review")
    )
    total_review: int = total_review_result.scalar_one()

    return {
        "total_batches": total_batches,
        "total_vouchers": total_vouchers,
        "total_verified": total_verified,
        "total_review": total_review,
    }

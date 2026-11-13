from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    BatchDetailResponse,
    BatchResponse,
    SearchResponse,
    StatsResponse,
    VoucherResponse,
)
from app.db.base import get_session
from app.db import repositories

router = APIRouter(prefix="/api/v1")


@router.get("/vouchers/review", response_model=list[VoucherResponse])
async def list_vouchers_for_review(
    limit: int = Query(default=50, ge=1, le=200),
    session: AsyncSession = Depends(get_session),
):
    """List vouchers that need manual review."""
    vouchers = await repositories.get_vouchers_by_status(session, status="review", limit=limit)
    result = []
    for v in vouchers:
        batch = await repositories.get_batch_by_id(session, v.batch_id)
        item = VoucherResponse(
            id=v.id,
            batch_id=v.batch_id,
            batch_code=batch.batch_code if batch else "",
            serial_number=v.serial_number,
            voucher_code=v.voucher_code,
            crop_image=v.crop_image,
            confidence=v.confidence,
            status=v.status,
            created_at=v.created_at,
        )
        result.append(item)
    return result


@router.get("/vouchers/{serial_number}", response_model=SearchResponse)
async def search_voucher(
    serial_number: str,
    session: AsyncSession = Depends(get_session),
):
    """Search for a voucher by serial number."""
    voucher = await repositories.get_voucher_by_serial(session, serial_number)
    if not voucher:
        return SearchResponse(found=False, voucher=None)

    batch = await repositories.get_batch_by_id(session, voucher.batch_id)
    voucher_resp = VoucherResponse(
        id=voucher.id,
        batch_id=voucher.batch_id,
        batch_code=batch.batch_code if batch else "",
        serial_number=voucher.serial_number,
        voucher_code=voucher.voucher_code,
        crop_image=voucher.crop_image,
        confidence=voucher.confidence,
        status=voucher.status,
        created_at=voucher.created_at,
    )
    return SearchResponse(found=True, voucher=voucher_resp)


@router.get("/batches", response_model=list[BatchResponse])
async def list_batches(
    limit: int = Query(default=10, ge=1, le=100),
    session: AsyncSession = Depends(get_session),
):
    """List recent batches."""
    batches = await repositories.get_recent_batches(session, limit=limit)
    return [
        BatchResponse(
            id=b.id,
            batch_code=b.batch_code,
            original_image=b.original_image,
            uploaded_by=b.uploaded_by,
            total_detected=b.total_detected,
            total_success=b.total_success,
            total_review=b.total_review,
            status=b.status,
            created_at=b.created_at,
        )
        for b in batches
    ]


@router.get("/batches/{batch_id}", response_model=BatchDetailResponse)
async def get_batch_detail(
    batch_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get batch detail with its vouchers."""
    batch = await repositories.get_batch_by_id(session, batch_id)
    if not batch:
        raise HTTPException(status_code=404, detail="Batch not found")

    vouchers = await repositories.get_vouchers_by_batch(session, batch_id)
    voucher_list = [
        VoucherResponse(
            id=v.id,
            batch_id=v.batch_id,
            batch_code=batch.batch_code,
            serial_number=v.serial_number,
            voucher_code=v.voucher_code,
            crop_image=v.crop_image,
            confidence=v.confidence,
            status=v.status,
            created_at=v.created_at,
        )
        for v in vouchers
    ]

    return BatchDetailResponse(
        id=batch.id,
        batch_code=batch.batch_code,
        original_image=batch.original_image,
        uploaded_by=batch.uploaded_by,
        total_detected=batch.total_detected,
        total_success=batch.total_success,
        total_review=batch.total_review,
        status=batch.status,
        created_at=batch.created_at,
        vouchers=voucher_list,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(session: AsyncSession = Depends(get_session)):
    """Get overall statistics."""
    stats = await repositories.get_overall_stats(session)
    return StatsResponse(**stats)


@router.delete("/batches/{batch_code}")
async def delete_batch(
    batch_code: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete a batch and its associated vouchers and files."""
    success = await repositories.delete_batch_by_code(session, batch_code)
    if not success:
        raise HTTPException(status_code=404, detail="Batch not found")
    return {"status": "ok", "message": f"Batch {batch_code} deleted successfully"}


@router.delete("/vouchers/{serial_number}")
async def delete_voucher(
    serial_number: str,
    session: AsyncSession = Depends(get_session),
):
    """Delete an individual voucher and its crop image."""
    success = await repositories.delete_voucher_by_serial(session, serial_number)
    if not success:
        raise HTTPException(status_code=404, detail="Voucher not found")
    return {"status": "ok", "message": f"Voucher {serial_number} deleted successfully"}

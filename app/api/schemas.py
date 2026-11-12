from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class VoucherResponse(BaseModel):
    id: int
    batch_id: int
    batch_code: str
    serial_number: Optional[str] = None
    voucher_code: Optional[str] = None
    crop_image: Optional[str] = None
    confidence: Optional[float] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class BatchResponse(BaseModel):
    id: int
    batch_code: str
    original_image: Optional[str] = None
    uploaded_by: Optional[int] = None
    total_detected: int = 0
    total_success: int = 0
    total_review: int = 0
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class BatchDetailResponse(BatchResponse):
    vouchers: list[VoucherResponse] = []


class StatsResponse(BaseModel):
    total_vouchers: int
    total_verified: int
    total_review: int
    total_failed: int
    total_batches: int


class SearchResponse(BaseModel):
    found: bool
    voucher: Optional[VoucherResponse] = None

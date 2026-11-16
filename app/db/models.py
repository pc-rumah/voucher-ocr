"""
SQLAlchemy ORM models for the Voucher OCR application.

Tables:
    voucher_batches  — one per upload / scanned image
    vouchers         — individual vouchers detected in a batch
    ocr_logs         — raw OCR engine output per processing attempt
"""

from datetime import datetime, timezone

from sqlalchemy import (
    BIGINT,
    INTEGER,
    TEXT,
    TIMESTAMP,
    VARCHAR,
    BigInteger,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# VoucherBatch
# ---------------------------------------------------------------------------


class VoucherBatch(Base):
    __tablename__ = "voucher_batches"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_code: Mapped[str] = mapped_column(VARCHAR(50), unique=True, nullable=False)
    original_image: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    uploaded_by: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    total_detected: Mapped[int] = mapped_column(INTEGER, default=0, server_default="0", nullable=False)
    total_success: Mapped[int] = mapped_column(INTEGER, default=0, server_default="0", nullable=False)
    total_review: Mapped[int] = mapped_column(INTEGER, default=0, server_default="0", nullable=False)
    total_failed: Mapped[int] = mapped_column(INTEGER, default=0, server_default="0", nullable=False)
    status: Mapped[str] = mapped_column(
        VARCHAR(20), default="processing", server_default="processing", nullable=False
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    vouchers: Mapped[list["Voucher"]] = relationship(
        "Voucher", back_populates="batch", cascade="all, delete-orphan"
    )
    ocr_logs: Mapped[list["OcrLog"]] = relationship(
        "OcrLog", back_populates="batch", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_voucher_batches_created_at", "created_at"),
        Index("ix_voucher_batches_status", "status"),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<VoucherBatch id={self.id} code={self.batch_code!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# Voucher
# ---------------------------------------------------------------------------


class Voucher(Base):
    __tablename__ = "vouchers"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("voucher_batches.id", ondelete="CASCADE"), nullable=False
    )
    serial_number: Mapped[str] = mapped_column(VARCHAR(100), nullable=False)
    voucher_code: Mapped[str | None] = mapped_column(VARCHAR(100), nullable=True)
    crop_image: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    status: Mapped[str] = mapped_column(
        VARCHAR(20), default="review", server_default="review", nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        TIMESTAMP(timezone=True),
        onupdate=func.now(),
        nullable=True,
    )

    # Relationships
    batch: Mapped["VoucherBatch"] = relationship("VoucherBatch", back_populates="vouchers")
    ocr_logs: Mapped[list["OcrLog"]] = relationship(
        "OcrLog", back_populates="voucher", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("serial_number", name="uq_vouchers_serial_number"),
        Index("ix_vouchers_batch_id", "batch_id"),
        Index("ix_vouchers_serial_number", "serial_number", unique=True),
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Voucher id={self.id} serial={self.serial_number!r} status={self.status!r}>"


# ---------------------------------------------------------------------------
# OcrLog
# ---------------------------------------------------------------------------


class OcrLog(Base):
    __tablename__ = "ocr_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    batch_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("voucher_batches.id", ondelete="CASCADE"), nullable=False
    )
    voucher_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("vouchers.id", ondelete="SET NULL"), nullable=True
    )
    serial_number: Mapped[str | None] = mapped_column(VARCHAR(100), nullable=True)
    raw_text: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    raw_result: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    confidence: Mapped[float | None] = mapped_column(Numeric(5, 4), nullable=True)
    engine: Mapped[str | None] = mapped_column(VARCHAR(50), default="paddleocr", nullable=True)
    processing_time_ms: Mapped[float | None] = mapped_column(Numeric(10, 2), nullable=True)
    error_message: Mapped[str | None] = mapped_column(TEXT, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationships
    batch: Mapped["VoucherBatch"] = relationship("VoucherBatch", back_populates="ocr_logs")
    voucher: Mapped["Voucher | None"] = relationship("Voucher", back_populates="ocr_logs")

    def __repr__(self) -> str:  # pragma: no cover
        return f"<OcrLog id={self.id} batch_id={self.batch_id} voucher_id={self.voucher_id}>"

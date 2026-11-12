from __future__ import annotations

import logging

from celery import Celery
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import CELERY_BROKER_URL, CELERY_RESULT_BACKEND, DATABASE_URL

logger = logging.getLogger(__name__)

celery_app = Celery("voucher_ocr", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)
celery_app.conf.task_serializer = "json"
celery_app.conf.result_serializer = "json"
celery_app.conf.accept_content = ["json"]
celery_app.conf.timezone = "Asia/Jakarta"


def _make_sync_session() -> tuple[object, Session]:
    """Create a synchronous SQLAlchemy engine and session for Celery tasks."""
    # Replace asyncpg driver with psycopg2 for sync operations
    sync_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql+psycopg2://")
    engine = create_engine(sync_url, pool_pre_ping=True)
    SyncSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    return engine, SyncSession()


@celery_app.task(name="process_voucher_batch", bind=True, max_retries=3)
def process_voucher_batch(self, batch_id: int, image_path: str):
    """Process a voucher batch: run OCR pipeline and persist results."""
    from app.ocr.pipeline import OCRPipeline
    from app.db import repositories as repos

    engine, session = _make_sync_session()
    try:
        # Fetch the batch record (sync)
        batch = session.execute(
            __import__("sqlalchemy").text(
                "SELECT batch_code FROM voucher_batches WHERE id = :id"
            ),
            {"id": batch_id},
        ).fetchone()

        if not batch:
            logger.error("Batch %s not found", batch_id)
            return

        batch_code = batch[0]

        # Run OCR pipeline
        pipeline = OCRPipeline()
        results = pipeline.process_batch(batch_code, image_path)

        total_success = 0
        total_review = 0

        for voucher_result in results:
            status = voucher_result.get("status", "review")
            if status == "verified":
                total_success += 1
            else:
                total_review += 1

            # Persist voucher
            session.execute(
                __import__("sqlalchemy").text(
                    """
                    INSERT INTO vouchers
                        (batch_id, serial_number, voucher_code, crop_image, confidence, status)
                    VALUES
                        (:batch_id, :serial_number, :voucher_code, :crop_image, :confidence, :status)
                    """
                ),
                {
                    "batch_id": batch_id,
                    "serial_number": voucher_result.get("serial_number"),
                    "voucher_code": voucher_result.get("voucher_code"),
                    "crop_image": voucher_result.get("crop_image"),
                    "confidence": voucher_result.get("confidence"),
                    "status": status,
                },
            )

            # Persist OCR log
            session.execute(
                __import__("sqlalchemy").text(
                    """
                    INSERT INTO ocr_logs
                        (batch_id, serial_number, raw_text, confidence, engine, processing_time_ms)
                    VALUES
                        (:batch_id, :serial_number, :raw_text, :confidence, :engine, :processing_time_ms)
                    """
                ),
                {
                    "batch_id": batch_id,
                    "serial_number": voucher_result.get("serial_number"),
                    "raw_text": voucher_result.get("raw_text"),
                    "confidence": voucher_result.get("confidence"),
                    "engine": voucher_result.get("engine", "paddleocr"),
                    "processing_time_ms": voucher_result.get("processing_time_ms"),
                },
            )

        # Update batch stats
        total_detected = len(results)
        session.execute(
            __import__("sqlalchemy").text(
                """
                UPDATE voucher_batches
                SET
                    total_detected = :total_detected,
                    total_success = :total_success,
                    total_review = :total_review,
                    status = 'done',
                    processed_at = NOW()
                WHERE id = :batch_id
                """
            ),
            {
                "total_detected": total_detected,
                "total_success": total_success,
                "total_review": total_review,
                "batch_id": batch_id,
            },
        )
        session.commit()
        logger.info("Batch %s processed: %s vouchers", batch_id, total_detected)

    except Exception as exc:
        session.rollback()
        logger.error("Batch %s failed: %s", batch_id, exc, exc_info=True)
        try:
            session.execute(
                __import__("sqlalchemy").text(
                    "UPDATE voucher_batches SET status = 'failed' WHERE id = :batch_id"
                ),
                {"batch_id": batch_id},
            )
            session.commit()
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=60)
    finally:
        session.close()
        engine.dispose()

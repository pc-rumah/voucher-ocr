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


def _send_telegram_notification(chat_id: int | None, text: str) -> None:
    """Send notification message to Telegram user using Bot API."""
    import json
    import os
    import urllib.request

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            logger.info("Telegram notification sent to %s (status %s)", chat_id, resp.status)
    except Exception as e:
        logger.warning("Failed to send Telegram notification to %s: %s", chat_id, e)


@celery_app.task(name="process_voucher_batch", bind=True, max_retries=3)
def process_voucher_batch(self, batch_id: int, image_path: str):
    """Process a voucher batch: run OCR pipeline and persist results."""
    from app.ocr.pipeline import OCRPipeline
    from app.db import repositories as repos

    engine, session = _make_sync_session()
    uploaded_by = None
    batch_code = None
    try:
        # Fetch the batch record (sync)
        batch = session.execute(
            __import__("sqlalchemy").text(
                "SELECT batch_code, uploaded_by FROM voucher_batches WHERE id = :id"
            ),
            {"id": batch_id},
        ).fetchone()

        if not batch:
            logger.error("Batch %s not found", batch_id)
            return

        batch_code = batch[0]
        uploaded_by = batch[1]

        # Run OCR pipeline
        pipeline = OCRPipeline()
        batch_output = pipeline.process_batch(batch_code, image_path)
        vouchers = batch_output.get("vouchers", []) if isinstance(batch_output, dict) else batch_output

        total_success = 0
        total_review = 0
        total_failed = 0

        for idx, voucher_result in enumerate(vouchers):
            status = voucher_result.get("status", "review")
            if status == "verified":
                total_success += 1
            elif status == "failed":
                total_failed += 1
            else:
                total_review += 1

            crop_image = voucher_result.get("crop_image") or voucher_result.get("crop_path")
            raw_text = voucher_result.get("raw_text") or str(voucher_result.get("raw_ocr", ""))
            confidence = voucher_result.get("confidence") or 0.0
            serial_number = voucher_result.get("serial_number") or f"UNKNOWN-{batch_code}-{idx+1}"

            # Persist voucher
            session.execute(
                __import__("sqlalchemy").text(
                    """
                    INSERT INTO vouchers
                        (batch_id, serial_number, voucher_code, crop_image, confidence, status)
                    VALUES
                        (:batch_id, :serial_number, :voucher_code, :crop_image, :confidence, :status)
                    ON CONFLICT (serial_number) DO UPDATE
                    SET voucher_code = EXCLUDED.voucher_code,
                        crop_image = EXCLUDED.crop_image,
                        confidence = EXCLUDED.confidence,
                        status = EXCLUDED.status,
                        updated_at = NOW()
                    """
                ),
                {
                    "batch_id": batch_id,
                    "serial_number": serial_number,
                    "voucher_code": voucher_result.get("voucher_code"),
                    "crop_image": crop_image,
                    "confidence": confidence,
                    "status": status,
                },
            )

            # Persist OCR log
            session.execute(
                __import__("sqlalchemy").text(
                    """
                    INSERT INTO ocr_logs
                        (batch_id, serial_number, raw_text, confidence, engine, processing_time_ms, error_message)
                    VALUES
                        (:batch_id, :serial_number, :raw_text, :confidence, :engine, :processing_time_ms, :error_message)
                    """
                ),
                {
                    "batch_id": batch_id,
                    "serial_number": serial_number,
                    "raw_text": raw_text,
                    "confidence": confidence,
                    "engine": voucher_result.get("engine", "paddleocr"),
                    "processing_time_ms": voucher_result.get("processing_time_ms") or 0.0,
                    "error_message": voucher_result.get("error"),
                },
            )

        # Update batch stats
        total_detected = len(vouchers)
        session.execute(
            __import__("sqlalchemy").text(
                """
                UPDATE voucher_batches
                SET
                    total_detected = :total_detected,
                    total_success = :total_success,
                    total_review = :total_review,
                    total_failed = :total_failed,
                    status = 'done',
                    processed_at = NOW()
                WHERE id = :batch_id
                """
            ),
            {
                "total_detected": total_detected,
                "total_success": total_success,
                "total_review": total_review,
                "total_failed": total_failed,
                "batch_id": batch_id,
            },
        )
        session.commit()
        logger.info("Batch %s processed: %s vouchers", batch_id, total_detected)

        # Send Telegram notification if uploaded_by is known
        if uploaded_by:
            unverified_count = total_review + total_failed
            lines = [
                "🔔 <b>Hasil Pemrosesan Gambar Batch Selesai</b>",
                f"Kode Batch: <code>{batch_code}</code>",
                f"Total Voucher Terdeteksi: <b>{total_detected}</b>",
                f"✅ Berhasil: <b>{total_success}</b>",
            ]
            if total_review > 0:
                lines.append(f"⚠️ Perlu Review: <b>{total_review}</b>")
            if total_failed > 0:
                lines.append(f"❌ Gagal: <b>{total_failed}</b>")

            failure_summary = (
                batch_output.get("failure_summary", {})
                if isinstance(batch_output, dict)
                else {}
            )
            if failure_summary and unverified_count > 0:
                lines.append("")
                lines.append(f"⚠️ <b>Alasan Tidak Lolos ({unverified_count} voucher):</b>")
                import html  # noqa: PLC0415
                for reason, count in failure_summary.items():
                    safe_reason = html.escape(str(reason))
                    lines.append(f"• {safe_reason}: <b>{count}</b> voucher")

            if unverified_count > 0:
                lines.append("")
                lines.append("<i>Gunakan /review untuk memeriksa voucher yang perlu verifikasi.</i>")
            elif total_detected > 0:
                lines.append("")
                lines.append("✨ <i>Semua voucher berhasil diverifikasi!</i>")

            _send_telegram_notification(uploaded_by, "\n".join(lines))

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
            if uploaded_by and batch_code:
                import html  # noqa: PLC0415
                safe_exc = html.escape(str(exc))
                _send_telegram_notification(
                    uploaded_by,
                    f"❌ <b>Pemrosesan Batch Gagal!</b>\n"
                    f"Kode Batch: <code>{batch_code}</code>\n"
                    f"Alasan: <i>{safe_exc}</i>",
                )
        except Exception:
            pass
        raise self.retry(exc=exc, countdown=60)
    finally:
        session.close()
        engine.dispose()

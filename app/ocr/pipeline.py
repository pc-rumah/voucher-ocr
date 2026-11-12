"""End-to-end OCR pipeline: load → detect → crop → recognise → classify."""

import logging
import os
import time
from typing import Dict, List

import cv2

from app.ocr.detector import VoucherDetector
from app.ocr.preprocessor import ImagePreprocessor
from app.ocr.recognizer import VoucherRecognizer

logger = logging.getLogger(__name__)


class OCRPipeline:
    """Orchestrates the full voucher extraction workflow for a single image.

    Args:
        originals_dir: Root directory where original images are stored.
        crops_dir: Root directory where cropped voucher images are saved.
        confidence_threshold: Minimum OCR confidence for 'verified' status.
    """

    def __init__(
        self,
        originals_dir: str,
        crops_dir: str,
        confidence_threshold: float = 0.7,
    ):
        self.originals_dir = originals_dir
        self.crops_dir = crops_dir
        self.confidence_threshold = confidence_threshold

        self._preprocessor = ImagePreprocessor()
        self._detector = VoucherDetector()
        self._recognizer = VoucherRecognizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_batch(self, batch_code: str, image_path: str) -> Dict:
        """Process a single image and extract all voucher data.

        Each detected voucher is:
        1. Cropped and saved to ``{crops_dir}/{batch_code}/voucher-{index:03d}.jpg``.
        2. Passed through OCR to extract serial_number and voucher_code.
        3. Classified as ``verified``, ``review``, or ``failed``.

        Args:
            batch_code: Identifier for this processing batch (used in paths).
            image_path: Absolute path to the source image.

        Returns:
            Summary dict with keys ``total_detected``, ``total_success``,
            ``total_review``, and ``vouchers`` (list of per-voucher dicts).
        """
        logger.info("Processing batch '%s', image: %s", batch_code, image_path)

        # Load source image
        img = self._preprocessor.load_image(image_path)

        # Detect voucher regions
        vouchers_meta = self._detector.detect_vouchers(img)

        # Prepare output directory for crops
        batch_crops_dir = os.path.join(self.crops_dir, batch_code)
        os.makedirs(batch_crops_dir, exist_ok=True)

        results: List[Dict] = []

        for meta in vouchers_meta:
            index: int = meta["index"]
            bbox = meta["bbox"]
            crop_filename = f"voucher-{index:03d}.jpg"
            crop_path = os.path.join(batch_crops_dir, crop_filename)

            voucher_result: Dict = {
                "index": index,
                "serial_number": None,
                "voucher_code": None,
                "crop_path": crop_path,
                "confidence": 0.0,
                "status": "failed",
                "raw_ocr": [],
                "error": None,
            }

            t_start = time.monotonic()
            try:
                # Crop
                crop = self._detector.crop_voucher(img, bbox)

                # Save crop
                cv2.imwrite(crop_path, crop)

                # Preprocess for OCR
                ocr_ready = self._preprocessor.preprocess_for_ocr(crop)

                # Run OCR
                ocr_data = self._recognizer.extract_voucher_data(ocr_ready)

                serial_number = ocr_data["serial_number"]
                voucher_code = ocr_data["voucher_code"]
                confidence = ocr_data["confidence"]
                raw_ocr = ocr_data["raw_ocr"]

                # Classify
                if confidence >= self.confidence_threshold and serial_number:
                    status = "verified"
                else:
                    status = "review"

                voucher_result.update(
                    {
                        "serial_number": serial_number,
                        "voucher_code": voucher_code,
                        "confidence": confidence,
                        "status": status,
                        "raw_ocr": raw_ocr,
                        "error": None,
                    }
                )

            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Voucher %d in batch '%s' failed: %s", index, batch_code, exc
                )
                voucher_result["status"] = "failed"
                voucher_result["error"] = str(exc)

            elapsed = time.monotonic() - t_start
            logger.info(
                "Voucher %d [%s] processed in %.3f s (confidence=%.3f)",
                index,
                voucher_result["status"],
                elapsed,
                voucher_result["confidence"],
            )

            results.append(voucher_result)

        total_detected = len(results)
        total_success = sum(1 for r in results if r["status"] == "verified")
        total_review = sum(1 for r in results if r["status"] == "review")

        return {
            "total_detected": total_detected,
            "total_success": total_success,
            "total_review": total_review,
            "vouchers": results,
        }

"""End-to-end OCR pipeline: load -> detect -> crop -> recognise -> classify."""

import logging
import os
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.config import CROPS_DIR, OCR_CONFIDENCE_THRESHOLD, ORIGINALS_DIR
from app.ocr.barcode import read_barcode
from app.ocr.detector import VoucherDetector
from app.ocr.preprocessor import ImagePreprocessor
from app.ocr.recognizer import VoucherRecognizer

logger = logging.getLogger(__name__)


def _upscale_if_needed(img: np.ndarray, min_height: int = 120) -> np.ndarray:
    """Upscale *img* jika tingginya di bawah *min_height* (untuk barcode & OCR).

    Args:
        img: BGR atau grayscale image.
        min_height: Tinggi minimum dalam piksel.

    Returns:
        Image yang sudah di-upscale (atau asli jika sudah cukup besar).
    """
    h, w = img.shape[:2]
    if h > 0 and h < min_height:
        scale = min_height / h
        img = cv2.resize(
            img,
            (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
    return img


def _split_crop(
    crop: np.ndarray, left_ratio: float = 0.60
) -> Tuple[np.ndarray, np.ndarray]:
    """Belah crop voucher menjadi area kiri (*left_ratio*) dan sisa kanan.

    Args:
        crop: Gambar crop voucher (BGR atau grayscale).
        left_ratio: Proporsi lebar untuk area kiri (default 0.60 = 60%).

    Returns:
        Tuple ``(left_region, right_region)``.
    """
    _, w = crop.shape[:2]
    ratio = max(0.1, min(0.95, float(left_ratio)))
    mid = max(1, min(w, int(round(w * ratio))))
    return crop[:, :mid], crop[:, mid:]


class OCRPipeline:
    """Orchestrates the full voucher extraction workflow for a single image.

    Args:
        originals_dir: Root directory where original images are stored.
        crops_dir: Root directory where cropped voucher images are saved.
        confidence_threshold: Minimum per-field OCR confidence for 'verified' status.
    """

    LEFT_CROP_RATIO = 0.60

    def __init__(
        self,
        originals_dir: str = ORIGINALS_DIR,
        crops_dir: str = CROPS_DIR,
        confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
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
        1. Validated, deskewed/cropped (best crop selected), and saved to
           ``{crops_dir}/{batch_code}/voucher-{index:03d}.jpg``.
        2. Read for barcode (serial number primary source).
        3. Passed through multi-pass OCR across image variants (60% left area
           first when barcode is known, with fallback to full crop).
        4. Classified as ``verified``, ``review``, or ``failed`` using per-field
           validity and per-field confidence.
        """
        logger.info("Processing batch '%s', image: %s", batch_code, image_path)

        img = self._preprocessor.load_image(image_path)
        vouchers_meta = self._detector.detect_vouchers(img)

        batch_crops_dir = os.path.join(self.crops_dir, batch_code)
        os.makedirs(batch_crops_dir, exist_ok=True)

        results: List[Dict] = []

        for meta in vouchers_meta:
            index: int = meta["index"]
            crop_filename = f"voucher-{index:03d}.jpg"
            crop_path = os.path.join(batch_crops_dir, crop_filename)

            voucher_result: Dict = {
                "index": index,
                "serial_number": None,
                "voucher_code": None,
                "serial_number_confidence": 0.0,
                "voucher_code_confidence": 0.0,
                "crop_path": crop_path,
                "confidence": 0.0,
                "status": "failed",
                "raw_ocr": [],
                "error": None,
            }

            t_start = time.monotonic()
            try:
                # 1. Select best validated crop (handles padding + skew/perspective)
                crop = self._detector.select_best_crop(img, meta)
                crop = _upscale_if_needed(crop, min_height=120)

                # Save best crop before barcode & OCR
                cv2.imwrite(crop_path, crop)

                # 2. Priority 1: decode barcode from best crop
                barcode_result = read_barcode(crop)
                barcode_val: Optional[str] = None
                barcode_y_ratio: Optional[float] = None
                if barcode_result:
                    barcode_val, barcode_y_ratio = barcode_result
                    logger.info(
                        "Barcode ditemukan pada voucher %d: %s (y_ratio=%.3f)",
                        index,
                        barcode_val,
                        barcode_y_ratio,
                    )

                # 3. Multi-pass OCR
                if barcode_val:
                    # Area kiri ~60% terlebih dahulu agar angka bawah barcode tidak mencemari
                    left_crop, _ = _split_crop(crop, left_ratio=self.LEFT_CROP_RATIO)
                    ocr_data = self._run_multipass_ocr(
                        left_crop,
                        barcode_value=barcode_val,
                        barcode_y_ratio=barcode_y_ratio,
                        is_left_crop=True,
                    )
                    # Fallback ke full crop jika kode voucher belum ketemu / pendek / conf rendah
                    vc = ocr_data.get("voucher_code")
                    vc_conf = float(ocr_data.get("voucher_code_confidence", 0.0))
                    if not vc or len(vc) < 10 or vc_conf < self.confidence_threshold:
                        full_data = self._run_multipass_ocr(
                            crop,
                            barcode_value=barcode_val,
                            barcode_y_ratio=barcode_y_ratio,
                            is_left_crop=False,
                        )
                        if self._is_better_voucher_result(full_data, ocr_data):
                            ocr_data = full_data
                else:
                    # Barcode tidak terbaca -> multi-pass OCR full crop
                    ocr_data = self._run_multipass_ocr(
                        crop,
                        barcode_value=None,
                        barcode_y_ratio=None,
                        is_left_crop=False,
                    )
                    # Fallback ke area kiri 60% jika voucher_code belum terbaca atau lemah
                    vc = ocr_data.get("voucher_code")
                    vc_conf = float(ocr_data.get("voucher_code_confidence", 0.0))
                    if not vc or len(vc) < 10 or vc_conf < self.confidence_threshold:
                        left_crop, _ = _split_crop(crop, left_ratio=self.LEFT_CROP_RATIO)
                        left_data = self._run_multipass_ocr(
                            left_crop,
                            barcode_value=None,
                            barcode_y_ratio=None,
                            is_left_crop=True,
                        )
                        ocr_data = self._merge_full_and_left(ocr_data, left_data)

                serial_number = ocr_data.get("serial_number")
                voucher_code = ocr_data.get("voucher_code")
                serial_conf = float(ocr_data.get("serial_number_confidence", 0.0))
                voucher_conf = float(ocr_data.get("voucher_code_confidence", 0.0))
                confidence = float(ocr_data.get("confidence", 0.0))
                serial_source = ocr_data.get("serial_source")
                raw_ocr = ocr_data.get("raw_ocr", [])

                # 4. Strict per-field validation for 'verified' status
                serial_valid = bool(serial_number) and (
                    serial_source == "barcode" or self._recognizer.is_valid_code(serial_number)
                )
                voucher_valid = bool(voucher_code) and self._recognizer.is_valid_code(
                    voucher_code
                )

                reasons: List[str] = []
                if not serial_valid:
                    serial_number = None if not serial_valid else serial_number
                    serial_conf = 0.0
                    reasons.append("Barcode / Serial tidak terbaca")
                if not voucher_valid:
                    voucher_code = None if not voucher_valid else voucher_code
                    voucher_conf = 0.0
                    reasons.append("Kode voucher tidak terdeteksi")

                pct = int(self.confidence_threshold * 100)
                if serial_valid and serial_conf < self.confidence_threshold:
                    reasons.append(f"Confidence rendah serial_number (< {pct}%)")
                if voucher_valid and voucher_conf < self.confidence_threshold:
                    reasons.append(f"Confidence rendah voucher_code (< {pct}%)")

                # Recompute overall confidence from validated fields
                if serial_valid or voucher_valid:
                    confidence = round((serial_conf + voucher_conf) / 2.0, 4)
                else:
                    confidence = 0.0

                if not reasons:
                    status = "verified"
                    err_msg = None
                else:
                    status = "review"
                    err_msg = "; ".join(reasons)

                voucher_result.update(
                    {
                        "serial_number": serial_number,
                        "voucher_code": voucher_code,
                        "serial_number_confidence": round(serial_conf, 4),
                        "voucher_code_confidence": round(voucher_conf, 4),
                        "confidence": confidence,
                        "status": status,
                        "raw_ocr": raw_ocr,
                        "error": err_msg,
                    }
                )

            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Voucher %d in batch '%s' failed: %s", index, batch_code, exc
                )
                voucher_result["status"] = "failed"
                voucher_result["error"] = f"Gagal proses: {exc}"

            elapsed = time.monotonic() - t_start
            logger.info(
                "Voucher %d [%s] processed in %.3f s (conf=%.3f, sn_conf=%.3f, vc_conf=%.3f)",
                index,
                voucher_result["status"],
                elapsed,
                voucher_result["confidence"],
                voucher_result["serial_number_confidence"],
                voucher_result["voucher_code_confidence"],
            )

            results.append(voucher_result)

        total_detected = len(results)
        total_success = sum(1 for r in results if r["status"] == "verified")
        total_review = sum(1 for r in results if r["status"] == "review")
        total_failed = sum(1 for r in results if r["status"] == "failed")

        failure_summary: Dict[str, int] = {}
        for r in results:
            if r.get("error"):
                for part in r["error"].split("; "):
                    part = part.strip()
                    if part:
                        failure_summary[part] = failure_summary.get(part, 0) + 1

        return {
            "total_detected": total_detected,
            "total_success": total_success,
            "total_review": total_review,
            "total_failed": total_failed,
            "failure_summary": failure_summary,
            "vouchers": results,
        }

    # ------------------------------------------------------------------
    # Multi-pass OCR orchestration
    # ------------------------------------------------------------------

    def _run_multipass_ocr(
        self,
        region: np.ndarray,
        barcode_value: Optional[str] = None,
        barcode_y_ratio: Optional[float] = None,
        is_left_crop: bool = False,
    ) -> Dict:
        """Run OCR across preprocessing variants and select the best structured result."""
        variants = self._preprocessor.generate_ocr_variants(region)
        pass_outputs: List[Dict] = []
        all_candidates: List[Dict] = []
        text_consensus: Counter[str] = Counter()

        for pass_idx, (variant_name, variant_img) in enumerate(variants):
            blocks = self._recognizer.extract_text_blocks(variant_img)
            pass_data = self._recognizer.extract_from_blocks(
                blocks,
                img_shape=variant_img.shape,
                barcode_value=barcode_value,
                barcode_y_ratio=barcode_y_ratio,
                is_left_crop=is_left_crop,
            )
            pass_data["variant"] = variant_name
            pass_outputs.append(pass_data)

            seen_in_pass = set()
            for cand in pass_data.get("candidates", []):
                cand_copy = {**cand, "variant": variant_name}
                all_candidates.append(cand_copy)
                if cand["text"] not in seen_in_pass:
                    seen_in_pass.add(cand["text"])
                    text_consensus[cand["text"]] += 1

            # Early stop only after >=2 passes when high-confidence consensus is reached
            if pass_idx >= 1 and self._has_high_confidence_consensus(
                pass_outputs, text_consensus, barcode_value=barcode_value
            ):
                break

        return self._select_best_multipass_result(
            pass_outputs=pass_outputs,
            all_candidates=all_candidates,
            text_consensus=text_consensus,
            barcode_value=barcode_value,
            barcode_y_ratio=barcode_y_ratio,
            is_left_crop=is_left_crop,
        )

    @staticmethod
    def _has_high_confidence_consensus(
        pass_outputs: List[Dict],
        text_consensus: Counter[str],
        barcode_value: Optional[str],
    ) -> bool:
        """Return True if at least 2 passes agree on long, high-confidence fields."""
        latest = pass_outputs[-1]
        vc = latest.get("voucher_code")
        vc_conf = float(latest.get("voucher_code_confidence", 0.0))
        if not vc or len(vc) < 13 or vc_conf < 0.95 or text_consensus[vc] < 2:
            return False

        if barcode_value:
            return True

        sn = latest.get("serial_number")
        sn_conf = float(latest.get("serial_number_confidence", 0.0))
        return bool(
            sn and sn != vc and len(sn) >= 12 and sn_conf >= 0.95 and text_consensus[sn] >= 2
        )

    def _select_best_multipass_result(
        self,
        pass_outputs: List[Dict],
        all_candidates: List[Dict],
        text_consensus: Counter[str],
        barcode_value: Optional[str],
        barcode_y_ratio: Optional[float],
        is_left_crop: bool,
    ) -> Dict:
        """Select optimal serial_number and voucher_code across all OCR passes."""
        barcode_clean = barcode_value.strip() if barcode_value else None
        has_barcode = bool(barcode_clean and len(barcode_clean) >= 6)

        # Deduplicate cross-pass candidates by text, keeping the best-scoring occurrence
        best_by_text: Dict[str, Dict] = {}
        for cand in all_candidates:
            txt = cand["text"]
            prev = best_by_text.get(txt)
            if prev is None or (
                cand["digit_ratio"],
                -cand["corrections"],
                cand["confidence"],
            ) > (
                prev["digit_ratio"],
                -prev["corrections"],
                prev["confidence"],
            ):
                best_by_text[txt] = cand

        unique_cands = list(best_by_text.values())

        def _consensus_bonus(text: str) -> float:
            return min(2, max(0, text_consensus[text] - 1)) * 0.45

        serial_number: Optional[str] = barcode_clean if has_barcode else None
        serial_conf: float = 1.0 if has_barcode else 0.0
        serial_source: Optional[str] = "barcode" if has_barcode else None
        voucher_code: Optional[str] = None
        voucher_conf: float = 0.0

        if has_barcode:
            vc_cands = [c for c in unique_cands if c["text"] != serial_number]
            if vc_cands:
                best_vc = max(
                    vc_cands,
                    key=lambda c: self._recognizer.score_as_voucher(
                        c, barcode_y_ratio=barcode_y_ratio, is_left_crop=is_left_crop
                    )
                    + _consensus_bonus(c["text"]),
                )
                voucher_code = best_vc["text"]
                voucher_conf = float(best_vc["confidence"])
        else:
            if is_left_crop:
                if unique_cands:
                    best_vc = max(
                        unique_cands,
                        key=lambda c: self._recognizer.score_as_voucher(
                            c, barcode_y_ratio=barcode_y_ratio, is_left_crop=True
                        )
                        + _consensus_bonus(c["text"]),
                    )
                    voucher_code = best_vc["text"]
                    voucher_conf = float(best_vc["confidence"])
            elif len(unique_cands) == 1:
                only = unique_cands[0]
                if only["rel_x"] <= 0.60:
                    voucher_code = only["text"]
                    voucher_conf = float(only["confidence"])
                else:
                    serial_number = only["text"]
                    serial_conf = float(only["confidence"])
                    serial_source = "ocr"
            elif len(unique_cands) >= 2:
                best_pair: Optional[Tuple[Dict, Dict]] = None
                best_score = float("-inf")
                for s_it in unique_cands:
                    for v_it in unique_cands:
                        if s_it["text"] == v_it["text"]:
                            continue
                        order_bonus = 1.0 if s_it["rel_x"] > v_it["rel_x"] else -1.0
                        y_align = 1.0 - abs(
                            s_it.get("rel_y", 0.5) - v_it.get("rel_y", 0.5)
                        )
                        total = (
                            self._recognizer.score_as_serial(s_it)
                            + _consensus_bonus(s_it["text"])
                            + self._recognizer.score_as_voucher(
                                v_it, is_left_crop=False
                            )
                            + _consensus_bonus(v_it["text"])
                            + order_bonus
                            + y_align * 0.8
                        )
                        if total > best_score:
                            best_score = total
                            best_pair = (s_it, v_it)
                if best_pair:
                    s_it, v_it = best_pair
                    serial_number = s_it["text"]
                    serial_conf = float(s_it["confidence"])
                    serial_source = "ocr"
                    voucher_code = v_it["text"]
                    voucher_conf = float(v_it["confidence"])

        # Fallback: inspect per-pass results if cross-pass candidates missed regex fallback
        if voucher_code is None:
            vc_passes = [
                p
                for p in pass_outputs
                if p.get("voucher_code") and p.get("voucher_code") != serial_number
            ]
            if vc_passes:
                best_p = max(
                    vc_passes,
                    key=lambda p: (
                        len(p["voucher_code"]),
                        p.get("voucher_code_confidence", 0.0),
                    ),
                )
                voucher_code = best_p["voucher_code"]
                voucher_conf = float(best_p.get("voucher_code_confidence", 0.0))

        # Pick most informative raw_ocr pass (containing winning texts or highest confidence)
        best_raw_pass = max(
            pass_outputs,
            key=lambda p: (
                int(bool(voucher_code and p.get("voucher_code") == voucher_code)),
                int(bool(serial_number and p.get("serial_number") == serial_number)),
                len(p.get("candidates", [])),
                p.get("confidence", 0.0),
            ),
            default={"raw_ocr": []},
        )

        serial_conf = round(max(0.0, min(1.0, serial_conf)), 4) if serial_number else 0.0
        voucher_conf = round(max(0.0, min(1.0, voucher_conf)), 4) if voucher_code else 0.0
        overall = (
            round((serial_conf + voucher_conf) / 2.0, 4)
            if (serial_number or voucher_code)
            else 0.0
        )

        return {
            "serial_number": serial_number,
            "voucher_code": voucher_code,
            "serial_number_confidence": serial_conf,
            "voucher_code_confidence": voucher_conf,
            "confidence": overall,
            "serial_source": serial_source,
            "raw_ocr": best_raw_pass.get("raw_ocr", []),
        }

    @staticmethod
    def _is_better_voucher_result(candidate: Dict, current: Dict) -> bool:
        """Compare two OCR results when looking for a better voucher_code."""
        cand_vc = candidate.get("voucher_code") or ""
        curr_vc = current.get("voucher_code") or ""
        if not cand_vc:
            return False
        if not curr_vc:
            return True
        cand_score = (
            12 <= len(cand_vc) <= 17,
            len(cand_vc) >= 10,
            len(cand_vc),
            float(candidate.get("voucher_code_confidence", 0.0)),
        )
        curr_score = (
            12 <= len(curr_vc) <= 17,
            len(curr_vc) >= 10,
            len(curr_vc),
            float(current.get("voucher_code_confidence", 0.0)),
        )
        return cand_score > curr_score

    def _merge_full_and_left(self, full_data: Dict, left_data: Dict) -> Dict:
        """Combine full-crop serial_number with improved left-60% voucher_code."""
        sn = full_data.get("serial_number")
        sn_conf = float(full_data.get("serial_number_confidence", 0.0))
        sn_source = full_data.get("serial_source")

        left_vc = left_data.get("voucher_code")
        if left_vc and left_vc != sn and self._is_better_voucher_result(left_data, full_data):
            vc = left_vc
            vc_conf = float(left_data.get("voucher_code_confidence", 0.0))
            raw_ocr = full_data.get("raw_ocr") or left_data.get("raw_ocr", [])
        else:
            vc = full_data.get("voucher_code")
            vc_conf = float(full_data.get("voucher_code_confidence", 0.0))
            raw_ocr = full_data.get("raw_ocr", [])

        overall = round((sn_conf + vc_conf) / 2.0, 4) if (sn or vc) else 0.0
        return {
            "serial_number": sn,
            "voucher_code": vc,
            "serial_number_confidence": round(sn_conf, 4),
            "voucher_code_confidence": round(vc_conf, 4),
            "confidence": overall,
            "serial_source": sn_source,
            "raw_ocr": raw_ocr,
        }

"""Text recognition for voucher crops using PaddleOCR."""

import logging
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Regex patterns for voucher field extraction
_RE_NUMERIC = re.compile(r"^\d+$")
_RE_VOUCHER_CODE = re.compile(r"\d{10,20}")

# Karakter OCR yang sering salah baca sebagai digit
_OCR_CHAR_CORRECTIONS: dict[str, str] = {
    "O": "0", "o": "0",   # huruf O -> angka 0
    "I": "1", "l": "1",   # huruf I/l -> angka 1
    "S": "5",              # huruf S -> angka 5
    "B": "8",              # huruf B -> angka 8
    "G": "6",              # huruf G -> angka 6
    "Z": "2",              # huruf Z -> angka 2
    "D": "0",              # huruf D -> angka 0
}

# ---------------------------------------------------------------------------
# Pairing constants
# ---------------------------------------------------------------------------
# Voucher code (kiri/tengah-kiri): umumnya 10–17 digit (Indosat 14 digit)
_VOUCHER_CODE_MIN_LEN = 8
_VOUCHER_CODE_IDEAL_MIN_LEN = 12
_VOUCHER_CODE_MAX_LEN = 17

# Serial number (kanan): umumnya 10–16 digit (Indosat 13 digit)
_SERIAL_NUMBER_MIN_LEN = 8
_SERIAL_NUMBER_IDEAL_MIN_LEN = 11
_SERIAL_NUMBER_MAX_LEN = 16

# Posisi relatif: area kiri sekitar 60% untuk voucher code, > 50% untuk serial
_LEFT_THRESHOLD = 0.60
_RIGHT_THRESHOLD = 0.50

# Rasio digit minimum agar teks tidak dianggap huruf biasa yang dipaksa jadi angka
_MIN_DIGIT_RATIO = 0.65


class VoucherRecognizer:
    """Wraps PaddleOCR to extract structured voucher data from image crops.

    Uses lazy initialisation so that the heavy PaddleOCR model is only
    loaded the first time OCR is actually needed.
    """

    def __init__(self):
        self._ocr = None  # Lazy-loaded on first use

    # ------------------------------------------------------------------
    # Text cleaning & validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_numeric_text(text: str) -> str:
        """Bersihkan teks OCR agar menjadi string digit murni.

        1. Strip spasi, titik pemisah grup, dan tanda hubung.
        2. Koreksi karakter huruf yang sering salah baca sebagai digit.
        3. Hapus semua karakter non-digit yang tersisa.
        """
        text = re.sub(r"[\s.\-]+", "", text)
        corrected = []
        for ch in text:
            if ch.isdigit():
                corrected.append(ch)
            elif ch in _OCR_CHAR_CORRECTIONS:
                corrected.append(_OCR_CHAR_CORRECTIONS[ch])
        return "".join(corrected)

    @staticmethod
    def _analyze_numeric_text(text: str) -> Tuple[str, float, int]:
        """Analyze *text* for numeric purity and character corrections.

        Returns:
            ``(clean_digits, native_digit_ratio, correction_count)``
        """
        compact = re.sub(r"[\s.\-]+", "", text.strip())
        if not compact:
            return "", 0.0, 0

        native_digits = sum(1 for ch in compact if ch.isdigit())
        digit_ratio = native_digits / float(len(compact))

        corrected: List[str] = []
        corrections = 0
        for ch in compact:
            if ch.isdigit():
                corrected.append(ch)
            elif ch in _OCR_CHAR_CORRECTIONS:
                corrected.append(_OCR_CHAR_CORRECTIONS[ch])
                corrections += 1

        clean = "".join(corrected)
        return clean, digit_ratio, corrections

    @staticmethod
    def is_valid_code(text: Optional[str], min_len: int = _VOUCHER_CODE_MIN_LEN) -> bool:
        """Return True if *text* is a plausible numeric voucher/serial code."""
        if not text:
            return False
        s = str(text).strip()
        if not _RE_NUMERIC.match(s) or len(s) < min_len:
            return False
        # Tolak pola artefak semua digit sama (mis. '00000000' atau '11111111')
        if len(set(s)) == 1:
            return False
        return True

    def extract_text_blocks(self, img: np.ndarray) -> List[Dict]:
        """Run OCR on *img* and return all detected text blocks.

        Args:
            img: Grayscale or BGR image crop.

        Returns:
            List of dicts: ``{"text": str, "confidence": float, "bbox": list}``
        """
        ocr = self._get_ocr()

        result = ocr.ocr(img, cls=True)
        blocks = []

        if not result:
            return blocks

        for line_group in result:
            if not line_group:
                continue
            for line in line_group:
                if not line:
                    continue
                bbox, (text, confidence) = line
                blocks.append(
                    {
                        "text": text.strip(),
                        "confidence": float(confidence),
                        "bbox": bbox,
                    }
                )

        return blocks

    def collect_numeric_candidates(
        self, blocks: List[Dict], img_shape: Tuple[int, ...]
    ) -> List[Dict]:
        """Extract validated numeric candidate items with position & format metrics."""
        img_h = img_shape[0] if len(img_shape) >= 1 and img_shape[0] > 0 else 1
        img_w = img_shape[1] if len(img_shape) >= 2 and img_shape[1] > 0 else 1

        numeric_items: List[Dict] = []
        for block in blocks:
            raw_text = str(block.get("text", "")).strip()
            if not raw_text:
                continue

            clean_text, digit_ratio, corrections = self._analyze_numeric_text(raw_text)
            conf = float(block.get("confidence", 0.0))

            # Guard false positives: reject mostly-alphabetic text (e.g. headers/dates)
            if digit_ratio < _MIN_DIGIT_RATIO:
                # Fallback: check if a long pure-digit substring is embedded after a label
                compact = re.sub(r"[\s.\-]+", "", raw_text)
                match = _RE_VOUCHER_CODE.search(compact)
                if not match:
                    continue
                clean_text = match.group()
                digit_ratio = 1.0
                corrections = 0
                conf = conf * 0.95

            if not self.is_valid_code(clean_text, min_len=_VOUCHER_CODE_MIN_LEN):
                continue

            bbox = block.get("bbox", [])
            x_center = sum(pt[0] for pt in bbox) / len(bbox) if bbox else img_w * 0.5
            y_center = sum(pt[1] for pt in bbox) / len(bbox) if bbox else img_h * 0.5

            numeric_items.append(
                {
                    "text": clean_text,
                    "raw_text": raw_text,
                    "confidence": conf,
                    "digit_ratio": digit_ratio,
                    "corrections": corrections,
                    "x": x_center,
                    "rel_x": x_center / float(img_w),
                    "y": y_center,
                    "rel_y": y_center / float(img_h),
                }
            )

        # Deduplicate by cleaned text, keeping highest-quality instance
        unique_items: List[Dict] = []
        seen_texts = set()
        for item in sorted(
            numeric_items,
            key=lambda it: (it["digit_ratio"], len(it["text"]), it["confidence"]),
            reverse=True,
        ):
            if item["text"] not in seen_texts:
                seen_texts.add(item["text"])
                unique_items.append(item)

        return unique_items

    def extract_from_blocks(
        self,
        blocks: List[Dict],
        img_shape: Tuple[int, ...],
        barcode_value: Optional[str] = None,
        barcode_y_ratio: Optional[float] = None,
        is_left_crop: bool = False,
    ) -> Dict:
        """Extract fields and per-field confidence from pre-extracted OCR *blocks*."""
        barcode_clean = barcode_value.strip() if barcode_value else None
        has_valid_barcode = bool(barcode_clean and len(barcode_clean) >= 6)

        serial_number: Optional[str] = barcode_clean if has_valid_barcode else None
        serial_conf: float = 1.0 if has_valid_barcode else 0.0
        serial_source: Optional[str] = "barcode" if has_valid_barcode else None

        voucher_code: Optional[str] = None
        voucher_conf: float = 0.0

        unique_items = self.collect_numeric_candidates(blocks, img_shape)

        if has_valid_barcode:
            candidates = [it for it in unique_items if it["text"] != serial_number]
            if candidates:
                best_vc = self._pick_best_voucher_item(
                    candidates,
                    barcode_y_ratio=barcode_y_ratio,
                    is_left_crop=is_left_crop,
                )
                if best_vc:
                    voucher_code = best_vc["text"]
                    voucher_conf = float(best_vc["confidence"])
        else:
            serial_item, voucher_item = self._assign_by_scoring(
                unique_items, is_left_crop=is_left_crop
            )
            if serial_item:
                serial_number = serial_item["text"]
                serial_conf = float(serial_item["confidence"])
                serial_source = "ocr"
            if voucher_item:
                voucher_code = voucher_item["text"]
                voucher_conf = float(voucher_item["confidence"])

        if voucher_code is None and blocks:
            # Regex fallback on compacted block text (10-20 consecutive digits)
            for block in sorted(blocks, key=lambda b: float(b.get("confidence", 0.0)), reverse=True):
                compact = re.sub(r"[\s.\-]+", "", str(block.get("text", "")))
                match = _RE_VOUCHER_CODE.search(compact)
                if match:
                    candidate = match.group()
                    if candidate != serial_number and self.is_valid_code(candidate):
                        voucher_code = candidate
                        voucher_conf = float(block.get("confidence", 0.0)) * 0.95
                        break

        serial_conf = round(max(0.0, min(1.0, serial_conf)), 4) if serial_number else 0.0
        voucher_conf = round(max(0.0, min(1.0, voucher_conf)), 4) if voucher_code else 0.0

        if serial_number or voucher_code:
            confidence = round((serial_conf + voucher_conf) / 2.0, 4)
        else:
            confidence = 0.0

        return {
            "serial_number": serial_number,
            "voucher_code": voucher_code,
            "serial_number_confidence": serial_conf,
            "voucher_code_confidence": voucher_conf,
            "confidence": confidence,
            "serial_source": serial_source,
            "raw_ocr": blocks,
            "candidates": unique_items,
        }

    def extract_voucher_data(
        self,
        img: np.ndarray,
        barcode_value: Optional[str] = None,
        barcode_y_ratio: Optional[float] = None,
        is_left_crop: bool = False,
    ) -> Dict:
        """Extract serial_number, voucher_code, and per-field confidence from *img*."""
        blocks = self.extract_text_blocks(img)
        img_shape = img.shape if hasattr(img, "shape") else (1, 1)
        return self.extract_from_blocks(
            blocks,
            img_shape=img_shape,
            barcode_value=barcode_value,
            barcode_y_ratio=barcode_y_ratio,
            is_left_crop=is_left_crop,
        )

    # ------------------------------------------------------------------
    # Scoring helpers (public for pipeline multi-pass selection)
    # ------------------------------------------------------------------

    def score_as_serial(
        self, item: dict, barcode_y_ratio: Optional[float] = None
    ) -> float:
        """Skor seberapa cocok *item* sebagai serial number (sisi kanan)."""
        score = 0.0

        # Posisi: semakin kanan -> semakin tinggi
        score += item["rel_x"] * 2.0
        if item["rel_x"] >= _RIGHT_THRESHOLD:
            score += 0.8

        # Panjang digit (ideal 11-16 digit, mis. 13 digit Indosat)
        n = len(item["text"])
        if _SERIAL_NUMBER_IDEAL_MIN_LEN <= n <= _SERIAL_NUMBER_MAX_LEN:
            score += 1.5 + (n - _SERIAL_NUMBER_IDEAL_MIN_LEN) / max(
                1, _SERIAL_NUMBER_MAX_LEN - _SERIAL_NUMBER_IDEAL_MIN_LEN
            ) * 0.5
        elif _SERIAL_NUMBER_MIN_LEN <= n < _SERIAL_NUMBER_IDEAL_MIN_LEN:
            score += 0.6
        elif n > _SERIAL_NUMBER_MAX_LEN:
            score += 0.2

        # Format kemurnian digit & OCR confidence
        score += item.get("digit_ratio", 1.0) * 0.8
        score -= item.get("corrections", 0) * 0.2
        score += item["confidence"] * 1.0

        if barcode_y_ratio is not None:
            y_proximity = 1.0 - abs(item.get("rel_y", 0.5) - barcode_y_ratio)
            score += y_proximity * 1.5

        return score

    def score_as_voucher(
        self,
        item: dict,
        barcode_y_ratio: Optional[float] = None,
        is_left_crop: bool = False,
    ) -> float:
        """Skor seberapa cocok *item* sebagai voucher code (sisi kiri / tengah-kiri)."""
        score = 0.0

        # Posisi horizontal: di full crop, area kiri <= 60% mendapat prioritas.
        # Pada sub-crop kiri 60%, teks kode biasanya berada di tengah sub-crop.
        if is_left_crop:
            score += 1.5
        else:
            score += (1.0 - item["rel_x"]) * 2.0
            if item["rel_x"] <= _LEFT_THRESHOLD:
                score += 0.8

        # Panjang digit: kode 12-17 digit (Indosat 14 digit) sangat diutamakan
        # dibanding potongan terpotong (8-11 digit)
        n = len(item["text"])
        if _VOUCHER_CODE_IDEAL_MIN_LEN <= n <= _VOUCHER_CODE_MAX_LEN:
            score += 2.0 + (n - _VOUCHER_CODE_IDEAL_MIN_LEN) / max(
                1, _VOUCHER_CODE_MAX_LEN - _VOUCHER_CODE_IDEAL_MIN_LEN
            ) * 0.5
        elif _VOUCHER_CODE_MIN_LEN <= n < _VOUCHER_CODE_IDEAL_MIN_LEN:
            score += 0.5 + 0.3 * (n - _VOUCHER_CODE_MIN_LEN) / max(
                1, _VOUCHER_CODE_IDEAL_MIN_LEN - _VOUCHER_CODE_MIN_LEN
            )
        elif n > _VOUCHER_CODE_MAX_LEN:
            score += 0.3

        # Kemurnian digit & confidence OCR
        score += item.get("digit_ratio", 1.0) * 0.8
        score -= item.get("corrections", 0) * 0.25
        score += item["confidence"] * 1.2

        # Y-alignment jika diketahui posisi vertikal barcode
        if barcode_y_ratio is not None:
            y_proximity = 1.0 - abs(item.get("rel_y", 0.5) - barcode_y_ratio)
            score += y_proximity * 3.0

        return score

    # Keep backwards-compatible private method aliases
    def _score_as_serial(self, item: dict) -> float:
        return self.score_as_serial(item)

    def _score_as_voucher(self, item: dict) -> float:
        return self.score_as_voucher(item)

    def _pick_best_voucher_item(
        self,
        candidates: list,
        barcode_y_ratio: Optional[float] = None,
        is_left_crop: bool = False,
    ) -> Optional[dict]:
        """Pilih item kandidat terbaik sebagai voucher code berdasarkan scoring."""
        if not candidates:
            return None

        best = max(
            candidates,
            key=lambda item: self.score_as_voucher(
                item,
                barcode_y_ratio=barcode_y_ratio,
                is_left_crop=is_left_crop,
            ),
        )
        logger.debug(
            "Voucher code terpilih: %r (rel_x=%.2f, rel_y=%.2f, len=%d, conf=%.3f)",
            best["text"],
            best["rel_x"],
            best.get("rel_y", -1),
            len(best["text"]),
            best["confidence"],
        )
        return best

    def _pick_best_voucher_code(
        self, candidates: list, barcode_y_ratio: Optional[float] = None
    ) -> Optional[str]:
        item = self._pick_best_voucher_item(candidates, barcode_y_ratio=barcode_y_ratio)
        return item["text"] if item else None

    def _assign_by_scoring(
        self, items: list, is_left_crop: bool = False
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Assign serial_number item dan voucher_code item tanpa barcode."""
        if not items:
            return None, None
        if len(items) == 1:
            item = items[0]
            if is_left_crop or item["rel_x"] <= _LEFT_THRESHOLD:
                return None, item
            return item, None

        scored = [
            {
                **it,
                "_score_serial": self.score_as_serial(it),
                "_score_voucher": self.score_as_voucher(it, is_left_crop=is_left_crop),
            }
            for it in items
        ]

        if is_left_crop:
            voucher_item = max(scored, key=lambda it: it["_score_voucher"])
            return None, voucher_item

        # Evaluate best disjoint pair (serial_item, voucher_item)
        best_pair: Optional[Tuple[dict, dict]] = None
        best_pair_score = float("-inf")
        for s_it in scored:
            for v_it in scored:
                if s_it is v_it:
                    continue
                # Bonus if serial is clearly to the right of voucher
                order_bonus = 1.0 if s_it["rel_x"] > v_it["rel_x"] else -1.0
                # Bonus if both lie near the same horizontal row
                y_align = 1.0 - abs(s_it.get("rel_y", 0.5) - v_it.get("rel_y", 0.5))
                total = (
                    s_it["_score_serial"]
                    + v_it["_score_voucher"]
                    + order_bonus
                    + y_align * 0.8
                )
                if total > best_pair_score:
                    best_pair_score = total
                    best_pair = (s_it, v_it)

        if best_pair:
            return best_pair

        serial_item = max(scored, key=lambda it: it["_score_serial"])
        return serial_item, None

    def _get_ocr(self):
        """Lazy-initialise and return the PaddleOCR instance."""
        if self._ocr is None:
            logger.info("Initialising PaddleOCR (PP-OCRv5 Mobile)…")
            import pyclipper  # noqa: F401, PLC0415 — Pre-load pyclipper to prevent zlib symbol conflict with paddlepaddle
            from paddleocr import PaddleOCR  # noqa: PLC0415

            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                show_log=False,
            )
            logger.info("PaddleOCR initialised.")
        return self._ocr

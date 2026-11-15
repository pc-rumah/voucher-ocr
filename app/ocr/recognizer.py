"""Text recognition for voucher crops using PaddleOCR."""

import logging
import re
from typing import List, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Regex patterns for voucher field extraction
_RE_NUMERIC = re.compile(r"^\d+$")
_RE_VOUCHER_CODE = re.compile(r"\d{10,20}")


class VoucherRecognizer:
    """Wraps PaddleOCR to extract structured voucher data from image crops.

    Uses lazy initialisation so that the heavy PaddleOCR model is only
    loaded the first time OCR is actually needed.
    """

    def __init__(self):
        self._ocr = None  # Lazy-loaded on first use

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def extract_voucher_data(self, img: np.ndarray, barcode_value: Optional[str] = None) -> Dict:
        """Extract serial_number, voucher_code and confidence from *img*.

        Heuristics
        ----------
        * **serial_number** — barcode value (pyzbar) jika tersedia, fallback ke
          longest purely-numeric string (≥ 8 digits) dari OCR.
        * **voucher_code** — longest purely-numeric string dari OCR yang berbeda
          dari serial_number (biasanya nomor telepon kiri pada voucher Indosat).
        * **confidence** — average of the top-2 text block confidences.

        Args:
            img: Grayscale or BGR voucher crop.
            barcode_value: Nilai barcode yang sudah dibaca (pyzbar). Jika diisi,
                langsung dipakai sebagai serial_number tanpa perlu heuristik OCR
                untuk field ini.

        Returns:
            ``{"serial_number": str|None, "voucher_code": str|None,
               "confidence": float, "raw_ocr": list}``
        """
        blocks = self.extract_text_blocks(img)

        # Barcode (Serial Number kanan) sebagai prioritas utama
        serial_number: Optional[str] = barcode_value.strip() if barcode_value else None
        voucher_code: Optional[str] = None
        confidence: float = 0.0

        if blocks:
            # Compute confidence from top-2 blocks (sorted by confidence desc)
            sorted_blocks = sorted(blocks, key=lambda b: b["confidence"], reverse=True)
            top_confs = [b["confidence"] for b in sorted_blocks[:2]]
            confidence = sum(top_confs) / len(top_confs)

            # Gather all numeric-only strings (>= 8 digits) with bounding box position
            img_w = img.shape[1] if hasattr(img, "shape") and len(img.shape) >= 2 else 1
            numeric_items = []
            for block in blocks:
                text = re.sub(r"[\s\-]+", "", block["text"])  # strip spaces & hyphens
                if _RE_NUMERIC.match(text) and len(text) >= 8:
                    bbox = block.get("bbox", [])
                    x_center = sum(pt[0] for pt in bbox) / len(bbox) if bbox else 0.0
                    numeric_items.append(
                        {
                            "text": text,
                            "confidence": block["confidence"],
                            "x": x_center,
                            "rel_x": x_center / img_w if img_w > 0 else 0.5,
                        }
                    )

            # Deduplicate items by text
            unique_items = []
            seen_texts = set()
            for item in sorted(numeric_items, key=lambda it: it["confidence"], reverse=True):
                if item["text"] not in seen_texts:
                    seen_texts.add(item["text"])
                    unique_items.append(item)

            if serial_number:
                # Barcode terbaca via pyzbar: serial_number sudah terisi.
                # Cari voucher_code dari teks yang berbeda dari serial_number (prefer sisi kiri).
                candidates = [it for it in unique_items if it["text"] != serial_number]
                if candidates:
                    candidates.sort(key=lambda it: it["x"])
                    voucher_code = candidates[0]["text"]
            else:
                # Barcode tidak terbaca: gunakan tata letak spasial voucher
                # - Sisi kanan (x lebih besar) = Serial Number (di bawah barcode)
                # - Sisi kiri (x lebih kecil) = Voucher Code (nomor isi ulang / gosok)
                if len(unique_items) >= 2:
                    by_x = sorted(unique_items, key=lambda it: it["x"])
                    voucher_code = by_x[0]["text"]
                    serial_number = by_x[-1]["text"]
                elif len(unique_items) == 1:
                    serial_number = unique_items[0]["text"]

            if voucher_code is None:
                # Regex fallback: scan all block texts for 10–20 digit sequences
                for block in blocks:
                    match = _RE_VOUCHER_CODE.search(block["text"])
                    if match:
                        candidate = match.group()
                        if candidate != serial_number:
                            voucher_code = candidate
                            break

        return {
            "serial_number": serial_number,
            "voucher_code": voucher_code,
            "confidence": confidence,
            "raw_ocr": blocks,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

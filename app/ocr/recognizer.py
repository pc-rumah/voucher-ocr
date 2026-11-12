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

    def extract_voucher_data(self, img: np.ndarray) -> Dict:
        """Extract serial_number, voucher_code and confidence from *img*.

        Heuristics
        ----------
        * **serial_number** — longest purely-numeric string (≥ 8 digits).
        * **voucher_code** — second-longest numeric string, or any string
          matching ``\\d{10,20}``.
        * **confidence** — average of the top-2 text block confidences.

        Args:
            img: Grayscale or BGR voucher crop.

        Returns:
            ``{"serial_number": str|None, "voucher_code": str|None,
               "confidence": float, "raw_ocr": list}``
        """
        blocks = self.extract_text_blocks(img)

        serial_number: Optional[str] = None
        voucher_code: Optional[str] = None
        confidence: float = 0.0

        if blocks:
            # Compute confidence from top-2 blocks (sorted by confidence desc)
            sorted_blocks = sorted(blocks, key=lambda b: b["confidence"], reverse=True)
            top_confs = [b["confidence"] for b in sorted_blocks[:2]]
            confidence = sum(top_confs) / len(top_confs)

            # Gather all numeric-only strings
            numeric_strings = []
            for block in blocks:
                text = re.sub(r"\s+", "", block["text"])  # strip internal spaces
                if _RE_NUMERIC.match(text):
                    numeric_strings.append((text, block["confidence"]))

            # Sort by length (longest first)
            numeric_strings.sort(key=lambda t: len(t[0]), reverse=True)

            # serial_number: longest numeric string with >= 8 digits
            for text, _ in numeric_strings:
                if len(text) >= 8:
                    serial_number = text
                    break

            # voucher_code: second longest numeric string, or regex fallback
            if serial_number and len(numeric_strings) >= 2:
                for text, _ in numeric_strings:
                    if text != serial_number:
                        voucher_code = text
                        break

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

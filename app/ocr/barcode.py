"""Barcode reading helper using pyzbar (optional dependency)."""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _pick_best_barcode(barcodes: list, region_height: int) -> Optional[Tuple[str, float]]:
    """Select the cleanest decoded barcode (preferring numeric serials >= 8 chars)."""
    valid: List[Tuple[bool, int, str, float, str]] = []
    for bc in barcodes:
        raw_bytes = getattr(bc, "data", b"")
        if not raw_bytes:
            continue
        value = raw_bytes.decode("utf-8", errors="replace").strip()
        if len(value) < 6:
            continue
        rect = bc.rect
        y_center = rect.top + rect.height / 2.0
        y_ratio = y_center / region_height if region_height > 0 else 0.5
        y_ratio = max(0.0, min(1.0, float(y_ratio)))
        is_numeric_serial = value.isdigit() and len(value) >= 8
        valid.append((is_numeric_serial, len(value), value, y_ratio, str(bc.type)))

    if not valid:
        return None

    # Prefer numeric serial >= 8 digits, then longer value
    valid.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, value, y_ratio, bc_type = valid[0]
    logger.debug(
        "Barcode ditemukan: %r (type=%s, y_ratio=%.3f)",
        value,
        bc_type,
        y_ratio,
    )
    return value, y_ratio


def _decode_variants(decode_fn, region: np.ndarray, orig_height: int) -> Optional[Tuple[str, float]]:
    """Try pyzbar decode on raw, CLAHE, and upscaled/Otsu variants of *region*."""
    # 1. Raw image (fastest, highest fidelity on crisp photos)
    barcodes = decode_fn(region)
    picked = _pick_best_barcode(barcodes, orig_height)
    if picked:
        return picked

    gray = _to_gray(region)

    # 2. CLAHE contrast boost for dark / uneven lighting
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    barcodes = decode_fn(clahe)
    picked = _pick_best_barcode(barcodes, orig_height)
    if picked:
        return picked

    # 3. Upscaled grayscale & Otsu threshold for low-res barcodes
    rh, rw = gray.shape[:2]
    if rh > 0 and rw > 0:
        scale = 2.0 if rh < 240 else 1.5
        up = cv2.resize(
            gray,
            (max(1, int(round(rw * scale))), max(1, int(round(rh * scale)))),
            interpolation=cv2.INTER_CUBIC,
        )
        barcodes = decode_fn(up)
        picked = _pick_best_barcode(barcodes, up.shape[0])
        if picked:
            return picked

        _, otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        barcodes = decode_fn(otsu)
        picked = _pick_best_barcode(barcodes, otsu.shape[0])
        if picked:
            return picked

    return None


def read_barcode(img: np.ndarray) -> Optional[Tuple[str, float]]:
    """Attempt to decode a barcode or QR code from *img*.

    Scan strategy:
    1. Scan the right half first (typical barcode placement on vouchers).
    2. If not found, fall back to the full crop.
    Each region tries raw, CLAHE, and upscaled/Otsu variants.

    Uses ``pyzbar`` as the primary decoder. Returns ``None`` silently if
    pyzbar is not installed, the image contains no recognisable barcode,
    or any other error occurs.

    Args:
        img: BGR or grayscale image (numpy array).

    Returns:
        Tuple ``(barcode_value, barcode_y_ratio)`` where ``barcode_y_ratio``
        is the vertical barcode centre relative to image height (0.0–1.0).
        Returns ``None`` if no valid barcode is found.
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return None

    try:
        from pyzbar.pyzbar import decode  # noqa: PLC0415

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return None

        # Priority 1: right half (typical serial barcode placement)
        right_half = img[:, w // 2 :]
        if right_half.size > 0:
            result = _decode_variants(decode, right_half, h)
            if result:
                return result

        # Fallback: full image (different voucher layouts)
        logger.debug("Barcode tidak ditemukan di kanan; fallback ke full image.")
        return _decode_variants(decode, img, h)

    except ImportError:
        logger.debug("pyzbar is not installed; barcode reading is disabled.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Barcode decoding failed: %s", exc)

    return None

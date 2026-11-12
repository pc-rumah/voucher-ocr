"""Barcode reading helper using pyzbar (optional dependency)."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def read_barcode(img: np.ndarray) -> Optional[str]:
    """Attempt to decode a barcode or QR code from *img*.

    Uses ``pyzbar`` as the primary decoder. Returns ``None`` silently if
    pyzbar is not installed, the image contains no recognisable barcode,
    or any other error occurs.

    Args:
        img: BGR or grayscale image (numpy array).

    Returns:
        Decoded barcode string, or ``None`` if none found.
    """
    try:
        from pyzbar.pyzbar import decode  # noqa: PLC0415

        barcodes = decode(img)
        if barcodes:
            first = barcodes[0]
            return first.data.decode("utf-8", errors="replace")
    except ImportError:
        logger.debug("pyzbar is not installed; barcode reading is disabled.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Barcode decoding failed: %s", exc)

    return None

"""Voucher region detection from full-page images."""

import logging
from typing import List, Dict, Tuple

import cv2
import numpy as np

from app.ocr.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class VoucherDetector:
    """Detects and crops individual voucher regions from an image.

    Supports images containing a single voucher or many vouchers laid out
    in a grid or scattered arrangement.
    """

    # Filtering constants
    MIN_AREA_RATIO = 0.05   # Voucher must be at least 5 % of image area
    MAX_AREA_RATIO = 0.80   # Voucher cannot be more than 80 % of image area
    MIN_ASPECT_RATIO = 0.3
    MAX_ASPECT_RATIO = 4.0

    def __init__(self):
        self._preprocessor = ImagePreprocessor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_vouchers(self, img: np.ndarray) -> List[Dict]:
        """Detect all voucher regions in *img*.

        Strategy
        --------
        1. Preprocess image for edge/contour detection.
        2. Find external contours.
        3. Filter by area and aspect ratio.
        4. Sort top-to-bottom, left-to-right.
        5. If nothing passes the filters, fall back to grid subdivision.

        Args:
            img: BGR source image.

        Returns:
            List of dicts: ``{"contour": ndarray, "bbox": (x,y,w,h), "index": int}``
        """
        image_area = img.shape[0] * img.shape[1]
        preprocessed = self._preprocessor.preprocess_for_detection(img)

        contours, _ = cv2.findContours(
            preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for cnt in contours:
            bbox = cv2.boundingRect(cnt)
            x, y, w, h = bbox
            area = w * h
            aspect = w / h if h > 0 else 0

            if not (self.MIN_AREA_RATIO * image_area <= area <= self.MAX_AREA_RATIO * image_area):
                continue
            if not (self.MIN_ASPECT_RATIO <= aspect <= self.MAX_ASPECT_RATIO):
                continue

            # Approximate to polygon; prefer 4-sided
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            candidates.append(
                {
                    "contour": approx if len(approx) == 4 else cnt,
                    "bbox": bbox,
                    "_sort_key": (y, x),
                }
            )

        # Sort top-to-bottom, left-to-right (row-major with 50 px row tolerance)
        candidates = self._sort_grid_order(candidates, img.shape)

        # A screenshot containing stacked yellow vouchers often has one
        # external contour around the phone screen. Use the voucher rows
        # inside that contour instead of saving the whole screen as one crop.
        image_area = img.shape[0] * img.shape[1]
        has_full_image_candidate = any(
            c["bbox"][2] * c["bbox"][3] > 0.60 * image_area
            for c in candidates
        )
        if has_full_image_candidate and len(candidates) > 1:
            color_candidates = self._yellow_row_fallback(img)
            if len(color_candidates) > 1:
                candidates = color_candidates

        if not candidates:
            color_candidates = self._yellow_row_fallback(img)
            if len(color_candidates) > 1:
                candidates = color_candidates
            else:
                logger.warning(
                    "No vouchers found via contour detection; falling back to grid."
                )
                candidates = self._grid_fallback(img)

        # Assign index
        vouchers = []
        for idx, cand in enumerate(candidates):
            vouchers.append(
                {
                    "contour": cand["contour"],
                    "bbox": cand["bbox"],
                    "index": idx,
                }
            )

        logger.info("Detected %d voucher(s).", len(vouchers))
        return vouchers

    def crop_voucher(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
    ) -> np.ndarray:
        """Crop a voucher region from *img* with optional padding.

        The crop is clamped to the image boundaries so it never reads
        outside the array.

        Args:
            img: Source BGR image.
            bbox: ``(x, y, w, h)`` bounding box from :meth:`detect_vouchers`.
            padding: Pixels to expand on each side.

        Returns:
            Cropped BGR image.
        """
        h_img, w_img = img.shape[:2]
        x, y, w, h = bbox

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w_img, x + w + padding)
        y2 = min(h_img, y + h + padding)

        return img[y1:y2, x1:x2]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sort_grid_order(self, candidates: List[Dict], shape: Tuple) -> List[Dict]:
        """Sort candidates in reading order (top-to-bottom, left-to-right).

        Uses a row-tolerance of 10 % of the image height so that vouchers
        on the same visual row group together before sorting by x.
        """
        if not candidates:
            return candidates

        row_tolerance = max(20, shape[0] * 0.10)

        def sort_key(c):
            y, x = c["_sort_key"]
            row = int(y / row_tolerance)
            return (row, x)

        return sorted(candidates, key=sort_key)

    def _grid_fallback(self, img: np.ndarray) -> List[Dict]:
        """Divide the image into a regular grid as a last resort.

        Tries grids in order of increasing cell count based on the image
        aspect ratio and picks the first configuration that produces cells
        whose aspect ratio looks voucher-like.

        Grid candidates (cols × rows): 1×1, 1×2, 2×1, 2×2, 2×3, 3×2, 3×3
        """
        h, w = img.shape[:2]
        img_aspect = w / h if h > 0 else 1.0

        grid_candidates = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3)]

        chosen_grid = (1, 1)
        for cols, rows in grid_candidates:
            cell_w = w / cols
            cell_h = h / rows
            cell_aspect = cell_w / cell_h if cell_h > 0 else 1.0
            if self.MIN_ASPECT_RATIO <= cell_aspect <= self.MAX_ASPECT_RATIO:
                chosen_grid = (cols, rows)
                break

        cols, rows = chosen_grid
        cell_w = w // cols
        cell_h = h // rows

        logger.info("Grid fallback: %d col(s) × %d row(s).", cols, rows)

        candidates = []
        for row in range(rows):
            for col in range(cols):
                x = col * cell_w
                y = row * cell_h
                # Last cell takes any remainder
                cw = cell_w if col < cols - 1 else w - x
                ch = cell_h if row < rows - 1 else h - y
                bbox = (x, y, cw, ch)
                # Synthetic contour from bbox corners
                contour = np.array(
                    [[x, y], [x + cw, y], [x + cw, y + ch], [x, y + ch]],
                    dtype=np.int32,
                ).reshape(-1, 1, 2)
                candidates.append(
                    {
                        "contour": contour,
                        "bbox": bbox,
                        "_sort_key": (y, x),
                    }
                )

        return candidates

    def _yellow_row_fallback(self, img: np.ndarray) -> List[Dict]:
        """Detect stacked yellow voucher rows inside a larger screenshot."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv,
            np.array((15, 50, 80), dtype=np.uint8),
            np.array((45, 255, 255), dtype=np.uint8),
        )
        # Ignore chat chrome around the embedded phone screenshot.
        left = int(img.shape[1] * 0.20)
        right = int(img.shape[1] * 0.80)
        mask[:, :left] = 0
        mask[:, right:] = 0

        row_pixels = (mask > 0).sum(axis=1)
        minimum_pixels = max(10, int((right - left) * 0.15))
        row_indexes = np.flatnonzero(row_pixels >= minimum_pixels)
        if row_indexes.size < 2:
            return []

        row_groups = np.split(
            row_indexes, np.flatnonzero(np.diff(row_indexes) > 1) + 1
        )
        row_groups = [group for group in row_groups if group.size]
        if len(row_groups) < 2:
            return []

        centers = [int((group[0] + group[-1]) / 2) for group in row_groups]
        candidates: List[Dict] = []
        for index, center in enumerate(centers):
            top = 0 if index == 0 else (centers[index - 1] + center) // 2
            bottom = (
                img.shape[0]
                if index == len(centers) - 1
                else (center + centers[index + 1]) // 2
            )
            if center < img.shape[0] * 0.05 or center > img.shape[0] * 0.80:
                continue
            points = np.flatnonzero(mask[top:bottom].any(axis=0))
            if points.size == 0:
                continue

            x = max(0, int(points[0]) - 5)
            right = min(img.shape[1], int(points[-1]) + 6)
            y = max(0, top - 2)
            bottom = min(img.shape[0], bottom + 2)
            candidates.append(
                {
                    "contour": np.array(
                        [[x, y], [right, y], [right, bottom], [x, bottom]],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2),
                    "bbox": (x, y, right - x, bottom - y),
                    "_sort_key": (y, x),
                }
            )

        logger.info("Yellow-row fallback: detected %d row(s).", len(candidates))
        return candidates

"""Voucher region detection from full-page images."""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.ocr.preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)


class VoucherDetector:
    """Detects and crops individual voucher regions from an image.

    Supports images containing a single voucher or many vouchers laid out
    in a grid, stacked screenshot rows, or scattered arrangement.
    """

    # Filtering constants
    MIN_AREA_RATIO = 0.05   # Voucher must be at least 5 % of image area
    MAX_AREA_RATIO = 0.80   # Voucher cannot be more than 80 % of image area
    MIN_ASPECT_RATIO = 0.3
    MAX_ASPECT_RATIO = 8.0

    # Minimum valid crop dimensions (px)
    MIN_CROP_WIDTH = 40
    MIN_CROP_HEIGHT = 20

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
        5. Detect stacked screenshot rows when a screen-sized contour or
           no valid contours are found; fall back to grid subdivision.

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

            # Approximate to polygon; prefer 4-sided convex quad
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            candidates.append(
                {
                    "contour": approx if len(approx) == 4 else cnt,
                    "bbox": bbox,
                    "_sort_key": (y, x),
                }
            )

        candidates = self._sort_grid_order(candidates, img.shape)

        # A screenshot containing stacked vouchers often has one external contour
        # around the phone screen. Prefer the individual voucher rows inside it.
        has_full_image_candidate = any(
            c["bbox"][2] * c["bbox"][3] > 0.55 * image_area
            for c in candidates
        )
        if has_full_image_candidate:
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

    def validate_crop(
        self,
        crop: Optional[np.ndarray],
        min_width: int = MIN_CROP_WIDTH,
        min_height: int = MIN_CROP_HEIGHT,
    ) -> bool:
        """Return True if *crop* has valid shape, size, and non-uniform content."""
        if crop is None or not isinstance(crop, np.ndarray) or crop.size == 0:
            return False
        if crop.ndim < 2:
            return False
        h, w = crop.shape[:2]
        if h < min_height or w < min_width:
            return False
        if float(np.std(crop)) <= 1.0:
            return False
        return True

    def crop_voucher(
        self, img: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
    ) -> np.ndarray:
        """Crop a voucher region from *img* with validated, clamped padding.

        The crop is clamped to the image boundaries so it never reads
        outside the array.

        Args:
            img: Source BGR image.
            bbox: ``(x, y, w, h)`` bounding box from :meth:`detect_vouchers`.
            padding: Pixels to expand on each side (clamped to >= 0).

        Returns:
            Cropped BGR image.
        """
        if img is None or img.size == 0:
            raise ValueError("Source image is empty.")

        h_img, w_img = img.shape[:2]
        x, y, w, h = [int(v) for v in bbox]
        if w <= 0 or h <= 0:
            raise ValueError(f"Invalid bounding box dimensions: {bbox}")

        safe_pad = max(0, min(int(padding), max(w, h) // 2))
        x1 = max(0, x - safe_pad)
        y1 = max(0, y - safe_pad)
        x2 = min(w_img, x + w + safe_pad)
        y2 = min(h_img, y + h + safe_pad)

        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Clamped crop region is empty for bbox={bbox}")

        return img[y1:y2, x1:x2]

    def correct_skew(
        self, crop: np.ndarray, max_angle: float = 15.0
    ) -> Tuple[np.ndarray, float]:
        """Estimate and correct small tilt/skew angles in *crop*.

        Uses row-projection variance of edge/foreground pixels across candidate
        angles so straight crops remain untouched and tilted crops align cleanly.

        Returns:
            ``(deskewed_crop, applied_angle_deg)``
        """
        if not self.validate_crop(crop):
            return crop, 0.0

        gray = self._preprocessor.to_grayscale(crop)
        h, w = gray.shape[:2]

        # Downscale only for fast angle search
        scale = min(1.0, 240.0 / max(h, w))
        small = (
            cv2.resize(gray, (max(16, int(w * scale)), max(16, int(h * scale))))
            if scale < 1.0
            else gray
        )

        # Edge map highlights horizontal text lines, borders, and barcode bars
        edges = cv2.Sobel(small, cv2.CV_32F, 0, 1, ksize=3)
        abs_edges = np.abs(edges)

        def _row_alignment_score(mat: np.ndarray) -> float:
            row_means = mat.mean(axis=1)
            return float(np.var(row_means))

        sh, sw = abs_edges.shape[:2]
        center_s = (sw / 2.0, sh / 2.0)
        base_score = _row_alignment_score(abs_edges)

        best_angle = 0.0
        best_score = base_score
        angles = np.arange(-max_angle, max_angle + 0.5, 1.0)
        for angle in angles:
            if abs(angle) < 1.0:
                continue
            rot_m = cv2.getRotationMatrix2D(center_s, float(angle), 1.0)
            rotated = cv2.warpAffine(
                abs_edges, rot_m, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            score = _row_alignment_score(rotated)
            if score > best_score:
                best_score = score
                best_angle = float(angle)

        # Apply rotation only when there is a meaningful tilt and clear improvement
        if abs(best_angle) < 1.5 or best_score <= base_score * 1.12:
            return crop, 0.0

        # Refine around best_angle in 0.5 deg steps
        refined_angle = best_angle
        for delta in (-0.5, 0.5):
            cand_angle = best_angle + delta
            if abs(cand_angle) > max_angle:
                continue
            rot_m = cv2.getRotationMatrix2D(center_s, cand_angle, 1.0)
            rotated = cv2.warpAffine(
                abs_edges, rot_m, (sw, sh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
            score = _row_alignment_score(rotated)
            if score > best_score:
                best_score = score
                refined_angle = cand_angle

        center = (w / 2.0, h / 2.0)
        rot_matrix = cv2.getRotationMatrix2D(center, refined_angle, 1.0)
        deskewed = cv2.warpAffine(
            crop,
            rot_matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return deskewed, refined_angle

    def select_best_crop(
        self, img: np.ndarray, meta: Dict, padding: int = 10
    ) -> np.ndarray:
        """Build, validate, and pick the best crop (axis vs perspective/deskewed).

        Args:
            img: Source BGR image.
            meta: Voucher metadata dict from :meth:`detect_vouchers`.
            padding: Padding in pixels around the detected bounding region.

        Returns:
            Best validated crop ready for saving, barcode decoding, and OCR.
        """
        bbox = meta["bbox"]
        contour = meta.get("contour")

        axis_crop = self.crop_voucher(img, bbox, padding=padding)
        candidates: List[Tuple[str, np.ndarray]] = []
        if self.validate_crop(axis_crop):
            candidates.append(("axis", axis_crop))

        # Perspective / oriented-rect candidate if contour is genuinely skewed or non-rectangular
        if contour is not None and len(contour) >= 4:
            rect = cv2.minAreaRect(contour)
            rw, rh = rect[1]
            angle = abs(rect[2])
            # OpenCV minAreaRect angle lies in [0, 90] or [-90, 0] depending on version
            tilt = min(angle, abs(90.0 - angle))
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            is_tilted_or_quad = tilt >= 1.5 or (len(approx) == 4 and self._is_perspective_quad(approx, bbox))

            if is_tilted_or_quad and rw >= self.MIN_CROP_WIDTH and rh >= self.MIN_CROP_HEIGHT:
                warped = self._preprocessor.correct_perspective(
                    img, contour, padding=max(2, min(padding, 8))
                )
                if self.validate_crop(warped) and self._aspect_compatible(warped, bbox):
                    candidates.append(("perspective", warped))

        # Residual skew correction candidate on axis_crop (if axis_crop is tilted)
        if self.validate_crop(axis_crop):
            deskewed, applied_angle = self.correct_skew(axis_crop)
            if abs(applied_angle) >= 1.5 and self.validate_crop(deskewed):
                candidates.append(("deskewed", deskewed))

        if not candidates:
            return axis_crop

        if len(candidates) == 1:
            return candidates[0][1]

        best_name, best_crop = max(candidates, key=lambda item: self._score_crop_quality(item[1], item[0]))
        logger.debug("Selected '%s' crop (shape=%s).", best_name, best_crop.shape)
        return best_crop

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_perspective_quad(approx: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Check whether a 4-point polygon deviates notably from its axis bounding box."""
        _, _, bw, bh = bbox
        bbox_area = max(1, bw * bh)
        quad_area = abs(cv2.contourArea(approx))
        return quad_area < 0.92 * bbox_area

    @staticmethod
    def _aspect_compatible(crop: np.ndarray, bbox: Tuple[int, int, int, int]) -> bool:
        """Verify warped crop wasn't rotated 90 degrees or overly distorted."""
        ch, cw = crop.shape[:2]
        _, _, bw, bh = bbox
        if ch <= 0 or cw <= 0 or bh <= 0 or bw <= 0:
            return False
        crop_aspect = cw / float(ch)
        bbox_aspect = bw / float(bh)
        # Ensure landscape/portrait orientation did not flip accidentally
        if (crop_aspect >= 1.2 and bbox_aspect <= 0.8) or (crop_aspect <= 0.8 and bbox_aspect >= 1.2):
            return False
        return True

    def _score_crop_quality(self, crop: np.ndarray, kind: str) -> float:
        """Score a candidate crop by horizontal text alignment and sharpness."""
        gray = self._preprocessor.to_grayscale(crop)
        # Horizontal edge alignment score (higher when text/bars are horizontal)
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        row_var = float(np.var(sobel_y.mean(axis=1)))
        # Sharpness via Laplacian variance (normalized mildly)
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        score = row_var * 1.5 + min(lap_var, 2000.0) * 0.05
        # Slight preference for unwarped axis crop when tied (avoids resampling blur)
        if kind == "axis":
            score *= 1.05
        return score

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
        """Divide the image into a regular grid as a last resort."""
        h, w = img.shape[:2]

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

        logger.info("Grid fallback: %d col(s) x %d row(s).", cols, rows)

        candidates = []
        for row in range(rows):
            for col in range(cols):
                x = col * cell_w
                y = row * cell_h
                cw = cell_w if col < cols - 1 else w - x
                ch = cell_h if row < rows - 1 else h - y
                bbox = (x, y, cw, ch)
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
        """Detect stacked voucher rows inside a larger screenshot or photo.

        Handles two common stacked layouts:
        1. White/light voucher cards separated by thin yellow divider bars
           (e.g. Indosat physical voucher sheets): extracts each gap between
           consecutive yellow bars so vouchers are never split mid-card.
        2. Bright yellow or white voucher cards separated by dark gaps.
        """
        h_img, w_img = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tolerate darker lighting as well as bright screenshots
        mask_yellow = cv2.inRange(
            hsv,
            np.array((15, 40, 40), dtype=np.uint8),
            np.array((45, 255, 255), dtype=np.uint8),
        )

        # Check central band (12%..88%) to ignore phone side chrome
        scan_left = int(w_img * 0.12)
        scan_right = int(w_img * 0.88)
        scan_w = max(1, scan_right - scan_left)

        yellow_central = mask_yellow[:, scan_left:scan_right]
        yellow_row_pixels = (yellow_central > 0).sum(axis=1)
        yellow_rows = np.flatnonzero(yellow_row_pixels >= int(scan_w * 0.30))

        if yellow_rows.size >= 2:
            yellow_bands = np.split(
                yellow_rows, np.flatnonzero(np.diff(yellow_rows) > 3) + 1
            )
            yellow_bands = [g for g in yellow_bands if g.size > 0]

            if len(yellow_bands) >= 3:
                band_heights = [int(g[-1] - g[0] + 1) for g in yellow_bands]
                gap_heights = [
                    int(yellow_bands[i + 1][0] - yellow_bands[i][-1] - 1)
                    for i in range(len(yellow_bands) - 1)
                ]
                med_band = float(np.median(band_heights)) if band_heights else 0.0
                med_gap = float(np.median(gap_heights)) if gap_heights else 0.0

                # Case 1: Thin yellow divider bars separating taller voucher bodies
                if med_gap >= 25 and med_band < 0.5 * med_gap:
                    candidates = self._extract_rows_from_gaps(
                        gray, yellow_bands, scan_left, scan_right
                    )
                    if len(candidates) >= 2:
                        logger.info(
                            "Yellow-separator row detection: detected %d voucher(s).",
                            len(candidates),
                        )
                        return candidates

        # Case 2: Solid yellow/white voucher cards separated by dark gaps
        mask_white = cv2.inRange(
            hsv,
            np.array((0, 0, 150), dtype=np.uint8),
            np.array((180, 50, 255), dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask_yellow, mask_white)
        mask[:, :scan_left] = 0
        mask[:, scan_right:] = 0

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        row_pixels = (mask > 0).sum(axis=1)
        minimum_pixels = max(10, int(scan_w * 0.20))
        row_indexes = np.flatnonzero(row_pixels >= minimum_pixels)
        if row_indexes.size < 2:
            return []

        row_groups = np.split(
            row_indexes, np.flatnonzero(np.diff(row_indexes) > 2) + 1
        )
        row_groups = [group for group in row_groups if group.size >= 20]
        if len(row_groups) < 2:
            return []

        candidates = []
        for group in row_groups:
            y1 = int(group[0])
            y2 = int(group[-1]) + 1
            if y2 - y1 < self.MIN_CROP_HEIGHT:
                continue
            points = np.flatnonzero(mask[y1:y2].any(axis=0))
            if points.size == 0:
                continue
            x1 = max(0, int(points[0]) - 4)
            x2 = min(w_img, int(points[-1]) + 5)
            cw = x2 - x1
            ch = y2 - y1
            aspect = cw / float(ch) if ch > 0 else 0.0
            if not (self.MIN_ASPECT_RATIO <= aspect <= self.MAX_ASPECT_RATIO):
                continue
            candidates.append(
                {
                    "contour": np.array(
                        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2),
                    "bbox": (x1, y1, cw, ch),
                    "_sort_key": (y1, x1),
                }
            )

        logger.info("Yellow/white-row fallback: detected %d row(s).", len(candidates))
        return candidates

    def _extract_rows_from_gaps(
        self,
        gray: np.ndarray,
        yellow_bands: List[np.ndarray],
        default_left: int,
        default_right: int,
    ) -> List[Dict]:
        """Build voucher bounding boxes from the gaps between yellow separator bars."""
        h_img, w_img = gray.shape[:2]
        candidates: List[Dict] = []

        for idx in range(len(yellow_bands) - 1):
            y1 = int(yellow_bands[idx][-1]) + 1
            y2 = int(yellow_bands[idx + 1][0])
            gap_h = y2 - y1
            if gap_h < self.MIN_CROP_HEIGHT:
                continue

            band_gray = gray[y1:y2, :]
            # Detect bright card columns across the full image width so voucher edges aren't clipped
            bright_threshold = max(60, int(np.percentile(band_gray, 75) * 0.45))
            col_bright = (band_gray > bright_threshold).mean(axis=0) > 0.35
            xs = np.flatnonzero(col_bright)
            if xs.size >= self.MIN_CROP_WIDTH:
                x1 = max(0, int(xs[0]) - 2)
                x2 = min(w_img, int(xs[-1]) + 3)
            else:
                x1, x2 = default_left, default_right

            cw = x2 - x1
            aspect = cw / float(gap_h) if gap_h > 0 else 0.0
            if cw < self.MIN_CROP_WIDTH or not (self.MIN_ASPECT_RATIO <= aspect <= self.MAX_ASPECT_RATIO):
                continue

            # Include tiny vertical margin into surrounding border without overlapping adjacent voucher text
            crop_y1 = max(0, y1 - 2)
            crop_y2 = min(h_img, y2 + 2)
            ch = crop_y2 - crop_y1

            candidates.append(
                {
                    "contour": np.array(
                        [[x1, crop_y1], [x2, crop_y1], [x2, crop_y2], [x1, crop_y2]],
                        dtype=np.int32,
                    ).reshape(-1, 1, 2),
                    "bbox": (x1, crop_y1, cw, ch),
                    "_sort_key": (crop_y1, x1),
                }
            )

        return candidates

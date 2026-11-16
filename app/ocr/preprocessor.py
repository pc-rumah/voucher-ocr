"""Image preprocessing utilities for voucher OCR."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class ImagePreprocessor:
    """Handles image loading and preprocessing for voucher detection and OCR."""

    def load_image(self, path: str) -> np.ndarray:
        """Load an image from disk using OpenCV.

        Args:
            path: Absolute or relative path to the image file.

        Returns:
            BGR image as a numpy array.

        Raises:
            FileNotFoundError: If the image cannot be loaded.
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {path}")
        return img

    @staticmethod
    def to_grayscale(img: np.ndarray) -> np.ndarray:
        """Return a 2D grayscale copy of *img*."""
        if img is None or img.size == 0:
            raise ValueError("Input image is empty.")
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.copy()

    @staticmethod
    def upscale_image(
        img: np.ndarray,
        min_height: int = 160,
        scale: Optional[float] = None,
        max_width: int = 2000,
    ) -> np.ndarray:
        """Upscale *img* using bicubic interpolation without shrinking larger images.

        Args:
            img: Input grayscale or BGR image.
            min_height: Minimum target height in pixels when *scale* is None.
            scale: Explicit scale factor (> 1.0). Overrides *min_height* if set.
            max_width: Upper bound on output width to avoid excessive memory usage.

        Returns:
            Upscaled image (or original copy if already large enough).
        """
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return img

        target_scale = 1.0
        if scale is not None and scale > 1.0:
            target_scale = float(scale)
        elif h < min_height:
            target_scale = min_height / float(h)

        if target_scale <= 1.0:
            return img

        if w * target_scale > max_width:
            target_scale = max(1.0, max_width / float(w))

        if target_scale <= 1.0:
            return img

        new_w = max(1, int(round(w * target_scale)))
        new_h = max(1, int(round(h * target_scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def preprocess_for_detection(self, img: np.ndarray) -> np.ndarray:
        """Prepare image for voucher contour detection.

        Pipeline:
            1. Convert to grayscale
            2. Fast Non-Local Means denoising
            3. Adaptive thresholding (Gaussian)
            4. Morphological closing to fill gaps between edges

        Args:
            img: BGR or grayscale input image.

        Returns:
            Binary (thresholded) image ready for contour detection.
        """
        gray = self.to_grayscale(img)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(
            gray, h=10, templateWindowSize=7, searchWindowSize=21
        )

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=4,
        )

        # Morphological closing to close small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

        return closed

    def preprocess_for_ocr(self, img: np.ndarray, min_height: int = 160) -> np.ndarray:
        """Enhance an image crop for OCR without destroying thin digit strokes.

        Pipeline:
            1. Convert to grayscale
            2. Bicubic upscale when height < *min_height*
            3. Mild CLAHE contrast enhancement
            4. Gentle unsharp masking

        Args:
            img: BGR or grayscale voucher crop.
            min_height: Minimum target height in pixels after upscaling.

        Returns:
            Sharpened, contrast-enhanced grayscale image.
        """
        return self.variant_upscale_sharpen(img, min_height=min_height)

    def variant_upscale_gray(self, img: np.ndarray, min_height: int = 160) -> np.ndarray:
        """Clean grayscale + bicubic upscale (preserves original stroke geometry)."""
        gray = self.to_grayscale(img)
        return self.upscale_image(gray, min_height=min_height)

    def variant_clahe(self, img: np.ndarray, min_height: int = 160) -> np.ndarray:
        """Upscaled grayscale + mild CLAHE contrast enhancement."""
        gray = self.variant_upscale_gray(img, min_height=min_height)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def variant_upscale_sharpen(
        self, img: np.ndarray, min_height: int = 180
    ) -> np.ndarray:
        """Upscaled grayscale + mild CLAHE + gentle unsharp masking."""
        gray = self.variant_upscale_gray(img, min_height=min_height)
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.2)
        return cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

    def variant_adaptive_threshold(
        self, img: np.ndarray, min_height: int = 160
    ) -> np.ndarray:
        """Upscaled grayscale + adaptive Gaussian thresholding (dark digits on light bg)."""
        gray = self.variant_upscale_gray(img, min_height=min_height)
        # Mild blur suppresses JPEG blocking noise without thinning digits
        smoothed = cv2.GaussianBlur(gray, (3, 3), 0)
        # Keep blockSize proportional to upscaled height (always odd >= 21)
        block_size = max(21, int(round(gray.shape[0] * 0.22)) | 1)
        return cv2.adaptiveThreshold(
            smoothed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=block_size,
            C=8,
        )

    def variant_otsu_threshold(
        self, img: np.ndarray, min_height: int = 160
    ) -> np.ndarray:
        """Upscaled grayscale + mild CLAHE + Otsu global thresholding."""
        enhanced = self.variant_clahe(img, min_height=min_height)
        smoothed = cv2.GaussianBlur(enhanced, (3, 3), 0)
        _, otsu = cv2.threshold(smoothed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return otsu

    def generate_ocr_variants(
        self, img: np.ndarray, min_height: int = 160
    ) -> List[Tuple[str, np.ndarray]]:
        """Generate ordered image preprocessing variants for multi-pass OCR.

        Args:
            img: Source BGR or grayscale crop.
            min_height: Minimum target height for upscaling.

        Returns:
            List of ``(variant_name, processed_image)`` tuples.
        """
        return [
            ("upscale_gray", self.variant_upscale_gray(img, min_height=min_height)),
            ("clahe", self.variant_clahe(img, min_height=min_height)),
            (
                "upscale_sharpen",
                self.variant_upscale_sharpen(img, min_height=max(min_height, 180)),
            ),
            (
                "adaptive_thresh",
                self.variant_adaptive_threshold(img, min_height=min_height),
            ),
            ("otsu_thresh", self.variant_otsu_threshold(img, min_height=min_height)),
        ]

    def correct_perspective(
        self,
        img: np.ndarray,
        contour: np.ndarray,
        padding: int = 0,
    ) -> np.ndarray:
        """Apply a 4-point perspective transform to deskew a voucher.

        Uses the approximated 4-point convex polygon when valid; otherwise
        falls back to the oriented minimum-area rectangle (which still deskews
        tilted vouchers with rounded corners) rather than an axis-aligned box.

        Args:
            img: Source BGR or grayscale image.
            contour: OpenCV contour array for the voucher region.
            padding: Optional pixel margin to expand around the rectified quad.

        Returns:
            Warped (rectified) image crop.
        """
        if contour is None or len(contour) < 3:
            return img

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        pts: Optional[np.ndarray] = None
        if len(approx) == 4 and cv2.isContourConvex(approx):
            cand_area = abs(cv2.contourArea(approx))
            cnt_area = max(1.0, abs(cv2.contourArea(contour)))
            if cand_area >= 0.5 * cnt_area:
                pts = approx.reshape(4, 2).astype(np.float32)

        if pts is None:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            pts = box.astype(np.float32)

        pts = self._order_points(pts)
        if padding > 0:
            pts = self._expand_quad(pts, padding=padding, img_shape=img.shape)

        (tl, tr, br, bl) = pts

        width_a = float(np.linalg.norm(br - bl))
        width_b = float(np.linalg.norm(tr - tl))
        max_width = max(int(round(width_a)), int(round(width_b)))

        height_a = float(np.linalg.norm(tr - br))
        height_b = float(np.linalg.norm(tl - bl))
        max_height = max(int(round(height_a)), int(round(height_b)))

        if max_width < 2 or max_height < 2:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 0 and h > 0:
                return img[y : y + h, x : x + w]
            return img

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(
            img,
            matrix,
            (max_width, max_height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _expand_quad(
        pts: np.ndarray, padding: int, img_shape: Tuple[int, ...]
    ) -> np.ndarray:
        """Expand ordered quad corners outward from their centroid, clamped to image."""
        center = pts.mean(axis=0)
        expanded = np.zeros_like(pts)
        h_img, w_img = img_shape[:2]
        for idx, pt in enumerate(pts):
            vec = pt - center
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                new_pt = pt + (vec / norm) * padding
            else:
                new_pt = pt
            expanded[idx, 0] = np.clip(new_pt[0], 0, max(0, w_img - 1))
            expanded[idx, 1] = np.clip(new_pt[1], 0, max(0, h_img - 1))
        return expanded

    @staticmethod
    def _order_points(pts: np.ndarray) -> np.ndarray:
        """Return (top-left, top-right, bottom-right, bottom-left)."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
        rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right has smallest diff
        rect[3] = pts[np.argmax(diff)]  # bottom-left has largest diff
        return rect

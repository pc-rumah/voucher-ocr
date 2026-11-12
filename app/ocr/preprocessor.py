"""Image preprocessing utilities for voucher OCR."""

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
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

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

    def preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Enhance an image crop for better OCR accuracy.

        Pipeline:
            1. Convert to grayscale
            2. CLAHE contrast enhancement
            3. Unsharp masking (sharpening)

        Args:
            img: BGR or grayscale voucher crop.

        Returns:
            Sharpened, contrast-enhanced grayscale image.
        """
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Unsharp masking (sharpen)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)

        return sharpened

    def correct_perspective(self, img: np.ndarray, contour: np.ndarray) -> np.ndarray:
        """Apply a 4-point perspective transform to deskew a voucher.

        If the contour does not approximate to exactly 4 points the bounding
        rectangle corners are used instead.

        Args:
            img: Source BGR or grayscale image.
            contour: OpenCV contour array for the voucher region.

        Returns:
            Warped (rectified) image crop.
        """
        # Approximate contour to polygon
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            # Fall back to bounding rect corners
            x, y, w, h = cv2.boundingRect(contour)
            pts = np.array(
                [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
            )

        # Order points: top-left, top-right, bottom-right, bottom-left
        pts = self._order_points(pts)

        (tl, tr, br, bl) = pts

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        max_width = max(int(width_a), int(width_b))

        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)
        max_height = max(int(height_a), int(height_b))

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(img, M, (max_width, max_height))
        return warped

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

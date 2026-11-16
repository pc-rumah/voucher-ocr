"""Unit tests for OCR preprocessing, crop validation, recognizer, and multi-pass pipeline."""

import os
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from app.ocr.barcode import read_barcode
from app.ocr.detector import VoucherDetector
from app.ocr.pipeline import OCRPipeline, _split_crop
from app.ocr.preprocessor import ImagePreprocessor
from app.ocr.recognizer import VoucherRecognizer


_C39_PATTERNS = {
    "0": "nnnwwnwnn", "1": "wnnwnnnnw", "2": "nnwwnnnnw", "3": "wnwwnnnnn",
    "4": "nnnwwnnnw", "5": "wnnwwnnnn", "6": "nnwwwnnnn", "7": "nnnwnnwnw",
    "8": "wnnwnnwnn", "9": "nnwwnnwnn", "*": "nwnnwnwnn",
}


def _render_code39(text: str, module: int = 2, height: int = 44) -> np.ndarray:
    """Render a Code 39 barcode image (dark bars on white background)."""
    bits = []
    for ch in f"*{text}*":
        for idx, width_kind in enumerate(_C39_PATTERNS[ch]):
            bar = 1 if idx % 2 == 0 else 0
            width = 3 if width_kind == "w" else 1
            bits.extend([bar] * (width * module))
        bits.extend([0] * module)
    quiet = [0] * (10 * module)
    row = np.array(quiet + bits + quiet, dtype=np.uint8)
    line = np.where(row == 1, 0, 255).astype(np.uint8)
    gray = np.tile(line, (height, 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _make_synthetic_voucher(
    voucher_code: str = "05902395508058",
    serial_number: str = "2001506833832",
    with_barcode: bool = True,
    tilt_deg: float = 0.0,
) -> np.ndarray:
    """Create a realistic synthetic voucher card (white card on dark background)."""
    card = np.full((140, 760, 3), 245, dtype=np.uint8)
    cv2.rectangle(card, (2, 2), (757, 137), (0, 210, 240), 4)
    cv2.putText(card, "PT Indosat Tbk", (55, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)
    cv2.rectangle(card, (35, 45), (390, 95), (100, 100, 100), 2)
    formatted = f"{voucher_code[:3]} {voucher_code[3:6]} {voucher_code[6:9]} {voucher_code[9:12]} {voucher_code[12:]}"
    cv2.putText(card, formatted, (48, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (70, 70, 70), 2)

    if with_barcode:
        bc = _render_code39(serial_number, module=1, height=40)
        bw = min(310, bc.shape[1])
        bc_resized = cv2.resize(bc, (bw, 38), interpolation=cv2.INTER_NEAREST)
        card[26:64, 420 : 420 + bw] = bc_resized

    cv2.putText(card, serial_number, (450, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (30, 30, 30), 2)

    if abs(tilt_deg) > 0.1:
        canvas = np.zeros((280, 880, 3), dtype=np.uint8)
        canvas[70:210, 60:820] = card
        rot = cv2.getRotationMatrix2D((440, 140), tilt_deg, 1.0)
        return cv2.warpAffine(canvas, rot, (880, 280), borderValue=(20, 20, 20))
    return card


class TestImagePreprocessor(unittest.TestCase):
    def setUp(self) -> None:
        self.prep = ImagePreprocessor()
        self.voucher = _make_synthetic_voucher()

    def test_generate_ocr_variants_includes_all_required_variants(self) -> None:
        variants = self.prep.generate_ocr_variants(self.voucher, min_height=160)
        names = [name for name, _ in variants]
        self.assertEqual(
            names,
            ["upscale_gray", "clahe", "upscale_sharpen", "adaptive_thresh", "otsu_thresh"],
        )
        for name, img in variants:
            self.assertEqual(img.ndim, 2, msg=f"{name} should be grayscale")
            self.assertGreaterEqual(img.shape[0], 160, msg=f"{name} not upscaled")
            # Digits should remain dark ink on bright background (mean > 150)
            self.assertGreater(float(np.mean(img)), 150.0, msg=f"{name} inverted")
            self.assertGreater(float(np.std(img)), 10.0, msg=f"{name} lost digit strokes")

    def test_correct_perspective_handles_tilted_and_degenerate_contours(self) -> None:
        tilted = _make_synthetic_voucher(tilt_deg=8.0)
        gray = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertTrue(contours)
        biggest = max(contours, key=cv2.contourArea)
        warped = self.prep.correct_perspective(tilted, biggest, padding=4)
        self.assertGreater(warped.shape[1], warped.shape[0])
        self.assertGreater(warped.shape[0], 80)

        # Degenerate tiny contour should not raise
        tiny = np.array([[[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int32)
        out = self.prep.correct_perspective(tilted, tiny)
        self.assertIsNotNone(out)


class TestVoucherDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.det = VoucherDetector()

    def test_validate_crop_and_padding_clamping(self) -> None:
        img = _make_synthetic_voucher()
        crop = self.det.crop_voucher(img, (0, 0, 200, 100), padding=25)
        self.assertTrue(self.det.validate_crop(crop))
        self.assertEqual(crop.shape[:2], (125, 225))

        self.assertFalse(self.det.validate_crop(None))
        self.assertFalse(self.det.validate_crop(np.zeros((10, 10, 3), dtype=np.uint8)))
        self.assertFalse(self.det.validate_crop(np.full((60, 100, 3), 128, dtype=np.uint8)))

        with self.assertRaises(ValueError):
            self.det.crop_voucher(img, (10, 10, 0, 50))

    def test_select_best_crop_deskews_tilted_voucher(self) -> None:
        tilted = _make_synthetic_voucher(tilt_deg=7.0)
        gray = cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = max(contours, key=cv2.contourArea)
        bbox = cv2.boundingRect(biggest)
        meta = {"contour": biggest, "bbox": bbox, "index": 0}

        best = self.det.select_best_crop(tilted, meta, padding=8)
        self.assertTrue(self.det.validate_crop(best))
        # Deskewed crop should have smaller height than raw tilted bounding box
        self.assertLess(best.shape[0], bbox[3])

    def test_stacked_yellow_separator_detection_does_not_split_vouchers(self) -> None:
        # Build a sheet with 4 stacked white vouchers separated by thin yellow bars
        h_card, w_card = 70, 400
        sheet = np.zeros((380, 560, 3), dtype=np.uint8)
        y = 10
        for _ in range(4):
            # Yellow separator bar
            sheet[y : y + 12, 80:480] = (0, 220, 240)
            y += 12
            # White voucher body with dark text blocks
            sheet[y : y + h_card, 80:480] = (240, 240, 240)
            cv2.putText(sheet, "059 023 955 080 58", (95, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
            cv2.putText(sheet, "2001506833832", (280, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
            y += h_card
        sheet[y : y + 12, 80:480] = (0, 220, 240)

        vouchers = self.det._yellow_row_fallback(sheet)
        self.assertEqual(len(vouchers), 4)
        for idx, v in enumerate(vouchers):
            _, vy, _, vh = v["bbox"]
            self.assertGreaterEqual(vh, 60)
            self.assertLessEqual(vh, 80)
            if idx > 0:
                self.assertGreater(vy, vouchers[idx - 1]["bbox"][1])


class TestBarcodeReader(unittest.TestCase):
    def test_read_barcode_normal_and_dark(self) -> None:
        voucher = _make_synthetic_voucher(serial_number="2001506833832", with_barcode=True)
        res = read_barcode(voucher)
        self.assertIsNotNone(res)
        value, y_ratio = res
        self.assertEqual(value, "2001506833832")
        self.assertGreater(y_ratio, 0.0)
        self.assertLess(y_ratio, 1.0)

        # Dark image
        dark = np.clip(voucher.astype(np.float32) * 0.35, 0, 255).astype(np.uint8)
        res_dark = read_barcode(dark)
        self.assertIsNotNone(res_dark)
        self.assertEqual(res_dark[0], "2001506833832")


class TestVoucherRecognizer(unittest.TestCase):
    def setUp(self) -> None:
        self.rec = VoucherRecognizer()

    def test_rejects_alphabetic_false_positives_and_separates_confidence(self) -> None:
        blocks = [
            {"text": "PT Indosat Tbk", "confidence": 0.98, "bbox": [[10, 5], [150, 5], [150, 20], [10, 20]]},
            {"text": "COKEMKAKS", "confidence": 0.95, "bbox": [[20, 30], [140, 30], [140, 50], [20, 50]]},
            {"text": "Masa Berlaku sampai 31 Desember 2027", "confidence": 0.96, "bbox": [[10, 60], [250, 60], [250, 75], [10, 75]]},
            {"text": "0000000000", "confidence": 0.99, "bbox": [[20, 80], [120, 80], [120, 95], [20, 95]]},
        ]
        # With valid barcode, no valid voucher_code exists -> voucher_code must be None, conf=0.0
        data = self.rec.extract_from_blocks(
            blocks, img_shape=(120, 400), barcode_value="2001506833832", barcode_y_ratio=0.45
        )
        self.assertEqual(data["serial_number"], "2001506833832")
        self.assertEqual(data["serial_number_confidence"], 1.0)
        self.assertIsNone(data["voucher_code"])
        self.assertEqual(data["voucher_code_confidence"], 0.0)
        self.assertEqual(data["confidence"], 0.5)

    def test_center_left_voucher_code_preferred_over_short_fragment(self) -> None:
        blocks = [
            {"text": "672104234", "confidence": 0.95, "bbox": [[60, 40], [150, 40], [150, 60], [60, 60]]},
            {"text": "059.023.955.080.58", "confidence": 0.91, "bbox": [[120, 40], [320, 40], [320, 60], [120, 60]]},
        ]
        data = self.rec.extract_from_blocks(
            blocks, img_shape=(120, 400), barcode_value="2001506833832", barcode_y_ratio=0.42
        )
        self.assertEqual(data["voucher_code"], "05902395508058")
        self.assertAlmostEqual(data["voucher_code_confidence"], 0.91, places=3)


class TestOCRPipelineLogic(unittest.TestCase):
    def test_split_crop_defaults_to_60_percent(self) -> None:
        img = np.zeros((100, 500, 3), dtype=np.uint8)
        left, right = _split_crop(img)
        self.assertEqual(left.shape[1], 300)
        self.assertEqual(right.shape[1], 200)

    def test_left_60_fallback_to_full_crop_and_strict_per_field_threshold(self) -> None:
        voucher = _make_synthetic_voucher(
            voucher_code="05902395508058", serial_number="2001506833832", with_barcode=True
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "voucher.png")
            crops_dir = os.path.join(tmpdir, "crops")
            cv2.imwrite(img_path, voucher)

            pipeline = OCRPipeline(crops_dir=crops_dir, confidence_threshold=0.70)

            # Simulate left-60% returning empty, full crop returning a low-confidence code (0.62)
            def fake_extract_blocks(variant_img):
                w = variant_img.shape[1]
                # Left 60% subcrop (~521px upscaled) is narrower than full crop (~869px upscaled)
                if w < 700:
                    return []
                return [
                    {
                        "text": "059.023.955.080.58",
                        "confidence": 0.62,
                        "bbox": [[150, 70], [480, 70], [480, 100], [150, 100]],
                    }
                ]

            with patch.object(pipeline._detector, "detect_vouchers", return_value=[{"contour": None, "bbox": (0, 0, voucher.shape[1], voucher.shape[0]), "index": 0}]), \
                 patch("app.ocr.pipeline.read_barcode", return_value=("2001506833832", 0.35)), \
                 patch.object(pipeline._recognizer, "extract_text_blocks", side_effect=fake_extract_blocks):
                res = pipeline.process_batch("TEST-BATCH", img_path)

            self.assertEqual(res["total_detected"], 1)
            v = res["vouchers"][0]
            self.assertEqual(v["serial_number"], "2001506833832")
            self.assertEqual(v["serial_number_confidence"], 1.0)
            self.assertEqual(v["voucher_code"], "05902395508058")
            self.assertAlmostEqual(v["voucher_code_confidence"], 0.62, places=3)
            # Average confidence is (1.0 + 0.62)/2 = 0.81 >= 0.70, BUT voucher_code_confidence < 0.70!
            # Status MUST be 'review', NOT 'verified'.
            self.assertEqual(v["status"], "review")
            self.assertIn("Confidence rendah", v["error"])

    def test_multipass_prefers_cleaner_higher_confidence_variant(self) -> None:
        voucher = _make_synthetic_voucher(
            voucher_code="08849381999521", serial_number="2001506833842", with_barcode=False
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = os.path.join(tmpdir, "voucher.png")
            crops_dir = os.path.join(tmpdir, "crops")
            cv2.imwrite(img_path, voucher)

            pipeline = OCRPipeline(crops_dir=crops_dir, confidence_threshold=0.70)
            call_count = {"n": 0}

            def fake_extract_blocks(variant_img):
                call_count["n"] += 1
                h, w = variant_img.shape[:2]
                if call_count["n"] == 1:
                    # Pass 1 (upscale_gray): truncated voucher code, noisy
                    return [
                        {"text": "849381999", "confidence": 0.75, "bbox": [[w * 0.15, h * 0.5], [w * 0.45, h * 0.5], [w * 0.45, h * 0.65], [w * 0.15, h * 0.65]]},
                        {"text": "2001506833842", "confidence": 0.94, "bbox": [[w * 0.62, h * 0.5], [w * 0.90, h * 0.5], [w * 0.90, h * 0.65], [w * 0.62, h * 0.65]]},
                    ]
                # Pass 2+ (clahe / sharpen): full 14-digit code with high confidence
                return [
                    {"text": "088 493 819 995 21", "confidence": 0.96, "bbox": [[w * 0.12, h * 0.5], [w * 0.52, h * 0.5], [w * 0.52, h * 0.65], [w * 0.12, h * 0.65]]},
                    {"text": "2001506833842", "confidence": 0.97, "bbox": [[w * 0.62, h * 0.5], [w * 0.90, h * 0.5], [w * 0.90, h * 0.65], [w * 0.62, h * 0.65]]},
                ]

            with patch.object(pipeline._detector, "detect_vouchers", return_value=[{"contour": None, "bbox": (0, 0, voucher.shape[1], voucher.shape[0]), "index": 0}]), \
                 patch("app.ocr.pipeline.read_barcode", return_value=None), \
                 patch.object(pipeline._recognizer, "extract_text_blocks", side_effect=fake_extract_blocks):
                res = pipeline.process_batch("MULTI", img_path)

            v = res["vouchers"][0]
            self.assertEqual(v["status"], "verified")
            self.assertEqual(v["serial_number"], "2001506833842")
            self.assertEqual(v["voucher_code"], "08849381999521")
            self.assertGreaterEqual(v["serial_number_confidence"], 0.94)
            self.assertGreaterEqual(v["voucher_code_confidence"], 0.95)


if __name__ == "__main__":
    unittest.main()

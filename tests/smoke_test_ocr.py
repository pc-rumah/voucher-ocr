"""End-to-end smoke tests for the voucher OCR pipeline across degraded conditions."""

import json
import os
import tempfile
import unittest

import cv2
import numpy as np

from app.ocr.pipeline import OCRPipeline
from tests.test_ocr_components import _make_synthetic_voucher


def _create_degraded_variants(base_img: np.ndarray) -> dict[str, np.ndarray]:
    """Generate normal, tilted, dark, blurry, and low-resolution test images."""
    h, w = base_img.shape[:2]

    # 1. Normal (framed inside dark border so contour detection can exercise crop)
    normal = np.zeros((h + 80, w + 80, 3), dtype=np.uint8)
    normal[40 : 40 + h, 40 : 40 + w] = base_img

    # 2. Tilted (miring +6 deg)
    rot_m = cv2.getRotationMatrix2D((normal.shape[1] / 2.0, normal.shape[0] / 2.0), 6.0, 1.0)
    miring = cv2.warpAffine(normal, rot_m, (normal.shape[1], normal.shape[0]), borderValue=(15, 15, 15))

    # 3. Dark (gelap, ~40% brightness)
    gelap = np.clip(normal.astype(np.float32) * 0.42, 0, 255).astype(np.uint8)

    # 4. Blurry (blur)
    blur = cv2.GaussianBlur(normal, (5, 5), sigmaX=1.2)

    # 5. Low-resolution (beresolusi rendah, 0.55x)
    lowres = cv2.resize(
        normal,
        (max(80, int(normal.shape[1] * 0.55)), max(40, int(normal.shape[0] * 0.55))),
        interpolation=cv2.INTER_AREA,
    )

    return {
        "normal": normal,
        "miring": miring,
        "gelap": gelap,
        "blur": blur,
        "lowres": lowres,
    }


class SmokeTestOCRPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.pipeline = OCRPipeline(
            crops_dir=os.path.join(cls.tmpdir.name, "crops"),
            confidence_threshold=0.70,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmpdir.cleanup()

    def test_smoke_conditions_with_barcode_and_ocr_fallback(self) -> None:
        expected_vc = "05902395508058"
        expected_sn = "2001506833832"

        # Case A: Voucher with real barcode + printed voucher code
        base_with_bc = _make_synthetic_voucher(
            voucher_code=expected_vc,
            serial_number=expected_sn,
            with_barcode=True,
        )
        variants = _create_degraded_variants(base_with_bc)

        report = []
        for label, img in variants.items():
            path = os.path.join(self.tmpdir.name, f"bc_{label}.png")
            cv2.imwrite(path, img)
            res = self.pipeline.process_batch(f"SMOKE-BC-{label.upper()}", path)
            self.assertGreaterEqual(res["total_detected"], 1, msg=f"{label}: no voucher detected")
            v = res["vouchers"][0]
            report.append(
                {
                    "case": f"barcode_{label}",
                    "status": v["status"],
                    "serial_number": v["serial_number"],
                    "voucher_code": v["voucher_code"],
                    "sn_conf": v["serial_number_confidence"],
                    "vc_conf": v["voucher_code_confidence"],
                    "confidence": v["confidence"],
                }
            )
            self.assertEqual(v["serial_number"], expected_sn, msg=f"{label}: serial mismatch {v}")
            self.assertEqual(v["voucher_code"], expected_vc, msg=f"{label}: voucher_code mismatch {v}")
            self.assertEqual(v["status"], "verified", msg=f"{label}: expected verified, got {v}")

        # Case B: Voucher without readable barcode -> OCR fallback must still read both fields
        base_no_bc = _make_synthetic_voucher(
            voucher_code=expected_vc,
            serial_number=expected_sn,
            with_barcode=False,
        )
        no_bc_path = os.path.join(self.tmpdir.name, "ocr_fallback_normal.png")
        normal_no_bc = _create_degraded_variants(base_no_bc)["normal"]
        cv2.imwrite(no_bc_path, normal_no_bc)
        res_fb = self.pipeline.process_batch("SMOKE-OCR-FALLBACK", no_bc_path)
        self.assertGreaterEqual(res_fb["total_detected"], 1)
        v_fb = res_fb["vouchers"][0]
        report.append(
            {
                "case": "ocr_fallback_normal",
                "status": v_fb["status"],
                "serial_number": v_fb["serial_number"],
                "voucher_code": v_fb["voucher_code"],
                "sn_conf": v_fb["serial_number_confidence"],
                "vc_conf": v_fb["voucher_code_confidence"],
                "confidence": v_fb["confidence"],
            }
        )
        self.assertEqual(v_fb["serial_number"], expected_sn)
        self.assertEqual(v_fb["voucher_code"], expected_vc)
        self.assertEqual(v_fb["status"], "verified")

        print("SMOKE_SYNTHETIC_REPORT=" + json.dumps(report))

    def test_smoke_real_sample_sheet_if_present(self) -> None:
        sample_path = "/data/originals/BATCH-20261115-29204D/original.jpg"
        if not os.path.exists(sample_path):
            self.skipTest(f"Sample image not present at {sample_path}")

        res = self.pipeline.process_batch("SMOKE-REAL-SHEET", sample_path)
        self.assertEqual(res["total_detected"], 15)
        self.assertGreaterEqual(res["total_success"], 14)
        self.assertEqual(res["total_failed"], 0)
        for v in res["vouchers"]:
            self.assertIsNotNone(v["voucher_code"], msg=f"Missing voucher_code on idx={v['index']}")
            self.assertEqual(len(v["voucher_code"]), 14, msg=f"Truncated voucher_code on idx={v['index']}: {v['voucher_code']}")


if __name__ == "__main__":
    unittest.main()

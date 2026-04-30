import json
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ValidationCliTests(unittest.TestCase):
    def test_validation_cli_writes_report_and_figures(self):
        script = Path("scripts/run_spherical_atlas_validation.py").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--out",
                    tmp,
                    "--quick",
                    "--n-s",
                    "8",
                    "--n-theta",
                    "6",
                    "--null-trials",
                    "4",
                    "--material-anchor-pole",
                    "--sh-orientation",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            ledger_path = Path(tmp) / "validation_claim_ledger.json"
            self.assertTrue(ledger_path.exists())
            self.assertTrue((Path(tmp) / "validation_claim_ledger.md").exists())
            self.assertTrue((Path(tmp) / "uniform_2x_transition_panel.png").exists())
            self.assertTrue((Path(tmp) / "local_bulge_transition_panel.png").exists())
            self.assertTrue((Path(tmp) / "exact_repeat_transition_panel.png").exists())

            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            self.assertIn("claim_level", ledger["claim_classification"])
            self.assertTrue(ledger["gauge"]["material_anchor_pole"])
            self.assertTrue(ledger["gauge"]["sh_orientation"])
            self.assertIn("material_anchor_face_index", ledger["gauge"])
            self.assertIn("decoder_transfer", ledger["uniform_2x"])
            self.assertIn("sh_orientation", ledger["uniform_2x"])

    @unittest.skipIf(importlib.util.find_spec("lapy") is None, "lapy is not installed")
    def test_validation_cli_can_run_conformal_backend(self):
        script = Path("scripts/run_spherical_atlas_validation.py").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--out",
                    tmp,
                    "--quick",
                    "--n-s",
                    "8",
                    "--n-theta",
                    "6",
                    "--null-trials",
                    "2",
                    "--parameterizer",
                    "conformal",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            ledger = json.loads((Path(tmp) / "validation_claim_ledger.json").read_text(encoding="utf-8"))
            self.assertEqual(ledger["config"]["parameterization_method"], "conformal")
            self.assertEqual(ledger["exact_repeat"]["source_map"]["info"]["method"], "conformal")
            self.assertEqual(
                ledger["exact_repeat"]["source_map"]["info"]["injectivity_certification"],
                "unavailable",
            )


if __name__ == "__main__":
    unittest.main()

"""Prepared-file workflows must not need source experiments or process OB twice."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline.batch import run_batch, validate_batch
from lumacam_pipeline.config import load_config, save_config
from lumacam_pipeline.gui import LumacamPipelineGUI
from lumacam_pipeline.presets import create_session
from lumacam_pipeline.prepared_inputs import prepared_ob_integral
from lumacam_pipeline.section import SectionState
from test_lumacam_pipeline import make_markerless_proposal


class PreparedInputTests(unittest.TestCase):
    def fixture(self, root):
        proposal, _, experiment, _ = make_markerless_proposal(root, values=(8, 16))
        ob = root / "averaged_ob.tiff"
        bg = root / "median_bg.tiff"
        # Nonuniform OB makes a second smoothing/binning observable.
        pixels = np.arange(1, 37, dtype=np.float32).reshape(6, 6)
        tifffile.imwrite(ob, pixels)
        tifffile.imwrite(bg, np.full((6, 6), 0.1, dtype=np.float32))
        config = {
            "project": {"data_path": str(proposal), "experiment": experiment, "variant": "3to1000"},
            "corrections": {"reference_input_mode": "prepared", "prepared_reference_path": str(ob),
                            "reference_dir": "/unavailable/original/ob", "reference_bin_x": 2, "reference_bin_y": 2},
            "background": {"enabled": True, "source_mode": "shared", "background_path": str(bg)},
        }
        return proposal, experiment, ob, bg, pixels, config

    def test_skip_preparation_individual_and_batch_use_same_files_without_smoothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, experiment, ob, bg, pixels, config = self.fixture(Path(temporary))
            expected = 12 / pixels - 0.1
            original_bytes = ob.read_bytes(), bg.read_bytes()
            session = create_session("exsitu", config)
            for section in ("project", "corrections", "background", "images", "average"):
                result = session.run(section)
                self.assertEqual(result.state, SectionState.COMPLETED, result.message)
            result_dir = session.artifacts.require("results_dir")
            np.testing.assert_allclose(tifffile.imread(next((result_dir / "images").glob("average*.tiff"))), expected, rtol=1e-6)
            config["project"]["analysis_name"] = "batch"
            with patch("lumacam_pipeline.batch.backends.nci_data") as backend:
                # Explicit prepared files must not call the reference preparation backend.
                report = validate_batch(proposal, [experiment], config=config, sections=("corrections", "background", "average"))
                self.assertTrue(report.valid, report.common_errors)
                backend.assert_not_called()
            result = run_batch(proposal, [experiment], preset="exsitu", config=config,
                               sections=("corrections", "background", "average"))
            self.assertEqual(result.failed_count, 0, result.results)
            directory = result.results[0].results_dir
            np.testing.assert_allclose(tifffile.imread(next((directory / "images").glob("average*.tiff"))), expected, rtol=1e-6)
            self.assertEqual((ob.read_bytes(), bg.read_bytes()), original_bytes)
            self.assertFalse((proposal / "analysis" / "shared_corrections").exists())
            _, saved = load_config(result.results[0].config_path)
            self.assertEqual(saved["corrections"]["prepared_reference_path"], str(ob))
            repeated = run_batch(proposal, [experiment], preset="exsitu", config=config,
                                 sections=("corrections", "background", "average"))
            self.assertEqual(repeated.results[0].status, "skipped")

    def test_changed_ob_is_a_different_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, ob, _, _, config = self.fixture(Path(temporary))
            session = create_session("exsitu", config)
            for section in ("project", "corrections"):
                self.assertEqual(session.run(section).state, SectionState.COMPLETED)
            other = ob.with_name("other.tiff")
            tifffile.imwrite(other, np.ones((6, 6), dtype=np.float32))
            session.configure("corrections", prepared_reference_path=str(other))
            result = session.run("corrections")
            self.assertEqual(result.state, SectionState.FAILED)
            self.assertIn("Analysis name", result.message)

    def test_gui_skip_prepare_save_reload_and_separate_preparation_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, experiment, ob, bg, _, _ = self.fixture(Path(temporary))
            gui = LumacamPipelineGUI("exsitu")
            gui.data_path.value = str(proposal)
            gui.experiment.value = experiment
            gui.variant.value = "3to1000"
            gui.prepared_ob_path.value = str(ob)
            gui.background_path.value = str(bg)
            gui.background_enabled.value = True
            gui.prepare_bg.value = False
            gui.preparation_bg_count.value = 3
            gui.background_count.value = 9
            gui.ob_integral.value = 123
            gui._save_config(None)
            restored = LumacamPipelineGUI("exsitu")
            restored.config_path.value = str(proposal / "analysis" / "lumacam_pipeline.ini")
            restored._load_config(None)
            self.assertEqual(restored.prepared_ob_path.value, str(ob))
            self.assertEqual(restored.background_path.value, str(bg))
            self.assertEqual(restored.ob_integral.value, 123)
            self.assertFalse(restored.prepare_bg.value)
            self.assertEqual(restored.preparation_bg_count.value, 3)
            self.assertEqual(restored.background_count.value, 9)
            restored.prepared_ob_path.value = str(ob.with_name("different.tiff"))
            self.assertEqual(restored.ob_integral.value, 0)
            restored.prepared_ob_browse_btn.click()
            self.assertEqual(restored.path_browser.mode, "file")
            restored.background_browse_btn.click()
            self.assertEqual(restored.path_browser.mode, "file")
            self.assertFalse((proposal / "analysis" / "shared_corrections").exists())

    def test_prepare_ob_bg_autofills_and_uses_separate_median_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, ob, _ = make_markerless_proposal(root, experiment="exp001_OB", values=(2, 2))
            _, _, bg, _ = make_markerless_proposal(root, experiment="exp002_BG", values=(2, 4, 18))
            gui = LumacamPipelineGUI("exsitu")
            gui.data_path.value = str(proposal)
            gui.variant.value = "3to1000"
            gui.shared_ob_experiment.value = ob
            gui.shared_background_experiment.value = bg
            gui.preparation_bg_count.value = 3
            gui.background_count.value = 1
            gui._prepare_shared_inputs()
            self.assertIn("ready", gui.shared_status_html.value)
            self.assertTrue(Path(gui.prepared_ob_path.value).is_file())
            np.testing.assert_allclose(tifffile.imread(gui.background_path.value), 2)
            self.assertEqual(gui.background_count.value, 1)
            self.assertEqual(gui.background_source.value, "shared")
            self.assertEqual(gui.session.config["corrections"]["reference_input_mode"], "prepared")

    def test_missing_or_non_2d_prepared_ob_is_rejected_before_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, experiment, ob, _, _, config = self.fixture(Path(temporary))
            for bad in (ob.with_name("missing.tiff"), ob.with_name("stack.tiff")):
                if bad.name == "stack.tiff":
                    tifffile.imwrite(bad, np.ones((2, 6, 6), dtype=np.float32), photometric="minisblack")
                config["corrections"]["prepared_reference_path"] = str(bad)
                report = validate_batch(proposal, [experiment], config=config)
                self.assertFalse(report.valid)
            self.assertFalse((proposal / "analysis").exists())

    def test_preparation_metadata_recovers_beam_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ob = root / "ob.tiff"
            (root / "ob_preparation.json").write_text(json.dumps({"reference_beam_integral_per_minute": 120}))
            self.assertEqual(prepared_ob_integral(ob), 120)
            gui = LumacamPipelineGUI()
            gui.prepared_ob_path.value = str(ob)
            self.assertEqual(gui.ob_integral.value, 120)
            gui._sync_config()
            self.assertEqual(gui.session.config["corrections"]["reference_beam_integral_per_minute"], 120)


if __name__ == "__main__":
    unittest.main()

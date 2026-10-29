"""White-spot filtering, provenance, GUI, and batch regression tests."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline import backends, create_session
from lumacam_pipeline.batch import run_batch, validate_batch
from lumacam_pipeline.config import load_config, save_config
from lumacam_pipeline.gui import LumacamPipelineGUI
from lumacam_pipeline.presets import PRESETS
from lumacam_pipeline.provenance import recipe_for
from lumacam_pipeline.section import SectionState
from lumacam_pipeline.sections.images import preview_white_spots
from lumacam_pipeline.white_spots import DEFAULTS, filter_settings, remove_white_spots
from test_lumacam_pipeline import make_markerless_proposal


class WhiteSpotAlgorithmTests(unittest.TestCase):
    def test_only_bright_outliers_are_replaced_without_mutating_input(self):
        source = np.full((15, 15), 100, dtype=np.uint16)
        source[7, 7], source[3, 3], source[0, 0] = 60000, 0, 60000
        before = source.copy()
        result, mask, stats = remove_white_spots(source, method="abs", threshold=500)
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(stats["replaced_pixels"], 2)
        self.assertEqual(result[3, 3], 0)
        self.assertEqual(result[7, 7], 100)
        self.assertEqual(result[0, 0], 100)
        np.testing.assert_array_equal(result[~mask], source[~mask])
        np.testing.assert_array_equal(source, before)

    def test_broad_edges_and_uniform_borders_are_not_smoothed(self):
        source = np.full((31, 31), 10, dtype=np.float32)
        source[:, 15:] = 100
        for method in ("mad", "abs", "prop"):
            result, mask, _ = remove_white_spots(source, method=method, threshold=8)
            np.testing.assert_array_equal(result, source)
            self.assertFalse(mask.any())

    def test_automatic_threshold_removes_large_spike_not_gaussian_noise(self):
        source = np.random.default_rng(20).normal(10, 0.3, size=(101, 101)).astype(np.float32)
        source[50, 50] = 1000
        result, mask, stats = remove_white_spots(source)
        self.assertTrue(mask[50, 50])
        self.assertLess(stats["replaced_pixels"], 5)
        self.assertGreater(stats["noise_sigma"], 0)
        self.assertLess(result[50, 50], 11)

    def test_nonfinite_and_negative_pixels_are_preserved(self):
        source = np.full((15, 15), -2, dtype=np.float32)
        source[7, 7], source[7, 6], source[2, 2], source[3, 3] = 100, np.nan, np.inf, -np.inf
        for method in ("mad", "abs", "prop"):
            result, mask, _ = remove_white_spots(source, method=method, threshold=8)
            self.assertEqual(result[7, 7], -2)
            np.testing.assert_array_equal(np.isnan(result), np.isnan(source))
            np.testing.assert_array_equal(np.isinf(result), np.isinf(source))
            self.assertFalse(mask[~np.isfinite(source)].any())
        all_invalid = np.full((4, 4), np.nan, dtype=np.float32)
        _, mask, stats = remove_white_spots(all_invalid)
        self.assertFalse(mask.any())
        self.assertIsNone(stats["threshold_min"])

    def test_invalid_parameters_are_rejected(self):
        for values in ({"white_spot_size": 2}, {"white_spot_size": 3.5}, {"white_spot_size": 33},
                       {"white_spot_threshold": 0}, {"white_spot_threshold": -1},
                       {"white_spot_threshold": float("nan")}, {"white_spot_threshold": float("inf")},
                       {"white_spot_method": "invalid"}, {"white_spot_enabled": "False"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                filter_settings(values)
        with self.assertRaises(ValueError):
            remove_white_spots(np.ones((2, 4, 4)))

    def test_absolute_and_count_noise_rules_match_pyerre_for_interior_bright_spikes(self):
        pyerre = Path(__file__).resolve().parents[3] / "pyerre" / "proc_functions" / "filters.py"
        if not pyerre.is_file():
            self.skipTest("Local pyerre reference is not installed.")
        spec = importlib.util.spec_from_file_location("pyerre_reference_filters", pyerre)
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        image = np.full((31, 31), 100, dtype=np.float32)
        image[15, 15] = 10000
        for mode in ("abs", "prop"):
            ours, _, _ = remove_white_spots(image, method=mode, threshold=8)
            theirs = reference.ro_filter_2D(image, 3, 8, type_threshold=mode)
            np.testing.assert_array_equal(ours[2:-2, 2:-2], theirs[2:-2, 2:-2])


class WhiteSpotPipelineTests(unittest.TestCase):
    def fixture(self, root, experiment="exp101_sample"):
        proposal, layout, name, _ = make_markerless_proposal(root, experiment=experiment, values=(10, 20))
        paths = sorted(path for path in (layout / "final" / name / "3to1000").glob("*.tiff")
                       if "sumAllRuns" not in path.name)
        originals = {}
        for index, path in enumerate(paths):
            image = tifffile.imread(path)
            image[2, 2] = 10000
            tifffile.imwrite(path, image, extratags=[(65001, "d", 1, 1.25, False)], resolution=(300, 300))
            originals[path] = path.read_bytes()
        config = {"project": {"data_path": str(proposal), "experiment": name, "variant": "3to1000"},
                  "images": {"white_spot_enabled": True, "white_spot_method": "abs",
                             "white_spot_threshold": 50, "frames_per_merge": 2}}
        return proposal, name, originals, config

    def run_sections(self, session, sections):
        for section in sections:
            outcome = session.run(section)
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)

    def test_source_frames_filtered_before_merging_with_saved_timing_and_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, name, originals, config = self.fixture(Path(temporary))
            session = create_session("exsitu", config)
            self.run_sections(session, ("project", "images", "average", "image_measurements"))
            np.testing.assert_allclose(session.artifacts.require("image_stack")[0], 15)
            np.testing.assert_allclose(session.artifacts.require("average_image"), 15)
            np.testing.assert_allclose(session.artifacts.require("image_metric_values"), [15])
            root = session.artifacts.require("results_dir")
            report = json.loads(session.artifacts.require("white_spot_report").read_text())
            self.assertEqual(report["frame_count"], 2)
            self.assertEqual(report["replaced_pixels"], 2)
            outputs = sorted((root / "images" / "white_spot_filtered_frames").glob("*.tiff"))
            self.assertEqual(len(outputs), 2)
            for output in outputs:
                with tifffile.TiffFile(output) as tif:
                    self.assertIn(65002, tif.pages[0].tags)
                    self.assertEqual(tif.pages[0].tags[65001].value, 1.25)
                    self.assertEqual(tif.pages[0].tags["XResolution"].value, (300, 1))
            times, images, _ = backends.nci_plots().load_corrected_images_with_time(str(outputs[0].parent))
            self.assertEqual(len(times), 2)
            self.assertEqual(float(images[0].mean()), 10)
            _, saved = load_config(root / "processing" / "config_used.ini")
            self.assertTrue(saved["images"]["white_spot_enabled"])
            for path, original in originals.items():
                self.assertEqual(path.read_bytes(), original)

    def test_filter_is_after_background_correction_not_before(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, _, config = self.fixture(Path(temporary))
            bg = Path(temporary) / "bg.tiff"
            tifffile.imwrite(bg, np.ones((6, 6), dtype=np.float32))
            config["background"] = {"enabled": True, "source_mode": "shared", "background_path": str(bg)}
            session = create_session("exsitu", config)
            self.run_sections(session, ("project", "corrections", "background", "images", "average"))
            np.testing.assert_allclose(session.artifacts.require("average_image"), 14)
            report = json.loads(session.artifacts.require("white_spot_report").read_text())
            self.assertTrue(all("/images/corrected_frames/" in row["source"] for row in report["frames"]))
            self.assertTrue(all(row["replaced_pixels"] == 1 for row in report["frames"]))

    def test_preview_writes_no_outputs_and_uses_current_corrected_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, config = self.fixture(Path(temporary))
            session = create_session("image_only", config)
            self.run_sections(session, ("project",))
            path, source, filtered, mask, stats = preview_white_spots(session)
            self.assertEqual(stats["replaced_pixels"], 1)
            self.assertGreater(source.max(), filtered.max())
            self.assertTrue(mask[2, 2])
            self.assertFalse(session.artifacts.require("results_dir").exists())
            session.configure("corrections", enabled=True)
            self.run_sections(session, ("corrections",))
            corrected, _, _, _, _ = preview_white_spots(session)
            self.assertNotEqual(path.parent, corrected.parent)
            self.assertEqual(corrected.parent, session.artifacts.require("processed_image_dir"))

    def test_finished_average_is_preserved_and_changed_filter_requires_alternative(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, config = self.fixture(Path(temporary))
            config["images"]["white_spot_enabled"] = False
            session = create_session("exsitu", config)
            self.run_sections(session, ("project", "images", "average"))
            average = session.artifacts.require("average_image_path")
            original = average.read_bytes()
            root = session.artifacts.require("results_dir")
            session.configure("images", white_spot_enabled=True)
            outcome = session.run("images")
            self.assertEqual(outcome.state, SectionState.FAILED)
            self.assertIn("Analysis name", outcome.message)
            self.assertEqual(average.read_bytes(), original)
            self.assertFalse((root / "images" / "white_spot_filtered_frames").exists())
            session.configure("project", analysis_name="despotted")
            self.run_sections(session, ("project", "images", "average"))
            report = session.artifacts.require("white_spot_report")
            original_report = report.read_bytes()
            session.configure("images", white_spot_threshold=100)
            self.assertEqual(session.run("images").state, SectionState.FAILED)
            self.assertEqual(report.read_bytes(), original_report)

    def test_batch_filters_each_experiment_and_resumes_with_saved_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, first, _, config = self.fixture(Path(temporary))
            _, second, _, _ = self.fixture(Path(temporary), experiment="exp102_sample")
            kwargs = dict(preset="exsitu", config=config, sections=("average",), batch_name="spots")
            result = run_batch(proposal, (first, second), **kwargs)
            self.assertEqual([job.status for job in result.results], ["completed", "completed"])
            for job in result.results:
                report = json.loads((job.results_dir / "processing" / "white_spot_filter.json").read_text())
                self.assertEqual(report["replaced_pixels"], 2)
                _, saved = load_config(job.config_path)
                self.assertTrue(saved["images"]["white_spot_enabled"])
            repeated = run_batch(proposal, (first, second), **kwargs)
            self.assertEqual([job.status for job in repeated.results], ["skipped", "skipped"])

    def test_batch_rejects_invalid_filter_before_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, experiment, _, config = self.fixture(Path(temporary))
            config["images"]["white_spot_size"] = 4
            report = validate_batch(proposal, (experiment,), config=config, sections=("average",))
            self.assertFalse(report.valid)
            self.assertIn("odd", report.common_errors[0])
            self.assertFalse((proposal / "analysis").exists())

    def test_disabled_filter_preserves_previous_average_recipes(self):
        session = create_session("exsitu")
        for key in DEFAULTS:
            session.config["images"].pop(key)
        old = recipe_for(session, "average")
        session.configure("images", **DEFAULTS)
        self.assertEqual(recipe_for(session, "average"), old)

    def test_crop_origin_is_preserved_for_filtered_and_unfiltered_frames(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temporary:
                _, _, _, config = self.fixture(Path(temporary))
                config["images"].update(white_spot_enabled=enabled, frames_per_merge=1,
                                        crop_roi={"x0": 3, "y0": 3, "r0": 1})
                session = create_session("exsitu", config)
                self.run_sections(session, ("project", "images"))
                image = session.artifacts.require("image_stack")[0]
                self.assertEqual(image.shape, (2, 2))
                self.assertEqual(image.crop_origin_xy, (2, 2))

    def test_gui_settings_roundtrip_and_preview_control_in_all_modes(self):
        for mode in PRESETS:
            gui = LumacamPipelineGUI(mode)
            self.assertFalse(gui.white_spot_enabled.value)
            self.assertNotEqual(gui.white_spot_preview_btn.layout.display, "none")
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, _, config = self.fixture(Path(temporary))
            path = Path(temporary) / "config.ini"
            save_config(path, "exsitu", config)
            gui = LumacamPipelineGUI("exsitu")
            gui.config_path.value = str(path)
            gui._load_config(None)
            self.assertTrue(gui.white_spot_enabled.value)
            self.assertEqual(gui.white_spot_threshold.value, 50)
            self.assertEqual(gui.white_spot_method.value, "abs")
            gui.white_spot_size.value = 5
            gui._sync_config()
            self.assertEqual(gui.session.section_config("images")["white_spot_size"], 5)
            gui._run("project")
            with patch.object(gui, "display") as display:
                gui._preview_white_spots()
                display.assert_called_once()


if __name__ == "__main__":
    unittest.main()

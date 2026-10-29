from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline.background import background_source, effective_background_settings
from lumacam_pipeline.batch import prepare_shared_ob_background, run_batch, validate_batch
from lumacam_pipeline.config import load_config, save_config
from lumacam_pipeline.gui import LumacamPipelineGUI
from lumacam_pipeline.presets import create_session
from lumacam_pipeline.provenance import recipe_for
from lumacam_pipeline.result_layout import snapshot_path
from lumacam_pipeline.section import SectionState
from test_lumacam_pipeline import make_markerless_proposal


class BackgroundSourceTests(unittest.TestCase):
    sections = ("corrections", "background", "images", "average")

    def fixture(self, root):
        proposal, _, ob, _ = make_markerless_proposal(root, experiment="exp001_OB", values=(2, 2))
        _, _, first, _ = make_markerless_proposal(root, experiment="exp101_sample", values=(4, 8, 12))
        _, _, second, _ = make_markerless_proposal(root, experiment="exp102_sample", values=(20, 24, 40))
        config = {
            "project": {"data_path": str(proposal), "variant": "3to1000"},
            "corrections": {"reference_bin_x": 1, "reference_bin_y": 1},
            "background": {"enabled": True, "source_mode": "first_frames", "n_images": 2,
                           "background_path": str(root / "unused_missing_shared_bg.tiff")},
        }
        return proposal, ob, (first, second), config

    def test_batch_each_experiment_uses_its_own_first_corrected_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, ob, experiments, config = self.fixture(Path(temporary))
            prepared = prepare_shared_ob_background(proposal, ob, config=config)
            self.assertIsNone(prepared.prepared_background_path)
            self.assertEqual(prepared.reference_tiff_count, 2)
            self.assertFalse((prepared.prepared_reference_path.parent / "background.tiff").exists())
            result = run_batch(proposal, experiments, preset="exsitu", config=config,
                               sections=self.sections, batch_name="first_frames")
            self.assertEqual([job.status for job in result.results], ["completed", "completed"])
            for job, expected_bg, expected_mean in zip(result.results, (3, 11), (4 / 3, 10 / 3)):
                directory = job.results_dir
                np.testing.assert_allclose(tifffile.imread(directory / "processing" / "background.tiff"), expected_bg)
                average = next((directory / "images").glob("average_*.tiff"))
                np.testing.assert_allclose(tifffile.imread(average), expected_mean)
                self.assertEqual(len(list((directory / "images" / "corrected_frames").glob("*.tiff"))), 3)
                metadata = json.loads((directory / "processing" / "background_metadata.json").read_text())
                extra = metadata["sections"]["background_processing"]["background_image"]["extra"]
                self.assertEqual(extra["valid_images_used"], 2)
                _, saved = load_config(job.config_path)
                self.assertEqual(saved["project"]["experiment"], job.experiment)
                self.assertEqual(saved["background"]["source_mode"], "first_frames")
                self.assertEqual(saved["background"]["n_images"], 2)
                recipes = json.loads((directory / "processing" / "recipes.json").read_text())
                self.assertEqual(recipes["background"]["settings"]["background_path"], "")
            resumed = run_batch(proposal, experiments, preset="exsitu", config=config,
                                sections=self.sections, batch_name="first_frames")
            self.assertEqual([job.status for job in resumed.results], ["skipped", "skipped"])

    def test_shared_bg_is_reused_for_both_experiments_without_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, ob, experiments, config = self.fixture(root)
            prepare_shared_ob_background(proposal, ob, config=config)
            bg = root / "prepared_background.tiff"
            tifffile.imwrite(bg, np.full((6, 6), 0.5, dtype=np.float32))
            # Count is irrelevant for an already prepared single TIFF.
            config["background"].update(source_mode="shared", background_path=str(bg), n_images=0)
            summary = run_batch(proposal, experiments, preset="exsitu", config=config,
                                sections=self.sections, batch_name="shared")
            self.assertEqual([job.status for job in summary.results], ["completed", "completed"])
            for job, mean in zip(summary.results, (3.5, 13.5)):
                average = next((job.results_dir / "images").glob("average_*.tiff"))
                np.testing.assert_allclose(tifffile.imread(average), mean)
                self.assertFalse((job.results_dir / "processing" / "background.tiff").exists())
                _, saved = load_config(job.config_path)
                self.assertEqual(saved["background"]["source_mode"], "shared")
                self.assertEqual(saved["background"]["background_path"], str(bg))

    def test_source_and_count_validation_before_batch_writes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, experiments, config = self.fixture(root)
            self.assertTrue(validate_batch(proposal, experiments, config=config, sections=self.sections).valid)
            for mode, path, count in (("shared", "", 2), ("shared", str(root / "missing"), 2),
                                      ("invalid", "", 2), ("first_frames", "", 0),
                                      ("first_frames", "", -1), ("first_frames", "", ""),
                                      ("first_frames", "", 1.5)):
                with self.subTest(mode=mode, count=count):
                    config["background"].update(source_mode=mode, background_path=path, n_images=count)
                    report = validate_batch(proposal, experiments, config=config, sections=self.sections)
                    self.assertFalse(report.valid)
                    self.assertTrue(report.common_errors)
            self.assertFalse((proposal / "analysis").exists())

    def test_individual_first_frames_uses_available_valid_frames_and_records_count(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, _, _ = make_markerless_proposal(Path(temporary), values=(0, 2, 4))
            session = create_session("exsitu", {
                "project": {"data_path": str(proposal), "variant": "3to1000"},
                "background": {"enabled": True, "source_mode": "first_frames", "n_images": 20,
                               "background_path": "/ignored/old/shared.tiff"},
                "images": {"frames_per_merge": 2},
            })
            for section in ("project", *self.sections):
                outcome = session.run(section)
                self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            result = session.artifacts.require("results_dir")
            np.testing.assert_allclose(tifffile.imread(result / "processing" / "background.tiff"), 3)
            metadata = json.loads((result / "processing" / "background_metadata.json").read_text())
            self.assertEqual(metadata["sections"]["background_processing"]["background_image"]["extra"]["valid_images_used"], 2)
            _, saved = load_config(snapshot_path(result))
            self.assertEqual(saved["background"]["n_images"], 20)

    def test_old_and_explicit_sources_have_equivalent_recipes(self):
        for path, expected in (("", "first_frames"), ("/shared/bg.tiff", "shared")):
            with self.subTest(mode=expected):
                old = {"background_path": path, "n_images": 2, "enabled": True}
                new = {**old, "source_mode": expected}
                self.assertEqual(background_source(old), expected)
                self.assertEqual(effective_background_settings(old), effective_background_settings(new))
                before = create_session("exsitu", {"background": old})
                before.config["background"].pop("source_mode", None)
                after = create_session("exsitu", {"background": new})
                self.assertEqual(recipe_for(before, "background"), recipe_for(after, "background"))
                for session in (before, after):
                    session.artifacts.put("analysis_image_dir", Path("/corrected"), "background")
                self.assertEqual(recipe_for(before, "average"), recipe_for(after, "average"))

    def test_old_batch_resumes_after_explicit_shared_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, experiments, config = self.fixture(root)
            bg = root / "background.tiff"
            tifffile.imwrite(bg, np.ones((6, 6), dtype=np.float32))
            config["background"].update(background_path=str(bg))
            config["background"].pop("source_mode")
            kwargs = dict(preset="exsitu", config=config, sections=self.sections, batch_name="legacy")
            original = run_batch(proposal, experiments, **kwargs)
            self.assertEqual([job.status for job in original.results], ["completed", "completed"])
            config["background"]["source_mode"] = "shared"
            repeated = run_batch(proposal, experiments, **kwargs)
            self.assertEqual([job.status for job in repeated.results], ["skipped", "skipped"])

    def test_changed_background_needs_an_alternative_analysis(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, experiments, config = self.fixture(root)
            first = run_batch(proposal, experiments[:1], preset="exsitu", config=config,
                              sections=self.sections, batch_name="first")
            original = next((first.results[0].results_dir / "images").glob("average_*.tiff"))
            original_bytes = original.read_bytes()
            changed = deepcopy(config)
            bg = root / "shared.tiff"
            tifffile.imwrite(bg, np.ones((6, 6), dtype=np.float32))
            changed["background"].update(source_mode="shared", background_path=str(bg))
            rejected = run_batch(proposal, experiments[:1], preset="exsitu", config=changed,
                                 sections=self.sections, batch_name="second")
            self.assertEqual(rejected.results[0].status, "failed")
            self.assertIn("Analysis name", rejected.results[0].message)
            self.assertEqual(original.read_bytes(), original_bytes)
            changed["project"]["analysis_name"] = "shared_bg"
            alternative = run_batch(proposal, experiments[:1], preset="exsitu", config=changed,
                                    sections=self.sections, batch_name="second")
            self.assertEqual(alternative.results[0].status, "completed")
            self.assertIn("alternatives/shared_bg", str(alternative.results[0].results_dir))

    def test_gui_ob_only_preparation_and_choice_saved_in_main_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, ob, _, _ = self.fixture(Path(temporary))
            gui = LumacamPipelineGUI("exsitu")
            self.assertEqual(gui.background_source.value, "shared")
            gui.data_path.value = str(proposal)
            gui._data_selected(proposal)
            gui.shared_ob_experiment.value = ob
            gui.background_source.value = "first_frames"
            gui.prepare_bg.value = False
            gui.background_count.value = 2
            gui.background_enabled.value = True
            self.assertEqual(gui.shared_background_experiment.layout.display, "none")
            self.assertEqual(gui.background_path_row.layout.display, "none")
            self.assertEqual(gui.prepare_shared_btn.description, "Prepare OB file")
            self.assertIn("first 2 valid corrected frames", gui.batch_background_info.value)
            gui._prepare_shared_inputs()
            self.assertIn("ready", gui.shared_status_html.value)
            _, saved = load_config(proposal / "analysis" / "lumacam_pipeline.ini")
            self.assertEqual(saved["background"]["source_mode"], "first_frames")
            self.assertEqual(saved["background"]["n_images"], 2)
            self.assertTrue(Path(saved["corrections"]["prepared_reference_path"]).is_file())
            self.assertFalse(list((proposal / "analysis" / "shared_corrections").rglob("background.tiff")))
            gui.background_source.value = "shared"
            gui._load_config(None)
            self.assertEqual(gui.background_source.value, "first_frames")
            self.assertEqual(gui.background_count.value, 2)

    def test_gui_infers_old_configs_and_honors_explicit_mode_with_retained_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.ini"
            gui = LumacamPipelineGUI("exsitu")
            for source in ({"background_path": "/shared/bg.tiff"}, {"background_path": ""},
                           {"source_mode": "first_frames", "background_path": "/shared/bg.tiff"}):
                with self.subTest(source=source):
                    save_config(path, "exsitu", {"background": source})
                    gui.config_path.value = str(path)
                    gui._load_config(None)
                    self.assertEqual(gui.background_source.value, background_source(source))
                    self.assertEqual(gui.background_path.value, source["background_path"])
                    self.assertEqual(gui.background_path_row.layout.display,
                                     "" if background_source(source) == "shared" else "none")


if __name__ == "__main__":
    unittest.main()

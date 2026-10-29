from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline.batch import _relocated_sentinel_matches, prepare_shared_ob_background, run_batch
from lumacam_pipeline.config import load_config, save_config
from lumacam_pipeline.gui import LumacamPipelineGUI
from lumacam_pipeline.presets import create_session
from lumacam_pipeline.restructure import make_plan, apply_plan, sha256
from lumacam_pipeline.result_layout import result_directory, snapshot_path
from lumacam_pipeline.section import SectionState
from test_lumacam_pipeline import make_markerless_proposal


class ResultStructureTests(unittest.TestCase):
    def run_sections(self, session, sections):
        for section in sections:
            outcome = session.run(section)
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)

    def test_final_intermediate_shared_background_and_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, ob, _ = make_markerless_proposal(Path(temporary), experiment="exp001_OB", values=(2, 2))
            _, _, bg, _ = make_markerless_proposal(Path(temporary), experiment="exp002_BG", values=(0.5, 0.5))
            _, _, sample, _ = make_markerless_proposal(Path(temporary), experiment="exp101_sample", values=(4, 4))
            config = {"project": {"data_path": str(proposal), "experiment": sample, "variant": "3to1000"},
                      "corrections": {"reference_bin_x": 1, "reference_bin_y": 1},
                      "background": {"n_images": 2}}
            prepared = prepare_shared_ob_background(proposal, ob, bg, preset="exsitu", config=config)
            shared = proposal / "analysis" / "shared_corrections" / "3to1000"
            self.assertEqual(prepared.prepared_reference_path, shared / "ob.tiff")
            self.assertEqual(prepared.prepared_background_path, shared / "background.tiff")
            session = create_session("exsitu", config)
            self.run_sections(session, ("project", "corrections", "background", "images", "average"))
            result = result_directory(proposal / "analysis", sample, "3to1000")
            self.assertTrue((result / "processing" / "intermediate" / "uncorrected").is_dir())
            self.assertTrue((result / "processing" / "intermediate" / "beam_ob_corrected").is_dir())
            self.assertTrue((result / "images" / "corrected_frames").is_dir())
            self.assertEqual(session.artifacts.require("background_path"), shared / "background.tiff")
            self.assertEqual(list((result / "processing").glob("background*.tif*")), [])
            self.assertTrue(snapshot_path(result).is_file())
            self.assertFalse((result / "lumacam_pipeline.ini").exists())
            np.testing.assert_allclose(tifffile.imread(session.artifacts.require("average_image_path")), 1.75)
            final_frame = next((result / "images" / "corrected_frames").glob("*.tiff"))
            final_hash = sha256(final_frame)
            session.configure("background", enabled=False)
            self.assertEqual(session.run("corrections").state, SectionState.FAILED)
            self.assertEqual(sha256(final_frame), final_hash)
            before = sha256(shared / "ob.tiff")
            config["corrections"]["reference_bin_x"] = 2
            with self.assertRaisesRegex(ValueError, "Reference set"):
                prepare_shared_ob_background(proposal, ob, bg, preset="exsitu", config=config)
            self.assertEqual(sha256(shared / "ob.tiff"), before)

    def test_without_background_final_frames_are_in_images_and_copies_are_kept(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, sample, _ = make_markerless_proposal(Path(temporary), values=(2, 4))
            session = create_session("exsitu", {"project": {"data_path": str(proposal), "variant": "3to1000"}})
            self.run_sections(session, ("project", "corrections"))
            result = session.artifacts.require("results_dir")
            self.assertEqual(session.artifacts.require("processed_image_dir"), result / "images" / "corrected_frames")
            self.assertEqual(len(list((result / "processing" / "intermediate" / "uncorrected").glob("*.tiff"))), 2)
            session.configure("background", enabled=True, n_images=1)
            self.run_sections(session, ("background", "images", "average"))
            np.testing.assert_allclose(tifffile.imread(session.artifacts.require("average_image_path")), 1)
            self.assertEqual(len(list((result / "processing" / "intermediate" / "beam_ob_corrected").glob("*.tiff"))), 2)

    def test_changed_recipe_requires_named_alternative_without_overwriting_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, sample, _ = make_markerless_proposal(Path(temporary), values=(2, 4))
            session = create_session("exsitu", {"project": {"data_path": str(proposal), "variant": "3to1000"}})
            self.run_sections(session, ("project", "corrections", "images", "average"))
            original = session.artifacts.require("average_image_path")
            original_hash = sha256(original)
            snapshot = snapshot_path(session.artifacts.require("results_dir"))
            original_config = snapshot.read_bytes()
            session.configure("corrections", reference_thickness_um=2)
            outcome = session.run("corrections")
            self.assertEqual(outcome.state, SectionState.FAILED)
            self.assertIn("Analysis name", outcome.message)
            self.assertEqual(snapshot.read_bytes(), original_config)
            self.assertEqual(sha256(original), original_hash)
            session.configure("project", analysis_name="check_reference")
            self.run_sections(session, ("project", "corrections", "images", "average"))
            self.assertIn("alternatives/check_reference", str(session.artifacts.require("results_dir")))
            self.assertEqual(sha256(original), original_hash)
            self.assertTrue(session.artifacts.require("average_collection_path").name.endswith("_check_reference.tiff"))

    def test_variants_have_separate_snapshots_and_batch_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, sample, _ = make_markerless_proposal(Path(temporary), values=(2, 4), variants=("3to1000", "5to1000"))
            for variant in ("3to1000", "5to1000"):
                summary = run_batch(proposal, (sample,), preset="exsitu",
                                    config={"project": {"variant": variant}}, sections=("average",), batch_name=variant)
                self.assertEqual(summary.results[0].status, "completed")
                self.assertEqual(summary.status_path.parent.parent, proposal / "analysis" / "batches")
                self.assertTrue(snapshot_path(result_directory(proposal / "analysis", sample, variant)).is_file())
            with self.assertRaises(ValueError):
                result_directory(proposal / "analysis", sample, "3to1000", "../escape")

    def test_loading_snapshot_or_old_main_path_saves_only_to_new_main_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, sample, _ = make_markerless_proposal(Path(temporary), values=(2,))
            session = create_session("exsitu", {"project": {"data_path": str(proposal), "variant": "3to1000", "experiment": sample}})
            self.run_sections(session, ("project", "images", "average"))
            snapshot = snapshot_path(session.artifacts.require("results_dir"))
            original = snapshot.read_bytes()
            gui = LumacamPipelineGUI("exsitu")
            gui.config_path.value = str(snapshot)
            gui._load_config(None)
            main = proposal / "analysis" / "lumacam_pipeline.ini"
            self.assertEqual(gui.config_path.value, str(main))
            gui.frames_per_merge.value = 2
            gui._save_config(None)
            self.assertEqual(snapshot.read_bytes(), original)
            self.assertEqual(load_config(main)[1]["images"]["frames_per_merge"], 2)
            gui.config_path.value = str(proposal / "analysis" / "proposal_summary" / "lumacam_pipeline.ini")
            gui._load_config(None)
            self.assertEqual(gui.config_path.value, str(main))


def legacy_fixture(root):
    proposal, _, experiment, _ = make_markerless_proposal(root, experiment="exp101_sample", values=(2,))
    analysis = proposal / "analysis"
    old = analysis / experiment
    summary = analysis / "proposal_summary"
    ref_dir = summary / "reference_cache" / "old_ref"
    bg_dir = summary / "shared_inputs" / "background_cache" / "old_bg"
    ref_dir.mkdir(parents=True)
    bg_dir.mkdir(parents=True)
    ref, bg = ref_dir / "processed_reference_3to1000.tiff", bg_dir / "processed_background_3to1000.tiff"
    tifffile.imwrite(ref, np.full((6, 6), 2, dtype=np.float32))
    tifffile.imwrite(bg, np.full((6, 6), 0.5, dtype=np.float32))
    (ref_dir / "reference_cache.json").write_text(json.dumps({"source_count": 2}))
    (bg_dir / "shared_background.json").write_text(json.dumps({"background_experiment": "exp002_BG", "prepared_background_path": str(bg)}))
    config = create_session("exsitu", {
        "project": {"data_path": str(proposal), "experiment": experiment, "variant": "3to1000"},
        "shared_setup": {"prepared_reference_path": str(ref), "prepared_background_path": str(bg)},
        "corrections": {"prepared_reference_path": str(ref)},
        "background": {"enabled": True, "background_path": str(bg)},
    }).config
    save_config(old / "lumacam_pipeline.ini", "exsitu", config)
    save_config(summary / "lumacam_pipeline.ini", "exsitu", config)
    for folder, value in ((f"uncorrected_{experiment}", 4), (f"obint_corrected_{experiment}", 2), (f"final_{experiment}", 1.5)):
        destination = old / "images" / folder / "frame.tiff"
        destination.parent.mkdir(parents=True)
        tifffile.imwrite(destination, np.full((6, 6), value, dtype=np.float32))
    tifffile.imwrite(old / "images" / f"average_{experiment}.tiff", np.full((6, 6), 1.5, dtype=np.float32))
    (old / "processing").mkdir()
    (old / "processing" / f"background_{experiment}.tif").write_bytes(bg.read_bytes())
    (old / "processing" / "processing_metadata.json").write_text(json.dumps({"output_dir": str(old / "images" / f"obint_corrected_{experiment}")}))
    return proposal, experiment


class RestructureTests(unittest.TestCase):
    def test_relocated_completion_requires_same_inputs_and_exact_section_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "exp101_sample" / "3to1000"
            config = create_session("exsitu", {"project": {"experiment": "exp101_sample", "variant": "3to1000"}}).config
            saved = snapshot_path(result)
            save_config(saved, "exsitu", config)
            (result / "images").mkdir()
            (result / "images" / "average_exp101_sample_3to1000.tiff").write_bytes(b"test")
            sentinel = result / "processing" / "batch_old.json"
            sections = ("project", "images", "average")
            sentinel.write_text(json.dumps({"status": "completed", "input_signature": "same",
                                           "sections": sections, "relocated_config_snapshot": str(saved)}))
            self.assertTrue(_relocated_sentinel_matches(sentinel, saved, "same", sections, "exsitu", config))
            self.assertFalse(_relocated_sentinel_matches(sentinel, saved, "changed", sections, "exsitu", config))
            config["average"]["method"] = "median"
            self.assertFalse(_relocated_sentinel_matches(sentinel, saved, "same", sections, "exsitu", config))

    def test_plan_apply_preserves_tiffs_updates_records_and_archives_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, experiment = legacy_fixture(Path(temporary))
            with patch("lumacam_pipeline.restructure.time.time", return_value=time.time() + 300):
                plan = make_plan(proposal)
            self.assertEqual(sum(item["duplicate"] for item in plan["files"]), 1)
            self.assertFalse((proposal / "analysis" / "lumacam_pipeline.ini").exists())
            report = apply_plan(plan)
            self.assertEqual(report["duplicate_backgrounds_archived"], 1)
            self.assertTrue((Path(report["recovery"]) / "original_records.tar.gz").is_file())
            for item in plan["files"]:
                if Path(item["source"]).suffix in {".tif", ".tiff"}:
                    self.assertEqual(sha256(item["target"]), item["sha256"])
            result = proposal / "analysis" / experiment / "3to1000"
            _, config = load_config(snapshot_path(result))
            self.assertEqual(config["background"]["background_path"], str(proposal / "analysis" / "shared_corrections" / "3to1000" / "background.tiff"))
            self.assertFalse((proposal / "analysis" / "proposal_summary" / "reference_cache").exists())
            self.assertTrue((result / "processing" / "recipes.json").is_file())
            self.assertIn("processing/intermediate/beam_ob_corrected", (result / "processing" / "processing_metadata.json").read_text())

    def test_changed_file_aborts_before_moves(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _ = legacy_fixture(Path(temporary))
            with patch("lumacam_pipeline.restructure.time.time", return_value=time.time() + 300):
                plan = make_plan(proposal)
            source = Path(plan["files"][0]["source"])
            source.write_bytes(source.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "changed since audit"):
                apply_plan(plan)
            self.assertFalse((proposal / ".trash").exists())


if __name__ == "__main__":
    unittest.main()

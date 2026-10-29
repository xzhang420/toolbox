"""Processing/visualization boundaries and relocation, using only temporary data."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline.batch import run_batch, validate_batch, discover_batch_experiments
from lumacam_pipeline.cli import main
from lumacam_pipeline.config import load_config, save_config
from lumacam_pipeline.handover import MANIFEST, load_processed_images, file_digest
from lumacam_pipeline.image_io import read_image
from lumacam_pipeline.inspection import inspect_layout, discover_for_config
from lumacam_pipeline.processing import create_processing_session, resolve_processing_paths
from lumacam_pipeline.section import SectionState
from lumacam_pipeline.visualization import open_visualization, load_visualization_settings, save_visualization_settings
from lumacam_pipeline.gui import LumacamPipelineGUI
from lumacam_pipeline.visualization_gui import VisualizationGUI
from test_lumacam_pipeline import make_markerless_proposal


class SplitWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def proposal(self, **kwargs):
        proposal, layout, experiment, runs = make_markerless_proposal(self.root, **kwargs)
        (layout / "tpx3Files").rename(proposal / "raw")
        (proposal / "reduced").mkdir()
        (proposal / "data").rename(proposal / "reduced/data")
        (proposal / "reduced/data/experiments").rename(proposal / "reduced/data/experiment")
        return proposal, experiment, runs

    def process(self, proposal, experiment, extra=None):
        values = {"project": {"data_path": str(proposal), "experiment": experiment, "variant": "3to1000"}}
        for section, options in (extra or {}).items():
            values.setdefault(section, {}).update(options)
        session = create_processing_session("exsitu", values)
        for section in ("project", "corrections", "images", "average"):
            outcome = session.run(section)
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        return session

    def plain(self, name="plain", count=3, timed=False):
        root = self.root / name
        root.mkdir()
        for i in range(count):
            tifffile.imwrite(root / f"frame_{i}.tiff", np.full((12, 12), i+1, np.float32), metadata=None,
                             description=f"AcquisitionTime:2026-08-20T12:0{i}:00+00:00" if timed else "No timing")
        return root

    def test_processing_publishes_without_raw_and_portable_copy_visualizes(self):
        proposal, experiment, _ = self.proposal()
        (proposal / "raw").rename(self.root / "offline_raw")
        source_hashes = {p: file_digest(p) for p in (proposal / "reduced/data").rglob("*.tiff")}
        session = self.process(proposal, experiment)
        result = session.artifacts.require("results_dir")
        self.assertTrue(result.is_relative_to(proposal / "reduced/analysis"))
        manifest = json.loads((result / MANIFEST).read_text())
        self.assertEqual(manifest["frame_count"], 2)
        self.assertTrue(all(row["timestamp"].endswith("+00:00") for row in manifest["frames"]))
        snapshot = (result / "processing/config_used.ini").read_bytes()
        copied = self.root / "collaborator"
        shutil.copytree(result, copied)
        proposal.rename(self.root / "offline_proposal")
        view = open_visualization(copied)
        for step in ("image_measurements", "selection", "image_plot", "individual_image_plots"):
            outcome = view.run(step)
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        self.assertEqual((copied / "processing/config_used.ini").read_bytes(), snapshot)
        self.assertTrue(list((copied / "visualization/overview/figures").rglob("*.png")))
        self.assertFalse((copied / "visualization/overview/processing").exists())
        for path, digest in source_hashes.items():
            self.assertEqual(file_digest(self.root / "offline_proposal" / path.relative_to(proposal)), digest)

    def test_cropped_merged_stack_is_the_handover_not_original_frames(self):
        proposal, experiment, _ = self.proposal(values=(2., 4., 6., 8.))
        session = self.process(proposal, experiment, {"images": {"frames_per_merge": 2,
                             "spatial_pixel_merge": 2, "pixel_size_um": 20}})
        data = load_processed_images(session.artifacts.require("results_dir"))
        self.assertEqual(len(data["images"]), 2)
        self.assertEqual(data["images"][0].shape, (3, 3))
        np.testing.assert_allclose(data["images"][0], 3)
        self.assertEqual(data["pixel_size_um"], 40)

    def test_processing_rejects_mixed_variants(self):
        proposal, experiment, _ = self.proposal(variants=("3to1000", "5to1000"))
        session = create_processing_session("exsitu", {"project": {"data_path": str(proposal), "experiment": experiment, "variant": ""}})
        outcome = session.run("project")
        self.assertEqual(outcome.state, SectionState.FAILED)
        self.assertIn("variant", outcome.message)

    def test_processing_cannot_run_visualization_sections(self):
        session = create_processing_session()
        self.assertEqual(session.run("image_plot").state, SectionState.FAILED)
        for mode in ("exsitu", "operando_nci", "tof"):
            gui = LumacamPipelineGUI(mode, workspace="processing")
            gui.show_all_sections.value = True
            self.assertNotIn("image_plot", gui._visible_sections)
            self.assertNotIn("electrochemistry", gui._visible_sections)
            self.assertNotIn("individual_image_plots", gui.batch_sections.value)

    def test_enabled_background_must_be_applied_before_publication(self):
        proposal, experiment, _ = self.proposal()
        session = create_processing_session("exsitu", {"project": {"data_path": str(proposal), "experiment": experiment, "variant": "3to1000"},
                                                       "background": {"enabled": True}})
        self.assertEqual(session.run("project").state, SectionState.COMPLETED)
        self.assertEqual(session.run("images").state, SectionState.FAILED)
        self.assertFalse(Path(session.artifacts.require("results_dir"), MANIFEST).exists())

    def test_untimed_images_are_viewable_but_cannot_synchronize(self):
        root = self.plain()
        view = open_visualization(root)
        self.assertFalse(view.artifacts.require("timing_available"))
        self.assertEqual(view.artifacts.require("image_times"), [None] * 3)
        self.assertIsNone(view.artifacts.get("pixel_size_um"))
        self.assertEqual(view.run("selection").state, SectionState.COMPLETED)
        self.assertEqual(view.run("image_plot").state, SectionState.COMPLETED)
        self.assertEqual(view.run("synchronization").state, SectionState.FAILED)
        self.assertEqual(view.run("nci_plating_metrics").state, SectionState.FAILED)
        with self.assertRaises(KeyError):
            view.run("corrections")

    def test_explicit_timing_and_pixel_size(self):
        view = open_visualization(self.plain(), config={"input": {"start_time": "2026-01-01T12:00:00+00:00",
                                                                  "interval_seconds": 60, "pixel_size_um": 3.5}})
        self.assertTrue(view.artifacts.require("timing_available"))
        self.assertEqual((view.artifacts.require("image_times")[1] - view.artifacts.require("image_times")[0]).total_seconds(), 60)
        self.assertEqual(view.artifacts.require("pixel_size_um"), 3.5)

    def test_profile_relocation_and_export_records(self):
        root = self.plain(timed=True)
        session = open_visualization(root)
        self.assertEqual(session.run("selection").state, SectionState.COMPLETED)
        for value in (10, 20):
            session.configure("image_plot", vmin=0, vmax=value)
            self.assertEqual(session.run("image_plot").state, SectionState.COMPLETED)
        records = list((root / "visualization/overview/records").glob("*/export.json"))
        self.assertEqual(len(records), 3)
        copied = self.root / "copy"
        shutil.copytree(root, copied)
        root.rename(self.root / "offline")
        reopened = load_visualization_settings(copied / "visualization/overview/settings.ini")
        self.assertEqual(reopened.artifacts.require("image_paths")[0], str(copied / "frame_0.tiff"))
        self.assertEqual(reopened.config["image_plot"]["vmax"], 20)
        self.assertEqual(reopened.artifacts.require("results_dir"), copied / "visualization/overview")
        self.assertFalse((copied / "processing/config_used.ini").exists())

    def test_modified_manifest_image_rejected(self):
        proposal, experiment, _ = self.proposal()
        result = self.process(proposal, experiment).artifacts.require("results_dir")
        tifffile.imwrite(result / "images/processed_frames/frame_00000.tiff", np.zeros((6, 6), np.float32))
        with self.assertRaisesRegex(ValueError, "changed since publication"):
            open_visualization(result)

    def test_profile_identity_and_directory_escape_protected(self):
        root = self.plain()
        view = open_visualization(root)
        save_visualization_settings(view)
        tifffile.imwrite(root / "frame_0.tiff", np.full((12, 12), 999, np.float32))
        with self.assertRaisesRegex(ValueError, "different/changed"):
            open_visualization(root)
        with self.assertRaises(ValueError):
            open_visualization(root, profile="../escape")
        with self.assertRaises(ValueError):
            load_processed_images(root, pattern="../*.tiff")

    def test_average_and_nested_variants_are_not_recursively_mixed(self):
        root = self.plain(count=1)
        (root / "another_variant").mkdir()
        tifffile.imwrite(root / "another_variant/other.tiff", np.ones((12, 12), np.float32))
        tifffile.imwrite(root / "exp_sumAllRuns.tiff", np.ones((12, 12), np.float32))
        self.assertEqual(len(load_processed_images(root)["images"]), 1)

    def test_fits_numeric_values_and_hdu_selection(self):
        from astropy.io import fits
        path = self.root / "image.fits"
        original = np.arange(12, dtype=np.int16).reshape(3, 4)
        hdu = fits.PrimaryHDU(original)
        hdu.header["BSCALE"] = 2
        hdu.header["BZERO"] = 10
        hdu.header["BUNIT"] = "count"
        hdu.writeto(path)
        array, metadata = read_image(path)
        np.testing.assert_array_equal(array, original * 2 + 10)
        self.assertEqual(metadata["unit"], "count")
        multiple = self.root / "multiple.fits"
        fits.HDUList([fits.PrimaryHDU(original), fits.ImageHDU(original+1)]).writeto(multiple)
        with self.assertRaisesRegex(ValueError, "fits_hdu"):
            read_image(multiple)
        np.testing.assert_array_equal(read_image(multiple, fits_hdu=1)[0], original+1)

    def test_inspection_is_read_only_and_new_layout_defaults_correct(self):
        proposal, experiment, _ = self.proposal()
        before = sorted(str(p.relative_to(proposal)) for p in proposal.rglob("*"))
        report = inspect_layout(proposal, experiment=experiment, variant="3to1000")
        self.assertEqual(report["layout"], "raw_reduced")
        self.assertEqual(report["analysis_root"], str(proposal / "reduced/analysis"))
        self.assertTrue(report["valid"])
        self.assertEqual(report["datasets"][0]["image_count"], 2)
        self.assertEqual(report["datasets"][0]["missing_timing_runs"], [])
        self.assertEqual(before, sorted(str(p.relative_to(proposal)) for p in proposal.rglob("*")))

    def test_explicit_layout_mapping_processing_and_batch(self):
        proposal, _, experiment, runs = make_markerless_proposal(self.root)
        image_root = proposal / "data/experiments/final"
        image_root.rename(proposal / "pictures")
        (proposal / "data/experiments/metadata").rename(proposal / "timing")
        config = {"project": {"data_path": str(proposal), "experiment": experiment, "variant": "3to1000"},
                  "layout": {"images_pattern": "pictures/{experiment}/{variant}/*.tiff",
                             "metadata_pattern": "timing/{experiment}/{run}", "run_regex": r"(?P<run>.+_\d{5})_sumImage"},
                  "workflow": {"role": "processing"}}
        dataset = discover_for_config(config)
        self.assertEqual(dataset.layout_type, "mapped")
        self.assertEqual(len(dataset.products), 2)
        gui = LumacamPipelineGUI("exsitu", workspace="processing")
        gui.data_path.value = str(proposal)
        gui.images_pattern.value = config["layout"]["images_pattern"]
        gui.metadata_pattern.value = config["layout"]["metadata_pattern"]
        gui.run_regex.value = config["layout"]["run_regex"]
        gui._populate_batch_choices()
        self.assertIn(experiment, gui.batch_unselected_experiments.options)
        self.assertEqual(gui._automatic_proposal_config_path(), proposal / "analysis/lumacam_pipeline.ini")
        session = create_processing_session("exsitu", config)
        for step in ("project", "corrections", "images"):
            outcome = session.run(step)
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        names = discover_batch_experiments(proposal, config=config)
        self.assertIn(experiment, names)
        # An explicit alternative prevents collisions with the individual result.
        config["project"]["analysis_name"] = "batch"
        summary = run_batch(proposal, [experiment], preset="exsitu", config=config,
                            sections=("corrections", "images", "average"), batch_name="mapped")
        self.assertEqual(summary.failed_count, 0, str(summary.results))

    def test_batch_publishes_each_experiment(self):
        proposal, experiment, _ = self.proposal()
        config = {"workflow": {"role": "processing"}, "project": {"variant": "3to1000"}}
        summary = run_batch(proposal, [experiment], preset="exsitu", config=config,
                            sections=("images", "average"), batch_name="processed")
        self.assertEqual(summary.failed_count, 0, str(summary.results))
        self.assertTrue((summary.results[0].results_dir / MANIFEST).is_file())
        self.assertTrue((summary.results[0].results_dir / "processing/config_used.ini").is_file())

    def test_processing_gui_loads_existing_main_without_autoselecting_ob(self):
        proposal, experiment, _ = self.proposal(experiment="exp001_OB")
        path = proposal / "reduced/analysis/lumacam_pipeline.ini"
        save_config(path, "exsitu", {"project": {"data_path": "../..", "experiment": "", "variant": "3to1000"},
                                    "corrections": {"beamhistory_path": "reduced/data/beamhistory"}})
        gui = LumacamPipelineGUI("exsitu", workspace="processing")
        gui.config_path.value = str(path)
        before = path.read_bytes()
        gui._load_config(None)
        self.assertEqual(gui.shared_ob_experiment.value, "")
        self.assertEqual(gui.data_path.value, str(proposal))
        self.assertEqual(gui.beam_path.value, str(proposal / "reduced/data/beamhistory"))
        self.assertEqual(gui.session.config["workflow"]["role"], "processing")
        self.assertEqual(path.read_bytes(), before)

    def test_visualization_gui_restores_settings_and_detects_input_changes(self):
        root = self.plain()
        gui = VisualizationGUI()
        gui.source.value = str(root)
        gui.open()
        self.assertIsNotNone(gui.session)
        gui.vmin.value, gui.vmax.value = "0", "12"
        gui.run("image_plot")
        saved = Path(gui.config_path.value)
        self.assertTrue(saved.is_file())
        other = VisualizationGUI("operando_nci")
        other.config_path.value = str(saved)
        other.load()
        self.assertEqual(other.vmax.value, "12.0")
        other.profile.value = "another"
        with self.assertRaisesRegex(ValueError, "changed"):
            other._sync()

    def test_cli_inspect_and_render_emit_json(self):
        proposal, experiment, _ = self.proposal()
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = main(["--data", str(proposal), "--experiment", experiment, "--variant", "3to1000", "inspect"])
        self.assertEqual(result, 0)
        self.assertTrue(json.loads(stream.getvalue())["valid"])
        root = self.plain()
        stream = io.StringIO()
        with redirect_stdout(stream):
            result = main(["--preset", "exsitu", "visualize", "--input", str(root), "--set", "image_plot.dpi=60"])
        self.assertEqual(result, 0, stream.getvalue())
        self.assertEqual(json.loads(stream.getvalue())["results"][-1]["state"], "completed")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
    def test_real_video_export_keeps_sources_unchanged(self):
        root = self.plain(count=2)
        before = {path: file_digest(path) for path in root.glob("*.tiff")}
        session = open_visualization(root, config={"image_video": {"dpi": 50, "fps": 2}})
        outcome = session.run("image_video")
        self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        self.assertGreater(session.artifacts.require("image_video_path").stat().st_size, 0)
        self.assertEqual(before, {path: file_digest(path) for path in before})

    def test_tof_exports_have_independent_records(self):
        path = self.root / "cube.npy"
        np.save(path, np.arange(48).reshape(3, 4, 4))
        before = file_digest(path)
        view = open_visualization(path, preset="tof", cube=True)
        outputs = []
        for _ in range(2):
            outcome = view.run("tof_cube")
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            outputs.append(view.artifacts.require("tof_spectrum_path"))
        self.assertNotEqual(*outputs)
        self.assertTrue(all(p.is_file() for p in outputs))
        self.assertEqual(before, file_digest(path))

    def test_echem_utc_alignment_and_combined_figure(self):
        from datetime import datetime
        from lumacam_pipeline import backends
        source = self.root / "echem.txt"
        source.touch()
        view = open_visualization(self.plain(timed=True), preset="operando_nci",
            config={"electrochemistry": {"path": str(source), "timestamp_timezone": "Europe/Zurich"}})
        values = ([datetime(2026, 8, 20, 14, i) for i in range(3)],
                  np.array([0., 1., 0.]), np.array([0., 1., 2.]), np.ones(3), None, np.array([0., .3, .6]))
        with patch.object(backends.nci_plots(), "load_echem_data", return_value=values):
            self.assertEqual(view.run("electrochemistry").state, SectionState.COMPLETED)
        self.assertEqual(view.run("image_measurements").state, SectionState.COMPLETED)
        outcome = view.run("synchronization")
        self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
        np.testing.assert_allclose(view.artifacts.require("synchronized_table")["time_offset_s"], 0)
        outcome = view.run("combined_plot")
        self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)

    def test_reference_filter_remains_active_for_preselected_samples(self):
        from lumacam_pipeline import backends
        proposal, experiment, _ = self.proposal(variants=("3to1000", "5to1000"))
        reference = proposal / "reduced/data/experiment/final" / experiment
        with patch.object(backends.nci_data(), "load_reference_image_series", return_value=[np.ones((6, 6))]) as loader:
            self.process(proposal, experiment, {"corrections": {"reference_dir": str(reference),
                                                               "reference_bin_x": 1, "reference_bin_y": 1}})
        self.assertEqual(loader.call_args.kwargs["file_filter"], "3to1000")

    def test_beam_correction_refuses_missing_metadata_and_reference_integral(self):
        proposal, experiment, runs = self.proposal()
        beam = self.root / "beam.txt"
        beam.write_text("2026-08-20T12:00:00Z,1\n2026-08-20T12:03:00Z,1\n")
        session = create_processing_session("exsitu", {"project": {"data_path": str(proposal), "experiment": experiment, "variant": "3to1000"},
            "corrections": {"beamhistory_path": str(beam)}})
        self.assertEqual(session.run("project").state, SectionState.COMPLETED)
        self.assertIn("reference beam integral", session.run("corrections").message)
        metadata = proposal / "reduced/data/experiment/metadata" / experiment / runs[0] / "acquisition.json"
        metadata.rename(metadata.with_suffix(".offline"))
        session.configure("corrections", reference_beam_integral_per_minute=60)
        outcome = session.run("corrections")
        self.assertEqual(outcome.state, SectionState.FAILED)
        self.assertIn("timing metadata", outcome.message)

    def test_nested_beamhistory_folder_is_not_treated_as_empty(self):
        from lumacam_pipeline.processing import available_beamhistory
        root = self.root / "beam"
        (root / "day1").mkdir(parents=True)
        self.assertIsNone(available_beamhistory(root))
        (root / "day1/file.dat").write_text("2026-01-01T00:00:00Z,1")
        self.assertEqual(available_beamhistory(root), str(root))

    def test_notebooks_are_thin_valid_launchers(self):
        import nbformat
        directory = Path(__file__).resolve().parents[1]
        for name in ("lumacam_processing.ipynb", "lumacam_visualization.ipynb"):
            notebook = nbformat.read(directory / name, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    compile(cell.source, name, "exec")

    def test_config_round_trip_preserves_nested_json_and_percent_paths(self):
        values = {"input": {"path": "../../100%data"},
                  "custom": {"options": {"enabled": True, "lower": None, "extra": [False, 2]}}}
        path = self.root / "settings.ini"
        save_config(path, "exsitu", values)
        self.assertEqual(load_config(path)[1], values)

    def test_shared_preparation_ignores_empty_default_beam_folder(self):
        from lumacam_pipeline.batch import prepare_shared_ob_background
        proposal, experiment, _ = self.proposal()
        beam = proposal / "reduced/data/beamhistory"
        beam.mkdir()
        config = {"workflow": {"role": "processing"}, "project": {"variant": "3to1000"},
                  "corrections": {"beamhistory_path": str(beam)}}
        result = prepare_shared_ob_background(proposal, experiment, config=config, prepare_background=False)
        self.assertEqual(result.reference_tiff_count, 2)
        self.assertIsNone(result.reference_beam_integral_per_minute)

    def test_shared_reference_timing_preflight_rejects_missing_metadata(self):
        from lumacam_pipeline.processing import prepare_reference_timing
        proposal, experiment, runs = self.proposal()
        beam = self.root / "beam.txt"
        beam.write_text("2026-08-20T12:00:00Z,1")
        metadata = proposal / "reduced/data/experiment/metadata" / experiment / runs[0] / "acquisition.json"
        metadata.rename(metadata.with_suffix(".offline"))
        config = {"workflow": {"role": "processing"}, "corrections": {
            "reference_dir": str(proposal / "reduced/data/experiment/final" / experiment),
            "beamhistory_path": str(beam)}}
        with self.assertRaisesRegex(ValueError, "JSON timing metadata"):
            prepare_reference_timing(config)


if __name__ == "__main__":
    unittest.main()

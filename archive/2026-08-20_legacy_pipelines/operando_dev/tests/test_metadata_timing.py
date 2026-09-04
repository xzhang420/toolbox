import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile


OPERANDO_DEV_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OPERANDO_DEV_DIR))

import operando_data  # noqa: E402
import operando_gui  # noqa: E402


def make_layout(root, tree_name="experiment", experiment="sample", run="sample_00000"):
    layout_root = root / "data" / tree_name
    metadata_dir = layout_root / "metadata" / experiment / run
    final_dir = layout_root / "final" / experiment / "3to1000"
    metadata_dir.mkdir(parents=True)
    final_dir.mkdir(parents=True)
    (layout_root / "metadata" / "layout.json").write_text(
        json.dumps({"layout_name": "lumacam-nci-proposal", "layout_version": 2}),
        encoding="utf-8",
    )
    image_path = final_dir / f"{run}_sumImage_3to1000.tiff"
    tifffile.imwrite(image_path, np.full((4, 4), 7.0, dtype=np.float32))
    return layout_root, metadata_dir, final_dir, image_path


class Version2MetadataTimingTests(unittest.TestCase):
    def test_gui_loads_the_development_backend(self):
        self.assertEqual(operando_gui.OPERANDO_DIR, OPERANDO_DEV_DIR)

    def test_gui_uses_intensity_labels_without_reference_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            gui = operando_gui.OperandoWorkflowGUI(
                campaign_dir=root,
                history_path=root / ".operando_gui_history.json",
            )
            self.assertEqual(
                gui._nci_metric_plot_kwargs()["thickness_axis_label"],
                "Mean NCI intensity (a.u.)",
            )
            gui.state["obint_result"] = {"reference_corrected": True}
            self.assertEqual(
                gui._nci_metric_plot_kwargs()["thickness_axis_label"],
                "Li thickness (μm)",
            )
            gui.state["echem_datetimes"] = []
            gui.show_cycle_numbers.value = True
            self.assertFalse(gui._event_plot_kwargs()["show_cycle_numbers"])

    def test_proposal_root_discovers_singular_and_plural_data_trees(self):
        for tree_name in ("experiment", "experiments"):
            with self.subTest(tree_name=tree_name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                layout_root, _, _, _ = make_layout(root, tree_name=tree_name)
                self.assertEqual(
                    Path(operando_data._find_v2_layout_root(root)),
                    layout_root,
                )
                choices = operando_data.list_nci_experiment_dirs(root)
                self.assertEqual(choices, [str(layout_root / "final" / "sample")])

    def test_acquisition_json_supplies_timing_after_raw_files_are_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, metadata_dir, final_dir, _ = make_layout(root)
            acquisition_path = metadata_dir / "acquisition.json"
            acquisition_path.write_text(
                json.dumps(
                    {
                        "run": "sample_00000",
                        "status": "acquired",
                        "started_at": "2026-07-27T15:39:36.425995+00:00",
                        "ended_at": "2026-07-27T15:40:42.226301+00:00",
                    }
                ),
                encoding="utf-8",
            )

            _, plan, intensity_enabled = operando_data._correction_plan(
                final_dir,
                file_filter="3to1000",
            )

            self.assertFalse(intensity_enabled)
            self.assertEqual(len(plan), 1)
            row = plan[0]
            self.assertEqual(row["exposure_timing_source"], "acquisition_json")
            self.assertEqual(row["tpx_file_count"], 0)
            self.assertEqual(row["exposure_window_count"], 1)
            self.assertAlmostEqual(row["total_effective_exposure_s"], 65.800306)
            self.assertEqual(row["timing_metadata_paths"], [str(acquisition_path)])
            expected_midpoint = datetime.fromtimestamp(
                (
                    operando_data.pd.Timestamp("2026-07-27T15:39:36.425995Z").timestamp()
                    + operando_data.pd.Timestamp("2026-07-27T15:40:42.226301Z").timestamp()
                )
                / 2
            )
            self.assertAlmostEqual(
                row["time"].timestamp(), expected_midpoint.timestamp(), places=5
            )

    def test_experiment_json_is_used_when_per_run_json_is_absent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout_root, metadata_dir, final_dir, _ = make_layout(root)
            experiment_path = metadata_dir.parent / "experiment.json"
            experiment_path.write_text(
                json.dumps(
                    {
                        "experiment": "sample",
                        "runs": {
                            "sample_00000": {
                                "status": "acquired",
                                "started_at": "2026-07-27T15:39:36+00:00",
                                "ended_at": "2026-07-27T15:40:36+00:00",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            windows = operando_data.get_exposure_windows(
                tpx_folder=layout_root / "tpx3Files" / "sample" / "sample_00000",
                metadata_dir=metadata_dir,
            )
            self.assertEqual(len(windows), 1)
            self.assertEqual(windows[0]["timing_metadata_kind"], "experiment_json")
            self.assertEqual(windows[0]["metadata_path"], str(experiment_path))
            self.assertEqual(windows[0]["duration_s"], 60.0)

    def test_detailed_manifest_remains_preferred_for_beam_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, metadata_dir, final_dir, _ = make_layout(root)
            base_timestamp = 1_700_000_000
            base_ns = int(base_timestamp * 1_000_000_000)
            frames = []
            cursor_ns = base_ns
            for index, duration_s in enumerate((2.0, 0.25)):
                duration_ns = int(duration_s * 1_000_000_000)
                end_ns = cursor_ns + duration_ns
                frames.append(
                    {
                        "frame_index": index,
                        "file_name": f"raw_{index:03d}.tpx3",
                        "start_unix_ns": cursor_ns,
                        "end_unix_ns": end_ns,
                        "midpoint_unix_ns": cursor_ns + duration_ns // 2,
                        "timing_source": "filesystem_birth_and_mtime",
                        "timing_quality": "measured",
                    }
                )
                cursor_ns = end_ns
            (metadata_dir / "tpx3_exposure_manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "lumacam-tpx3-exposure-manifest",
                        "schema_version": 1,
                        "experiment": "sample",
                        "run": "sample_00000",
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )
            (metadata_dir / "acquisition.json").write_text(
                json.dumps(
                    {
                        "started_at": "2030-01-01T00:00:00+00:00",
                        "ended_at": "2030-01-01T00:01:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            beam_path = root / "beam.csv"
            beam_path.write_text(
                "".join(
                    f"{datetime.fromtimestamp(base_timestamp + offset).strftime('%m/%d/%Y %I:%M:%S %p')},100\n"
                    for offset in range(-2, 8)
                ),
                encoding="utf-8",
            )

            _, plan, intensity_enabled = operando_data._correction_plan(
                final_dir,
                beamhistory_path=beam_path,
                reference_beam_integral_per_minute=6000.0,
                file_filter="3to1000",
            )
            self.assertTrue(intensity_enabled)
            self.assertEqual(plan[0]["exposure_timing_source"], "manifest")
            self.assertEqual(plan[0]["tpx_file_count"], 2)
            self.assertEqual(plan[0]["exposure_window_count"], 2)
            self.assertAlmostEqual(plan[0]["total_effective_exposure_s"], 2.25)
            self.assertAlmostEqual(plan[0]["relative_factor"], 1.0)

    def test_pipeline_runs_with_only_final_images_and_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, metadata_dir, final_dir, _ = make_layout(root)
            (metadata_dir / "acquisition.json").write_text(
                json.dumps(
                    {
                        "run": "sample_00000",
                        "status": "acquired",
                        "started_at": "2026-07-27T15:39:36+00:00",
                        "ended_at": "2026-07-27T15:40:36+00:00",
                    }
                ),
                encoding="utf-8",
            )
            result = operando_data.run_operando_pipeline(
                exp_dir=final_dir,
                beamhistory_path=None,
                reference_dir=None,
                reference_beamhistory_path=None,
                file_filter="3to1000",
                results_dir=root / "analysis" / "results_sample",
                corrected_dir=root / "analysis" / "results_sample" / "images" / "corrected",
                uncorrected_dir=root / "analysis" / "results_sample" / "images" / "uncorrected",
            )
            self.assertFalse(result["intensity_corrected"])
            self.assertFalse(result["reference_corrected"])
            self.assertEqual(result["timing_sources"], ["acquisition_json"])
            self.assertEqual(
                result["timing_metadata_paths"],
                [str(metadata_dir / "acquisition.json")],
            )
            self.assertEqual(result["beam_log"][0]["exposure_timing_source"], "acquisition_json")
            processing_metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
            timing_extra = processing_metadata["sections"]["obint_correction"]["extra"]
            self.assertEqual(timing_extra["timing_sources"], ["acquisition_json"])
            corrected = next(Path(result["corrected_dir"]).glob("*.tif*"))
            np.testing.assert_allclose(tifffile.imread(corrected), np.full((4, 4), 7.0))
            with tifffile.TiffFile(corrected) as tif:
                self.assertIn(65002, tif.pages[0].tags)


if __name__ == "__main__":
    unittest.main()

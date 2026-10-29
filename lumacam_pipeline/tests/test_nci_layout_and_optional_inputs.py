import os
import json
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import tifffile
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from lumacam_pipeline import backends, discover_dataset
from lumacam_pipeline.batch import _prepare_shared_reference


operando_data = backends.nci_data()
operando_plots = backends.nci_plots()
exsitu_pipeline = backends.exsitu_pipeline()
discover_exsitu_images = exsitu_pipeline.discover_exsitu_images
run_exsitu_pipeline = exsitu_pipeline.run_exsitu_pipeline


def make_v2_experiment(root, experiment, run_names, image_value=10.0, variants=1):
    layout_root = root / "data" / "experiments"
    (layout_root / "metadata").mkdir(parents=True, exist_ok=True)
    timestamp = 1_700_000_000
    for run_index, run_name in enumerate(run_names):
        tpx3_dir = layout_root / "tpx3Files" / experiment / run_name
        final_dir = layout_root / "final" / experiment / run_name
        tpx3_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)
        for tpx_index in range(3):
            path = tpx3_dir / f"raw_{tpx_index:03d}.tpx3"
            path.write_bytes(b"raw")
            os.utime(path, (timestamp + run_index * 60 + tpx_index, timestamp + run_index * 60 + tpx_index))
        for variant_index in range(variants):
            tifffile.imwrite(
                final_dir / f"{run_name}_sumImage_{variant_index + 3}to1000.tiff",
                np.full((4, 4), image_value + run_index, dtype=np.float32),
            )
    return layout_root


def write_exposure_manifest(layout_root, experiment, run_name, base_timestamp):
    metadata_dir = layout_root / "metadata" / experiment / run_name
    metadata_dir.mkdir(parents=True, exist_ok=True)
    durations = [2.0, 2.0, 0.25]
    frames = []
    cursor_ns = int(base_timestamp * 1e9)
    for index, duration in enumerate(durations):
        duration_ns = int(duration * 1e9)
        end_ns = cursor_ns + duration_ns
        frames.append(
            {
                "frame_index": index,
                "file_name": f"raw_{index:03d}.tpx3",
                "start_unix_ns": cursor_ns,
                "end_unix_ns": end_ns,
                "midpoint_unix_ns": cursor_ns + duration_ns // 2,
                "duration_s": duration,
                "timing_source": "filesystem_birth_and_mtime",
                "timing_quality": "measured",
            }
        )
        cursor_ns = end_ns
    payload = {
        "schema": "lumacam-tpx3-exposure-manifest",
        "schema_version": 1,
        "experiment": experiment,
        "run": run_name,
        "frames": frames,
        "summary": {
            "file_count": len(frames),
            "total_effective_exposure_s": sum(durations),
        },
    }
    (metadata_dir / "tpx3_exposure_manifest.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )


def write_processing_exclusion(layout_root, experiment, run_name, frame_index=2):
    metadata_dir = layout_root / "metadata" / experiment / run_name
    manifest_path = metadata_dir / "tpx3_exposure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = dict(manifest["frames"][frame_index])
    frame.update(
        {
            "reason": "missing_global_timestamp",
            "converter_message": "Supplied file does not contain a global timestamp!",
            "raw_tpx3_retained": True,
        }
    )
    input_duration = float(manifest["summary"]["total_effective_exposure_s"])
    excluded_duration = float(frame["duration_s"])
    payload = {
        "schema": "lumacam-processing-exclusions",
        "schema_version": 1,
        "experiment": experiment,
        "run": run_name,
        "policy": "manifest-proven-final-partial-missing-global-timestamp-only",
        "source_exposure_manifest": str(manifest_path),
        "raw_tpx3_retained": True,
        "input_frame_count": len(manifest["frames"]),
        "processed_frame_count": len(manifest["frames"]) - 1,
        "excluded_frame_count": 1,
        "input_effective_exposure_s": input_duration,
        "excluded_effective_exposure_s": excluded_duration,
        "included_effective_exposure_s": input_duration - excluded_duration,
        "excluded_frames": [frame],
    }
    path = metadata_dir / "processing_exclusions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_run_processing_exclusion(layout_root, experiment, run_name):
    metadata_dir = layout_root / "metadata" / experiment / run_name
    manifest_path = metadata_dir / "tpx3_exposure_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames = []
    for source in manifest["frames"]:
        frame = dict(source)
        frame.update({"reason": "manual_run_exclusion", "raw_tpx3_retained": True})
        frames.append(frame)
    input_duration = float(manifest["summary"]["total_effective_exposure_s"])
    payload = {
        "schema": "lumacam-processing-exclusions",
        "schema_version": 1,
        "experiment": experiment,
        "run": run_name,
        "run_excluded": True,
        "run_exclusion_reason": "manual_run_exclusion",
        "policy": "operator-specified-whole-run-exclusion",
        "source_exposure_manifest": str(manifest_path),
        "raw_tpx3_retained": True,
        "input_frame_count": len(frames),
        "processed_frame_count": 0,
        "excluded_frame_count": len(frames),
        "input_effective_exposure_s": input_duration,
        "excluded_effective_exposure_s": input_duration,
        "included_effective_exposure_s": 0.0,
        "excluded_frames": frames,
    }
    path = metadata_dir / "processing_exclusions.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_constant_beam_history(path, base_timestamp, value=100.0):
    lines = []
    for offset in range(-10, 21):
        timestamp = datetime.fromtimestamp(base_timestamp + offset)
        lines.append(f"{timestamp.strftime('%m/%d/%Y %I:%M:%S %p')},{value}\n")
    path.write_text("".join(lines), encoding="utf-8")


class BeamHistoryParsingTests(unittest.TestCase):
    def test_folder_combines_multiple_beam_history_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "beamhistory"
            folder.mkdir()
            (folder / "part_1.dat").write_text(
                "2026-07-20.12:00:00 100\n"
                "2026-07-20.12:00:05 110\n"
                "2026-07-20.12:00:10 120\n",
                encoding="utf-8",
            )
            (folder / "part_2.csv").write_text(
                "2026-07-20T12:00:10,125\n"
                "2026-07-20T12:00:15,130\n"
                "2026-07-20T12:00:20,140\n",
                encoding="utf-8",
            )

            with self.assertWarnsRegex(UserWarning, "overlapping timestamp"):
                beam = operando_data.load_beamhistory(folder, warn_on_gaps=False)

            self.assertEqual(len(beam), 5)
            np.testing.assert_allclose(beam["beam_count"], [100, 110, 125, 130, 140])
            self.assertEqual(beam.attrs["beamhistory_cadence"]["source_file_count"], 2)
            self.assertEqual(beam.attrs["beamhistory_cadence"]["overlapping_row_count"], 1)
            self.assertEqual(len(beam.attrs["beamhistory_source_files"]), 2)

    def test_mixed_human_readable_timestamp_formats_are_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "beam.csv"
            path.write_text(
                "07/20/2026 12:00:00 PM,100\n"
                "2026-07-20.12:00:05,110\n"
                "2026-07-20T12:00:10,120\n"
                "2026/07/20 12:00:15,130\n"
                "20.07.2026 12:00:20,140\n",
                encoding="utf-8",
            )
            beam = operando_data.load_beamhistory(path, warn_on_gaps=False)

            self.assertEqual(len(beam), 5)
            self.assertEqual(beam["timestamp"].iloc[0], datetime(2026, 7, 20, 12, 0, 0))
            self.assertEqual(beam["timestamp"].iloc[-1], datetime(2026, 7, 20, 12, 0, 20))
            np.testing.assert_allclose(beam["beam_count"], [100, 110, 120, 130, 140])
            self.assertEqual(
                beam.attrs["beamhistory_cadence"]["nominal_interval_s"], 5.0
            )

    def test_new_whitespace_format_and_60_second_integration(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "beam.dat"
            start = datetime(2026, 7, 20, 12, 0, 0)
            path.write_text(
                "".join(
                    f"{(start + timedelta(seconds=offset)).strftime('%Y-%m-%d.%H:%M:%S')} 100\n"
                    for offset in range(0, 61, 5)
                ),
                encoding="utf-8",
            )
            beam = operando_data.load_beamhistory(path)
            windows = [
                {
                    "start": start + timedelta(seconds=offset),
                    "end": start + timedelta(seconds=offset + 2),
                    "midpoint": start + timedelta(seconds=offset + 1),
                    "duration_s": 2.0,
                }
                for offset in range(0, 60, 2)
            ]

            self.assertEqual(len(windows), 30)
            self.assertAlmostEqual(
                operando_data.mean_beam_integral_for_windows(beam, windows),
                6000.0,
            )

    def test_unix_units_and_timezone_aware_iso_are_accepted(self):
        base = 1_700_000_000
        cases = {
            "seconds": [base, base + 5, base + 10],
            "milliseconds": [base * 1000, (base + 5) * 1000, (base + 10) * 1000],
            "microseconds": [base * 1_000_000, (base + 5) * 1_000_000, (base + 10) * 1_000_000],
            "nanoseconds": [base * 1_000_000_000, (base + 5) * 1_000_000_000, (base + 10) * 1_000_000_000],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = [datetime.fromtimestamp(base + offset) for offset in (0, 5, 10)]
            for name, timestamps in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.csv"
                    path.write_text(
                        "".join(f"{timestamp},{100 + index}\n" for index, timestamp in enumerate(timestamps)),
                        encoding="utf-8",
                    )
                    beam = operando_data.load_beamhistory(path, warn_on_gaps=False)
                    self.assertEqual(beam["timestamp"].tolist(), expected)

            iso_path = root / "aware_iso.csv"
            iso_path.write_text(
                "2023-11-14T22:13:20Z,100\n"
                "2023-11-14T22:13:25Z,101\n"
                "2023-11-14T22:13:30Z,102\n",
                encoding="utf-8",
            )
            iso = operando_data.load_beamhistory(iso_path, warn_on_gaps=False)
            self.assertEqual(iso["timestamp"].tolist(), expected)

    def test_long_timestamp_gap_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "gapped.dat"
            path.write_text(
                "2026-07-20.12:00:00 100\n"
                "2026-07-20.12:00:05 101\n"
                "2026-07-20.12:00:20 102\n"
                "2026-07-20.12:00:25 103\n",
                encoding="utf-8",
            )
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                beam = operando_data.load_beamhistory(path)

            cadence = beam.attrs["beamhistory_cadence"]
            self.assertEqual(cadence["nominal_interval_s"], 5.0)
            self.assertEqual(cadence["long_gap_count"], 1)
            self.assertEqual(cadence["longest_gap_s"], 15.0)
            self.assertTrue(any("longest gap is 15 s" in str(item.message) for item in seen))


class NciLayoutCompatibilityTests(unittest.TestCase):

    def test_v2_resolver_omits_whole_run_processing_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = ["sample_00006", "sample_00007", "sample_00008"]
            layout_root = make_v2_experiment(root, "sample", runs)
            for index, run in enumerate(runs):
                write_exposure_manifest(
                    layout_root, "sample", run, 1_700_000_000 + index * 60
                )
            write_run_processing_exclusion(layout_root, "sample", runs[1])

            layout = operando_data.resolve_nci_experiment(
                layout_root / "final" / "sample"
            )

            self.assertEqual(
                [run["name"] for run in layout["runs"]], [runs[0], runs[2]]
            )
            self.assertEqual(layout["excluded_runs"], [runs[1]])

            dataset = discover_dataset(layout_root / "final" / "sample")
            self.assertEqual({product.run for product in dataset.image_products()}, {runs[0], runs[2]})
            self.assertEqual(dataset.metadata["excluded_runs"], [runs[1]])
            self.assertEqual(
                {record.group_label for record in discover_exsitu_images(layout_root / "final" / "sample")},
                {runs[0], runs[2]},
            )

    def test_exsitu_cannot_bypass_exclusions_with_flat_variant_or_empty_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = "sample_00007"
            layout_root = make_v2_experiment(root, "sample", [run])
            write_exposure_manifest(layout_root, "sample", run, 1_700_000_000)
            exclusion_path = write_run_processing_exclusion(layout_root, "sample", run)
            final_root = layout_root / "final" / "sample"
            variant_dir = final_root / "3to1000"
            variant_dir.mkdir()
            tifffile.imwrite(variant_dir / f"{run}_sumImage_3to1000.tiff", np.ones((4, 4)))
            for directory in (final_root, variant_dir):
                with self.assertRaisesRegex(ValueError, "No non-excluded"):
                    discover_exsitu_images(directory)
            exclusion_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "processing-exclusion schema"):
                discover_exsitu_images(variant_dir)

    def test_reference_cache_changes_when_a_run_is_excluded_without_tiff_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = ["reference_00000", "reference_00001"]
            layout_root = make_v2_experiment(root, "reference", runs, image_value=10.0)
            for index, run in enumerate(runs):
                write_exposure_manifest(layout_root, "reference", run, 1_700_000_000 + index * 60)
            config = {"corrections": {
                "reference_dir": str(layout_root / "final" / "reference"),
                "file_filter": "3to1000", "reference_bin_y": 1, "reference_bin_x": 1,
            }}
            before = _prepare_shared_reference(root, config, root / "batch.log")
            self.assertAlmostEqual(float(tifffile.imread(before).mean()), 10.5)
            write_run_processing_exclusion(layout_root, "reference", runs[1])
            with self.assertRaisesRegex(ValueError, "Reference set"):
                _prepare_shared_reference(root, config, root / "batch.log")
            config["shared_setup"] = {"reference_set": "excluded_run"}
            after = _prepare_shared_reference(root, config, root / "batch.log")
            self.assertNotEqual(before, after)
            self.assertAlmostEqual(float(tifffile.imread(after).mean()), 10.0)

    def test_manifest_windows_weight_short_final_frame_by_its_actual_duration(self):
        base = datetime(2025, 1, 1, 12, 0, 0)
        beamhistory = operando_data.pd.DataFrame(
            {
                "timestamp": [
                    base,
                    base + timedelta(seconds=2),
                    base + timedelta(seconds=3),
                ],
                "beam_count": [100.0, 200.0, 200.0],
            }
        )
        windows = [
            {
                "start": base,
                "end": base + timedelta(seconds=2),
                "midpoint": base + timedelta(seconds=1),
                "duration_s": 2.0,
            },
            {
                "start": base + timedelta(seconds=2),
                "end": base + timedelta(seconds=2.25),
                "midpoint": base + timedelta(seconds=2.125),
                "duration_s": 0.25,
            },
        ]
        value = operando_data.mean_beam_integral_for_windows(beamhistory, windows)
        self.assertAlmostEqual(value, (6000.0 * 2.0 + 12000.0 * 0.25) / 2.25)

    def test_processing_exclusion_removes_tail_from_beam_integral(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name = "sample_00000"
            layout_root = make_v2_experiment(root, "sample", [run_name])
            base_timestamp = 1_700_000_000
            write_exposure_manifest(layout_root, "sample", run_name, base_timestamp)
            exclusions_path = write_processing_exclusion(
                layout_root, "sample", run_name
            )
            metadata_dir = layout_root / "metadata" / "sample" / run_name
            windows = operando_data.get_exposure_windows(metadata_dir=metadata_dir)
            beam = operando_data.pd.DataFrame(
                {
                    "timestamp": [
                        datetime.fromtimestamp(base_timestamp - 1),
                        datetime.fromtimestamp(base_timestamp + 4),
                        datetime.fromtimestamp(base_timestamp + 5),
                    ],
                    "beam_count": [100.0, 1000.0, 1000.0],
                }
            )

            self.assertEqual([window["file_name"] for window in windows], [
                "raw_000.tpx3", "raw_001.tpx3"
            ])
            self.assertAlmostEqual(
                operando_data.mean_beam_integral_for_windows(beam, windows),
                6000.0,
            )
            beam_metrics = operando_data.beam_metrics_for_windows(beam, windows)
            self.assertAlmostEqual(beam_metrics["total_beam_integral"], 400.0)
            self.assertEqual(len(beam_metrics["tpx3_beam_integrals"]), 2)
            self.assertAlmostEqual(
                beam_metrics["tpx3_beam_integrals"][0]["total_beam_integral"],
                200.0,
            )
            provenance = operando_data.exposure_group_provenance(windows)
            self.assertEqual(provenance["tpx_file_count"], 2)
            self.assertEqual(provenance["input_tpx_file_count"], 3)
            self.assertEqual(provenance["excluded_tpx_file_count"], 1)
            self.assertEqual(provenance["excluded_tpx_files"], ["raw_002.tpx3"])
            self.assertEqual(provenance["processing_exclusions_path"], str(exclusions_path))

    def test_processing_exclusion_must_match_exposure_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name = "sample_00000"
            layout_root = make_v2_experiment(root, "sample", [run_name])
            write_exposure_manifest(layout_root, "sample", run_name, 1_700_000_000)
            exclusions_path = write_processing_exclusion(
                layout_root, "sample", run_name
            )
            exclusions = json.loads(exclusions_path.read_text(encoding="utf-8"))
            exclusions["excluded_frames"][0]["file_name"] = "unknown.tpx3"
            exclusions_path.write_text(json.dumps(exclusions), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "absent from the exposure manifest"):
                operando_data.get_exposure_windows(
                    metadata_dir=layout_root / "metadata" / "sample" / run_name
                )

    def test_v2_resolver_pairs_separate_raw_and_final_trees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout_root = make_v2_experiment(root, "sample", ["sample_00000"], variants=2)
            selected = layout_root / "final" / "sample"

            layout = operando_data.resolve_nci_experiment(selected)
            self.assertEqual(layout["layout_type"], "v2")
            self.assertEqual(layout["experiment"], "sample")
            self.assertEqual(len(layout["runs"]), 1)
            self.assertEqual(
                Path(layout["runs"][0]["tpx_dir"]),
                layout_root / "tpx3Files" / "sample" / "sample_00000",
            )

            _, plan, intensity_enabled = operando_data._correction_plan(selected)
            self.assertFalse(intensity_enabled)
            self.assertEqual(len(plan), 2)
            self.assertEqual([row["tpx_file_count"] for row in plan], [3, 3])
            self.assertEqual(plan[0]["time"], plan[1]["time"])

    def test_current_photon_range_final_layout_pairs_each_tiff_with_its_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout_root = make_v2_experiment(
                root, "sample", ["sample_00000", "sample_00001"]
            )
            variant_dir = layout_root / "final" / "sample" / "3to1000"
            variant_dir.mkdir()
            for run_name in ("sample_00000", "sample_00001"):
                old_dir = layout_root / "final" / "sample" / run_name
                source = next(old_dir.glob("*.tiff"))
                source.replace(variant_dir / source.name)
                old_dir.rmdir()

            layout = operando_data.resolve_nci_experiment(variant_dir)
            self.assertEqual([run["name"] for run in layout["runs"]], ["sample_00000", "sample_00001"])
            self.assertTrue(all(len(run["image_files"]) == 1 for run in layout["runs"]))
            self.assertTrue(
                Path(layout["runs"][0]["image_files"][0]).name.startswith("sample_00000_")
            )

    def test_operando_prefers_manifest_after_raw_tpx3_files_are_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name = "sample_00000"
            layout_root = make_v2_experiment(root, "sample", [run_name])
            base_timestamp = 1_700_000_000
            write_exposure_manifest(layout_root, "sample", run_name, base_timestamp)
            exclusions_path = write_processing_exclusion(
                layout_root, "sample", run_name
            )
            shutil.rmtree(layout_root / "tpx3Files" / "sample" / run_name)
            beam_path = root / "beam.csv"
            write_constant_beam_history(beam_path, base_timestamp)

            result = operando_data.run_operando_pipeline(
                exp_dir=layout_root / "final" / "sample",
                beamhistory_path=beam_path,
                reference_beam_integral_per_minute=6000.0,
                reference_dir=None,
                reference_beamhistory_path=None,
                results_dir=root / "results_operando",
            )
            self.assertTrue(result["intensity_corrected"])
            self.assertEqual(result["beam_log"][0]["exposure_timing_source"], "manifest")
            self.assertAlmostEqual(result["beam_log"][0]["total_effective_exposure_s"], 4.0)
            self.assertEqual(result["beam_log"][0]["tpx_file_count"], 2)
            self.assertEqual(result["beam_log"][0]["input_tpx_file_count"], 3)
            self.assertEqual(result["beam_log"][0]["excluded_tpx_file_count"], 1)
            self.assertEqual(result["beam_log"][0]["excluded_tpx_files"], "raw_002.tpx3")
            self.assertAlmostEqual(result["beam_log"][0]["total_beam_integral"], 400.0)
            self.assertAlmostEqual(
                result["beam_log"][0]["effective_exposure_fraction"], 4.0 / 4.25
            )
            self.assertAlmostEqual(
                result["beam_log"][0]["relative_factor"], 4.0 / 4.25
            )
            self.assertEqual(
                len(json.loads(result["beam_log"][0]["tpx3_beam_integrals_json"])),
                2,
            )
            self.assertEqual(
                result["beam_log"][0]["processing_exclusions_path"],
                str(exclusions_path),
            )
            np.testing.assert_allclose(
                tifffile.imread(
                    Path(result["corrected_dir"])
                    / result["beam_log"][0]["corrected_filename"]
                ),
                np.full((4, 4), 10.625),
            )

    def test_operando_accepts_new_five_second_beam_history_format(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name = "sample_00000"
            layout_root = make_v2_experiment(root, "sample", [run_name])
            base_timestamp = 1_700_000_000
            write_exposure_manifest(layout_root, "sample", run_name, base_timestamp)
            beam_path = root / "beam.dat"
            beam_path.write_text(
                "".join(
                    f"{datetime.fromtimestamp(base_timestamp + offset).strftime('%Y-%m-%d.%H:%M:%S')} 100\n"
                    for offset in (-5, 0, 5, 10)
                ),
                encoding="utf-8",
            )

            result = operando_data.run_operando_pipeline(
                exp_dir=layout_root / "final" / "sample",
                beamhistory_path=beam_path,
                reference_beam_integral_per_minute=6000.0,
                reference_dir=None,
                reference_beamhistory_path=None,
                results_dir=root / "results_operando",
            )

            self.assertTrue(result["intensity_corrected"])
            self.assertAlmostEqual(result["beam_log"][0]["relative_factor"], 1.0)
            np.testing.assert_allclose(
                tifffile.imread(
                    Path(result["corrected_dir"])
                    / result["beam_log"][0]["corrected_filename"]
                ),
                np.full((4, 4), 10.0),
            )

    def test_operando_runs_without_beam_reference_or_background(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout_root = make_v2_experiment(root, "sample", ["sample_00000", "sample_00001"])
            results = root / "analysis" / "results_sample"
            corrected = results / "images" / "corrected"
            uncorrected = results / "images" / "uncorrected"

            result = operando_data.run_operando_pipeline(
                exp_dir=layout_root / "tpx3Files" / "sample",
                beamhistory_path=None,
                reference_dir=None,
                reference_beamhistory_path=None,
                results_dir=results,
                corrected_dir=corrected,
                uncorrected_dir=uncorrected,
            )

            self.assertFalse(result["intensity_corrected"])
            self.assertFalse(result["reference_corrected"])
            self.assertEqual(result["source_layout_type"], "v2")
            self.assertEqual(len(list(corrected.glob("*.tif*"))), 2)
            self.assertEqual(len(list(uncorrected.glob("*.tif*"))), 2)
            for path in corrected.glob("*.tif*"):
                np.testing.assert_allclose(tifffile.imread(path), np.full((4, 4), 10.0) if "00000" in path.name else np.full((4, 4), 11.0))
                with tifffile.TiffFile(path) as tif:
                    self.assertIn(65002, tif.pages[0].tags)

    def test_legacy_layout_is_still_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "legacy_sample"
            run = experiment / "legacy_sample_00000"
            (run / "tpx3Files").mkdir(parents=True)
            (run / "final").mkdir()
            (run / "tpx3Files" / "raw.tpx3").write_bytes(b"raw")
            tifffile.imwrite(run / "final" / "image.tif", np.ones((4, 4), dtype=np.float32))
            (experiment / "final").mkdir()
            tifffile.imwrite(
                experiment / "final" / "legacy_sum.tif",
                np.ones((4, 4), dtype=np.float32),
            )

            layout = operando_data.resolve_nci_experiment(experiment)
            self.assertEqual(layout["layout_type"], "legacy")
            self.assertEqual(len(layout["runs"]), 1)
            self.assertEqual(layout["runs"][0]["name"], "legacy_sample_00000")

            records = discover_exsitu_images(experiment)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].group_label, "legacy_sample_00000")

    def test_operando_keeps_ob_correction_when_beam_histories_are_blank(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout_root = make_v2_experiment(root, "sample", ["sample_00000"], image_value=10.0)
            make_v2_experiment(root, "open_beam", ["open_beam_00000"], image_value=2.0)
            result = operando_data.run_operando_pipeline(
                exp_dir=layout_root / "final" / "sample",
                beamhistory_path=None,
                reference_dir=layout_root / "final" / "open_beam",
                reference_beamhistory_path=None,
                results_dir=root / "results",
                reference_bin_y=1,
                reference_bin_x=1,
            )
            self.assertFalse(result["intensity_corrected"])
            self.assertTrue(result["reference_corrected"])
            output = next(Path(result["corrected_dir"]).glob("*.tif*"))
            np.testing.assert_allclose(tifffile.imread(output), np.full((4, 4), 5.0))

    def test_reference_image_is_normalized_for_its_excluded_exposure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout_root = make_v2_experiment(
                root, "sample", ["sample_00000"], image_value=10.0
            )
            make_v2_experiment(
                root,
                "open_beam",
                ["open_beam_00000"],
                image_value=2.0 * 4.0 / 4.25,
            )
            base_timestamp = 1_700_000_000
            write_exposure_manifest(
                layout_root, "sample", "sample_00000", base_timestamp
            )
            write_exposure_manifest(
                layout_root, "open_beam", "open_beam_00000", base_timestamp
            )
            write_processing_exclusion(
                layout_root, "open_beam", "open_beam_00000"
            )
            beam_path = root / "beam.csv"
            write_constant_beam_history(beam_path, base_timestamp)

            result = operando_data.run_operando_pipeline(
                exp_dir=layout_root / "final" / "sample",
                beamhistory_path=beam_path,
                reference_dir=layout_root / "final" / "open_beam",
                reference_beamhistory_path=beam_path,
                reference_bin_y=1,
                reference_bin_x=1,
                results_dir=root / "results",
            )

            output = next(Path(result["corrected_dir"]).glob("*.tif*"))
            np.testing.assert_allclose(
                tifffile.imread(output), np.full((4, 4), 5.0), rtol=1e-6
            )


class ExSituOptionalCorrectionTests(unittest.TestCase):
    def test_v2_discovery_and_no_optional_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            layout_root = make_v2_experiment(root, "exsitu", ["exsitu_00000", "exsitu_00001"])
            data_dir = layout_root / "final" / "exsitu"
            records = discover_exsitu_images(data_dir)
            self.assertEqual(len(records), 2)
            self.assertTrue(all(record.layout_type == "v2" for record in records))

            result = run_exsitu_pipeline(
                data_dir=data_dir,
                beamhistory_path=None,
                ob_path=None,
                bg_path=None,
                results_dir=root / "results",
                verbose=False,
            )
            self.assertFalse(result["intensity_corrected"])
            self.assertFalse(result["ob_corrected"])
            self.assertFalse(result["background_corrected"])
            self.assertIsNone(result["final_dir"])
            np.testing.assert_allclose(
                tifffile.imread(result["average_path"]),
                np.full((4, 4), 10.5, dtype=np.float32),
            )

    def test_external_background_gets_ob_correction_without_beam_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample = root / "sample"
            ob = root / "ob"
            background = root / "background"
            sample.mkdir()
            ob.mkdir()
            background.mkdir()
            tifffile.imwrite(sample / "sample.tif", np.full((4, 4), 10.0, dtype=np.float32))
            tifffile.imwrite(ob / "ob.tif", np.full((4, 4), 2.0, dtype=np.float32))
            tifffile.imwrite(background / "bg.tif", np.full((4, 4), 2.0, dtype=np.float32))

            result = run_exsitu_pipeline(
                data_dir=sample,
                beamhistory_path=None,
                ob_path=ob,
                bg_path=background,
                results_dir=root / "results",
                ob_bin_y=1,
                ob_bin_x=1,
                clip_negatives=False,
                verbose=False,
            )
            self.assertFalse(result["intensity_corrected"])
            self.assertTrue(result["ob_corrected"])
            self.assertTrue(result["background_corrected"])
            self.assertEqual(result["background_source_count"], 1)
            np.testing.assert_allclose(
                tifffile.imread(result["average_path"]),
                np.full((4, 4), 4.0, dtype=np.float32),
            )

    def test_exsitu_prefers_manifest_after_raw_tpx3_files_are_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_name = "exsitu_00000"
            layout_root = make_v2_experiment(root, "exsitu", [run_name])
            base_timestamp = 1_700_000_000
            write_exposure_manifest(layout_root, "exsitu", run_name, base_timestamp)
            write_processing_exclusion(layout_root, "exsitu", run_name)
            shutil.rmtree(layout_root / "tpx3Files" / "exsitu" / run_name)
            beam_path = root / "beam.csv"
            write_constant_beam_history(beam_path, base_timestamp)

            result = run_exsitu_pipeline(
                data_dir=layout_root / "final" / "exsitu",
                beamhistory_path=beam_path,
                reference_beam_integral_per_minute=6000.0,
                results_dir=root / "results_exsitu",
                verbose=False,
            )
            self.assertTrue(result["intensity_corrected"])
            rows = operando_data.load_obint_beam_log(result["beam_log_path"])
            self.assertEqual(rows[0]["exposure_timing_source"], "manifest")
            self.assertAlmostEqual(rows[0]["total_effective_exposure_s"], 4.0)
            self.assertEqual(rows[0]["tpx_file_count"], 2)
            self.assertEqual(rows[0]["input_tpx_file_count"], 3)
            self.assertEqual(rows[0]["excluded_tpx_file_count"], 1)
            self.assertEqual(rows[0]["excluded_tpx_files"], "raw_002.tpx3")
            self.assertAlmostEqual(rows[0]["total_beam_integral"], 400.0)
            self.assertAlmostEqual(rows[0]["effective_exposure_fraction"], 4.0 / 4.25)
            self.assertAlmostEqual(rows[0]["relative_factor"], 4.0 / 4.25)


class NciOnlyPlotTests(unittest.TestCase):
    def test_final_nci_panel_saves_without_electrochemistry(self):
        def plot_images_func(img_series, img_indices, **_kwargs):
            fig, axes = plt.subplots(1, len(img_indices), squeeze=False)
            for axis, image_index in zip(axes[0], img_indices):
                axis.imshow(img_series[image_index])
            return fig, axes

        with tempfile.TemporaryDirectory() as temporary:
            times = [datetime(2025, 1, 1) + timedelta(minutes=index) for index in range(3)]
            images = [np.full((4, 4), index, dtype=np.float32) for index in range(3)]
            event_context = {
                "events": [],
                "event_times": [],
                "img_indices": [],
                "matched_count": 0,
                "total_events_found": 0,
            }
            result = operando_plots.save_selected_event_outputs(
                event_context=event_context,
                selected_image_indices=[0, 2],
                img_datetimes=times,
                img_series=images,
                avg_thickness_series=np.asarray([0.0, 1.0, 2.0]),
                echem_datetimes=[],
                potential_series=np.asarray([], dtype=float),
                echem_thickness_series=np.asarray([], dtype=float),
                output_plot_dir=temporary,
                exp_name="nci_only",
                roi_small={"x0": 2, "y0": 2, "r0": 1},
                event_plot_kwargs={},
                plot_images_func=plot_images_func,
                save_potential=False,
            )
            self.assertTrue(Path(result["saved_paths"]["images"]).is_file())
            self.assertTrue(Path(result["saved_paths"]["summary"]).is_file())
            self.assertNotIn("potential", result["saved_paths"])


if __name__ == "__main__":
    unittest.main()

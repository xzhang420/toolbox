"""Spatial methods, presentation-only windows, native and matched CSV exports."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout
import io
import json

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from lumacam_pipeline.measurements import measure_stack, region_mask
from lumacam_pipeline.time_window import bounds, image_indices
from lumacam_pipeline.csv_export import native_tables, comparison_table
from lumacam_pipeline.event_selection import selected_records
from lumacam_pipeline.visualization import open_visualization, load_visualization_settings, save_visualization_settings
from lumacam_pipeline.visualization_gui import VisualizationGUI
from lumacam_pipeline.handover import file_digest
from lumacam_pipeline.section import SectionState
from lumacam_pipeline import backends


class MeasurementTests(unittest.TestCase):
    def test_uniform_and_partial_coverage_area_identity(self):
        image = np.array([[0., 0.], [4., 4.]])
        cfg = dict(methods=["mean", "area_normalized", "active_mean", "median", "trimmed_mean"],
                   electrode_area_mm2=4., active_threshold=.1)
        table, _, cols, _ = measure_stack([image], [None], cfg, pixel_size_um=1000, unit="um")
        self.assertEqual(table[cols["mean"]][0], 2.)
        self.assertEqual(table[cols["area_normalized"]][0], 2.)
        self.assertEqual(table[cols["active_mean"]][0], 4.)
        self.assertEqual(table.active_fraction_of_valid[0], .5)
        cfg["electrode_area_mm2"] = 8
        table, _, _, _ = measure_stack([image], [None], cfg, pixel_size_um=1000, unit="µm")
        self.assertEqual(table.nci_area_normalized_um[0], 1.)
        np.testing.assert_array_equal(image, [[0, 0], [4, 4]])

    def test_negative_nonfinite_empty_and_outlier_values(self):
        cfg = dict(methods=["mean", "area_normalized", "active_mean"], electrode_area_mm2=4, active_threshold=100)
        table, _, _, _ = measure_stack([np.array([[-1., 1.], [np.nan, 2.]])], [None], cfg,
                                       pixel_size_um=1000, unit="um")
        self.assertAlmostEqual(table.nci_mean_um[0], 2/3)
        self.assertTrue(np.isnan(table.nci_area_normalized_um[0]))
        self.assertTrue(np.isnan(table.nci_active_mean_um[0]))
        self.assertEqual(table.quality_flags[0], "nonfinite_pixels;empty_active_area")
        table, _, _, _ = measure_stack([np.array([[-10., 1., 1., 1., 100.]])], [],
            dict(methods=["mean", "median", "trimmed_mean"], trim_fraction=.2))
        self.assertEqual(table.image_trimmed_mean_value[0], 1)
        self.assertEqual(table.image_median_value[0], 1)
        self.assertAlmostEqual(table.image_mean_value[0], 18.6)
        table, _, _, _ = measure_stack([np.full((2, 2), np.nan), np.zeros((2, 2))], [], {})
        self.assertIn("no_valid_pixels", table.quality_flags[0])
        self.assertEqual(table.quality_flags[1], "all_zero_region")

    def test_regions_and_validation(self):
        self.assertEqual(region_mask((5, 5), {}).sum(), 25)
        self.assertEqual(region_mask((5, 5), {"roi": dict(x0=2, y0=2, r0=1)}).sum(), 5)
        self.assertEqual(region_mask((5, 5), dict(region="rectangle", roi=dict(x_min=1, x_max=3, y_min=1, y_max=4))).sum(), 6)
        points = [[1, 1], [3, 1], [3, 3], [1, 3]]
        for vertices in (points, points[::-1]):
            self.assertEqual(region_mask((5, 5), dict(region="polygon", roi=dict(vertices=vertices))).sum(), 9)
        with self.assertRaises(ValueError):
            region_mask((5, 5), {"roi": dict(x0=100, y0=100, r0=1)})
        for cfg in (dict(methods=["active_mean"]), dict(methods=["area_normalized"], electrode_area_mm2=1),
                    dict(trim_fraction=.5), dict(primary_method="median"), dict(methods=["unknown"])):
            with self.assertRaises(ValueError):
                measure_stack([np.ones((2, 2))], [], cfg)


class WindowExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root/"images"; self.source.mkdir()
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(5):
            tifffile.imwrite(self.source/f"frame_{i}.tif", np.full((12, 12), i+1, np.float32), metadata=None,
                description=f"AcquisitionTime:{(self.start+timedelta(hours=i)).isoformat()}")
        self.config = {"reports": {"enabled": False}, "visualization": {"export_layout": "flat"},
            "input": {"unit": "um", "pixel_size_um": 100}, "image_measurements": {"methods": ["mean", "median"]},
            "combined_plot": {"image_methods": ["mean", "median"], "pause_gap_minutes": 180, "dpi": 70},
            "selection": {"indices": "0,1,2,3,4"}, "image_plot": {"dpi": 70}}
        self.view = open_visualization(self.source, config=self.config, preset="operando_nci")

    def run_ok(self, step):
        result = self.view.run(step)
        self.assertEqual(result.state, SectionState.COMPLETED, result.message)
        return result

    def echem(self):
        for name, value in {"echem_times": [self.start+timedelta(minutes=30*i) for i in range(9)],
            "echem_potential": np.arange(9)*.1, "echem_thickness": np.arange(9)+10.,
            "echem_current": [-1.]*9, "echem_capacity": np.arange(9)/3.66, "echem_cycles": [1]*9}.items():
            self.view.artifacts.put(name, value, "electrochemistry")

    def test_csv_native_and_window_values_no_reset(self):
        before = {p: file_digest(p) for p in self.source.glob("*.tif")}
        self.run_ok("image_measurements"); self.echem()
        self.run_ok("csv_export")
        paths = self.view.artifacts.require("csv_export_paths")
        image_full = pd.read_csv(paths["thickness"])
        ec_full = pd.read_csv(paths["electrochemistry"])
        self.assertEqual(len(image_full), 5); self.assertEqual(len(ec_full), 9)
        self.assertIn("nci_median_um", image_full)
        self.assertIn("capacity_mAh", ec_full)
        self.assertTrue(image_full.image_time_utc.str.endswith("+00:00").all())
        measured = self.view.artifacts.require("image_metrics_table")
        self.view.configure("time_window", enabled=True, start=1, end=3)
        self.run_ok("time_window")
        self.assertIs(self.view.artifacts.require("image_metrics_table"), measured)
        self.view.configure("csv_export", scope="window")
        self.run_ok("csv_export")
        paths = self.view.artifacts.require("csv_export_paths")
        cropped = pd.read_csv(paths["electrochemistry"])
        self.assertEqual(cropped.ec_thickness_um.iloc[0], 12.)
        self.assertEqual(cropped.elapsed_h.iloc[0], 1.)
        pd.testing.assert_frame_equal(cropped, ec_full.iloc[2:7].reset_index(drop=True))
        self.assertEqual(pd.read_csv(paths["thickness"]).frame.tolist(), [1, 2, 3])
        self.assertEqual(before, {p: file_digest(p) for p in before})
        restored = load_visualization_settings(save_visualization_settings(self.view))
        self.assertEqual(bounds(restored), bounds(self.view))
        self.assertEqual(restored.config["image_measurements"]["methods"], ["mean", "median"])

    def test_matching_gaps_endpoints_and_tolerance(self):
        self.run_ok("image_measurements"); self.echem()
        images, ec = native_tables(self.view)
        ec = ec.iloc[[1, 4, 7]].copy()  # .5h, 2h, 3.5h; images 0,1,2,3,4
        matched = comparison_table(images, ec, 2000)
        self.assertEqual(matched.matched.tolist(), [False, True, True, True, False])
        self.assertEqual(matched.time_offset_s.iloc[1], 1800.)
        self.assertEqual(comparison_table(images, ec, 10).matched.tolist(), [False, False, True, False, False])
        for tolerance in (-1, float("nan")):
            with self.assertRaises(ValueError):
                comparison_table(images, ec, tolerance)

    def test_absolute_window_category_flags_and_invalid_bounds(self):
        self.view.configure("time_window", enabled=True, mode="absolute", start="2026-01-01T02:00:00+01:00",
                            end="2026-01-01T04:00:00+01:00", apply_panels=False)
        self.assertEqual(bounds(self.view), (self.start+timedelta(hours=1), self.start+timedelta(hours=3)))
        self.assertEqual(image_indices(self.view, range(5), "panels"), list(range(5)))
        self.assertEqual(image_indices(self.view, range(5), "videos"), [1, 2, 3])
        self.view.configure("time_window", mode="elapsed", start=10, end=12)
        self.assertEqual(self.view.run("time_window").state, SectionState.FAILED)

    def test_windowed_panels_and_curves_keep_labels_and_methods(self):
        import matplotlib.pyplot as plt
        self.run_ok("image_measurements"); self.echem(); self.run_ok("selection")
        self.view.configure("time_window", enabled=True, start=1, end=3)
        self.assertEqual([row["label"] for row in selected_records(self.view, "panels")], ["b", "c", "d"])
        with patch("matplotlib.pyplot.close"):
            self.run_ok("image_plot")
            text = [t.get_text() for ax in plt.gcf().axes for t in ax.texts]
            self.assertTrue(all(label in text for label in ["b", "c", "d"]))
            self.run_ok("combined_plot")
            fig = plt.gcf()
            self.assertEqual(fig.axes[0].get_xlim(), (1., 3.))
            labels = [line.get_label() for line in fig.axes[1].lines]
            self.assertIn("ROI median (µm)", labels)
            self.assertIn("ROI mean (µm)", labels)
            method_lines = [line for line in fig.axes[1].lines if line.get_label().startswith("ROI")]
            self.assertEqual(len({line.get_color() for line in method_lines}), len(method_lines))
            ec_line = next(line for line in fig.axes[1].lines if line.get_label() == "EC thickness")
            np.testing.assert_array_equal(ec_line.get_ydata(), np.arange(9)+10.)
        plt.close("all")

    def test_gui_roundtrip_window_does_not_invalidate_measurements(self):
        gui = VisualizationGUI("operando_nci"); gui.session = self.view; gui._populate()
        gui.region.value = "rectangle"; gui.rectangle.value = "1, 9, 2, 10"
        gui.methods.value = ("mean", "median"); gui.primary_method.value = "median"
        gui.plot_methods.value = ("mean", "median")
        gui._sync(); self.run_ok("image_measurements")
        table = self.view.artifacts.require("image_metrics_table")
        gui.window_slider.value = (1., 3.); gui._sync()
        self.assertIs(table, self.view.artifacts.require("image_metrics_table"))
        restored = VisualizationGUI(); restored.session = load_visualization_settings(save_visualization_settings(self.view))
        restored._populate(); restored._sync()
        self.assertEqual(restored.region.value, "rectangle")
        self.assertEqual(restored.primary_method.value, "median")
        self.assertEqual(restored.window_slider.value, (1., 3.))
        gui.reset_window()
        self.assertFalse(self.view.config["time_window"]["enabled"])

    def test_optional_echem_and_untimed_images_export(self):
        self.view.artifacts.put("image_times", [None]*5, "dataset")
        self.run_ok("image_measurements"); self.run_ok("csv_export")
        paths = self.view.artifacts.require("csv_export_paths")
        self.assertEqual(set(paths), {"thickness"})
        self.assertTrue(pd.read_csv(paths["thickness"]).elapsed_h.isna().all())
        self.view.configure("time_window", enabled=True)
        self.assertEqual(self.view.run("time_window").state, SectionState.FAILED)

    def test_video_passes_full_data_and_separate_window(self):
        self.run_ok("image_measurements"); self.echem()
        self.view.configure("time_window", enabled=True, start=1, end=3)
        self.view.configure("combined_video", enabled=True)
        with patch.object(backends.nci_plots(), "make_potential_image_video_uniform_time") as make:
            self.run_ok("combined_video")
        self.assertEqual(len(make.call_args.args[0]), 9)
        self.assertEqual(len(make.call_args.args[2]), 5)
        np.testing.assert_array_equal(make.call_args.kwargs["echem_thickness_series"], np.arange(9)+10.)
        self.assertEqual(make.call_args.kwargs["time_window"], bounds(self.view))

    def test_cli_new_settings_and_reset_window(self):
        from lumacam_pipeline.cli import main
        settings = save_visualization_settings(self.view)
        with redirect_stdout(io.StringIO()) as output:
            status = main(["--config", str(settings), "visualize", "--set", "time_window.enabled=false",
                "--set", 'image_measurements.methods=["mean","median"]', "--set", 'csv_export.scope="full"',
                "--sections", "time_window", "image_measurements", "csv_export"])
        self.assertEqual(status, 0, output.getvalue())
        self.assertFalse(load_visualization_settings(settings).config["time_window"]["enabled"])
        payload = json.loads(output.getvalue())
        self.assertEqual([row["state"] for row in payload["results"]], ["completed"]*3)

    def test_old_circular_settings_survive_gui_load(self):
        self.view.configure("image_measurements", roi=dict(x0=5, y0=6, r0=4))
        gui = VisualizationGUI(); gui.session = self.view; gui._populate(); gui._sync()
        self.assertEqual(gui.region.value, "circle")
        self.assertEqual(self.view.config["image_measurements"]["roi"], dict(x0=5, y0=6, r0=4))

    def test_individual_figures_window_accepts_list_indices(self):
        self.view.configure("individual_image_plots", indices=[0, 1, 2, 3, 4], dpi=40)
        self.view.configure("time_window", enabled=True, start=1, end=2)
        self.run_ok("individual_image_plots")
        paths = self.view.artifacts.require("individual_image_plot_paths")
        self.assertEqual(len(paths), 2)
        self.assertTrue(paths[0].name.startswith("frame_00001_"))
        self.assertFalse(self.view.artifacts.has("selected_image_indices"))

    def test_counts_cannot_share_axis_with_ec_micrometres(self):
        self.view.artifacts.put("image_unit", "count", "dataset")
        self.run_ok("image_measurements"); self.echem()
        self.view.configure("combined_plot", thickness_traces="both")
        outcome = self.view.run("combined_plot")
        self.assertEqual(outcome.state, SectionState.FAILED)
        self.assertIn("Do not relabel counts", outcome.message)
        self.view.configure("combined_plot", thickness_traces="nci")
        self.run_ok("combined_plot")


if __name__ == "__main__":
    unittest.main()

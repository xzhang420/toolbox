"""New renderer compatibility, explicit playback gaps and settings round trips."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline.visualization import (open_visualization, FrameFigureSection,
    FrameIndividualSection, _draw_frame, save_visualization_settings, load_visualization_settings)
from lumacam_pipeline.visualization_gui import VisualizationGUI
from lumacam_pipeline.video_timing import video_plan
from lumacam_pipeline.handover import file_digest


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = self.root / "input"
        source.mkdir()
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i, seconds in enumerate([0, 60, 120, 600]):
            tifffile.imwrite(source / f"frame_{i}.tif",
                np.linspace(0, i+1, 144, dtype=np.float32).reshape(12, 12), metadata=None,
                description=f"AcquisitionTime:{(self.start+timedelta(seconds=seconds)).isoformat()}")
        self.view = open_visualization(source, config={"reports": {"enabled": False},
            "visualization": {"export_layout": "flat"}, "input": {"unit": "um", "pixel_size_um": 20},
            "image_plot": {"dpi": 50}, "individual_image_plots": {"dpi": 50},
            "image_video": {"dpi": 50, "figsize": [4, 4]}})

    def run_ok(self, section):
        result = self.view.run(section)
        self.assertEqual(result.state.value, "completed", result.message)
        return result

    def test_sequence_and_elapsed_gap_plan(self):
        rows, settings = video_plan(self.view, {"fps": 10})
        self.assertEqual([r["image_index"] for r in rows], [0, 1, 2, 3])
        self.assertEqual(rows[-1]["playback_seconds"], .3)
        rows, settings = video_plan(self.view, {"fps": 10, "timing_mode": "elapsed_time", "dt_seconds": 60})
        self.assertEqual(len(rows), 11)
        self.assertEqual([r["image_index"] for r in rows], [0, 1, 2] + [None]*7 + [3])
        self.assertEqual(settings["image_tolerance_seconds"], 45)
        self.assertEqual(settings["experimental_seconds_per_playback_second"], 600)
        self.assertEqual(rows[3]["acquisition_utc"], "")

    def test_gap_only_window_preserves_experiment_origin(self):
        self.view.configure("time_window", enabled=True, start=180/3600, end=300/3600)
        rows, _ = video_plan(self.view, {"timing_mode": "elapsed_time", "dt_seconds": 60})
        self.assertEqual([r["elapsed_seconds"] for r in rows], [180, 240, 300])
        self.assertTrue(all(r["image_index"] is None for r in rows))

    def test_timing_invalid_inputs_and_no_timestamps(self):
        for extra in ({"dt_seconds": 0}, {"dt_seconds": float("nan")},
                      {"dt_seconds": .000001}, {"image_tolerance_seconds": -1}, {"fps": float("inf")}):
            with self.assertRaises(ValueError):
                video_plan(self.view, {"timing_mode": "elapsed_time", **extra})
        self.view.artifacts.put("image_times", [self.start]*4, "dataset")
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            video_plan(self.view, {"timing_mode": "elapsed_time"})
        self.view.artifacts.put("image_times", [None]*4, "dataset")
        self.assertEqual(len(video_plan(self.view, {})[0]), 4)
        with self.assertRaisesRegex(ValueError, "real acquisition"):
            video_plan(self.view, {"timing_mode": "elapsed_time"})

    def test_regular_sampling_and_no_extrapolation_past_end(self):
        self.view.artifacts.put("image_times", [self.start+timedelta(seconds=60*i) for i in range(4)], "dataset")
        rows, _ = video_plan(self.view, {"timing_mode": "elapsed_time"})
        self.assertEqual([r["image_index"] for r in rows], [0, 1, 2, 3])
        rows, _ = video_plan(self.view, {"timing_mode": "elapsed_time", "dt_seconds": 70})
        self.assertEqual([r["elapsed_seconds"] for r in rows], [0, 70, 140])

    def test_individual_scales_explicit_override_and_panel_preservation(self):
        self.run_ok("selection")
        panel = self.run_ok("image_plot").outputs["image_plot_path"]
        checksum = file_digest(panel)
        original = FrameFigureSection.clean_panel
        for mode, explicit, expected in (("global", {}, [4]*4), ("individual", {}, [1, 2, 3, 4]),
                                         ("individual", {"vmin": -1, "vmax": 8}, [8]*4)):
            limits = []
            def capture(renderer, session, config, images, indices):
                limits.append(config["vmax"])
                return original(renderer, session, config, images, indices)
            self.view.configure("individual_image_plots", color_scale=mode, **explicit)
            with patch.object(FrameFigureSection, "clean_panel", capture):
                outcome = self.run_ok("individual_image_plots")
            self.assertEqual(limits, expected)
            self.assertEqual(len(outcome.outputs["individual_image_plot_paths"]), 4)
            self.assertEqual(file_digest(panel), checksum)
        self.view.configure("individual_image_plots", color_scale="typo")
        self.assertEqual(self.view.run("individual_image_plots").state.value, "failed")

    def test_surface_geometry_and_no_source_mutation(self):
        import matplotlib.pyplot as plt
        images = self.view.artifacts.require("image_stack")
        before = images[0].copy()
        fig = plt.figure()
        try:
            ax = fig.add_subplot(projection="3d")
            _draw_frame(ax, images[0], self.view, {"render_3d": True, "surface_stride": 3,
                        "surface_zlim": [-1, 5]}, 0, (0, 4))
            self.assertEqual(ax.get_xlim(), (0, 220))
            self.assertEqual(ax.get_ylim(), (220, 0))
            self.assertEqual(ax.get_zlim(), (-1, 5))
            self.assertEqual(ax.get_xlabel(), "x (µm)")
        finally:
            plt.close(fig)
        np.testing.assert_array_equal(images[0], before)
        self.run_ok("selection")
        self.view.configure("image_plot", render_3d=True)
        self.assertTrue(self.run_ok("image_plot").outputs["image_plot_path"].name.endswith("_3d.png"))
        self.view.configure("individual_image_plots", render_3d=True, color_scale="individual")
        self.assertTrue(all(p.name.endswith("_3d.png") for p in self.run_ok("individual_image_plots").outputs["individual_image_plot_paths"]))

    def test_gui_save_load_new_options(self):
        gui = VisualizationGUI()
        gui.session = self.view
        gui._populate()
        gui.render_3d.value = True
        gui.surface_elev.value = "45"
        gui.surface_zmin.value, gui.surface_zmax.value = "-1", "6"
        gui.individual_color_scale.value = "individual"
        gui.video_timing.value = "elapsed_time"
        gui.video_dt.value, gui.video_tolerance.value = "60", "40"
        gui._sync()
        saved = save_visualization_settings(self.view)
        gui.session = load_visualization_settings(saved)
        gui._populate()
        self.assertTrue(gui.render_3d.value)
        self.assertEqual(gui.individual_color_scale.value, "individual")
        self.assertEqual(gui.video_timing.value, "elapsed_time")
        self.assertEqual(float(gui.video_dt.value), 60)
        self.assertEqual(float(gui.video_tolerance.value), 40)
        self.assertEqual(gui.session.config["image_plot"]["surface_zlim"], [-1, 6])
        gui.surface_zmin.value = gui.surface_zmax.value = ""
        gui._sync()
        gui.session = load_visualization_settings(save_visualization_settings(gui.session))
        gui._populate()
        self.assertEqual(gui.surface_zmin.value, "")
        # INI serializes None as an empty string, including API-only reloads.
        self.assertEqual(gui.session.run("selection").state.value, "completed")
        self.assertEqual(gui.session.run("image_plot").state.value, "completed")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg required")
    def test_real_2d_and_3d_mp4_frame_counts_and_audit(self):
        for surface, mode, count in ((False, "image_sequence", 4), (False, "elapsed_time", 11), (True, "elapsed_time", 11)):
            self.view.configure("image_video", render_3d=surface, timing_mode=mode, dt_seconds=60)
            output = self.run_ok("image_video").outputs
            probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                "-show_entries", "stream=nb_read_frames", "-of", "json", str(output["image_video_path"])],
                check=True, capture_output=True, text=True)
            self.assertEqual(int(json.loads(probe.stdout)["streams"][0]["nb_read_frames"]), count)
            table = pd.read_csv(output["image_video_timing_path"])
            self.assertEqual(len(table), count)
            self.assertEqual((table.status == "no_acquisition_data").sum(), 7 if mode == "elapsed_time" else 0)
            settings = json.loads(output["image_video_timing_settings_path"].read_text())
            self.assertEqual(settings["frame_count"], count)


if __name__ == "__main__":
    unittest.main()

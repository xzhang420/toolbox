"""Synthetic regressions for older run names and potentiostat exports."""
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline import backends
from lumacam_pipeline.handover import load_processed_images
from lumacam_pipeline.section import PipelineSession, SectionState
from lumacam_pipeline.sections.electrochemistry import SynchronizationSection
from lumacam_pipeline.visualization import open_visualization


class LegacyOperandoInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def echem(self):
        path = self.root / "cell.txt"
        path.write_text("time/s\tEwe/V\t<I>/mA\tCapacity/mA.h\tcycle number\n"
                        "07/12/2025 12:09:46.2088\t0.1\t1\t0\t1\n"
                        "07/12/2025 12:10:46.2088\t0.2\t1\t0.01\t1\n")
        return path

    def test_single_file_and_folder_load_identically(self):
        path = self.echem()
        (self.root / "slides.pptx").write_bytes(b"not an echem export")
        loader = backends.nci_plots().load_echem_data
        single = loader(path, report_summary=False, include_cycle=True)
        folder = loader(self.root, report_summary=False, include_cycle=True)
        self.assertEqual(single[0], folder[0])
        self.assertEqual(single[0][0], datetime(2025, 12, 7, 12, 9, 46, 208800))
        for a, b in zip(single[1:], folder[1:]):
            np.testing.assert_allclose(a, b)

    def test_unsupported_echem_file_is_actionable(self):
        path = self.root / "slides.pptx"
        path.touch()
        with self.assertRaisesRegex(ValueError, "TXT or CSV"):
            backends.nci_plots().load_echem_data(path)

    def test_legacy_times_sort_images_and_provenance_together(self):
        for name, minute, value in (("exp100_cont_0.tif", 2, 2), ("exp200_0.tif", 1, 1)):
            tifffile.imwrite(self.root / name, np.full((4, 4), value, np.float32), metadata=None,
                             extratags=[(65000, 's', 0, f"2025-12-07 12:0{minute}:00", False)])
        dataset = load_processed_images(self.root, timestamp_timezone="Europe/Zurich")
        self.assertEqual([a[0, 0] for a in dataset["images"]], [1, 2])
        self.assertEqual(dataset["paths"][0].name, "exp200_0.tif")
        self.assertEqual(dataset["records"][0]["path"], "exp200_0.tif")
        self.assertEqual(dataset["times"][0].hour, 11)
        self.assertEqual(dataset["times"], sorted(dataset["times"]))

    def test_synchronization_sorts_all_columns_and_marks_uncovered_images(self):
        section = SynchronizationSection()
        session = PipelineSession([section])
        t = datetime(2025, 12, 7, 12)
        values = {"image_times": [t-timedelta(seconds=5), t, t+timedelta(seconds=10), t+timedelta(seconds=20)],
                  "image_metric_values": [0, 1, 2, 3],
                  "echem_times": [t+timedelta(seconds=10), t], "echem_potential": [20, 10],
                  "echem_thickness": [200, 100], "echem_cycles": [2, 1]}
        for key, value in values.items(): session.artifacts.put(key, value, "test")
        outcome = section.run(session, {"allow_extrapolation": False})
        table = outcome.outputs["synchronized_table"]
        self.assertIn("2/4", outcome.message)
        np.testing.assert_allclose(table["echem_thickness"], [np.nan, 100, 200, np.nan])
        np.testing.assert_allclose(table["cycle"], [np.nan, 1, 2, np.nan])

    def test_visualization_refuses_nonoverlapping_echem_dates(self):
        path = self.root / "frame.tif"
        tifffile.imwrite(path, np.ones((4, 4), np.float32), metadata=None,
                         description="AcquisitionTime:2026-12-07T11:10:00Z")
        view = open_visualization(path, preset="operando_nci", config={
            "electrochemistry": {"path": str(self.echem()), "timestamp_timezone": "Europe/Zurich"},
            "image_measurements": {"label": "Recorded thickness (um)"}})
        self.assertFalse(view.config["synchronization"]["allow_extrapolation"])
        self.assertEqual(view.config["combined_plot"]["thickness_traces"], "nci")
        self.assertEqual(view.config["combined_video"]["thickness_traces"], "nci")
        self.assertEqual(view.run("electrochemistry").state, SectionState.COMPLETED)
        self.assertEqual(view.run("image_measurements").state, SectionState.COMPLETED)
        self.assertEqual(view.artifacts.require("image_metric_label"), "Recorded thickness (um)")
        for action in ("synchronization", "combined_plot", "combined_video"):
            outcome = view.run(action)
            self.assertEqual(outcome.state, SectionState.FAILED)
            self.assertIn("do not overlap", outcome.message)

    def test_explicit_thickness_units_enable_comparable_echem_trace(self):
        path = self.root / "frame.tif"
        tifffile.imwrite(path, np.ones((4,4), np.float32))
        view = open_visualization(path, preset="operando_nci", config={"input":{"unit":"um"}})
        for section in ("combined_plot", "combined_video"):
            self.assertEqual(view.config[section]["thickness_traces"], "both")

    def test_combined_video_uses_only_shared_times_and_explicit_image_origin(self):
        module = backends.nci_plots()
        t = datetime(2025, 12, 7, 12)
        captured = []
        class Writer:
            def __init__(self, **kwargs): pass
            @contextmanager
            def saving(self, figure, *args, **kwargs):
                self.figure = figure
                yield
            def grab_frame(self):
                axis = self.figure.axes[0]
                marker = axis.lines[1]
                captured.append((float(marker.get_xdata()[0])*3600,
                                 self.figure.axes[1].images[0].origin))
        with patch.object(module, "FFMpegWriter", Writer):
            module.make_potential_image_video_uniform_time(
                [t+timedelta(seconds=s) for s in (10,20)], [1,2],
                [t,t+timedelta(seconds=30)], [np.ones((4,4)),np.ones((4,4))*2],
                str(self.root/'test.mp4'), dt_seconds=5, scalebar_length=0,
                scale_label="", limit_to_overlap=True, image_origin="upper")
        self.assertEqual(captured, [(10., "upper"), (15., "upper")])


if __name__ == "__main__":
    unittest.main()

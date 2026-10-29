from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline.event_selection import current_events, panel_label
from lumacam_pipeline.visualization import open_visualization, FrameFigureSection
from lumacam_pipeline.reports import create_report
from lumacam_pipeline.section import SectionState


class EventTests(unittest.TestCase):
    def setUp(self):
        self.start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.times = [self.start + timedelta(seconds=i*10) for i in range(31)]

    def test_ocv_and_half_cycles_no_distant_images(self):
        current = [0]*3 + [-1]*12 + [1]*12 + [0]*4
        context = current_events(self.times, current, self.times, plating_current_sign=-1)
        self.assertEqual(context["img_indices"], [2, 14, 26])
        self.assertEqual([e["caption"] for e in context["events"]], ["OCV", "Plating 1", "Stripping 1"])
        # A missing-image gap cannot select a point several minutes earlier.
        context = current_events(self.times, current, self.times[:4], tolerance_seconds=20)
        self.assertEqual(context["img_indices"], [2])
        self.assertEqual(len(context["skipped_events"]), 2)

    def test_pauses_short_pulses_final_incomplete_and_bad_images(self):
        current = [0]*3 + [-1]*6 + [0]*3 + [-1]*6 + [1] + [-1]*2 + [1]*10
        valid = np.ones(len(self.times), bool); valid[20] = False
        ctx = current_events(self.times, current, self.times, valid_images=valid)
        self.assertEqual(ctx["events"][-1]["status"], "last recorded; completion unconfirmed")
        self.assertTrue(ctx["events"][1]["interrupted"])
        self.assertEqual(len(ctx["ignored_segments"]), 2)
        self.assertNotIn(20, ctx["img_indices"])
        self.assertIn("negative current", ctx["events"][1]["kind"])

    def test_no_ocv_invented_under_load_and_label_alphabet(self):
        ctx = current_events(self.times, [-1]*15+[1]*16, self.times)
        self.assertNotIn("ocv", [e["kind"] for e in ctx["events"]])
        self.assertEqual([panel_label(i) for i in (0, 25, 26, 51, 52)], ["a", "z", "aa", "az", "ba"])

    def test_short_opposite_pulse_still_separates_distinct_phases(self):
        current = [0]*3 + [-1]*10 + [1] + [0]*3 + [-1]*10 + [0]*4
        ctx = current_events(self.times, current, self.times, plating_current_sign=-1)
        self.assertEqual(ctx["img_indices"], [2, 12, 26])
        self.assertEqual([e['caption'] for e in ctx['events']], ['OCV', 'Plating', 'Plating'])

    def test_flat_profile_points_clean_panel_combined_report(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); source = root / "input"; source.mkdir()
            for i in range(6):
                tifffile.imwrite(source/f"{i}.tif", np.full((32, 32), 1+i/10, np.float32), metadata=None,
                                 description=f"AcquisitionTime:{self.times[i*5].isoformat()}")
            cfg = {"reports": {"enabled": False}, "visualization": {"export_layout": "flat"},
                   "input": {"pixel_size_um": 100, "unit": "um"},
                   "image_plot": {"scalebar_length_um": 500, "scalebar_label": "0.5 mm"},
                   "event_detection": {"plating_current_sign": -1},
                   "combined_plot": {"thickness_traces": "nci"}}
            view = open_visualization(source, config=cfg, preset="operando_nci")
            for name, value in {"echem_times": self.times, "echem_current": [0]*3+[-1]*12+[1]*12+[0]*4,
                                "echem_potential": np.linspace(0, 1, 31), "echem_thickness": np.ones(31)}.items():
                view.artifacts.put(name, value, "fixture")
            for section in ("image_measurements", "event_detection", "selection", "image_plot", "combined_plot"):
                outcome = view.run(section)
                self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            profile = view.artifacts.require("results_dir")
            self.assertTrue((profile/"figures/images.png").is_file())
            self.assertTrue((profile/"tables/image_measurements.csv").is_file())
            import matplotlib.pyplot as plt
            with patch("matplotlib.pyplot.close"):
                FrameFigureSection().run(view, view.section_config("image_plot"))
                figure = plt.gcf()
                texts = [text.get_text() for ax in figure.axes for text in ax.texts]
                self.assertEqual(texts.count("0.5 mm"), 1)
                self.assertTrue(all(not ax.get_xlabel() for ax in figure.axes[:-1]))
            plt.close("all")
            # Re-render above must not invalidate the recorded PNG hash.
            md = create_report(profile)["markdown"].read_text()
            self.assertIn("Minimum / maximum", md)
            self.assertNotIn("#### Image measurements\n", md)
            self.assertIn("#### Electrochemistry and image measurements", md)
            for record in (profile/"records").glob("*/export.json"):
                row = json.loads(record.read_text())
                if row["section"] == "combined_plot":
                    self.assertEqual(len(row["outputs"]), 2)  # figure and points only


if __name__ == "__main__":
    unittest.main()

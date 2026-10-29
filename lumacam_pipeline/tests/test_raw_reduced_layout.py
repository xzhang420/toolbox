"""Exercise raw/reduced discovery and default outputs through public workflows."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline import backends, create_session, discover_dataset, find_layout_root
from lumacam_pipeline.layout import LayoutError
from lumacam_pipeline.gui import LumacamPipelineGUI
from lumacam_pipeline.batch import discover_batch_experiments, _proposal_paths, _batch_dir
from lumacam_pipeline.metadata_ranges import inspect_metadata_time_ranges, save_metadata_time_ranges_csv
from lumacam_pipeline.result_layout import shared_directory
from lumacam_pipeline.section import SectionState
from test_lumacam_pipeline import make_markerless_proposal


class RawReducedPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        proposal, layout, self.experiment, self.runs = make_markerless_proposal(Path(self.temp.name))
        self.proposal = proposal
        (layout / "tpx3Files").rename(proposal / "raw")
        (proposal / "reduced").mkdir()
        (proposal / "data").rename(proposal / "reduced/data")
        (proposal / "reduced/data/experiments").rename(proposal / "reduced/data/experiment")
        self.layout = proposal / "reduced/data/experiment"
        for name in ("analysis", "documentation", "data/echem", "data/beamhistory"):
            (proposal / "reduced" / name).mkdir(parents=True, exist_ok=True)

    def test_proposal_and_raw_run_discover_matching_images_and_metadata(self):
        raw_run = self.proposal / "raw" / self.experiment / self.runs[0]
        for entry in (self.proposal, self.proposal / "reduced", raw_run, self.layout):
            dataset = discover_dataset(entry, experiment=self.experiment)
            self.assertEqual(dataset.proposal_root, self.proposal)
            self.assertEqual(dataset.layout_root, self.layout)
            self.assertEqual(dataset.runs[0].tpx3_dir, raw_run)
            self.assertEqual(find_layout_root(entry), self.layout)
        selected = discover_dataset(raw_run)
        self.assertEqual(selected.selected_run, self.runs[0])
        resolved = backends.nci_data().resolve_nci_experiment(raw_run)
        self.assertEqual(resolved["runs"][0]["tpx_dir"], str(raw_run))

    def test_project_and_batch_outputs_remain_under_reduced_analysis(self):
        session = create_session("nci_images_only", {"project": {
            "data_path": str(self.proposal), "experiment": self.experiment, "variant": "3to1000"}})
        self.assertEqual(session.run("project").state, SectionState.COMPLETED)
        analysis = self.proposal / "reduced/analysis"
        self.assertTrue(Path(session.artifacts.require("results_dir")).is_relative_to(analysis))
        self.assertEqual(_proposal_paths(self.proposal), (self.proposal, self.layout))
        self.assertTrue(_batch_dir(self.proposal, "batch").is_relative_to(analysis))
        self.assertTrue(shared_directory(self.proposal, {}).is_relative_to(analysis))
        self.assertTrue(discover_batch_experiments(self.proposal))
        report = inspect_metadata_time_ranges(self.proposal)
        self.assertEqual(report.proposal_root, self.proposal)
        saved = save_metadata_time_ranges_csv(report)
        self.assertTrue(Path(saved).is_relative_to(analysis))

    def test_gui_saves_config_and_browses_inputs_inside_reduced(self):
        gui = LumacamPipelineGUI("nci_images_only")
        gui.data_path.value = str(self.proposal)
        self.assertEqual(gui._automatic_proposal_config_path(), self.proposal / "reduced/analysis/lumacam_pipeline.ini")
        self.assertEqual(gui._proposal_subfolder("echem"), self.proposal / "reduced/data/echem")
        self.assertEqual(gui._proposal_subfolder("beamhistory"), self.proposal / "reduced/data/beamhistory")
        self.assertEqual(gui._proposal_subfolder("results"), self.proposal / "reduced/analysis")

    def test_nci_and_exsitu_processing_write_only_to_reduced(self):
        for mode in ("nci_images_only", "exsitu"):
            with self.subTest(mode=mode):
                session = create_session(mode, {"project": {
                    "data_path": str(self.proposal), "experiment": self.experiment,
                    "variant": "3to1000", "analysis_name": mode}})
                for section in ("project", "corrections", "images", "average"):
                    outcome = session.run(section)
                    self.assertEqual(outcome.state, SectionState.COMPLETED, str(outcome))
                self.assertTrue(Path(session.artifacts.require("results_dir")).is_relative_to(self.proposal / "reduced/analysis"))
        self.assertFalse((self.proposal / "analysis").exists())
        self.assertEqual(list((self.proposal / "reduced").rglob("*.tpx3")), [])

    def test_reduced_images_remain_readable_when_raw_is_offline(self):
        (self.proposal / "raw").rename(Path(self.temp.name) / "offline_raw")
        dataset = discover_dataset(self.proposal, experiment=self.experiment)
        self.assertTrue(dataset.image_products())

    def test_conflicting_layout_and_incoming_selection_are_rejected(self):
        incoming = self.proposal / "raw/.incoming/exp/run"
        incoming.mkdir(parents=True)
        with self.assertRaises(LayoutError):
            discover_dataset(incoming)
        (self.proposal / "data/experiment").mkdir(parents=True)
        with self.assertRaises(LayoutError):
            find_layout_root(self.proposal)


if __name__ == "__main__":
    unittest.main()

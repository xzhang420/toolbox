"""Exercise mode filtering through real ipywidgets and temporary proposals."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lumacam_pipeline.batch import BATCH_SECTION_LABELS, normalize_batch_sections, run_batch
from lumacam_pipeline.config import load_config, save_config
from lumacam_pipeline.gui import LumacamPipelineGUI
from lumacam_pipeline.modes import EC_SECTIONS, LEGACY_MODE_ALIASES, MODE_OPTIONS, MODE_SECTIONS
from lumacam_pipeline.presets import PRESETS
from lumacam_pipeline.section import SectionOutcome, SectionState
from test_lumacam_pipeline import make_markerless_proposal


def titles(gui):
    return tuple(gui.accordion.get_title(i) for i in range(len(gui.accordion.children)))


def displayed_widgets(widget):
    if getattr(getattr(widget, "layout", None), "display", None) == "none":
        return
    yield widget
    for child in getattr(widget, "children", ()):
        yield from displayed_widgets(child)


class GuiModeTests(unittest.TestCase):
    def test_every_mode_filters_panels_status_and_batch_choices(self):
        self.assertEqual(set(MODE_SECTIONS), set(PRESETS) - set(LEGACY_MODE_ALIASES))
        for mode, sections in MODE_SECTIONS.items():
            with self.subTest(mode=mode):
                gui = LumacamPipelineGUI(mode)
                self.assertIn("Processing mode", gui.preset_label.value)
                self.assertFalse(gui.show_all_sections.value)
                self.assertEqual(gui._visible_sections, sections)
                self.assertEqual(
                    {section for _, section in gui.batch_sections.options},
                    (set(BATCH_SECTION_LABELS) - {"project"}) & sections,
                )
                for section, implementation in gui.session.sections.items():
                    status_title = f"<td>{implementation.title}</td>"
                    self.assertEqual(status_title in gui.status_html.value, section in sections)
                self.assertEqual(
                    "Electrochemistry and synchronization" in titles(gui),
                    mode == "operando_nci",
                )
                self.assertEqual(
                    "Prepare OB/BG files (optional)" in titles(gui),
                    mode in {"operando_nci", "nci_images_only", "exsitu"},
                )
                self.assertEqual("Time-of-flight" in titles(gui), mode == "tof")
                # No visible batch choice pulls in an invisible prerequisite.
                for _, section in gui.batch_sections.options:
                    self.assertLessEqual(set(normalize_batch_sections((section,))), sections)

    def test_exsitu_has_no_irrelevant_controls_or_help(self):
        gui = LumacamPipelineGUI("exsitu")
        displayed = list(displayed_widgets(gui.widget))
        for control in (
            gui.echem_path, gui.echem_browse_btn, gui.sync_btn, gui.events_btn,
            gui.combined_plot_btn, gui.combined_video_btn, gui.nci_outputs_btn,
            gui.photon_btn, gui.tof_path, gui.tof_btn, gui.plating_btn,
        ):
            self.assertNotIn(control, displayed)
        for control in (
            gui.prepare_shared_btn, gui.run_correction_btn, gui.run_background_btn,
            gui.average_btn, gui.measure_btn, gui.individual_plots_btn, gui.image_video_btn,
        ):
            self.assertIn(control, displayed)
        help_text = " ".join(
            str(widget.value) for widget in displayed if isinstance(widget, gui.w.HTML)
            and "lumacam-term-help" in str(widget.value)
        )
        for term in ("EC events", "Plating metrics", "Photon-count cube", "Combined output"):
            self.assertNotIn(term, help_text)

    def test_show_all_reveals_controls_and_all_status_rows(self):
        gui = LumacamPipelineGUI("exsitu")
        gui.show_all_sections.value = True
        self.assertEqual(gui._visible_sections, set(gui.session.sections))
        self.assertIn("Photon-count and time-of-flight", titles(gui))
        displayed = list(displayed_widgets(gui.widget))
        self.assertIn(gui.echem_path, displayed)
        self.assertIn(gui.combined_plot_btn, displayed)
        self.assertIn(gui.photon_btn, displayed)
        self.assertIn(gui.tof_btn, displayed)
        for implementation in gui.session.sections.values():
            self.assertIn(f"<td>{implementation.title}</td>", gui.status_html.value)
        gui.show_all_sections.value = False
        self.assertEqual(gui._visible_sections, MODE_SECTIONS["exsitu"])

    def test_show_all_does_not_select_corrections_for_a_new_image_only_session(self):
        gui = LumacamPipelineGUI("image_only")
        self.assertNotIn("corrections", gui.batch_sections.value)
        gui.show_all_sections.value = True
        self.assertIn("corrections", {section for _, section in gui.batch_sections.options})
        self.assertNotIn("corrections", gui.batch_sections.value)

    def test_mode_switch_preserves_entered_settings_and_completed_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(Path(temporary), values=(2, 4))
            gui = LumacamPipelineGUI("operando_nci")
            gui.data_path.value = str(proposal)
            gui.experiment.value = experiment
            gui.variant.value = "3to1000"
            gui.background_enabled.value = True
            gui.background_count.value = 2
            gui.reference_path.value = "/saved/ob.tiff"
            gui.echem_path.value = "/saved/echem.csv"
            gui._sync_config()
            self.assertEqual(gui.session.run("project").state, SectionState.COMPLETED)
            self.assertEqual(gui.session.run("images").state, SectionState.COMPLETED)
            session = gui.session
            dataset = session.artifacts.require("dataset")
            stack = session.artifacts.require("image_stack")
            outcomes = dict(session.outcomes)
            config = deepcopy(session.config)
            # These not-yet-saved GUI edits must survive too.
            gui.roi_r.value = 42
            gui.frames_per_merge.value = 3
            gui.preset.value = "exsitu"
            gui.show_all_sections.value = True
            gui.preset.value = "tof"
            gui.show_all_sections.value = False
            gui.preset.value = "operando_nci"
            self.assertIs(gui.session, session)
            self.assertIs(session.artifacts.require("dataset"), dataset)
            self.assertIs(session.artifacts.require("image_stack"), stack)
            self.assertEqual(session.outcomes, outcomes)
            for key in ("project", "corrections", "background", "electrochemistry"):
                self.assertEqual(session.config[key], config[key])
            self.assertEqual(gui.echem_path.value, "/saved/echem.csv")
            self.assertEqual(gui.reference_path.value, "/saved/ob.tiff")
            self.assertTrue(gui.background_enabled.value)
            self.assertEqual(gui.roi_r.value, 42)
            self.assertEqual(gui.frames_per_merge.value, 3)

    def test_new_exsitu_mode_keeps_default_average_collection(self):
        gui = LumacamPipelineGUI("operando_nci")
        gui.preset.value = "exsitu"
        self.assertTrue(gui.session.section_config("average")["collect_in_proposal"])
        gui.session.section_config("average")["collect_in_proposal"] = False
        gui.preset.value = "operando_nci"
        gui.preset.value = "exsitu"
        self.assertFalse(gui.session.section_config("average")["collect_in_proposal"])

    def test_tof_starts_with_derived_products_without_resetting_custom_variant(self):
        gui = LumacamPipelineGUI("tof")
        self.assertEqual(gui.variant.value, "")
        gui = LumacamPipelineGUI("operando_nci")
        gui.preset.value = "tof"
        self.assertEqual(gui.variant.value, "")
        self.assertIn("derived", gui.session.section_config("project")["product_kinds"])
        gui.preset.value = "exsitu"
        gui.variant.value = "5to1000"
        gui.preset.value = "tof"
        self.assertEqual(gui.variant.value, "5to1000")

    def test_chosen_data_settings_are_preserved_even_before_discovery(self):
        gui = LumacamPipelineGUI("operando_nci")
        gui.data_path.value = "/chosen/proposal"
        gui.preset.value = "tof"
        self.assertEqual(gui.variant.value, "3to1000")
        self.assertEqual(gui.session.section_config("project")["product_kinds"], "final")

    def test_batch_choices_survive_modes_but_hidden_steps_never_execute(self):
        gui = LumacamPipelineGUI("operando_nci")
        gui._batch_all_names = ("exp401_sample",)
        gui._set_batch_selected(("exp401_sample",))
        gui.batch_sections.value = ("average", "combined_plot", "combined_video")
        gui.preset.value = "exsitu"
        experiments, sections = gui._batch_inputs()
        self.assertEqual(experiments, ("exp401_sample",))
        self.assertEqual(sections, ("average",))
        self.assertFalse(set(normalize_batch_sections(sections)) & EC_SECTIONS)
        self.assertEqual(gui.session.config["batch"]["sections"], ["average"])
        gui.show_all_sections.value = True
        self.assertEqual(set(gui.batch_sections.value), {"average", "combined_plot", "combined_video"})
        gui.batch_sections.value = ("average", "combined_plot")
        gui.show_all_sections.value = False
        gui.preset.value = "operando_nci"
        self.assertEqual(set(gui.batch_sections.value), {"average", "combined_plot"})
        gui.batch_sections.value = ()
        gui.preset.value = "exsitu"
        gui.preset.value = "operando_nci"
        self.assertEqual(gui.batch_sections.value, ())

    def test_hidden_individual_action_is_blocked_until_show_all(self):
        gui = LumacamPipelineGUI("exsitu")
        outcome = SectionOutcome("electrochemistry", SectionState.COMPLETED, "test")
        with patch.object(gui.session, "run", return_value=outcome) as run:
            gui._run("electrochemistry")
            run.assert_not_called()
            gui.show_all_sections.value = True
            gui._run("electrochemistry")
            run.assert_called_once_with("electrochemistry")

    def test_config_restores_mode_override_and_remembered_batch_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mode.ini"
            gui = LumacamPipelineGUI("operando_nci")
            gui.batch_sections.value = ("average", "combined_plot")
            gui.echem_path.value = "/saved/echem.csv"
            gui.preset.value = "exsitu"
            gui.config_path.value = str(path)
            gui._save_config(None)
            preset, values = load_config(path)
            self.assertEqual(preset, "exsitu")
            self.assertEqual(values["batch"]["sections"], ["average"])
            loaded = LumacamPipelineGUI("tof")
            loaded.config_path.value = str(path)
            loaded._load_config(None)
            self.assertEqual(loaded._visible_sections, MODE_SECTIONS["exsitu"])
            self.assertEqual(loaded.batch_sections.value, ("average",))
            self.assertEqual(loaded.echem_path.value, "/saved/echem.csv")
            loaded.show_all_sections.value = True
            self.assertEqual(set(loaded.batch_sections.value), {"average", "combined_plot"})
            loaded._save_config(None)
            restored = LumacamPipelineGUI("image_only")
            restored.config_path.value = str(path)
            restored._load_config(None)
            self.assertTrue(restored.show_all_sections.value)
            self.assertEqual(restored._visible_sections, set(restored.session.sections))

    def test_old_config_and_explicit_empty_batch_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.ini"
            for selection in ([], ["average", "combined_plot"]):
                save_config(path, "exsitu", {"batch": {"sections": selection}})
                gui = LumacamPipelineGUI("operando_nci")
                gui.show_all_sections.value = True
                gui.config_path.value = str(path)
                gui._load_config(None)
                self.assertFalse(gui.show_all_sections.value)
                self.assertEqual(gui.batch_sections.value, ("average",) if selection else ())
                gui.show_all_sections.value = True
                self.assertEqual(set(gui.batch_sections.value), set(selection))

    def test_open_or_folded_panel_is_retained_when_possible(self):
        gui = LumacamPipelineGUI("operando_nci")
        gui.accordion.selected_index = titles(gui).index("Image products")
        gui.preset.value = "exsitu"
        self.assertEqual(gui.accordion.get_title(gui.accordion.selected_index), "Image products")
        gui.accordion.selected_index = None
        gui.show_all_sections.value = True
        self.assertIsNone(gui.accordion.selected_index)
        gui.accordion.selected_index = titles(gui).index("Electrochemistry and synchronization")
        gui.show_all_sections.value = False
        self.assertEqual(gui.accordion.selected_index, 0)

    def test_exsitu_batch_after_mode_switch_ignores_saved_echem_and_copies_average(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(Path(temporary), values=(2, 4))
            gui = LumacamPipelineGUI("operando_nci")
            gui.data_path.value = str(proposal)
            gui.experiment.value = experiment
            gui._populate_batch_choices()
            gui._set_batch_selected((experiment,))
            gui.echem_path.value = "/missing/echem.csv"
            gui.batch_sections.value = ("average", "combined_plot")
            gui.preset.value = "exsitu"
            gui._run_batch(None)
            self.assertIn("Batch finished", gui.batch_status_html.value)
            average = proposal / "analysis" / "proposal_summary" / "averaged_images" / f"average_{experiment}_3to1000.tiff"
            self.assertTrue(average.is_file())
            import tifffile
            np.testing.assert_allclose(tifffile.imread(average), 3)
            _, config = load_config(proposal / "analysis" / experiment / "3to1000" / "processing" / "config_used.ini")
            self.assertFalse(set(config["batch"]["sections"]) & EC_SECTIONS)

    def test_view_preferences_do_not_change_batch_resume_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(Path(temporary), values=(2, 4))
            config = {
                "project": {"variant": "3to1000"},
                "gui": {"show_all_sections": False, "batch_section_selection": ["average"]},
            }
            first = run_batch(proposal, (experiment,), preset="exsitu", config=config,
                              sections=("average",), batch_name="view_only")
            self.assertEqual(first.results[0].status, "completed")
            config["gui"] = {"show_all_sections": True,
                             "batch_section_selection": ["average", "electrochemistry"]}
            resumed = run_batch(proposal, (experiment,), preset="exsitu", config=config,
                                sections=("average",), batch_name="view_only")
            self.assertEqual(resumed.results[0].status, "skipped")


if __name__ == "__main__":
    unittest.main()

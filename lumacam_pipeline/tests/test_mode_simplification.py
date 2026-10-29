"""Three GUI choices, readable mode label, and lossless legacy configuration loading."""

from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lumacam_pipeline.batch import run_batch
from lumacam_pipeline.config import load_config, save_config
from lumacam_pipeline.gui import LumacamPipelineGUI, _create_gui_session
from lumacam_pipeline.modes import LEGACY_MODE_ALIASES, MODE_OPTIONS, MODE_SECTIONS
from lumacam_pipeline.presets import create_session
from test_lumacam_pipeline import make_markerless_proposal


class ModeSimplificationTests(unittest.TestCase):
    def test_only_three_named_choices_with_separate_untruncated_label(self):
        for mode in MODE_SECTIONS:
            gui = LumacamPipelineGUI(mode)
            self.assertEqual(tuple(gui.preset.options), MODE_OPTIONS)
            self.assertEqual([label for label, _ in gui.preset.options], ["Operando NCI", "Ex situ NCI", "Time of flight"])
            self.assertEqual(gui.preset.description, "")
            self.assertEqual(gui.mode_selector.children, (gui.preset_label, gui.preset))
            self.assertIn("Processing mode", gui.preset_label.value)
            self.assertIn("white-space:normal", gui.preset_label.value)
            self.assertIn("overflow:visible", gui.preset_label.value)
            self.assertEqual(gui.mode_selector.layout.width, "100%")

    def test_legacy_constructor_preserves_original_processing_defaults(self):
        for old, new in LEGACY_MODE_ALIASES.items():
            with self.subTest(mode=old):
                expected = create_session(old).config
                expected["average"].setdefault("collect_in_proposal", False)
                gui = LumacamPipelineGUI(old)
                self.assertEqual(gui.preset.value, new)
                self.assertEqual(gui.session.preset, new)
                self.assertEqual(gui.session.config, expected)
                self.assertEqual(gui._visible_sections, MODE_SECTIONS[new])
                self.assertFalse(gui.session.section_config("electrochemistry")["enabled"])
                if old == "image_only":
                    self.assertNotIn("corrections", gui.batch_sections.value)
                # Python/CLI compatibility is independent of the simplified GUI.
                self.assertEqual(create_session(old).preset, old)

    def test_legacy_configs_roundtrip_without_enabling_new_processing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.ini"
            for old, new in LEGACY_MODE_ALIASES.items():
                with self.subTest(mode=old):
                    values = {"images": {"frames_per_merge": 3, "white_spot_enabled": True},
                              "background": {"enabled": True, "source_mode": "first_frames", "n_images": 4},
                              "batch": {"sections": ["average"]}}
                    save_config(path, old, values)
                    original = path.read_bytes()
                    gui = LumacamPipelineGUI("tof")
                    gui.config_path.value = str(path)
                    gui._load_config(None)
                    self.assertEqual(path.read_bytes(), original)
                    self.assertEqual(gui.preset.value, new)
                    self.assertEqual(gui.frames_per_merge.value, 3)
                    self.assertEqual(gui.background_count.value, 4)
                    self.assertTrue(gui.white_spot_enabled.value)
                    self.assertEqual(gui.batch_sections.value, ("average",))
                    self.assertFalse(gui.session.section_config("electrochemistry")["enabled"])
                    gui._save_config(None)
                    saved_mode, saved = load_config(path)
                    self.assertEqual(saved_mode, new)
                    self.assertFalse(saved["average"]["collect_in_proposal"])
                    reopened = LumacamPipelineGUI("tof")
                    reopened.config_path.value = str(path)
                    reopened._load_config(None)
                    self.assertEqual(reopened.preset.value, new)
                    for section in ("images", "average", "corrections", "background", "electrochemistry"):
                        self.assertEqual(reopened.session.config[section], gui.session.config[section])

    def test_explicit_legacy_settings_override_defaults(self):
        config = {"average": {"collect_in_proposal": True},
                  "electrochemistry": {"enabled": True, "path": "/saved/custom.csv"},
                  "corrections": {"enabled": True}}
        for old in LEGACY_MODE_ALIASES:
            session = _create_gui_session(old, config)
            for section, values in config.items():
                for key, value in values.items():
                    self.assertEqual(session.config[section][key], value)

    def test_old_batch_can_resume_after_equivalent_gui_mode_conversion(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(Path(temporary), values=(2, 4))
            for old, new in LEGACY_MODE_ALIASES.items():
                with self.subTest(mode=old):
                    config = {"project": {"variant": "3to1000", "analysis_name": old}}
                    kwargs = dict(sections=("average",), batch_name=old)
                    original = run_batch(proposal, (experiment,), preset=old, config=config, **kwargs)
                    self.assertEqual(original.results[0].status, "completed")
                    session = _create_gui_session(old, config)
                    converted = run_batch(proposal, (experiment,), preset=new, config=session.config, **kwargs)
                    self.assertEqual(converted.results[0].status, "skipped")
                    changed = deepcopy(session.config)
                    changed["images"]["frames_per_merge"] = 2
                    changed_result = run_batch(proposal, (experiment,), preset=new, config=changed, **kwargs)
                    self.assertEqual(changed_result.results[0].status, "failed")
                    self.assertIn("Analysis name", changed_result.results[0].message)


if __name__ == "__main__":
    unittest.main()

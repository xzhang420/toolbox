import json
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tifffile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from lumacam_pipeline import (  # noqa: E402
    LayoutError,
    backends,
    create_session,
    discover_dataset,
    find_layout_root,
)
from lumacam_pipeline.section import SectionState  # noqa: E402
from lumacam_pipeline.gui import LumacamPipelineGUI  # noqa: E402
from lumacam_pipeline.config import load_config, save_config  # noqa: E402
from lumacam_pipeline.batch import (  # noqa: E402
    discover_batch_experiments,
    prepare_shared_ob_background,
    run_batch,
    validate_batch,
)
from lumacam_pipeline.metadata_ranges import (  # noqa: E402
    inspect_metadata_time_ranges,
    save_metadata_time_ranges_csv,
)


def make_markerless_proposal(
    root,
    variants=("3to1000",),
    values=(1.0, 2.0),
    experiment="exp001",
):
    proposal = root / "proposal"
    layout = proposal / "data" / "experiments"
    runs = [f"{experiment}_{index:05d}" for index in range(len(values))]
    for name in ("derived", "logs", ".work"):
        (layout / name).mkdir(parents=True, exist_ok=True)
    metadata_root = layout / "metadata" / experiment
    metadata_root.mkdir(parents=True)
    (metadata_root / "experiment.json").write_text(
        json.dumps(
            {
                "experiment": experiment,
                "expected_runs": runs,
                "runs": {run: {"status": "completed"} for run in runs},
            }
        ),
        encoding="utf-8",
    )
    for index, (run, value) in enumerate(zip(runs, values)):
        (layout / "tpx3Files" / experiment / run).mkdir(parents=True)
        run_metadata = metadata_root / run
        run_metadata.mkdir(parents=True)
        (run_metadata / "acquisition.json").write_text(
            json.dumps(
                {
                    "run": run,
                    "status": "completed",
                    "started_at": f"2026-08-20T12:{index:02d}:00+00:00",
                    "ended_at": f"2026-08-20T12:{index:02d}:30+00:00",
                }
            ),
            encoding="utf-8",
        )
        for variant in variants:
            final = layout / "final" / experiment / variant
            final.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                final / f"{run}_sumImage_{variant}.tiff",
                np.full((6, 6), value, dtype=np.float32),
            )
    for variant in variants:
        tifffile.imwrite(
            layout / "final" / experiment / variant / f"{experiment}_sumImage_{variant}_sumAllRuns.tiff",
            np.full((6, 6), sum(values), dtype=np.float32),
        )
    return proposal, layout, experiment, runs


class MarkerlessLayoutTests(unittest.TestCase):
    def test_image_products_loads_multi_frame_exposure_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, layout, experiment, runs = make_markerless_proposal(
                Path(temporary), values=(4.0,), experiment="exp401_GPE_Cell7Au"
            )
            base_ns = 1_700_000_000_000_000_000
            frames = []
            for index in range(3):
                start_ns = base_ns + index * 2_000_000_000
                end_ns = start_ns + 2_000_000_000
                frames.append(
                    {
                        "frame_index": index,
                        "start_unix_ns": start_ns,
                        "end_unix_ns": end_ns,
                        "midpoint_unix_ns": start_ns + 1_000_000_000,
                    }
                )
            manifest = {
                "schema": "lumacam-tpx3-exposure-manifest",
                "schema_version": 1,
                "frames": frames,
            }
            (layout / "metadata" / experiment / runs[0] / "tpx3_exposure_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            session = create_session(
                "nci_images_only",
                {
                    "project": {
                        "data_path": str(proposal),
                        "experiment": "exp401",
                        "variant": "3to1000",
                    }
                },
            )
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            outcome = session.run("images")
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            self.assertEqual(len(session.artifacts.require("image_stack")), 1)
            self.assertEqual(len(session.artifacts.require("image_times")), 1)

    def test_short_experiment_id_resolves_to_full_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(
                Path(temporary), experiment="exp401_GPE_Cell7Au"
            )
            dataset = discover_dataset(proposal, experiment="exp401", variant="3to1000")
            self.assertEqual(dataset.experiment, experiment)

    def test_ambiguous_experiment_prefix_lists_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, _, _ = make_markerless_proposal(
                root, values=(1.0,), experiment="exp401_GPE_Cell7Au"
            )
            make_markerless_proposal(
                root, values=(2.0,), experiment="exp402_GPE_Cell7Au"
            )
            with self.assertRaisesRegex(LayoutError, "ambiguous"):
                discover_dataset(proposal, experiment="exp40", variant="3to1000")

    def test_short_id_remains_selected_during_correction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, selected, _ = make_markerless_proposal(
                root, values=(1.0,), experiment="exp401_GPE_Cell7Au"
            )
            make_markerless_proposal(
                root, values=(2.0,), experiment="exp402_GPE_Cell7Au"
            )
            session = create_session(
                "nci_images_only",
                {
                    "project": {
                        "data_path": str(proposal),
                        "experiment": "exp401",
                        "variant": "3to1000",
                    }
                },
            )
            project = session.run("project")
            self.assertEqual(project.state, SectionState.COMPLETED, project.message)
            self.assertIn("matching per-run TIFF", project.message)
            self.assertNotIn("v2", project.message)
            self.assertNotIn("selected image", project.message)
            self.assertEqual(session.artifacts.require("experiment"), selected)
            correction = session.run("corrections")
            self.assertEqual(correction.state, SectionState.COMPLETED, correction.message)
            self.assertIn(selected, str(session.artifacts.require("processed_image_dir")))

    def test_markerless_layout_is_authoritative_and_aggregate_is_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, layout, experiment, runs = make_markerless_proposal(Path(temporary))
            self.assertFalse((layout / "metadata" / "layout.json").exists())
            self.assertEqual(find_layout_root(proposal), layout)
            dataset = discover_dataset(proposal)
            self.assertEqual(dataset.layout_type, "v2")
            self.assertEqual(dataset.experiment, experiment)
            self.assertEqual([run.name for run in dataset.runs], runs)
            self.assertEqual(len(dataset.products), 3)
            self.assertEqual(len(dataset.image_products()), 2)
            aggregate = [product for product in dataset.products if product.aggregate]
            self.assertEqual(len(aggregate), 1)
            self.assertIsNone(aggregate[0].run)
            self.assertFalse((layout / "metadata" / "layout.json").exists())

    def test_shared_nci_backend_accepts_markerless_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, runs = make_markerless_proposal(Path(temporary))
            resolved = backends.nci_data().resolve_nci_experiment(str(proposal))
            self.assertEqual(resolved["layout_type"], "v2")
            self.assertEqual(resolved["experiment"], experiment)
            self.assertEqual([run["name"] for run in resolved["runs"]], runs)


class MetadataTimeRangeTests(unittest.TestCase):
    def test_manifest_range_is_used_without_accessing_raw_tpx3_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, layout, experiment, runs = make_markerless_proposal(
                root, values=(4.0,), experiment="exp401_GPE_Cell7Au"
            )
            start_ns = 1_700_000_000_123_456_000
            end_ns = start_ns + 2_500_000_000
            metadata_dir = layout / "metadata" / experiment / runs[0]
            (metadata_dir / "tpx3_exposure_manifest.json").write_text(
                json.dumps(
                    {
                        "frames": [
                            {
                                "start_unix_ns": start_ns,
                                "end_unix_ns": end_ns,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            raw_path = layout / "tpx3Files" / experiment / runs[0] / "untrusted.tpx3"
            raw_path.write_bytes(b"raw")

            original_stat = Path.stat

            def reject_raw_stat(path, *args, **kwargs):
                if path.suffix.lower() == ".tpx3":
                    raise AssertionError("Raw TPX3 files must not be inspected")
                return original_stat(path, *args, **kwargs)

            with patch.object(Path, "stat", reject_raw_stat):
                report = inspect_metadata_time_ranges(proposal)

            expected_start = datetime.fromtimestamp(
                start_ns // 1_000_000_000, tz=timezone.utc
            ).replace(microsecond=(start_ns % 1_000_000_000) // 1000)
            expected_end = datetime.fromtimestamp(
                end_ns // 1_000_000_000, tz=timezone.utc
            ).replace(microsecond=(end_ns % 1_000_000_000) // 1000)
            self.assertEqual(report.start_utc, expected_start)
            self.assertEqual(report.end_utc, expected_end)
            self.assertEqual(report.experiments[0].sources, ("exposure_manifest",))
            self.assertEqual(report.experiments[0].exposure_frame_count, 1)
            csv_path = save_metadata_time_ranges_csv(report, root / "ranges.csv")
            self.assertTrue(csv_path.is_file())
            self.assertIn("exp401_GPE_Cell7Au", csv_path.read_text(encoding="utf-8"))

    def test_acquisition_json_is_the_metadata_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, _, _ = make_markerless_proposal(
                Path(temporary), values=(1.0,), experiment="exp402_sample"
            )
            report = inspect_metadata_time_ranges(proposal, experiments=("exp402",))
            row = report.experiments[0]
            self.assertEqual(row.sources, ("acquisition_json",))
            self.assertEqual(row.timed_run_count, 1)
            self.assertEqual(row.start_utc.isoformat(), "2026-08-20T12:00:00+00:00")
            self.assertEqual(row.end_utc.isoformat(), "2026-08-20T12:00:30+00:00")


class GuiLayoutTests(unittest.TestCase):
    def test_all_dashboard_labels_and_button_rows_use_full_content_layouts(self):
        gui = LumacamPipelineGUI("operando_nci")

        def walk(widget):
            yield widget
            for child in getattr(widget, "children", ()):
                yield from walk(child)

        widgets = list(walk(gui.widget))
        described = [widget for widget in widgets if getattr(widget, "description", "")]
        buttons = [widget for widget in described if isinstance(widget, gui.w.Button)]
        fields = [widget for widget in described if not isinstance(widget, gui.w.Button)]
        rows = [widget for widget in widgets if isinstance(widget, gui.w.HBox)]

        self.assertGreaterEqual(len(fields), 20)
        self.assertGreaterEqual(len(buttons), 20)
        self.assertTrue(all(widget.style.description_width == "initial" for widget in fields))
        self.assertTrue(all(widget.layout.width == "auto" for widget in buttons))
        self.assertTrue(all(widget.layout.flex_flow == "row wrap" for widget in rows))

    def test_batch_transfer_lists_commit_and_retain_explicit_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, exp401, _ = make_markerless_proposal(
                root, values=(1.0,), experiment="exp401_sample"
            )
            _, _, exp402, _ = make_markerless_proposal(
                root, values=(2.0,), experiment="exp402_sample"
            )
            gui = LumacamPipelineGUI("nci_images_only")
            gui.data_path.value = str(proposal)
            gui._data_selected(proposal)

            self.assertEqual(gui.batch_selected_experiments.options, ())
            self.assertEqual(
                set(gui.batch_unselected_experiments.options), {exp401, exp402}
            )

            gui.batch_unselected_experiments.value = (exp401,)
            gui._add_batch_experiments()
            self.assertEqual(gui._batch_selected_names(), (exp401,))
            self.assertNotIn(exp401, gui.batch_unselected_experiments.options)

            gui.batch_filter.value = "exp402"
            gui._populate_batch_choices()
            self.assertEqual(gui._batch_selected_names(), (exp401,))
            self.assertEqual(gui.batch_unselected_experiments.options, (exp402,))

            gui._add_all_batch_experiments()
            self.assertEqual(gui._batch_selected_names(), (exp401, exp402))
            gui.metadata_time_scope.value = "batch"
            self.assertEqual(gui._metadata_time_selection(), (exp401, exp402))

            gui.batch_selected_experiments.value = (exp401,)
            gui._remove_batch_experiments()
            self.assertEqual(gui._batch_selected_names(), (exp402,))

    def test_dashboard_has_clickable_inline_term_definitions(self):
        gui = LumacamPipelineGUI("operando_nci")
        gui.show_all_sections.value = True

        def walk(widget):
            yield widget
            for child in getattr(widget, "children", ()):
                yield from walk(child)

        help_widgets = [
            widget
            for widget in walk(gui.widget)
            if isinstance(widget, gui.w.HTML)
            and "lumacam-term-help" in str(widget.value)
        ]
        self.assertGreaterEqual(len(help_widgets), 10)
        self.assertTrue(all("<details" in widget.value for widget in help_widgets))
        self.assertTrue(all("<summary" in widget.value for widget in help_widgets))
        self.assertTrue(any("Frames/merge" in widget.value for widget in help_widgets))
        self.assertTrue(any("Missing artifacts" in widget.value for widget in help_widgets))

    def test_section_status_can_be_folded_or_shown(self):
        gui = LumacamPipelineGUI("operando_nci")
        self.assertIsNone(gui.status_accordion.selected_index)
        gui.status_accordion.selected_index = 0
        self.assertEqual(gui.status_accordion.selected_index, 0)
        self.assertIn("Section status", gui.status_accordion.get_title(0))

    def test_shared_preparation_updates_proposal_main_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, ob_name, _ = make_markerless_proposal(
                root, experiment="exp001_OB", values=(2.0, 2.0)
            )
            _, _, background_name, _ = make_markerless_proposal(
                root, experiment="exp002_BG", values=(0.5, 0.5)
            )
            gui = LumacamPipelineGUI("exsitu")
            gui.data_path.value = str(proposal)
            gui._data_selected(proposal)
            gui.variant.value = "3to1000"
            gui.shared_ob_experiment.value = "exp001"
            gui.shared_background_experiment.value = "exp002"
            gui.background_count.value = 2

            gui._prepare_shared_inputs()

            main_config = (
                proposal / "analysis" / "lumacam_pipeline.ini"
            )
            self.assertTrue(main_config.is_file())
            _, values = load_config(main_config)
            self.assertEqual(values["shared_setup"]["ob_experiment"], ob_name)
            self.assertEqual(
                values["shared_setup"]["background_experiment"], background_name
            )
            self.assertTrue(Path(values["background"]["background_path"]).is_file())
            self.assertTrue(values["background"]["enabled"])
            self.assertIn("ready", gui.shared_status_html.value)
            self.assertEqual(gui.config_path.value, str(main_config))
        self.assertIn("<table", gui.status_html.value)

    def test_batch_selection_is_restored_into_selected_box_from_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, exp401, _ = make_markerless_proposal(
                root, values=(1.0,), experiment="exp401_sample"
            )
            _, _, exp402, _ = make_markerless_proposal(
                root, values=(2.0,), experiment="exp402_sample"
            )
            config_path = root / "saved_batch.ini"
            save_config(
                config_path,
                "nci_images_only",
                {
                    "project": {
                        "data_path": str(proposal),
                        "experiment": exp401,
                        "variant": "3to1000",
                    },
                    "batch": {
                        "name": "saved_batch",
                        "experiments": [exp402],
                        "sections": ["images"],
                    },
                },
            )
            gui = LumacamPipelineGUI("operando_nci")
            gui.config_path.value = str(config_path)
            gui._load_config(None)

            self.assertEqual(gui._batch_selected_names(), (exp402,))
            self.assertIn(exp401, gui.batch_unselected_experiments.options)
            self.assertEqual(gui.batch_sections.value, ("images",))

    def test_inline_browser_selects_a_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "proposal"
            selected.mkdir()
            gui = LumacamPipelineGUI("operando_nci")
            gui.path_browser.show(
                gui.data_path,
                mode="directory",
                start=root,
                on_select=gui._data_selected,
            )
            gui.path_browser.entries.value = str(selected)
            gui.path_browser._choose_selected(None)
            self.assertEqual(gui.data_path.value, str(selected.resolve()))
            self.assertEqual(gui.path_browser.widget.layout.display, "none")

    def test_beam_history_browser_accepts_files_or_folders(self):
        gui = LumacamPipelineGUI("operando_nci")

        gui.beam_browse_btn.click()
        self.assertEqual(gui.path_browser.mode, "either")
        self.assertNotEqual(gui.path_browser.current_button.layout.display, "none")
        self.assertEqual(gui.path_browser.current_button.description, "Use current folder")
        gui.path_browser.hide()

        gui.beam_path.value = "/example/shared_beamhistory"
        gui.session.section_config("corrections")["reference_beamhistory_path"] = "/old/ob.csv"
        gui._sync_config()
        self.assertEqual(
            gui.session.section_config("corrections")["reference_beamhistory_path"],
            gui.beam_path.value,
        )

    def test_data_browser_populates_experiment_and_variant_choices(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(
                Path(temporary),
                variants=("3to1000", "5to1000"),
                experiment="exp401_GPE_Cell7Au",
            )
            gui = LumacamPipelineGUI("operando_nci")
            gui.data_path.value = str(proposal)
            gui._data_selected(proposal)
            self.assertIn(experiment, gui.experiment.options)
            self.assertEqual(gui.experiment.value, experiment)
            self.assertEqual(set(gui.variant.options), {"3to1000", "5to1000"})
            gui.experiment.value = "exp401"
            self.assertEqual(gui.experiment.value, "exp401")

    def test_gui_displays_and_saves_metadata_only_time_ranges(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(
                Path(temporary), values=(1.0,), experiment="exp401_GPE_Cell7Au"
            )
            gui = LumacamPipelineGUI("nci_images_only")
            gui.data_path.value = str(proposal)
            gui.experiment.value = "exp401"
            gui.metadata_time_scope.value = "current"
            gui._inspect_metadata_times()
            self.assertIn(experiment, gui.metadata_time_html.value)
            self.assertIn("JSON metadata only", gui.metadata_time_html.value)
            self.assertIsNotNone(gui._metadata_time_report)
            gui._save_metadata_times(None)
            self.assertTrue(
                (
                    proposal
                    / "analysis"
                    / "proposal_summary"
                    / "metadata_datetime_ranges.csv"
                ).is_file()
            )

    def test_each_run_updates_the_proposal_main_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(
                Path(temporary), experiment="exp401_GPE_Cell7Au"
            )
            gui = LumacamPipelineGUI("nci_images_only")
            gui.data_path.value = str(proposal)
            gui.experiment.value = "exp401"
            gui.variant.value = "3to1000"

            gui._run("project")
            experiment_root = proposal / "analysis" / experiment / "3to1000"
            automatic = (
                proposal / "analysis" / "lumacam_pipeline.ini"
            )
            self.assertTrue(automatic.is_file())
            preset, values = load_config(automatic)
            self.assertEqual(preset, "operando_nci")
            self.assertEqual(values["project"]["experiment"], experiment)

            gui.frames_per_merge.value = 2
            gui._run("images")
            _, values = load_config(automatic)
            self.assertEqual(values["images"]["frames_per_merge"], 2)
            self.assertEqual(gui.config_path.value, str(automatic))
            self.assertIn(str(automatic), gui.proposal_config_html.value)
            self.assertEqual(gui.session.artifacts.require("results_dir"), experiment_root)
            self.assertFalse((experiment_root / "lumacam_pipeline.ini").exists())


class SectionIndependenceTests(unittest.TestCase):
    def test_exsitu_average_copies_to_proposal_and_refreshes_by_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(
                Path(temporary), values=(2.0, 4.0), variants=("3to1000", "5to1000")
            )
            session = create_session("exsitu", {
                "project": {"data_path": str(proposal), "variant": "3to1000"},
                "average": {"filename": "custom_average.tiff"},
            })
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            self.assertEqual(session.run("images").state, SectionState.COMPLETED)
            outcome = session.run("average")
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            source = session.artifacts.require("average_image_path")
            copied = session.artifacts.require("average_collection_path")
            self.assertEqual(copied, proposal / "analysis" / "proposal_summary"
                             / "averaged_images" / f"average_{experiment}_3to1000.tiff")
            self.assertEqual(source.read_bytes(), copied.read_bytes())
            session.artifacts.put("image_stack", [np.full((6, 6), 9.0)], "images")
            self.assertEqual(session.run("average").state, SectionState.COMPLETED)
            np.testing.assert_allclose(tifffile.imread(copied), 9.0)
            session.configure("project", variant="5to1000")
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            self.assertEqual(session.run("images").state, SectionState.COMPLETED)
            self.assertEqual(session.run("average").state, SectionState.COMPLETED)
            self.assertNotEqual(session.artifacts.require("average_collection_path"), copied)
            self.assertTrue(copied.exists())

    def test_exsitu_average_collection_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, _, _ = make_markerless_proposal(Path(temporary))
            session = create_session("exsitu", {
                "project": {"data_path": str(proposal), "variant": "3to1000"},
                "average": {"collect_in_proposal": False},
            })
            session.run("project")
            session.run("images")
            self.assertEqual(session.run("average").state, SectionState.COMPLETED)
            self.assertIsNone(session.artifacts.require("average_collection_path"))
            self.assertFalse((proposal / "analysis" / "proposal_summary"
                              / "averaged_images").exists())

    def test_exsitu_images_and_average_do_not_require_echem_or_corrections(self):
        with tempfile.TemporaryDirectory() as temporary:
            image_dir = Path(temporary) / "sample"
            image_dir.mkdir()
            tifffile.imwrite(image_dir / "a.tiff", np.full((4, 4), 2, dtype=np.float32))
            tifffile.imwrite(image_dir / "b.tiff", np.full((4, 4), 4, dtype=np.float32))
            session = create_session(
                "image_only",
                {"project": {"data_path": str(image_dir), "variant": "default"}},
            )
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            self.assertEqual(session.status("electrochemistry"), SectionState.NOT_CONFIGURED)
            self.assertEqual(session.run("images").state, SectionState.COMPLETED)
            outcome = session.run("average")
            self.assertEqual(outcome.state, SectionState.COMPLETED)
            self.assertIn("2 source TIFF(s)", outcome.message)
            self.assertEqual(outcome.metadata["source_tiff_count"], 2)
            np.testing.assert_allclose(session.artifacts.require("average_image"), 3.0)
            average_path = session.artifacts.require("average_image_path")
            self.assertTrue(average_path.is_file())
            self.assertEqual(average_path.parent.name, "images")
            self.assertEqual(session.run("selection").state, SectionState.COMPLETED)
            self.assertEqual(session.run("image_plot").state, SectionState.COMPLETED)
            self.assertEqual(session.artifacts.require("image_plot_path").parent.name, "figures")
            self.assertEqual(session.status("synchronization"), SectionState.WAITING)

    def test_individual_image_figures_use_named_experiment_subfolder(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, _, experiment, _ = make_markerless_proposal(
                Path(temporary), experiment="exp401_GPE_Cell7Au"
            )
            session = create_session(
                "nci_images_only",
                {
                    "project": {
                        "data_path": str(proposal),
                        "experiment": experiment,
                        "variant": "3to1000",
                    },
                    "individual_image_plots": {
                        "enabled": True,
                        "format": "png",
                        "color_scale": "global",
                    },
                },
            )
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            self.assertEqual(session.run("images").state, SectionState.COMPLETED)
            outcome = session.run("individual_image_plots")
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            paths = session.artifacts.require("individual_image_plot_paths")
            self.assertEqual(len(paths), 2)
            expected_parent = proposal / "analysis" / experiment / "3to1000" / "figures" / "individual_images"
            self.assertTrue(all(path.parent == expected_parent for path in paths))
            self.assertTrue(all(path.name.startswith(f"{experiment}_image_") for path in paths))
            self.assertTrue(all(path.is_file() for path in paths))

    def test_markerless_correction_preserves_metadata_timing_without_optional_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, layout, _, _ = make_markerless_proposal(Path(temporary))
            session = create_session(
                "nci_images_only",
                {"project": {"data_path": str(proposal), "variant": "3to1000"}},
            )
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            correction = session.run("corrections")
            self.assertEqual(correction.state, SectionState.COMPLETED, correction.message)
            result = session.artifacts.require("correction_result")
            self.assertFalse(result["intensity_corrected"])
            self.assertFalse(result["reference_corrected"])
            self.assertEqual(result["timing_sources"], ["acquisition_json"])
            experiment_root = proposal / "analysis" / "exp001" / "3to1000"
            self.assertEqual(session.artifacts.require("results_dir"), experiment_root)
            self.assertEqual(Path(result["corrected_dir"]).parent, experiment_root / "images")
            self.assertEqual(Path(result["metadata_path"]).parent, experiment_root / "processing")
            self.assertFalse((layout / "metadata" / "layout.json").exists())


class PhotonHyperspectralTests(unittest.TestCase):
    def test_cube_averages_only_per_run_exact_photon_products(self):
        with tempfile.TemporaryDirectory() as temporary:
            proposal, layout, experiment, runs = make_markerless_proposal(
                Path(temporary), variants=("2to2", "3to3"), values=(2.0, 4.0)
            )
            session = create_session(
                "image_only",
                {
                    "project": {"data_path": str(proposal), "variant": "2to2"},
                    "photon_hyperspectral": {"enabled": True, "photon_min": 2, "photon_max": 3},
                },
            )
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            outcome = session.run("photon_hyperspectral")
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            cube = session.artifacts.require("photon_histogram_cube")
            self.assertEqual(cube.shape, (2, 6, 6))
            np.testing.assert_allclose(cube[0], 3.0)
            np.testing.assert_allclose(cube[1], 3.0)
            histogram_path = session.artifacts.require("photon_histogram_path")
            self.assertTrue(histogram_path.is_file())
            self.assertEqual(histogram_path.parent, proposal / "analysis" / experiment / "2to2" / "tables" / "photon_hyperspectral")
            feature_path = next(iter(session.artifacts.require("photon_feature_paths").values()))
            self.assertEqual(feature_path.parent, proposal / "analysis" / experiment / "2to2" / "images" / "photon_hyperspectral")
            self.assertFalse((layout / "metadata" / "layout.json").exists())


class TimeOfFlightTests(unittest.TestCase):
    def test_reconstructed_cube_runs_without_nci_or_electrochemistry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_dir = root / "derived_images"
            image_dir.mkdir()
            tifffile.imwrite(image_dir / "overview.tiff", np.ones((4, 4), dtype=np.float32))
            cube_path = root / "tof_cube.tiff"
            cube = np.stack(
                [np.full((4, 4), value, dtype=np.float32) for value in (1, 2, 3)], axis=0
            )
            tifffile.imwrite(cube_path, cube)
            session = create_session(
                "tof",
                {
                    "project": {
                        "data_path": str(image_dir),
                        "variant": "default",
                        "product_kinds": "final",
                    },
                    "tof_cube": {
                        "enabled": True,
                        "path": str(cube_path),
                        "axis_values": [10, 20, 30],
                        "bands": [[10, 20]],
                    },
                },
            )
            self.assertEqual(session.run("project").state, SectionState.COMPLETED)
            outcome = session.run("tof_cube")
            self.assertEqual(outcome.state, SectionState.COMPLETED, outcome.message)
            np.testing.assert_array_equal(session.artifacts.require("tof_axis"), [10, 20, 30])
            paths = session.artifacts.require("tof_feature_paths")
            self.assertIn("tof_band_10_to_20.tiff", paths)
            self.assertTrue(paths["tof_mean_axis_value.tiff"].is_file())
            experiment_root = root / "analysis" / "derived_images" / "default"
            self.assertEqual(paths["tof_mean_axis_value.tiff"].parent, experiment_root / "images" / "tof")
            self.assertEqual(session.artifacts.require("tof_spectrum_path").parent, experiment_root / "tables" / "tof")
            self.assertEqual(session.status("electrochemistry"), SectionState.NOT_CONFIGURED)


class BatchProcessingTests(unittest.TestCase):
    def test_shared_ob_and_background_are_prepared_once_and_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, sample, _ = make_markerless_proposal(
                root, experiment="exp101_sample", values=(4.0,)
            )
            _, _, ob_name, _ = make_markerless_proposal(
                root, experiment="exp001_OB", values=(2.0, 2.0)
            )
            _, _, background_name, _ = make_markerless_proposal(
                root, experiment="exp002_BG", values=(0.5, 0.5)
            )
            config = {
                "project": {
                    "data_path": str(proposal),
                    "variant": "3to1000",
                    "product_kinds": "final",
                },
                "corrections": {
                    "reference_bin_y": 1,
                    "reference_bin_x": 1,
                    "beamhistory_path": "",
                    "reference_beamhistory_path": "",
                },
                "background": {"n_images": 2},
            }

            prepared = prepare_shared_ob_background(
                proposal,
                "exp001",
                "exp002",
                preset="exsitu",
                config=config,
            )

            self.assertEqual(prepared.ob_experiment, ob_name)
            self.assertEqual(prepared.background_experiment, background_name)
            self.assertEqual(prepared.reference_tiff_count, 2)
            self.assertEqual(prepared.background_source_tiff_count, 2)
            self.assertEqual(prepared.background_tiff_count, 2)
            self.assertTrue(prepared.prepared_reference_path.is_file())
            self.assertTrue(prepared.prepared_background_path.is_file())
            np.testing.assert_allclose(
                tifffile.imread(prepared.prepared_reference_path), 2.0
            )
            np.testing.assert_allclose(
                tifffile.imread(prepared.prepared_background_path), 0.25
            )
            self.assertEqual(config["shared_setup"]["ob_experiment"], ob_name)
            self.assertEqual(
                Path(config["background"]["background_path"]),
                prepared.prepared_background_path,
            )

            summary = run_batch(
                proposal,
                (sample,),
                preset="exsitu",
                config=config,
                sections=("corrections", "background", "images", "average"),
                batch_name="prepared_shared_inputs",
            )
            self.assertEqual(summary.results[0].status, "completed")
            average = (
                proposal / "analysis" / sample / "3to1000" / "images" / f"average_{sample}_3to1000.tiff"
            )
            np.testing.assert_allclose(tifffile.imread(average), 1.75)
            from lumacam_pipeline.average_collection import collect_completed_batch
            collection = proposal / "analysis" / "proposal_summary" / "averaged_images"
            copied = collection / f"average_{sample}_3to1000.tiff"
            self.assertEqual(copied.read_bytes(), average.read_bytes())
            # Older finished batches can be collected without processing images again.
            copied.unlink()
            self.assertEqual(collect_completed_batch(summary.status_path), [copied])
            self.assertFalse(copied.exists())
            collect_completed_batch(summary.status_path, apply=True)
            self.assertEqual(copied.read_bytes(), average.read_bytes())

    def test_batch_validates_runs_resumes_and_reuses_one_reference_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, layout, exp101, _ = make_markerless_proposal(
                root, experiment="exp101_sample_a", values=(10.0, 12.0)
            )
            _, _, exp102, _ = make_markerless_proposal(
                root, experiment="exp102_sample_b", values=(14.0, 16.0)
            )
            _, _, reference, _ = make_markerless_proposal(
                root, experiment="exp001_OB", values=(2.0,)
            )
            config = {
                "project": {
                    "data_path": str(proposal),
                    "variant": "3to1000",
                    "product_kinds": "final",
                    "include_aggregates": False,
                },
                "corrections": {
                    "reference_dir": str(layout / "final" / reference),
                    "reference_bin_y": 1,
                    "reference_bin_x": 1,
                    "file_filter": "3to1000",
                    "beamhistory_path": "",
                    "reference_beamhistory_path": "",
                },
                "individual_image_plots": {
                    "format": "png",
                    "color_scale": "global",
                },
            }
            discovered = discover_batch_experiments(proposal, hide_setup=True)
            self.assertIn(exp101, discovered)
            self.assertIn(exp102, discovered)
            self.assertNotIn(reference, discovered)

            report = validate_batch(
                proposal,
                ("exp101", "exp102"),
                preset="nci_images_only",
                config=config,
            )
            self.assertTrue(report.valid, report.common_errors)
            self.assertEqual([row.experiment for row in report.rows], [exp101, exp102])

            summary = run_batch(
                proposal,
                (exp101, exp102),
                preset="nci_images_only",
                config=config,
                batch_name="shared_ob",
            )
            self.assertEqual([result.status for result in summary.results], ["completed", "completed"])
            self.assertTrue(summary.status_path.is_file())
            self.assertTrue(summary.recipe_path.is_file())
            recipe_preset, recipe_values = load_config(summary.recipe_path)
            self.assertEqual(recipe_preset, "nci_images_only")
            self.assertEqual(recipe_values["batch"]["experiments"], [exp101, exp102])
            self.assertIn("individual_image_plots", recipe_values["batch"]["sections"])
            caches = list(
                (proposal / "analysis" / "shared_corrections").glob(
                    "*/ob.tiff"
                )
            )
            self.assertEqual(len(caches), 1)
            for experiment in (exp101, exp102):
                experiment_root = proposal / "analysis" / experiment / "3to1000"
                config_path = experiment_root / "processing" / "config_used.ini"
                self.assertTrue(config_path.is_file())
                _, saved = load_config(config_path)
                self.assertEqual(saved["project"]["experiment"], experiment)
                self.assertEqual(
                    Path(saved["corrections"]["prepared_reference_path"]), caches[0]
                )
                figures = list(
                    (experiment_root / "figures" / "individual_images").glob("*.png")
                )
                self.assertEqual(len(figures), 2)
                self.assertTrue(
                    (experiment_root / "processing" / "batch_shared_ob.json").is_file()
                )

            resumed = run_batch(
                proposal,
                (exp101, exp102),
                preset="nci_images_only",
                config=config,
                batch_name="shared_ob",
            )
            self.assertEqual([result.status for result in resumed.results], ["skipped", "skipped"])

    def test_batch_continues_after_one_experiment_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            proposal, _, first, _ = make_markerless_proposal(
                root, experiment="exp101_sample_a", values=(1.0,)
            )
            _, _, second, _ = make_markerless_proposal(
                root, experiment="exp102_sample_b", values=(2.0,)
            )
            config = {
                "project": {
                    "data_path": str(proposal),
                    "variant": "3to1000",
                    "product_kinds": "final",
                }
            }
            summary = run_batch(
                proposal,
                (first, second),
                preset="nci_images_only",
                config=config,
                sections=("images", "individual_image_plots"),
                batch_name="failure_isolation",
                experiment_overrides={
                    first: {"individual_image_plots": {"indices": "99"}}
                },
            )
            self.assertEqual(
                [result.status for result in summary.results], ["failed", "completed"]
            )
            self.assertEqual(summary.results[0].failed_section, "individual_image_plots")
            self.assertTrue(
                list(
                    (
                        proposal
                        / "analysis"
                        / second
                        / "3to1000"
                        / "figures"
                        / "individual_images"
                    ).glob("*.png")
                )
            )


if __name__ == "__main__":
    unittest.main()

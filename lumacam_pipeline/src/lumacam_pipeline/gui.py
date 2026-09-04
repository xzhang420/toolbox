"""Notebook dashboard with independent, unnumbered processing sections."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from html import escape
from io import StringIO
from pathlib import Path

from .batch import (
    BATCH_SECTION_LABELS,
    DEFAULT_BATCH_SECTIONS,
    discover_batch_experiments,
    run_batch,
    save_batch_plan,
    validate_batch,
)
from .config import load_config, save_config
from .layout import (
    LayoutError,
    discover_dataset,
    find_layout_root,
    list_experiments,
    resolve_experiment_name,
)
from .metadata_ranges import inspect_metadata_time_ranges, save_metadata_time_ranges_csv
from .path_browser import InlinePathBrowser
from .presets import PRESETS, create_session


class LumacamPipelineGUI:
    """Small ipywidgets dashboard backed by :class:`PipelineSession`."""

    def __init__(self, preset="operando_nci"):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except Exception as exc:
            raise RuntimeError("ipywidgets and IPython are required for the notebook GUI") from exc
        self.w = widgets
        self.display = display
        self._loading_config = False
        self._metadata_time_report = None
        self.session = create_session(preset)
        self._build_widgets(preset)
        self._wire()
        self.refresh_status()

    def _text(self, description, value="", width="95%"):
        return self.w.Text(
            description=description,
            value=str(value),
            style={"description_width": "initial"},
            layout=self.w.Layout(width=width, min_width="280px"),
        )

    def _choice(self, description, value="", width="95%"):
        return self.w.Combobox(
            description=description,
            options=(),
            value=str(value),
            ensure_option=False,
            style={"description_width": "initial"},
            layout=self.w.Layout(width=width, min_width="280px"),
        )

    def _int(self, description, value, width="260px"):
        return self.w.IntText(
            description=description,
            value=value,
            style={"description_width": "initial"},
            layout=self.w.Layout(width=width, min_width="220px", margin="2px 12px 2px 0"),
        )

    def _float(self, description, value, width="280px"):
        return self.w.FloatText(
            description=description,
            value=value,
            style={"description_width": "initial"},
            layout=self.w.Layout(width=width, min_width="240px", margin="2px 12px 2px 0"),
        )

    def _checkbox(self, description, value=False):
        minimum = max(220, len(description) * 8 + 48)
        return self.w.Checkbox(
            description=description,
            value=value,
            style={"description_width": "initial"},
            layout=self.w.Layout(width="auto", min_width=f"{minimum}px"),
        )

    def _button(self, description, style=""):
        minimum = max(120, len(description) * 8 + 32)
        return self.w.Button(
            description=description,
            button_style=style,
            layout=self.w.Layout(
                width="auto", min_width=f"{minimum}px", margin="2px 8px 2px 0"
            ),
        )

    def _row(self, children):
        return self.w.HBox(
            children,
            layout=self.w.Layout(
                width="100%", display="flex", flex_flow="row wrap", align_items="center"
            ),
        )

    def _section(self, children):
        return self.w.VBox(children, layout=self.w.Layout(width="100%"))

    def _path_row(self, field, button):
        field.layout.width = "auto"
        field.layout.flex = "1 1 520px"
        return self._row([field, button])

    def _term_help(self, terms):
        """Return compact, browser-native click-to-expand terminology help."""

        entries = []
        for term, description in terms:
            entries.append(
                "<details style='display:inline-block;vertical-align:top;margin:2px 12px 2px 0'>"
                "<summary style='cursor:pointer;color:#1864ab'>"
                f"{escape(term)} &#9432;</summary>"
                "<span style='display:block;max-width:520px;padding:4px 8px;"
                "color:#343a40;background:#f8f9fa;border-left:3px solid #74c0fc'>"
                f"{escape(description)}</span></details>"
            )
        return self.w.HTML(
            "<div class='lumacam-term-help' style='width:100%;margin:3px 0 8px 0'>"
            "<small style='color:#666;margin-right:10px'>Click a term for help:</small>"
            + "".join(entries)
            + "</div>",
            layout=self.w.Layout(width="100%"),
        )

    def _build_widgets(self, preset):
        self.preset = self.w.Dropdown(
            description="Preset",
            options=sorted(PRESETS),
            value=preset,
            style={"description_width": "initial"},
            layout=self.w.Layout(width="420px", min_width="280px"),
        )
        self.config_path = self._text(
            "Config", str(Path(__file__).resolve().parents[2] / "lumacam_pipeline_config.ini")
        )
        self.config_browse_btn = self._button("Browse")
        self.load_config_btn = self._button("Load config")
        self.save_config_btn = self._button("Save config")
        self.data_path = self._text("Data")
        self.data_browse_btn = self._button("Browse")
        self.experiment = self._choice("Experiment")
        self.variant = self._choice("Variant", "3to1000")
        self.result_root = self._text("Results")
        self.result_browse_btn = self._button("Browse")
        self.include_aggregates = self._checkbox("Include all-runs aggregate products")
        self.discover_btn = self._button("Discover data", "primary")
        self.metadata_time_scope = self.w.Dropdown(
            description="Time-range scope",
            options=(
                ("Entire proposal", "proposal"),
                ("Current experiment", "current"),
                ("Selected batch experiments", "batch"),
            ),
            value="proposal",
            style={"description_width": "initial"},
            layout=self.w.Layout(width="360px", min_width="280px"),
        )
        self.metadata_time_margin = self._int("Download margin (min)", 5, width="280px")
        self.inspect_metadata_time_btn = self._button("Inspect metadata date range")
        self.save_metadata_time_btn = self._button("Save date-range CSV")
        self.metadata_time_html = self.w.HTML(
            "<small>Date ranges are read from JSON metadata only.</small>",
            layout=self.w.Layout(
                width="100%", max_height="420px", overflow="auto", border="1px solid #eee"
            ),
        )

        self.beam_path = self._text("Beam history")
        self.beam_browse_btn = self._button("Browse")
        self.reference_path = self._text("Reference")
        self.reference_browse_btn = self._button("Browse")
        self.reference_beam_path = self._text("Reference beam")
        self.reference_beam_browse_btn = self._button("Browse")
        self.run_correction_btn = self._button("Run corrections")
        self.background_enabled = self._checkbox("Enable background correction")
        self.background_path = self._text("Background")
        self.background_browse_btn = self._button("Browse")
        self.background_count = self._int("First images", 7)
        self.run_background_btn = self._button("Run background")

        self.frames_per_merge = self._int("Frames/merge", 1)
        self.spatial_merge = self._int("Spatial merge", 1)
        self.load_images_btn = self._button("Load images", "primary")
        self.average_btn = self._button("Write average")

        self.roi_x = self._int("ROI x", 256, width="210px")
        self.roi_y = self._int("ROI y", 256, width="210px")
        self.roi_r = self._int("ROI radius", 111, width="240px")
        self.calibrated = self._checkbox("Treat source images as calibrated NCI thickness")
        self.measure_btn = self._button("Measure images")
        self.plating_btn = self._button("Calculate NCI plating metrics")
        self.photon_btn = self._button("Build photon-count cube")
        self.tof_path = self._text("ToF cube")
        self.tof_browse_btn = self._button("Browse")
        self.tof_btn = self._button("Analyze reconstructed ToF cube")

        self.echem_path = self._text("EC data")
        self.echem_browse_btn = self._button("Browse")
        self.echem_conv = self._float("Capacity→μm", 3.66)
        self.load_echem_btn = self._button("Load electrochemistry")
        self.sync_btn = self._button("Synchronize")
        self.events_btn = self._button("Detect EC events")

        self.selected_indices = self._text("Indices")
        self.select_btn = self._button("Select images")
        self.image_plot_btn = self._button("Save image figure")
        self.individual_plot_indices = self._text("Individual indices (blank = all)")
        self.individual_plot_format = self.w.Dropdown(
            description="Individual format",
            options=("png", "pdf", "svg", "tiff"),
            value="png",
            style={"description_width": "initial"},
            layout=self.w.Layout(width="300px", min_width="240px"),
        )
        self.individual_color_scale = self.w.Dropdown(
            description="Individual color scale",
            options=(("Same scale for all", "global"), ("Scale each image", "individual")),
            value="global",
            style={"description_width": "initial"},
            layout=self.w.Layout(width="360px", min_width="280px"),
        )
        self.individual_plots_btn = self._button("Save individual image figures")
        self.combined_plot_btn = self._button("Save combined figure")
        self.nci_outputs_btn = self._button("Save established NCI/EC outputs")
        self.image_video_btn = self._button("Export image video")
        self.combined_video_btn = self._button("Export synchronized video")

        self.batch_name = self._text("Batch name", "batch_3to1000")
        self.batch_filter = self._text("Experiment filter")
        self.batch_hide_setup = self._checkbox("Hide OB, background, test, and focus data", True)
        self.batch_discover_btn = self._button("Discover batch experiments", "primary")
        self.batch_unselected_experiments = self.w.SelectMultiple(
            description="Currently unselected",
            options=(),
            value=(),
            rows=12,
            style={"description_width": "initial"},
            layout=self.w.Layout(width="100%", min_width="340px"),
        )
        self.batch_selected_experiments = self.w.SelectMultiple(
            description="Selected",
            options=(),
            value=(),
            rows=12,
            style={"description_width": "initial"},
            layout=self.w.Layout(width="100%", min_width="340px"),
        )
        self.batch_add_btn = self._button("Add highlighted →", "primary")
        self.batch_add_all_btn = self._button("Add all shown →")
        self.batch_remove_btn = self._button("← Remove highlighted")
        self.batch_clear_btn = self._button("Clear selected")
        self._batch_all_names = ()
        self._batch_visible_names = ()
        self.batch_sections = self.w.SelectMultiple(
            description="Processing sections",
            options=tuple((label, section) for section, label in BATCH_SECTION_LABELS.items() if section != "project"),
            value=DEFAULT_BATCH_SECTIONS,
            rows=11,
            style={"description_width": "initial"},
            layout=self.w.Layout(width="100%", min_width="420px"),
        )
        self.batch_overwrite = self._checkbox("Re-run jobs already completed with the same recipe")
        self.batch_validate_btn = self._button("Validate batch")
        self.batch_run_btn = self._button("Run or resume batch", "primary")
        self.batch_status_html = self.w.HTML(
            "<small>Select explicit experiments, then validate before running.</small>",
            layout=self.w.Layout(width="100%", overflow="auto"),
        )

        self.output = self.w.Output(
            layout=self.w.Layout(
                width="100%", border="1px solid #ddd", max_height="280px", overflow="auto"
            )
        )
        self.status_html = self.w.HTML(layout=self.w.Layout(width="100%", overflow="auto"))
        self.proposal_config_html = self.w.HTML(
            "<small>Automatic proposal config: available after selecting a proposal and experiment.</small>"
        )
        self.path_browser = InlinePathBrowser(self.w)

        project_help = self._term_help([
            ("Preset", "Chooses a sensible set of available processing sections for the data type."),
            ("Config", "A file containing saved paths and processing settings. Proposal copies update automatically."),
            ("Variant", "The reconstructed TIFF product name or energy window, for example 3to1000."),
            ("Aggregate product", "One TIFF summed across all runs; excluded from the time series by default."),
            ("Metadata date range", "The acquisition window read from JSON metadata, without inspecting raw TPX3 files."),
            ("Time-range scope", "Chooses whether the reported acquisition window covers the proposal, one experiment, or the batch."),
            ("Download margin", "Extra time added before and after acquisition when choosing beam-history data."),
            ("TPX3", "The detector's raw event-data file format."),
        ])
        batch_help = self._term_help([
            ("Batch", "The same selected processing recipe applied to several experiments."),
            ("Experiment filter", "Limits the left box only; experiments already in the selected box stay selected."),
            ("Processing sections", "The actions that each selected experiment will run."),
            ("Prerequisite", "A required earlier action that the pipeline adds automatically when necessary."),
            ("Resume", "Skips an unchanged experiment that already completed with the same recipe."),
            ("Reference / OB", "An open-beam or reference image reused for normalization across the batch."),
        ])
        image_help = self._term_help([
            ("Frames/merge", "Averages this many consecutive time points into each output frame."),
            ("Spatial merge", "Averages neighbouring pixel blocks; 2 means 2 by 2 pixels."),
            ("Average image", "One image containing the mean of every currently loaded frame."),
        ])
        correction_help = self._term_help([
            ("Beam history", "Recorded beam intensity over time, used to compensate images when the beam changes."),
            ("Reference / OB", "An open-beam or reference image used to remove the illumination and detector pattern."),
            ("Reference beam", "Beam-history data covering the time when the reference was acquired."),
            ("Background correction", "Subtracts a background image estimated from the chosen input images."),
            ("First images", "The number of earliest background images averaged to estimate the background."),
        ])
        measurement_help = self._term_help([
            ("NCI", "Neutron capture imaging."),
            ("ROI", "Region of interest: the circular image area used for measurements."),
            ("Calibrated NCI thickness", "Marks source values as physical NCI thickness rather than arbitrary intensity."),
            ("Plating metrics", "NCI-derived measurements intended for lithium-plating analysis."),
        ])
        spectral_help = self._term_help([
            ("Photon-count cube", "A stack that groups image events by detected photon count."),
            ("ToF cube", "A reconstructed three-dimensional time-of-flight dataset."),
        ])
        electrochemistry_help = self._term_help([
            ("EC data", "Electrochemistry time-series data such as time, voltage, current, and capacity."),
            ("Capacity→μm", "Factor converting electrochemical capacity to an estimated thickness in micrometres."),
            ("Synchronize", "Aligns image acquisition times with electrochemical measurement times."),
            ("EC events", "Automatically identified electrochemical transitions used for plots and image selection."),
        ])
        output_help = self._term_help([
            ("Indices", "Zero-based image numbers to select; leave blank where the control says blank means all."),
            ("Color scale", "Maps image values to colours; one shared scale makes frames directly comparable."),
            ("Combined output", "A figure or video showing images together with synchronized electrochemistry."),
            ("Synchronized video", "A video in which image frames and electrochemical position advance together."),
        ])
        status_help = self._term_help([
            ("Ready", "All required inputs for this section are available."),
            ("Completed", "The most recent run of this section finished successfully."),
            ("Waiting for input", "Run the listed prerequisite section or provide its required data."),
            ("Not configured", "This optional section is disabled or its optional input path is blank."),
            ("Failed", "The most recent run stopped with an error; read Details and the output box."),
            ("Missing artifacts", "Results from other sections that this section needs before it can run."),
        ])

        project = self._section([
            project_help,
            self.preset, self._path_row(self.config_path, self.config_browse_btn),
            self._row([self.load_config_btn, self.save_config_btn]),
            self._path_row(self.data_path, self.data_browse_btn),
            self.experiment, self.variant,
            self._path_row(self.result_root, self.result_browse_btn),
            self.include_aggregates, self.discover_btn, self.proposal_config_html,
            self._row([
                self.metadata_time_scope,
                self.metadata_time_margin,
                self.inspect_metadata_time_btn,
                self.save_metadata_time_btn,
            ]),
            self.metadata_time_html,
        ])
        batch = self._section([
            batch_help,
            self.w.HTML(
                "<small>The current pipeline settings are shared across the selected experiments. "
                "The Reference field in Corrections is prepared once and reused.</small>"
            ),
            self.batch_name,
            self._row([self.batch_filter, self.batch_hide_setup, self.batch_discover_btn]),
            self.w.HBox(
                [
                    self.w.VBox(
                        [
                            self.batch_unselected_experiments,
                            self._row([self.batch_add_btn, self.batch_add_all_btn]),
                        ],
                        layout=self.w.Layout(width="100%", min_width="360px", flex="1 1 440px"),
                    ),
                    self.w.VBox(
                        [
                            self.batch_selected_experiments,
                            self._row([self.batch_remove_btn, self.batch_clear_btn]),
                        ],
                        layout=self.w.Layout(width="100%", min_width="360px", flex="1 1 440px"),
                    ),
                ],
                layout=self.w.Layout(
                    width="100%", display="flex", flex_flow="row wrap", align_items="flex-start"
                ),
            ),
            self.batch_sections,
            self.batch_overwrite,
            self._row([self.batch_validate_btn, self.batch_run_btn]),
            self.batch_status_html,
        ])
        corrections = self._section([
            correction_help,
            self._path_row(self.beam_path, self.beam_browse_btn),
            self._path_row(self.reference_path, self.reference_browse_btn),
            self._path_row(self.reference_beam_path, self.reference_beam_browse_btn),
            self.run_correction_btn, self.background_enabled,
            self._path_row(self.background_path, self.background_browse_btn),
            self.background_count, self.run_background_btn,
        ])
        images = self._section([
            image_help,
            self._row([self.frames_per_merge, self.spatial_merge]),
            self._row([self.load_images_btn, self.average_btn]),
        ])
        measurements = self._section([
            measurement_help,
            self._row([self.roi_x, self.roi_y, self.roi_r]), self.calibrated,
            self._row([self.measure_btn, self.plating_btn]),
        ])
        spectral = self._section([
            spectral_help,
            self.photon_btn, self._path_row(self.tof_path, self.tof_browse_btn), self.tof_btn
        ])
        electrochemistry = self._section([
            electrochemistry_help,
            self._path_row(self.echem_path, self.echem_browse_btn), self.echem_conv,
            self._row([self.load_echem_btn, self.sync_btn, self.events_btn]),
        ])
        outputs = self._section([
            output_help,
            self.selected_indices,
            self._row([self.select_btn, self.image_plot_btn, self.combined_plot_btn]),
            self.individual_plot_indices,
            self._row([
                self.individual_plot_format,
                self.individual_color_scale,
                self.individual_plots_btn,
            ]),
            self.nci_outputs_btn,
            self._row([self.image_video_btn, self.combined_video_btn]),
        ])
        self.accordion = self.w.Accordion(
            children=[project, batch, images, corrections, measurements, spectral, electrochemistry, outputs],
            layout=self.w.Layout(width="100%"),
        )
        for index, title in enumerate([
            "Project and data", "Batch processing", "Image products", "Corrections", "Image measurements",
            "Photon-count and time-of-flight", "Electrochemistry and synchronization",
            "Figures, video, and export",
        ]):
            self.accordion.set_title(index, title)
        self.widget = self.w.VBox(
            [
                self.path_browser.widget,
                self.accordion,
                self.w.HTML("<h4>Section status</h4>"),
                status_help,
                self.status_html,
                self.output,
            ],
            layout=self.w.Layout(width="100%"),
        )

    def _wire(self):
        self.preset.observe(self._change_preset, names="value")
        self.experiment.observe(self._experiment_changed, names="value")
        self.config_browse_btn.on_click(
            lambda _: self._browse_path(
                self.config_path,
                mode="file",
                title="Select a pipeline config",
                extensions=(".ini",),
            )
        )
        self.data_browse_btn.on_click(
            lambda _: self._browse_path(
                self.data_path,
                mode="directory",
                title="Select a proposal or image folder",
                category="data",
                on_select=self._data_selected,
            )
        )
        self.result_browse_btn.on_click(
            lambda _: self._browse_path(
                self.result_root,
                mode="directory",
                title="Select a results folder",
                category="results",
            )
        )
        self.beam_browse_btn.on_click(
            lambda _: self._browse_path(
                self.beam_path,
                mode="file",
                title="Select beam-history data",
                category="beamhistory",
            )
        )
        self.reference_browse_btn.on_click(
            lambda _: self._browse_path(
                self.reference_path,
                mode="either",
                title="Select a reference file or folder",
                category="reference",
            )
        )
        self.reference_beam_browse_btn.on_click(
            lambda _: self._browse_path(
                self.reference_beam_path,
                mode="file",
                title="Select reference beam-history data",
                category="beamhistory",
            )
        )
        self.background_browse_btn.on_click(
            lambda _: self._browse_path(
                self.background_path,
                mode="either",
                title="Select a background file or folder",
                category="reference",
            )
        )
        self.tof_browse_btn.on_click(
            lambda _: self._browse_path(
                self.tof_path,
                mode="file",
                title="Select a reconstructed ToF cube",
                extensions=(".tif", ".tiff", ".npy", ".npz"),
            )
        )
        self.echem_browse_btn.on_click(
            lambda _: self._browse_path(
                self.echem_path,
                mode="either",
                title="Select an electrochemistry file or folder",
                category="echem",
            )
        )
        self.load_config_btn.on_click(self._load_config)
        self.save_config_btn.on_click(self._save_config)
        self.discover_btn.on_click(lambda _: self._run("project"))
        self.inspect_metadata_time_btn.on_click(self._inspect_metadata_times)
        self.save_metadata_time_btn.on_click(self._save_metadata_times)
        self.run_correction_btn.on_click(lambda _: self._run("corrections"))
        self.run_background_btn.on_click(lambda _: self._run("background"))
        self.load_images_btn.on_click(lambda _: self._run("images"))
        self.average_btn.on_click(lambda _: self._run("average"))
        self.measure_btn.on_click(lambda _: self._run("image_measurements"))
        self.plating_btn.on_click(lambda _: self._run("nci_plating_metrics"))
        self.photon_btn.on_click(lambda _: self._run("photon_hyperspectral"))
        self.tof_btn.on_click(lambda _: self._run("tof_cube"))
        self.load_echem_btn.on_click(lambda _: self._run("electrochemistry"))
        self.sync_btn.on_click(lambda _: self._run("synchronization"))
        self.events_btn.on_click(lambda _: self._run("event_detection"))
        self.select_btn.on_click(lambda _: self._run("selection"))
        self.image_plot_btn.on_click(lambda _: self._run("image_plot"))
        self.individual_plots_btn.on_click(lambda _: self._run("individual_image_plots"))
        self.combined_plot_btn.on_click(lambda _: self._run("combined_plot"))
        self.nci_outputs_btn.on_click(lambda _: self._run("nci_event_outputs"))
        self.image_video_btn.on_click(lambda _: self._run("image_video"))
        self.combined_video_btn.on_click(lambda _: self._run("combined_video"))
        self.batch_discover_btn.on_click(self._populate_batch_choices)
        self.batch_add_btn.on_click(self._add_batch_experiments)
        self.batch_add_all_btn.on_click(self._add_all_batch_experiments)
        self.batch_remove_btn.on_click(self._remove_batch_experiments)
        self.batch_clear_btn.on_click(self._clear_batch_experiments)
        self.batch_validate_btn.on_click(self._validate_batch)
        self.batch_run_btn.on_click(self._run_batch)

    def _proposal_subfolder(self, category):
        raw = str(self.data_path.value).strip()
        if not raw:
            return None
        selected = Path(raw).expanduser()
        layout_root = find_layout_root(selected)
        if layout_root is not None:
            proposal = layout_root.parent.parent if layout_root.parent.name == "data" else layout_root.parent
        elif selected.is_dir():
            proposal = selected
        else:
            return None
        relatives = {
            "results": ("analysis",),
            "beamhistory": ("data", "beamhistory"),
            "reference": ("data", "reference"),
            "echem": ("data", "echem"),
        }
        candidate = proposal.joinpath(*relatives.get(category, ()))
        return candidate if candidate.is_dir() else proposal

    def _browse_path(
        self,
        target,
        *,
        mode,
        title,
        extensions=(),
        category=None,
        on_select=None,
    ):
        current = str(target.value).strip()
        start = current or self._proposal_subfolder(category)
        if not start and category == "data" and Path("/data01").is_dir():
            start = Path("/data01")
        self.path_browser.show(
            target,
            mode=mode,
            title=title,
            extensions=extensions,
            start=start,
            on_select=on_select,
        )

    def _data_selected(self, selected):
        self._metadata_time_report = None
        self.metadata_time_html.value = "<small>Date ranges are read from JSON metadata only.</small>"
        self._populate_project_choices(selected)
        self._populate_batch_choices()

    def _metadata_time_selection(self):
        scope = self.metadata_time_scope.value
        if scope == "proposal":
            return None
        if scope == "current":
            experiment = str(self.experiment.value).strip()
            if not experiment:
                raise ValueError("Enter or select the current experiment first.")
            return (experiment,)
        experiments = self._batch_selected_names()
        if not experiments:
            raise ValueError("Select experiments in Batch processing first.")
        return experiments

    @staticmethod
    def _display_datetime(value, local=False):
        if value is None:
            return "unavailable"
        shown = value.astimezone() if local else value
        zone = shown.tzname() or ("local" if local else "UTC")
        return shown.strftime("%Y-%m-%d %H:%M:%S") + f" {zone}"

    def _render_metadata_times(self, report):
        margin = max(0, int(self.metadata_time_margin.value))
        padded_start, padded_end = report.padded_utc_range(margin)
        summary = (
            f"<b>Metadata acquisition range</b><br>"
            f"Exact UTC: <code>{escape(self._display_datetime(report.start_utc))}</code> to "
            f"<code>{escape(self._display_datetime(report.end_utc))}</code><br>"
            f"Local time: <code>{escape(self._display_datetime(report.start_utc, True))}</code> to "
            f"<code>{escape(self._display_datetime(report.end_utc, True))}</code><br>"
            f"Suggested beam-history window ({margin}-minute margin): "
            f"<code>{escape(self._display_datetime(padded_start))}</code> to "
            f"<code>{escape(self._display_datetime(padded_end))}</code><br>"
            f"Timed experiments: {report.timed_experiment_count}/{len(report.experiments)}. "
            "Source: JSON metadata only; raw TPX3 timestamps were not inspected."
        )
        rows = []
        for item in report.experiments:
            source = ", ".join(item.sources) or "missing"
            missing = len(item.missing_runs)
            color = "#087f5b" if item.start_utc is not None and not missing else "#b36b00"
            rows.append(
                f"<tr><td>{escape(item.experiment)}</td>"
                f"<td>{escape(self._display_datetime(item.start_utc))}</td>"
                f"<td>{escape(self._display_datetime(item.end_utc))}</td>"
                f"<td>{item.timed_run_count}/{item.run_count}</td>"
                f"<td>{item.exposure_frame_count}</td>"
                f"<td style='color:{color}'>{escape(source)}"
                + (f"; {missing} run(s) missing" if missing else "")
                + "</td></tr>"
            )
        self.metadata_time_html.value = (
            summary
            + "<table style='width:100%;margin-top:8px'>"
            "<tr><th align='left'>Experiment</th><th align='left'>Start UTC</th>"
            "<th align='left'>End UTC</th><th align='left'>Timed runs</th>"
            "<th align='left'>Frames</th><th align='left'>Metadata source</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    def _inspect_metadata_times(self, _=None):
        try:
            report = inspect_metadata_time_ranges(
                self.data_path.value,
                experiments=self._metadata_time_selection(),
            )
            self._metadata_time_report = report
            self._render_metadata_times(report)
            message = (
                f"Metadata range: {self._display_datetime(report.start_utc)} to "
                f"{self._display_datetime(report.end_utc)}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.metadata_time_html.value = (
                f"<small style='color:#c92a2a'>{escape(message)}</small>"
            )
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _save_metadata_times(self, _):
        try:
            self._metadata_time_report = inspect_metadata_time_ranges(
                self.data_path.value,
                experiments=self._metadata_time_selection(),
            )
            self._render_metadata_times(self._metadata_time_report)
            path = save_metadata_time_ranges_csv(
                self._metadata_time_report,
                margin_minutes=max(0, int(self.metadata_time_margin.value)),
            )
            message = f"Saved metadata date ranges: {path}"
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _populate_batch_choices(self, _=None):
        try:
            all_names = tuple(
                discover_batch_experiments(
                    self.data_path.value,
                    hide_setup=False,
                    filter_text="",
                )
            )
            visible_names = tuple(
                discover_batch_experiments(
                    self.data_path.value,
                    hide_setup=self.batch_hide_setup.value,
                    filter_text=self.batch_filter.value,
                )
            )
            retained = tuple(name for name in self._batch_selected_names() if name in all_names)
            self._batch_all_names = all_names
            self._batch_visible_names = visible_names
            self._set_batch_selected(retained)
            self._show_batch_selection_status("Experiment list updated.")
        except Exception as exc:
            self._batch_all_names = ()
            self._batch_visible_names = ()
            self.batch_unselected_experiments.options = ()
            self.batch_unselected_experiments.value = ()
            self.batch_selected_experiments.options = ()
            self.batch_selected_experiments.value = ()
            self.batch_status_html.value = (
                f"<small style='color:#c92a2a'>{escape(type(exc).__name__ + ': ' + str(exc))}</small>"
            )

    def _batch_selected_names(self):
        """Return every experiment committed to the right-hand batch list."""

        return tuple(self.batch_selected_experiments.options)

    def _set_batch_selected(self, names):
        requested = set(names)
        ordered = tuple(name for name in self._batch_all_names if name in requested)
        self.batch_selected_experiments.options = ordered
        self.batch_selected_experiments.value = ()
        self.batch_unselected_experiments.options = tuple(
            name for name in self._batch_visible_names if name not in requested
        )
        self.batch_unselected_experiments.value = ()

    def _show_batch_selection_status(self, prefix=""):
        selected_count = len(self._batch_selected_names())
        available_count = len(self.batch_unselected_experiments.options)
        lead = f"{escape(prefix)} " if prefix else ""
        self.batch_status_html.value = (
            f"<small>{lead}{selected_count} selected; "
            f"{available_count} currently unselected experiment(s) shown.</small>"
        )

    def _add_batch_experiments(self, _=None):
        chosen = set(self.batch_unselected_experiments.value)
        if chosen:
            self._set_batch_selected((*self._batch_selected_names(), *chosen))
        self._show_batch_selection_status()

    def _add_all_batch_experiments(self, _=None):
        self._set_batch_selected(
            (*self._batch_selected_names(), *self.batch_unselected_experiments.options)
        )
        self._show_batch_selection_status()

    def _remove_batch_experiments(self, _=None):
        removed = set(self.batch_selected_experiments.value)
        self._set_batch_selected(
            name for name in self._batch_selected_names() if name not in removed
        )
        self._show_batch_selection_status()

    def _clear_batch_experiments(self, _=None):
        self._set_batch_selected(())
        self._show_batch_selection_status()

    def _batch_inputs(self):
        self._sync_config()
        experiments = self._batch_selected_names()
        sections = tuple(self.batch_sections.value)
        if not experiments:
            raise ValueError("Select at least one experiment in Batch processing.")
        if not sections:
            raise ValueError("Select at least one processing section.")
        return experiments, sections

    def _render_batch_rows(self, rows, heading):
        table_rows = []
        for row in rows:
            if hasattr(row, "valid"):
                status = "valid" if row.valid else "invalid"
                message = row.message
            else:
                status = row.status
                message = row.message
            color = "#087f5b" if status in {"valid", "completed", "skipped"} else "#c92a2a"
            table_rows.append(
                f"<tr><td>{escape(row.experiment)}</td>"
                f"<td style='color:{color}'>{escape(status)}</td>"
                f"<td>{escape(str(message))}</td></tr>"
            )
        self.batch_status_html.value = (
            f"<b>{escape(heading)}</b><table style='width:100%'>"
            "<tr><th align='left'>Experiment</th><th align='left'>State</th>"
            "<th align='left'>Details</th></tr>" + "".join(table_rows) + "</table>"
        )

    def _validate_batch(self, _):
        try:
            experiments, sections = self._batch_inputs()
            report = validate_batch(
                self.data_path.value,
                experiments,
                preset=self.preset.value,
                config=self.session.config,
                sections=sections,
            )
            target = save_batch_plan(
                report,
                preset=self.preset.value,
                config=self.session.config,
                batch_name=self.batch_name.value,
            )
            heading = f"Validation {'passed' if report.valid else 'failed'}; plan: {target}"
            if report.common_errors:
                heading += " | " + " | ".join(report.common_errors)
            self._render_batch_rows(report.rows, heading)
            message = heading
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.batch_status_html.value = f"<small style='color:#c92a2a'>{escape(message)}</small>"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _run_batch(self, _):
        try:
            experiments, sections = self._batch_inputs()

            def progress(event, payload):
                experiment = escape(str(payload.get("experiment", "")))
                self.batch_status_html.value = (
                    f"<small>{escape(event.replace('_', ' ').title())}: {experiment}</small>"
                )

            summary = run_batch(
                self.data_path.value,
                experiments,
                preset=self.preset.value,
                config=self.session.config,
                sections=sections,
                batch_name=self.batch_name.value,
                overwrite=self.batch_overwrite.value,
                progress=progress,
            )
            self._render_batch_rows(
                summary.results,
                f"Batch finished; status: {summary.status_path}",
            )
            message = (
                f"Batch finished: {len(summary.results)} job(s), "
                f"{summary.failed_count} failed. Status: {summary.status_path}"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.batch_status_html.value = f"<small style='color:#c92a2a'>{escape(message)}</small>"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _populate_project_choices(self, data_path=None):
        raw = str(data_path or self.data_path.value).strip()
        if not raw:
            self.experiment.options = ()
            self.variant.options = ()
            return
        layout_root = find_layout_root(raw)
        if layout_root is None:
            self.experiment.options = ()
            self.variant.options = ()
            return
        names = tuple(list_experiments(layout_root))
        self.experiment.options = names
        current = str(self.experiment.value).strip()
        if current:
            try:
                resolve_experiment_name(current, names)
            except LayoutError:
                self.experiment.value = names[0] if len(names) == 1 else ""
        elif len(names) == 1:
            self.experiment.value = names[0]
        self._populate_variant_choices()

    def _populate_variant_choices(self):
        raw_data = str(self.data_path.value).strip()
        experiment = str(self.experiment.value).strip()
        if not raw_data or not experiment or find_layout_root(raw_data) is None:
            return
        try:
            dataset = discover_dataset(raw_data, experiment=experiment)
        except (OSError, ValueError):
            return
        variants = tuple(sorted({product.variant for product in dataset.products if product.kind == "final"}))
        self.variant.options = variants
        current = str(self.variant.value).strip()
        if not current and len(variants) == 1:
            self.variant.value = variants[0]

    def _experiment_changed(self, change):
        if str(change.get("new", "")).strip():
            self._populate_variant_choices()

    def _change_preset(self, change):
        if self._loading_config:
            return
        self.session = create_session(change["new"])
        self.refresh_status()

    def _load_config(self, _):
        try:
            preset, values = load_config(self.config_path.value)
            self._loading_config = True
            try:
                self.preset.value = preset
            finally:
                self._loading_config = False
            self.session = create_session(preset, values)
            project = self.session.section_config("project")
            corrections = self.session.section_config("corrections")
            background = self.session.section_config("background")
            images = self.session.section_config("images")
            measurements = self.session.section_config("image_measurements")
            tof = self.session.section_config("tof_cube")
            echem = self.session.section_config("electrochemistry")
            selection = self.session.section_config("selection")
            individual = self.session.section_config("individual_image_plots")
            batch_values = self.session.config.pop("batch", {})
            self.data_path.value = str(project.get("data_path", ""))
            self.experiment.value = str(project.get("experiment", ""))
            self.variant.value = str(project.get("variant", ""))
            self.result_root.value = str(project.get("result_root", ""))
            self.include_aggregates.value = bool(project.get("include_aggregates", False))
            self.beam_path.value = str(corrections.get("beamhistory_path", ""))
            self.reference_path.value = str(corrections.get("reference_dir", ""))
            self.reference_beam_path.value = str(corrections.get("reference_beamhistory_path", ""))
            self.background_enabled.value = bool(background.get("enabled", False))
            self.background_path.value = str(background.get("background_path", ""))
            self.background_count.value = int(background.get("n_images", 7))
            self.frames_per_merge.value = int(images.get("frames_per_merge", 1))
            self.spatial_merge.value = int(images.get("spatial_pixel_merge", 1))
            roi = measurements.get("roi") or {}
            self.roi_x.value = int(roi.get("x0", 256))
            self.roi_y.value = int(roi.get("y0", 256))
            self.roi_r.value = int(roi.get("r0", 111))
            self.calibrated.value = bool(measurements.get("calibrated", False))
            self.tof_path.value = str(tof.get("path", ""))
            self.echem_path.value = str(echem.get("path", ""))
            self.echem_conv.value = float(echem.get("capacity_to_thickness", 3.66))
            self.selected_indices.value = str(selection.get("indices", ""))
            self.individual_plot_indices.value = str(individual.get("indices", ""))
            self.individual_plot_format.value = str(individual.get("format", "png"))
            self.individual_color_scale.value = str(individual.get("color_scale", "global"))
            self.batch_name.value = str(batch_values.get("name", self.batch_name.value))
            self._populate_project_choices()
            self._populate_batch_choices()
            restored_experiments = tuple(
                value
                for value in batch_values.get("experiments", ())
                if value in self._batch_all_names
            )
            if restored_experiments:
                self._set_batch_selected(restored_experiments)
                self._show_batch_selection_status("Restored batch selection.")
            restored_sections = tuple(
                value
                for value in batch_values.get("sections", ())
                if value in {option[1] for option in self.batch_sections.options}
                and value != "project"
            )
            if restored_sections:
                self.batch_sections.value = restored_sections
            message = f"Loaded {self.config_path.value}"
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)
        self.refresh_status()

    def _save_config(self, _):
        self._sync_config()
        try:
            path = save_config(self.config_path.value, self.preset.value, self.session.config)
            message = f"Saved {path}"
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _sync_config(self):
        self.session.configure(
            "project", data_path=self.data_path.value, experiment=self.experiment.value,
            variant=self.variant.value, result_root=self.result_root.value,
            include_aggregates=self.include_aggregates.value,
        )
        self.session.configure(
            "corrections", beamhistory_path=self.beam_path.value,
            reference_dir=self.reference_path.value,
            reference_beamhistory_path=self.reference_beam_path.value,
            file_filter=self.variant.value,
            prepared_reference_path="",
        )
        self.session.configure(
            "background", enabled=self.background_enabled.value,
            background_path=self.background_path.value, n_images=self.background_count.value,
        )
        self.session.configure(
            "images", frames_per_merge=self.frames_per_merge.value,
            spatial_pixel_merge=self.spatial_merge.value,
        )
        roi = {"x0": self.roi_x.value, "y0": self.roi_y.value, "r0": self.roi_r.value}
        self.session.configure("image_measurements", roi=roi, calibrated=self.calibrated.value)
        self.session.configure("nci_plating_metrics", roi=roi)
        self.session.configure("tof_cube", path=self.tof_path.value)
        self.session.configure(
            "electrochemistry", path=self.echem_path.value,
            capacity_to_thickness=self.echem_conv.value,
        )
        self.session.configure("selection", indices=self.selected_indices.value)
        self.session.configure(
            "individual_image_plots",
            indices=self.individual_plot_indices.value,
            format=self.individual_plot_format.value,
            color_scale=self.individual_color_scale.value,
        )

    def _automatic_proposal_config_path(self):
        data_path = str(self.data_path.value).strip()
        if not data_path:
            return None
        layout_root = find_layout_root(data_path)
        if layout_root is None:
            return None
        proposal_root = (
            layout_root.parent.parent if layout_root.parent.name == "data" else layout_root.parent
        )
        experiment = self.session.artifacts.get("experiment")
        if not experiment:
            requested = str(self.experiment.value).strip()
            available = list_experiments(layout_root)
            if requested:
                try:
                    experiment = resolve_experiment_name(requested, available)
                except LayoutError:
                    return None
            elif len(available) == 1:
                experiment = available[0]
            else:
                return None
        experiment_name = Path(str(experiment)).name
        if experiment_name in {"", ".", ".."}:
            return None
        return proposal_root / "analysis" / experiment_name / "lumacam_pipeline.ini"

    def _save_automatic_proposal_config(self):
        path = self._automatic_proposal_config_path()
        if path is None:
            return None
        saved = save_config(path, self.preset.value, self.session.config)
        self.proposal_config_html.value = (
            f"<small>Automatic proposal config: <code>{escape(str(saved))}</code></small>"
        )
        return saved

    def _run(self, section_id):
        self._sync_config()
        if section_id in {
            "corrections", "electrochemistry", "event_detection", "nci_plating_metrics",
            "photon_hyperspectral", "tof_cube", "combined_plot", "nci_event_outputs",
            "image_video", "combined_video", "individual_image_plots",
        }:
            self.session.configure(section_id, enabled=True)
        automatic_config_path = None
        automatic_config_error = None
        try:
            automatic_config_path = self._save_automatic_proposal_config()
        except Exception as exc:
            automatic_config_error = f"{type(exc).__name__}: {exc}"
        capture = StringIO()
        with redirect_stdout(capture), redirect_stderr(capture):
            outcome = self.session.run(section_id)
        if section_id == "project" and outcome.state.value == "completed":
            resolved_experiment = self.session.artifacts.get("experiment")
            if resolved_experiment:
                self.experiment.value = str(resolved_experiment)
                # Canonicalizing the successful input must not invalidate the
                # project artifacts that were just discovered.
                self.session.section_config("project")["experiment"] = str(resolved_experiment)
            try:
                automatic_config_path = self._save_automatic_proposal_config()
                automatic_config_error = None
                if automatic_config_path is not None:
                    self.config_path.value = str(automatic_config_path)
            except Exception as exc:
                automatic_config_error = f"{type(exc).__name__}: {exc}"
        with self.output:
            self.output.clear_output(wait=True)
            detail = capture.getvalue()
            if detail:
                print(detail.rstrip())
            print(f"{outcome.state.value}: {outcome.message}")
            if automatic_config_path is not None:
                print(f"Proposal config updated: {automatic_config_path}")
            elif automatic_config_error:
                print(f"Warning: proposal config was not saved: {automatic_config_error}")
        self.refresh_status()

    def refresh_status(self):
        rows = []
        colors = {
            "ready": "#087f5b", "completed": "#1864ab", "not configured": "#666",
            "waiting for input": "#b36b00", "failed": "#c92a2a",
        }
        for row in self.session.summary():
            state = row["state"]
            missing = escape(", ".join(row["missing"]))
            details = escape(row.get("message", ""))
            rows.append(
                f"<tr><td>{escape(row['title'])}</td>"
                f"<td style='color:{colors.get(state, '#333')}'>{escape(state)}</td>"
                f"<td>{missing}</td><td>{details}</td></tr>"
            )
        self.status_html.value = (
            "<table style='width:100%'><tr><th align='left'>Section</th><th align='left'>State</th>"
            "<th align='left'>Missing artifacts</th><th align='left'>Details</th></tr>"
            + "".join(rows)
            + "</table>"
        )

    def show(self):
        self.display(self.widget)
        return self


def launch_lumacam_gui(preset="operando_nci"):
    return LumacamPipelineGUI(preset=preset).show()

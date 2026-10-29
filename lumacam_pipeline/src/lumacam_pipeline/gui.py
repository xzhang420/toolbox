"""Notebook dashboard with independent, unnumbered processing sections."""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from html import escape
from io import StringIO
from .proposal_paths import analysis_root_for_proposal, reduced_root_for_proposal, proposal_root_for_layout

from pathlib import Path

from .background import background_source
from .batch import (
    BATCH_SECTION_LABELS,
    DEFAULT_BATCH_SECTIONS,
    discover_batch_experiments,
    normalize_batch_sections,
    prepare_shared_ob_background,
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
from .modes import LEGACY_MODE_ALIASES, MODE_DESCRIPTIONS, MODE_OPTIONS, display_mode, visible_sections
from .path_browser import InlinePathBrowser
from .presets import PRESETS, create_session
from .prepared_inputs import prepared_ob_integral
from .sections.images import preview_white_spots


def _create_gui_session(preset, config=None):
    """Load legacy defaults before changing the displayed/saved mode name."""
    session = create_session(preset, config)
    if preset in LEGACY_MODE_ALIASES:
        # An absent flag meant False in legacy image modes. Materialize it so
        # saving/reopening as Ex situ cannot silently enable average collection.
        session.section_config("average").setdefault("collect_in_proposal", False)
    session.preset = display_mode(preset)
    return session


class LumacamPipelineGUI:
    """Small ipywidgets dashboard backed by :class:`PipelineSession`."""

    def __init__(self, preset="operando_nci", *, workspace="legacy"):
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except Exception as exc:
            raise RuntimeError("ipywidgets and IPython are required for the notebook GUI") from exc
        self.w = widgets
        if workspace not in {"legacy", "processing"}:
            raise ValueError("Use the separate VisualizationGUI for analysis and rendering.")
        self.workspace = workspace
        self.display = display
        self._loading_config = False
        self._updating_visibility = False
        self._metadata_time_report = None
        self.session = _create_gui_session(preset)
        if self.workspace == "processing":
            self.session.config["workflow"] = {"role": "processing"}
        self._batch_section_selection = self._default_batch_sections()
        self._build_widgets(self.session.preset)
        if self.workspace == "processing":
            self._add_processing_controls()
        self._wire()
        self._update_background_controls()
        self._apply_mode_visibility()
        self.refresh_status()

    def _default_batch_sections(self):
        sections = set(DEFAULT_BATCH_SECTIONS) & visible_sections(self.session.preset, self.session.sections)
        if self.workspace == "processing":
            from .processing import PROCESSING_SECTIONS
            sections &= PROCESSING_SECTIONS
        if not self.session.section_config("corrections").get("enabled", True):
            sections.discard("corrections")
        return sections

    def _add_processing_controls(self):
        self.load_images_btn.description = "Load and publish processed images"
        self.load_images_btn.layout.min_width = "340px"
        self.inspect_layout_btn = self._button("Inspect layout (read-only)")
        self.images_pattern = self._text("Images pattern (optional)")
        self.metadata_pattern = self._text("Metadata pattern (optional)")
        self.run_regex = self._text("Run-name regex (optional)")
        self.processing_pixel_size = self._text("Input pixel size (μm; blank if unknown)")
        self.processing_unit = self._text("Image units (blank if unknown)")
        self.qc_frame = self._int("QC frame (0-based)", 0)
        self.qc_button = self._button("Preview processed image (QC)")
        project = self._panels[0][1]
        project.children = (*project.children, self.inspect_layout_btn, self.images_pattern,
                            self.metadata_pattern, self.run_regex)
        images = self._panels[3][1]
        images.children = (*images.children, self.processing_pixel_size, self.processing_unit,
                           self._row([self.qc_frame, self.qc_button]),
                           self.w.HTML("<small>Load images publishes a portable stack in images/processed_frames/ and processing/processed_dataset.json. Use the visualization notebook for figures and videos.</small>"))
        self.inspect_layout_btn.on_click(self._inspect_processing_layout)
        self.qc_button.on_click(self._preview_processed_frame)

    def _inspect_processing_layout(self, _=None):
        import json
        from .inspection import inspect_layout
        self._sync_config()
        with self.output:
            self.output.clear_output(wait=True)
            try:
                print(json.dumps(inspect_layout(self.data_path.value, config=self.session.config), indent=2))
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")

    def _preview_processed_frame(self, _=None):
        import matplotlib.pyplot as plt
        with self.output:
            self.output.clear_output(wait=True)
            try:
                images = self.session.artifacts.require("image_stack")
                index = self.qc_frame.value
                if not 0 <= index < len(images):
                    raise ValueError(f"Frame must be between 0 and {len(images)-1}")
                figure, axis = plt.subplots()
                figure.colorbar(axis.imshow(images[index], cmap="inferno"), ax=axis)
                axis.set_title(f"Processed frame {index}; {len(images)} loaded frames")
                self.display(figure)
                plt.close(figure)
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")

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
        # A separate wrapping label avoids the inline widget-label ellipsis in
        # narrow notebook panes, independent of the browser's flexbox sizing.
        self.preset_label = self.w.HTML(
            "<div style='white-space:normal;overflow:visible;text-overflow:clip'>Processing mode</div>",
            layout=self.w.Layout(width="100%"),
        )
        self.preset = self.w.Dropdown(
            description="",
            options=MODE_OPTIONS,
            value=preset,
            style={"description_width": "initial"},
            layout=self.w.Layout(width="100%", min_width="260px"),
        )
        self.mode_selector = self.w.VBox(
            [self.preset_label, self.preset],
            layout=self.w.Layout(width="100%", max_width="480px", min_width="260px"),
        )
        self.show_all_sections = self._checkbox("Show all sections")
        self.mode_info = self.w.HTML(layout=self.w.Layout(width="100%"))
        self.config_path = self._text("Config (load proposal INI, or leave blank for new setup)")
        self.config_browse_btn = self._button("Browse")
        self.load_config_btn = self._button("Load config")
        self.save_config_btn = self._button("Save config")
        self.data_path = self._text("Data")
        self.data_browse_btn = self._button("Browse")
        self.experiment = self._choice("Experiment")
        self.variant = self._choice("Variant", self.session.section_config("project")["variant"])
        self.result_root = self._text("Results")
        self.analysis_name = self._text("Analysis name (blank = default)")
        self.result_browse_btn = self._button("Browse")
        self.include_aggregates = self._checkbox("Discover all-runs sums for preview only")
        self.discover_btn = self._button("Discover data", "primary")
        self.shared_ob_experiment = self._choice("OB experiment")
        self.reference_set = self._text("Preparation name (optional)")
        self.prepare_bg = self._checkbox("Also prepare a shared BG", preset == "exsitu")
        self.preparation_bg_count = self._int("BG frames for median", 7, width="420px")
        self.shared_background_experiment = self._choice("BG experiment")
        self.prepare_shared_btn = self._button("Prepare shared OB and BG", "primary")
        self.shared_status_html = self.w.HTML(
            "<small>Optional: skip this section if you already have processed OB/BG TIFFs.</small>",
            layout=self.w.Layout(width="100%", overflow="auto"),
        )
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
        self.reference_path = self._text("Original OB folder (legacy)")
        self.prepared_ob_path = self._text("Prepared OB TIFF")
        self.prepared_ob_browse_btn = self._button("Browse")
        self.ob_integral = self._float("OB beam integral / min (0 = automatic)", 0, width="480px")
        self.reference_browse_btn = self._button("Browse")
        self.run_correction_btn = self._button("Run corrections")
        self.background_enabled = self._checkbox("Enable background correction")
        self.background_clip = self._checkbox("Clip negative BG values to zero", False)
        self.background_source = self.w.Dropdown(
            description="BG source",
            options=(("Prepared BG TIFF", "shared"),
                     ("First X frames per experiment", "first_frames")),
            value="shared" if preset == "exsitu" else "first_frames",
            style={"description_width": "initial"},
            layout=self.w.Layout(width="95%", min_width="280px"),
        )
        self.background_source_help = self.w.HTML(layout=self.w.Layout(width="100%"))
        self.batch_background_info = self.w.HTML(layout=self.w.Layout(width="100%"))
        self.background_path = self._text("Prepared BG TIFF")
        self.background_browse_btn = self._button("Browse")
        self.background_path_row = self._path_row(self.background_path, self.background_browse_btn)
        self.background_count = self._int("First images", 7, width="420px")
        self.run_background_btn = self._button("Run background")

        self.frames_per_merge = self._int("Frames/merge", 1)
        self.spatial_merge = self._int("Spatial merge", 1)
        self.white_spot_enabled = self._checkbox("Remove white spots / hot pixels", False)
        self.white_spot_size = self._int("White-spot window (px)", 3, width="340px")
        self.white_spot_method = self.w.Dropdown(
            description="Threshold rule",
            options=(("Automatic noise (MAD)", "mad"), ("Absolute difference", "abs"),
                     ("Pyerre count-noise", "prop")),
            value="mad", style={"description_width": "initial"},
            layout=self.w.Layout(width="380px", min_width="280px"),
        )
        self.white_spot_threshold = self._float("White-spot threshold", 8.0, width="340px")
        self.white_spot_preview_index = self._int("Preview frame (0-based)", 0, width="340px")
        self.white_spot_preview_btn = self._button("Preview white-spot removal")
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
            "<small>Proposal main config: available after selecting a proposal.</small>"
        )
        self.path_browser = InlinePathBrowser(self.w)

        project_help = self._term_help([
            ("Processing mode", "Shows the tools relevant to your data type. Switching keeps your settings and completed results."),
            ("Show all sections", "Reveals advanced tools, batch choices and status rows hidden by the mode; it does not run them."),
            ("Config", "The proposal-level main file containing shared paths and processing settings."),
            ("Variant", "The reconstructed TIFF product name or energy window, for example 3to1000."),
            ("Analysis name", "Optional label for an alternative recipe. Results go under the experiment/variant/alternatives/name folder; existing results are kept."),
            ("Aggregate product", "One TIFF summed across all runs; excluded from the time series by default."),
            ("Metadata date range", "The acquisition window read from JSON metadata, without inspecting raw TPX3 files."),
            ("Time-range scope", "Chooses whether the reported acquisition window covers the proposal, one experiment, or the batch."),
            ("Download margin", "Extra time added before and after acquisition when choosing beam-history data."),
            ("TPX3", "The detector's raw event-data file format."),
        ])
        shared_help = self._term_help([
            ("Shared OB", "Open-beam TIFFs normalized and averaged once for all selected experiments."),
            ("Preparation name", "Optional output name, such as day2. Leave blank for the first preparation. Use a new name when changing the preparation settings to keep earlier files."),
            ("Shared BG", "Background TIFFs corrected with the shared OB, then combined for subtraction."),
            ("Prepare", "Builds shared OB, plus shared BG when selected, and records settings in the proposal config."),
            ("BG source", "Use one shared BG for all experiments, or estimate each experiment's own BG from its first X corrected frames."),
            ("Beam history", "One file or folder covering OB, BG, and sample acquisitions; each measurement uses its own acquisition times."),
            ("Skip preparation", "If averaged OB and median BG TIFFs already exist, select them directly in Use prepared OB/BG and correct samples."),
            ("First images", "X valid corrected TIFFs used for the pixel-wise median. Unreadable and all-zero images are skipped; if fewer than X remain, all valid images are used."),
        ])
        batch_help = self._term_help([
            ("Batch", "The same selected processing recipe applied to several experiments."),
            ("Experiment filter", "Limits the left box only; experiments already in the selected box stay selected."),
            ("Processing sections", "The actions that each selected experiment will run."),
            ("Prerequisite", "A required earlier action that the pipeline adds automatically when necessary."),
            ("Resume", "Skips an unchanged experiment that already completed with the same recipe."),
            ("Reference / OB", "An open-beam or reference image reused for normalization across the batch."),
            ("BG source", "Uses the selection in Use prepared OB/BG and correct samples for every experiment. Include Background correction in the batch steps to apply it."),
        ])
        image_help = self._term_help([
            ("White-spot removal", "Replaces bright outliers with their local spatial median before cropping/merging. Off by default. Original files stay untouched; inspect a preview because real small bright features can also be removed."),
            ("White-spot window", "Odd neighbourhood width in pixels; start with 3 (3 by 3). Larger windows detect larger spots but can remove real detail."),
            ("White-spot threshold", "Higher means fewer replacements. Automatic noise uses a multiplier of robust residual noise (start at 8); absolute uses current image units; Pyerre count-noise uses threshold times sqrt(max(local median, 0)), intended for count images."),
            ("Frames/merge", "Averages this many consecutive time points into each output frame."),
            ("Spatial merge", "Averages neighbouring pixel blocks; 2 means 2 by 2 pixels."),
            ("Average image", "One image containing the mean of every currently loaded frame."),
        ])
        correction_help = self._term_help([
            ("Beam history", "Recorded beam intensity over time, used to compensate images when the beam changes."),
            ("Reference / OB", "An open-beam or reference image used to remove the illumination and detector pattern."),
            ("Background correction", "Subtracts a background image estimated from the chosen input images."),
            ("First images", "The number of earliest corrected background images used for the pixel-wise median."),
        ])
        measurement_help = self._term_help([
            ("ROI", "Region of interest: the circular image area used for measurements."),
        ])
        calibration_help = self._term_help([
            ("NCI", "Neutron capture imaging."),
            ("Calibrated NCI thickness", "Marks source values as physical NCI thickness rather than arbitrary intensity."),
        ])
        plating_help = self._term_help([
            ("Plating metrics", "NCI-derived measurements intended for lithium-plating analysis."),
        ])
        photon_help = self._term_help([
            ("Photon-count cube", "A stack that groups image events by detected photon count."),
        ])
        tof_help = self._term_help([
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
        ])
        combined_output_help = self._term_help([
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
            self.mode_selector, self.show_all_sections, self.mode_info,
            self._path_row(self.config_path, self.config_browse_btn),
            self._row([self.load_config_btn, self.save_config_btn]),
            self._path_row(self.data_path, self.data_browse_btn),
            self.experiment, self.variant,
            self._path_row(self.result_root, self.result_browse_btn),
            self.analysis_name,
            self.include_aggregates, self.discover_btn, self.proposal_config_html,
            self._path_row(self.beam_path, self.beam_browse_btn),
            self._row([
                self.metadata_time_scope,
                self.metadata_time_margin,
                self.inspect_metadata_time_btn,
                self.save_metadata_time_btn,
            ]),
            self.metadata_time_html,
        ])
        shared_setup = self._section([
            self.w.HTML(
                "<b>Optional preparation — skip if you already have processed OB/BG TIFFs.</b>"
                "<p>Select the proposal, variant and optional beam history in Project and data. "
                "Choose an OB experiment below; optionally add a BG experiment. "
                "Preparation averages the processed OB frames and takes the pixel-wise median "
                "of the selected corrected BG frames. Output paths fill the next section.</p>"
            ),
            shared_help,
            self.shared_ob_experiment,
            self.prepare_bg, self.shared_background_experiment, self.preparation_bg_count,
            self.reference_set,
            self.prepare_shared_btn,
            self.shared_status_html,
        ])
        batch = self._section([
            batch_help,
            self.batch_background_info,
            self.w.HTML(
                "<small>The current pipeline settings are shared across the selected experiments. "
                "The selected prepared OB and BG files are reused without preparing them again.</small>"
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
        self.legacy_reference = self.w.Accordion(children=[self._section([
            self.w.HTML("<small>Compatibility for older configs that process an original OB stack. "
                        "A selected Prepared OB TIFF takes precedence.</small>"),
            self._path_row(self.reference_path, self.reference_browse_btn),
            self.ob_integral,
        ])], selected_index=None)
        self.legacy_reference.set_title(0, "Advanced / legacy OB settings")
        corrections = self._section([
            self.w.HTML(
                "<p>Select the <b>processed, averaged OB TIFF</b> and, if needed, the "
                "<b>processed median BG TIFF</b>. You can use existing files and skip preparation. "
                "Leave OB blank to skip OB division. BG must be in the same corrected units as "
                "the samples. Select the sample experiment in Project and data, then run the "
                "corrections below. These paths also apply to batches.</p>"
                "<small>OB beam normalization is loaded from its preparation metadata when available; "
                "for an external OB with beam correction, enter its beam integral in Advanced settings.</small>"
            ),
            correction_help,
            self._path_row(self.prepared_ob_path, self.prepared_ob_browse_btn),
            self.background_enabled,
            self.background_source,
            self.background_source_help,
            self.background_path_row,
            self.background_count,
            self.background_clip,
            self.w.HTML('<small>Clipping biases low-signal means and variation metrics. Clear this option for signed quantitative analysis.</small>'),
            self.legacy_reference,
            self._row([self.run_correction_btn, self.run_background_btn]),
        ])
        images = self._section([
            image_help,
            self.white_spot_enabled,
            self._row([self.white_spot_size, self.white_spot_method, self.white_spot_threshold]),
            self._row([self.white_spot_preview_index, self.white_spot_preview_btn]),
            self.w.HTML("<small>Applies on <b>Load images</b>, including batches, after any corrections already run. "
                        "Preview before applying. To change finished results, use a new <b>Analysis name</b> and "
                        "rerun the intended corrections. Filtered frames feed averages, plots and videos.</small>"),
            self._row([self.frames_per_merge, self.spatial_merge]),
            self._row([self.load_images_btn, self.average_btn]),
        ])
        measurements = self._section([
            measurement_help,
            self._row([self.roi_x, self.roi_y, self.roi_r]),
            calibration_help, self.calibrated, plating_help,
            self._row([self.measure_btn, self.plating_btn]),
        ])
        tof_path_row = self._path_row(self.tof_path, self.tof_browse_btn)
        spectral = self._section([
            photon_help, self.photon_btn, tof_help, tof_path_row, self.tof_btn
        ])
        electrochemistry = self._section([
            electrochemistry_help,
            self._path_row(self.echem_path, self.echem_browse_btn), self.echem_conv,
            self._row([self.load_echem_btn, self.sync_btn, self.events_btn]),
        ])
        outputs = self._section([
            output_help, combined_output_help,
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
        # Remove irrelevant accordion panes as well as controls in mixed panes.
        self._panels = (
            ("Project and data", project, {"project"}),
            ("Prepare OB/BG files (optional)", shared_setup, {"corrections", "background"}),
            ("Use prepared OB/BG and correct samples", corrections, {"corrections", "background"}),
            ("Image products", images, {"images", "average"}),
            ("Image measurements", measurements, {"image_measurements", "nci_plating_metrics"}),
            ("Photon-count and time-of-flight", spectral, {"photon_hyperspectral", "tof_cube"}),
            ("Electrochemistry and synchronization", electrochemistry, {"electrochemistry", "synchronization", "event_detection"}),
            ("Figures, video, and export", outputs, {"image_plot", "image_video", "combined_plot"}),
            ("Batch processing", batch, {"project"}),
        )
        self._mode_controls = (
            ({"nci_plating_metrics"}, (self.plating_btn, plating_help)),
            ({"corrections"}, (self.calibrated, calibration_help)),
            ({"photon_hyperspectral"}, (self.photon_btn, photon_help)),
            ({"tof_cube"}, (tof_path_row, self.tof_btn, tof_help)),
            ({"combined_plot"}, (self.combined_plot_btn,)),
            ({"nci_event_outputs"}, (self.nci_outputs_btn,)),
            ({"combined_video"}, (self.combined_video_btn,)),
            ({"combined_plot", "combined_video", "nci_event_outputs"}, (combined_output_help,)),
        )
        self.accordion = self.w.Accordion(
            children=[project], selected_index=0,
            layout=self.w.Layout(width="100%"),
        )
        status_panel = self._section([status_help, self.status_html])
        self.status_accordion = self.w.Accordion(
            children=[status_panel],
            selected_index=None,
            layout=self.w.Layout(width="100%"),
        )
        self.status_accordion.set_title(0, "Section status — click to show or fold")
        self.widget = self.w.VBox(
            [
                self.path_browser.widget,
                self.accordion,
                self.status_accordion,
                self.output,
            ],
            layout=self.w.Layout(width="100%"),
        )

    def _wire(self):
        self.preset.observe(self._change_preset, names="value")
        self.background_source.observe(self._update_background_controls, names="value")
        self.prepare_bg.observe(self._update_background_controls, names="value")
        self.prepared_ob_path.observe(self._prepared_ob_changed, names="value")
        self.background_count.observe(self._update_background_controls, names="value")
        self.show_all_sections.observe(self._change_visibility, names="value")
        self.batch_sections.observe(self._remember_batch_sections, names="value")
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
                mode="either",
                title="Select a beam-history file or folder",
                category="beamhistory",
            )
        )
        self.prepared_ob_browse_btn.on_click(
            lambda _: self._browse_path(
                self.prepared_ob_path, mode="file", title="Select a processed, averaged OB TIFF",
                category="reference", extensions=(".tif", ".tiff"),
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
        self.background_browse_btn.on_click(
            lambda _: self._browse_path(
                self.background_path,
                mode="file",
                extensions=(".tif", ".tiff"),
                title="Select a processed median BG TIFF",
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
        self.prepare_shared_btn.on_click(self._prepare_shared_inputs)
        self.inspect_metadata_time_btn.on_click(self._inspect_metadata_times)
        self.save_metadata_time_btn.on_click(self._save_metadata_times)
        self.run_correction_btn.on_click(lambda _: self._run("corrections"))
        self.run_background_btn.on_click(lambda _: self._run("background"))
        self.load_images_btn.on_click(lambda _: self._run("images"))
        self.white_spot_preview_btn.on_click(self._preview_white_spots)
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
            proposal = proposal_root_for_layout(layout_root)
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
        candidate = reduced_root_for_proposal(proposal).joinpath(*relatives.get(category, ()))
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
            discovery_config = self._discovery_config()
            all_names = tuple(
                discover_batch_experiments(
                    self.data_path.value,
                    hide_setup=False,
                    filter_text="",
                    config=discovery_config,
                )
            )
            visible_names = tuple(
                discover_batch_experiments(
                    self.data_path.value,
                    hide_setup=self.batch_hide_setup.value,
                    filter_text=self.batch_filter.value,
                    config=discovery_config,
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
        hidden = set(normalize_batch_sections(sections)) - self._visible_sections
        if hidden:
            raise ValueError(
                "Batch requires hidden sections: " + ", ".join(sorted(hidden))
                + ". Enable Show all sections to include them."
            )
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

    def _preview_white_spots(self, _=None):
        self._sync_config()
        with self.output:
            self.output.clear_output(wait=True)
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                path, original, filtered, mask, stats = preview_white_spots(
                    self.session, self.white_spot_preview_index.value,
                )
                finite = original[np.isfinite(original)]
                limits = np.percentile(finite, [1, 99.5]) if finite.size else (0, 1)
                low, high = float(limits[0]), float(limits[1])
                if high <= low:
                    high = low + 1
                figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
                for axis, image, title in zip(axes[:2], (original, filtered), ("Before", "After")):
                    axis.imshow(image, cmap="gray", vmin=low, vmax=high)
                    axis.set_title(title)
                axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title(f"Replaced pixels: {stats['replaced_pixels']}")
                for axis in axes:
                    axis.set_axis_off()
                self.display(figure)
                plt.close(figure)
                print(f"{path.name}: {stats['replaced_pixels']} pixel(s) replaced "
                      f"({stats['replaced_fraction']:.2%}). Preview only; no TIFFs saved.")
                if stats["noise_sigma"] == 0:
                    print("Automatic noise estimate is zero (flat/sparse image). Check the mask; "
                          "use an absolute threshold if real features are being removed.")
            except Exception as exc:
                print(f"{type(exc).__name__}: {exc}")
        self.refresh_status()

    def _prepared_ob_changed(self, change):
        # Never carry a different OB's normalization into the new selection.
        try:
            self.ob_integral.value = float(prepared_ob_integral(change["new"]) or 0) if change["new"] else 0
        except (OSError, ValueError, TypeError):
            self.ob_integral.value = 0

    def _update_background_controls(self, _=None):
        shared = self.background_source.value == "shared"
        self.shared_background_experiment.layout.display = "" if self.prepare_bg.value else "none"
        self.preparation_bg_count.layout.display = "" if self.prepare_bg.value else "none"
        self.background_path_row.layout.display = "" if shared else "none"
        self.prepare_shared_btn.description = "Prepare OB and BG files" if self.prepare_bg.value else "Prepare OB file"
        self.background_count.description = "First images (per experiment)"
        self.background_count.layout.display = "none" if shared else ""
        text = (
            "Select a processed median BG TIFF. This single image is subtracted from every "
            "corrected sample frame; it is not processed or averaged again."
            if shared else
            "Each experiment gets its own median BG from its first X valid beam/OB-corrected "
            "frames, before frame merging. No separate BG experiment or file is used. "
            "Those frames stay in the processed image series."
        )
        self.background_source_help.value = f"<small>{text}</small>"
        choice = "one shared BG" if shared else f"each experiment's first {self.background_count.value} valid corrected frames"
        self.batch_background_info.value = (
            f"<small><b>BG source:</b> {escape(choice)}. Set this in Use prepared OB/BG and correct samples; "
            "include <b>Background correction</b> in the batch processing sections to apply it.</small>"
        )

    def _prepare_shared_inputs(self, _=None):
        self._sync_config()
        capture = StringIO()
        try:
            with redirect_stdout(capture), redirect_stderr(capture):
                preparation_config = deepcopy(self.session.config)
                preparation_config.setdefault("background", {})["n_images"] = self.preparation_bg_count.value
                preparation_config.setdefault("corrections", {})["reference_input_mode"] = "source"
                result = prepare_shared_ob_background(
                    self.data_path.value,
                    self.shared_ob_experiment.value,
                    self.shared_background_experiment.value,
                    preset=self.preset.value,
                    config=preparation_config,
                    prepare_background=self.prepare_bg.value,
                )
            self.shared_ob_experiment.value = result.ob_experiment
            self.session.config.setdefault("shared_setup", {}).update(preparation_config["shared_setup"])
            self.reference_path.value = ""
            self.prepared_ob_path.value = str(result.prepared_reference_path)
            self.ob_integral.value = float(result.reference_beam_integral_per_minute or 0)
            if result.prepared_background_path is not None:
                self.shared_background_experiment.value = result.background_experiment
                self.background_enabled.value = True
                self.background_source.value = "shared"
                self.background_path.value = str(result.prepared_background_path)
            self._sync_config()
            automatic = self._save_automatic_proposal_config()
            if automatic is not None:
                self.config_path.value = str(automatic)
            cache_label = "reused cached" if result.reused_background_cache else "prepared"
            bg_message = (
                f"BG: {result.background_tiff_count}/{result.background_source_tiff_count} corrected TIFF(s) "
                f"used for median; {cache_label} background: {result.prepared_background_path}"
                if result.prepared_background_path is not None else
                "No shared BG prepared. Choose a prepared BG file or per-experiment background in the next section."
            )
            message = (
                f"Shared inputs ready. OB: {result.reference_tiff_count} TIFF(s) averaged; "
                f"{bg_message}"
            )
            self.shared_status_html.value = (
                "<small style='color:#087f5b'><b>ready</b><br>"
                f"OB: {result.reference_tiff_count} TIFF(s) averaged<br>"
                f"{escape(bg_message)}<br>"
                f"Reference: <code>{escape(str(result.prepared_reference_path))}</code><br>"
                f"Config: <code>{escape(str(automatic or 'not saved'))}</code></small>"
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.shared_status_html.value = (
                f"<small style='color:#c92a2a'><b>failed</b>: {escape(message)}</small>"
            )
        with self.output:
            self.output.clear_output(wait=True)
            detail = capture.getvalue()
            if detail:
                print(detail.rstrip())
            print(message)
        self.refresh_status()

    def _validate_batch(self, _):
        try:
            experiments, sections = self._batch_inputs()
            automatic = self._save_automatic_proposal_config()
            if automatic is not None:
                self.config_path.value = str(automatic)
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
            automatic = self._save_automatic_proposal_config()
            if automatic is not None:
                self.config_path.value = str(automatic)

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
            if summary.report_error:
                message += "\n" + summary.report_error
            elif (summary.batch_dir / "overview.pdf").is_file():
                message += f"\nOverview: {summary.batch_dir / 'overview.md'} and overview.pdf"
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.batch_status_html.value = f"<small style='color:#c92a2a'>{escape(message)}</small>"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _discovery_config(self):
        """Read current path controls without saving or changing active artifacts."""
        config = deepcopy(self.session.config)
        config.setdefault("project", {}).update(data_path=self.data_path.value, variant=self.variant.value)
        if self.workspace == "processing":
            config["layout"] = {"images_pattern": self.images_pattern.value,
                                "metadata_pattern": self.metadata_pattern.value,
                                "run_regex": self.run_regex.value}
        return config

    def _populate_project_choices(self, data_path=None):
        raw = str(data_path or self.data_path.value).strip()
        if not raw:
            self.experiment.options = ()
            self.shared_ob_experiment.options = ()
            self.shared_background_experiment.options = ()
            self.variant.options = ()
            return
        layout_root = find_layout_root(raw)
        discovery_config = self._discovery_config()
        discovery_config["project"]["data_path"] = raw
        if discovery_config.get("layout", {}).get("images_pattern"):
            from .inspection import mapped_experiments
            names = tuple(mapped_experiments(discovery_config))
            self.experiment.options = names
            self.shared_ob_experiment.options = names
            self.shared_background_experiment.options = names
            self._populate_variant_choices()
            return
        if layout_root is None:
            self.experiment.options = ()
            self.shared_ob_experiment.options = ()
            self.shared_background_experiment.options = ()
            self.variant.options = ()
            return
        names = tuple(list_experiments(layout_root))
        self.experiment.options = names
        self.shared_ob_experiment.options = names
        self.shared_background_experiment.options = names
        if self.workspace != "processing" and not str(self.shared_ob_experiment.value).strip():
            ob_candidates = [
                name for name in names
                if "_ob" in name.casefold()
                or "openbeam" in name.casefold()
                or "open_beam" in name.casefold()
            ]
            if len(ob_candidates) == 1:
                self.shared_ob_experiment.value = ob_candidates[0]
        if self.workspace != "processing" and not str(self.shared_background_experiment.value).strip():
            background_candidates = [
                name for name in names
                if "_bg" in name.casefold() or "background" in name.casefold()
            ]
            if len(background_candidates) == 1:
                self.shared_background_experiment.value = background_candidates[0]
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
        if self.workspace == "processing" and self.images_pattern.value.strip() and raw_data and experiment:
            from .inspection import _mapped_records
            try:
                config = {"project": {"data_path": raw_data},
                          "layout": {"images_pattern": self.images_pattern.value,
                                     "run_regex": self.run_regex.value}}
                rows = _mapped_records(config)[1]
                name = resolve_experiment_name(experiment, sorted({fields["experiment"] for _, fields in rows}))
                self.variant.options = tuple(sorted({fields["variant"] for _, fields in rows if fields["experiment"] == name}))
                if not self.variant.value and len(self.variant.options) == 1:
                    self.variant.value = self.variant.options[0]
            except (OSError, ValueError):
                pass
            return
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
        self.session.preset = change["new"]
        # A mode change is not a new analysis: retain widgets, artifacts, paths
        # and explicit settings. Seed new options (e.g. ex-situ average copies).
        for section, defaults in PRESETS[change["new"]].items():
            current = self.session.config.setdefault(section, {})
            for key, value in defaults.items():
                if key not in current:
                    current[key] = deepcopy(value)
        # Before selecting data, untouched product defaults may follow the mode
        # (ToF uses derived products). Never change discovered input selections.
        project = self.session.section_config("project")
        previous = PRESETS[change["old"]]["project"]
        defaults = PRESETS[change["new"]]["project"]
        if not self.session.artifacts.has("dataset") and not self.data_path.value.strip():
            if project.get("product_kinds") == previous["product_kinds"]:
                project["product_kinds"] = defaults["product_kinds"]
            if self.variant.value == previous["variant"] and project.get("variant") == previous["variant"]:
                self.variant.value = defaults["variant"]
                project["variant"] = defaults["variant"]
        self._apply_mode_visibility()
        self.refresh_status()

    def _change_visibility(self, _):
        if not self._loading_config:
            self._apply_mode_visibility()
            self.refresh_status()

    def _remember_batch_sections(self, change):
        if self._updating_visibility or self._loading_config:
            return
        # Keep choices from other modes, but only visible choices may execute.
        options = {section for _, section in self.batch_sections.options}
        self._batch_section_selection.difference_update(options)
        self._batch_section_selection.update(change["new"])

    def _apply_mode_visibility(self):
        self._visible_sections = visible_sections(
            self.preset.value, self.session.sections,
            show_all=self.show_all_sections.value,
        )
        if self.workspace == "processing":
            from .processing import PROCESSING_SECTIONS
            self._visible_sections &= PROCESSING_SECTIONS
        selected_index = self.accordion.selected_index
        selected_panel = (
            self.accordion.children[selected_index] if selected_index is not None else None
        )
        panels = [panel for panel in self._panels if panel[2] & self._visible_sections]
        self.accordion.children = tuple(panel[1] for panel in panels)
        for index, (title, _, _) in enumerate(panels):
            if title == "Photon-count and time-of-flight" and not self.show_all_sections.value:
                title = "Time-of-flight"
            self.accordion.set_title(index, title)
        if selected_panel is None:
            self.accordion.selected_index = None
        elif selected_panel in self.accordion.children:
            self.accordion.selected_index = self.accordion.children.index(selected_panel)
        else:
            self.accordion.selected_index = 0
        for section_ids, controls in self._mode_controls:
            for control in controls:
                control.layout.display = "" if section_ids & self._visible_sections else "none"
        self._updating_visibility = True
        try:
            self.batch_sections.options = tuple(
                (label, section) for section, label in BATCH_SECTION_LABELS.items()
                if section != "project" and section in self._visible_sections
            )
            self.batch_sections.value = tuple(
                section for _, section in self.batch_sections.options
                if section in self._batch_section_selection
            )
            self.batch_sections.rows = min(11, max(4, len(self.batch_sections.options)))
        finally:
            self._updating_visibility = False
        description = (
            "All sections are shown, including advanced tools outside this mode."
            if self.show_all_sections.value else MODE_DESCRIPTIONS[self.preset.value]
        )
        self.mode_info.value = (
            f"<small>{escape(description)} Settings and completed results are retained; "
            "only selected actions run.</small>"
        )

    def _load_config(self, _):
        try:
            requested = Path(self.config_path.value).expanduser()
            if (not requested.exists() and requested.name == "lumacam_pipeline.ini"
                    and requested.parent.name == "proposal_summary"):
                replacement = requested.parent.parent / "lumacam_pipeline.ini"
                if replacement.is_file():
                    self.config_path.value = str(replacement)
            preset, values = load_config(self.config_path.value)
            if self.workspace == "processing":
                if values.get("workflow", {}).get("role") == "visualization":
                    raise ValueError("Load the proposal processing INI, not visualization settings.ini.")
                from .processing import resolve_processing_paths
                values = resolve_processing_paths(values, self.config_path.value)
                values["workflow"] = {"role": "processing"}
                layout = values.get("layout", {})
                for control, key in ((self.images_pattern, "images_pattern"), (self.metadata_pattern, "metadata_pattern"), (self.run_regex, "run_regex")):
                    control.value = str(layout.get(key, ""))
                self.processing_pixel_size.value = str(values.get("images", {}).get("pixel_size_um") or "")
                self.processing_unit.value = str(values.get("images", {}).get("unit", ""))
            restored_session = _create_gui_session(preset, values)
            self._loading_config = True
            try:
                self.preset.value = restored_session.preset
            finally:
                self._loading_config = False
            self.session = restored_session
            gui_values = self.session.config.get("gui", {})
            self._loading_config = True
            try:
                self.show_all_sections.value = bool(gui_values.get("show_all_sections", False))
            finally:
                self._loading_config = False
            project = self.session.section_config("project")
            corrections = self.session.section_config("corrections")
            if corrections.get("reference_beam_integral_per_minute") == "":
                corrections["reference_beam_integral_per_minute"] = None
            background = self.session.section_config("background")
            images = self.session.section_config("images")
            measurements = self.session.section_config("image_measurements")
            tof = self.session.section_config("tof_cube")
            echem = self.session.section_config("electrochemistry")
            selection = self.session.section_config("selection")
            individual = self.session.section_config("individual_image_plots")
            shared = self.session.config.setdefault("shared_setup", {})
            batch_values = self.session.config.setdefault("batch", {})
            self.data_path.value = str(project.get("data_path", ""))
            self.experiment.value = str(project.get("experiment", ""))
            self.shared_ob_experiment.value = str(shared.get("ob_experiment", ""))
            self.shared_background_experiment.value = str(
                shared.get("background_experiment", "")
            )
            self.variant.value = str(project.get("variant", ""))
            self.result_root.value = str(project.get("result_root", ""))
            self.analysis_name.value = str(project.get("analysis_name", ""))
            self.reference_set.value = str(shared.get("reference_set", ""))
            self.include_aggregates.value = bool(project.get("include_aggregates", False))
            self.beam_path.value = str(corrections.get("beamhistory_path", ""))
            self.reference_path.value = str(corrections.get("reference_dir", ""))
            self.prepared_ob_path.value = str(corrections.get("prepared_reference_path", "") or "")
            self.ob_integral.value = float(corrections.get("reference_beam_integral_per_minute") or
                                           (prepared_ob_integral(self.prepared_ob_path.value) if self.prepared_ob_path.value else 0) or 0)
            self.legacy_reference.selected_index = 0 if self.reference_path.value and not self.prepared_ob_path.value else None
            self.prepare_bg.value = bool(shared.get("prepare_background", bool(shared.get("background_experiment"))))
            self.preparation_bg_count.value = int(shared.get("background_n_images", background.get("n_images", 7)))
            self.background_enabled.value = bool(background.get("enabled", False))
            self.background_clip.value = bool(background.get("clip_negatives", False))
            self.background_path.value = str(background.get("background_path", ""))
            self.background_count.value = int(background.get("n_images", 7))
            self.background_source.value = background_source(background)
            self._update_background_controls()
            self.frames_per_merge.value = int(images.get("frames_per_merge", 1))
            self.spatial_merge.value = int(images.get("spatial_pixel_merge", 1))
            self.white_spot_enabled.value = bool(images.get("white_spot_enabled", False))
            self.white_spot_size.value = int(images.get("white_spot_size", 3))
            self.white_spot_method.value = str(images.get("white_spot_method", "mad"))
            self.white_spot_threshold.value = float(images.get("white_spot_threshold", 8.0))
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
            default_sections = self._default_batch_sections()
            saved_sections = gui_values.get(
                "batch_section_selection", batch_values.get("sections", default_sections),
            )
            self._batch_section_selection = set(saved_sections) & (
                set(BATCH_SECTION_LABELS) - {"project"}
            )
            self._apply_mode_visibility()
            main_config = self._automatic_proposal_config_path()
            if main_config is not None:
                self.config_path.value = str(main_config)
            self.batch_overwrite.value = bool(batch_values.get("overwrite", False))
            prepared_background = str(shared.get("prepared_background_path", "") or "")
            if prepared_background and self.background_source.value == "shared":
                self.shared_status_html.value = (
                    "<small>Prepared shared inputs: "
                    f"<code>{escape(prepared_background)}</code></small>"
                )
            message = f"Loaded {self.config_path.value}"
            if preset in LEGACY_MODE_ALIASES:
                label = dict((value, label) for label, value in MODE_OPTIONS)[self.preset.value]
                message += f" (legacy mode {preset!r} now displays as {label}; processing settings retained)"
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)
        self.refresh_status()

    def _save_config(self, _):
        self._sync_config()
        try:
            main_config = self._automatic_proposal_config_path()
            if main_config is not None:
                self.config_path.value = str(main_config)
            path = save_config(self.config_path.value, self.preset.value, self.session.config)
            message = f"Saved {path}"
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
        with self.output:
            self.output.clear_output(wait=True)
            print(message)

    def _sync_config(self):
        if self.workspace == "processing":
            self.session.config["workflow"] = {"role": "processing"}
            mapping = {"images_pattern": self.images_pattern.value,
                                             "metadata_pattern": self.metadata_pattern.value,
                                             "run_regex": self.run_regex.value}
            if mapping != self.session.config.get("layout", {}):
                self.session.configure("project", input_mapping=mapping)
            self.session.config["layout"] = mapping
            self.session.section_config("images").update(
                pixel_size_um=float(self.processing_pixel_size.value) if self.processing_pixel_size.value.strip() else None,
                unit=self.processing_unit.value)
        corrections = self.session.section_config("corrections")
        prepared = self.prepared_ob_path.value.strip()
        prepared_mode = bool(prepared) or not self.reference_path.value.strip()
        integral = self.ob_integral.value or None
        if not prepared_mode and any(
            str(corrections.get(key, "") or "") != str(value)
            for key, value in (("beamhistory_path", self.beam_path.value),
                               ("reference_dir", self.reference_path.value),
                               ("file_filter", self.variant.value))
        ):
            integral = None
        self.session.configure(
            "project", data_path=self.data_path.value, experiment=self.experiment.value,
            variant=self.variant.value, result_root=self.result_root.value,
            analysis_name=self.analysis_name.value,
            include_aggregates=self.include_aggregates.value,
        )
        self.session.configure(
            "corrections", beamhistory_path=self.beam_path.value,
            reference_dir="" if prepared_mode else self.reference_path.value,
            reference_beamhistory_path=self.beam_path.value,
            prepared_reference_path=prepared,
            reference_input_mode="prepared" if prepared_mode else "source",
            reference_beam_integral_per_minute=integral,
            file_filter=self.variant.value,
        )
        self.session.configure(
            "background", enabled=self.background_enabled.value, clip_negatives=self.background_clip.value,
            background_path=self.background_path.value, n_images=self.background_count.value,
            source_mode=self.background_source.value,
        )
        self.session.configure(
            "images", frames_per_merge=self.frames_per_merge.value,
            spatial_pixel_merge=self.spatial_merge.value,
            white_spot_enabled=self.white_spot_enabled.value,
            white_spot_size=self.white_spot_size.value,
            white_spot_method=self.white_spot_method.value,
            white_spot_threshold=self.white_spot_threshold.value,
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
        self.session.config.setdefault("shared_setup", {}).update(
            {
                "ob_experiment": self.shared_ob_experiment.value,
                "background_experiment": self.shared_background_experiment.value,
                "reference_set": self.reference_set.value,
                "prepare_background": self.prepare_bg.value,
                "background_n_images": self.preparation_bg_count.value,
            }
        )
        self.session.config.setdefault("batch", {}).update(
            {
                "name": self.batch_name.value,
                "experiments": list(self._batch_selected_names()),
                "sections": list(self.batch_sections.value),
                "overwrite": self.batch_overwrite.value,
            }
        )
        self.session.config.setdefault("gui", {}).update(
            {
                "show_all_sections": self.show_all_sections.value,
                "batch_section_selection": sorted(self._batch_section_selection),
            }
        )

    def _automatic_proposal_config_path(self):
        data_path = str(self.data_path.value).strip()
        if not data_path:
            return None
        if self.workspace == "processing" and self.images_pattern.value.strip():
            return analysis_root_for_proposal(Path(data_path).expanduser().resolve()) / "lumacam_pipeline.ini"
        layout_root = find_layout_root(data_path)
        if layout_root is None:
            return None
        proposal_root = (
            proposal_root_for_layout(layout_root)
        )
        return analysis_root_for_proposal(proposal_root) / "lumacam_pipeline.ini"

    def _save_automatic_proposal_config(self):
        path = self._automatic_proposal_config_path()
        if path is None:
            return None
        saved = save_config(path, self.preset.value, self.session.config)
        self.proposal_config_html.value = (
            f"<small>Proposal main config: <code>{escape(str(saved))}</code></small>"
        )
        return saved

    def _run(self, section_id):
        if section_id not in self._visible_sections:
            with self.output:
                self.output.clear_output(wait=True)
                print("This section is hidden in the current mode. Enable Show all sections to use it.")
            return
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
            if row["section"] not in self._visible_sections:
                continue
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


def launch_processing_gui(preset="operando_nci"):
    return LumacamPipelineGUI(preset=preset, workspace="processing").show()

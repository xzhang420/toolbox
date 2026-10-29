"""Notebook controls for standalone analysis; never start acquisition corrections."""

from copy import deepcopy
from html import escape
import json
from pathlib import Path

from .modes import MODE_OPTIONS, visible_sections
from .path_browser import InlinePathBrowser
from .visualization import open_visualization, load_visualization_settings, save_visualization_settings
from .measurements import METHODS, method_names, region_mask


class VisualizationGUI:
    def __init__(self, preset="exsitu"):
        import ipywidgets as w
        from IPython.display import display
        self.w, self.display = w, display
        self.session = None
        self.output = w.Output()
        self.browser = InlinePathBrowser(w)
        self.mode = w.Dropdown(options=MODE_OPTIONS, value=preset, description="Mode",
                               style={"description_width": "initial"}, layout=w.Layout(width="100%"))
        self.source = self._text("Processed dataset / image folder")
        self.profile = self._text("Visualization profile", "overview")
        self.output_root = self._text("Output parent (blank = beside dataset)")
        self.export_layout = w.Dropdown(options=[("Simple folders (refresh current files)", "flat"),
                                                ("Keep every export in dated folders", "versioned")],
                                       value="flat", description="Export layout",
                                       style={"description_width": "initial"}, layout=w.Layout(width="100%"))
        self.config_path = self._text("Visualization settings.ini")
        self.pattern = self._text("Image pattern (blank = direct children)")
        self.fits_hdu = self._text("FITS image HDU (blank = unique image)")
        self.pixel_size = self._text("Pixel size (μm; blank = unknown)")
        self.units = self._text("Image units (blank = metadata/unknown)")
        self.timezone = self._text("Timezone for timestamps without an offset")
        self.start_time = self._text("Optional acquisition start (ISO 8601)")
        self.interval = self._text("Frame interval (seconds; requires start)")
        self.open_btn = self._button("Open images", lambda _: self.open())
        self.tof_btn = self._button("Open reconstructed ToF cube", lambda _: self.open(cube=True))
        self.load_btn = self._button("Load settings", lambda _: self.load())
        self.save_btn = self._button("Save settings", lambda _: self.save())
        self.frame = w.IntSlider(description="Preview frame", min=0, max=0,
                                 style={"description_width": "initial"}, layout=w.Layout(width="100%"))
        self.preview_btn = self._button("Preview frame", lambda _: self.preview())
        self.indices = self._text("Figure frame indices (blank = detected events, otherwise representative)")
        self.count = self._text("Representative frame count", "4")
        self.cmap = self._text("Colormap", "inferno")
        self.vmin, self.vmax = self._text("Colour minimum (blank = automatic)"), self._text("Colour maximum (blank = automatic)")
        self.scale = self._text("Scale-bar length (μm; 0 = hidden)", "0")
        self.dpi = self._text("Figure/video DPI", "150")
        self.columns = self._text("Figure columns (0 = automatic)", "0")
        self.figure_format = self._text("Figure format", "png")
        self.fps = self._text("Video frames per second", "10")
        self.render_3d = self._check("3D surfaces for image panels, individual figures and image-only videos", False)
        self.surface_elev = self._text("3D elevation (degrees)", "35")
        self.surface_azim = self._text("3D azimuth (degrees)", "-60")
        self.surface_stride = self._text("3D display stride (pixels; measurements unchanged)", "2")
        self.surface_zmin = self._text("3D height minimum (blank = colour limits)")
        self.surface_zmax = self._text("3D height maximum (blank = colour limits)")
        self.individual_color_scale = self._choice("Individual-figure colour scaling",
            [("Global: same scale for all", "global"), ("Per image: automatic contrast", "individual")], "global")
        self.video_timing = self._choice("Image-only video timing",
            [("Image sequence: one frame per image", "image_sequence"),
             ("Elapsed-time playback: show acquisition gaps", "elapsed_time")], "image_sequence")
        self.video_dt = self._text("Experimental seconds per video frame (blank = median interval)")
        self.video_tolerance = self._text("Maximum time-to-image difference (seconds; blank = automatic)")
        self.roi_x, self.roi_y, self.roi_r = self._text("ROI centre x (blank = whole image)"), self._text("ROI centre y"), self._text("ROI radius (pixels)")
        self.region = self._choice("Measurement region", [("Whole image", "whole"), ("Circle", "circle"),
            ("Rectangle", "rectangle"), ("Polygon", "polygon")], "whole")
        self.rectangle = self._text("Rectangle: x_min, x_max, y_min, y_max")
        self.polygon = self._text("Polygon vertices: [[x,y], [x,y], ...]")
        self.methods = w.SelectMultiple(options=[(label, key) for key, label in METHODS.items()], value=("mean",),
            description="Calculate methods", rows=5, style={"description_width": "initial"}, layout=w.Layout(width="100%"))
        self.primary_method = self._choice("Primary method (panel values / video)", [(label, key) for key, label in METHODS.items()], "mean")
        self.electrode_area = self._text("Electrode area (mm²; required for electrode-area mean)")
        self.active_threshold = self._text("Active threshold (image units; required for active-area mean)")
        self.pattern_floor = self._text("Minimum reliable mean signal (optional, image units)")
        self.boundary_confirmed = self._check("Electrode boundary independently confirmed", False)
        self.electrode_diameter = self._text('Known electrode diameter (mm; for centre estimation)')
        self.baseline_indices = self._text("Baseline frame indices (optional, comma-separated)")
        self.trim_fraction = self._text("Trim fraction per tail (0 to <0.5)", "0.1")
        self.plot_methods = w.SelectMultiple(options=[(label, key) for key, label in METHODS.items()], value=(),
            description="Methods on combined plot (blank = primary)", rows=5,
            style={"description_width": "initial"}, layout=w.Layout(width="100%"))
        self.show_ec_thickness = self._check("Include electrochemical thickness on combined plot", False)
        self.window_enabled = self._check("Enable time window", False)
        self.window_mode = self._choice("Window units", [("Elapsed hours from first image", "elapsed"),
            ("Absolute date/time (ISO 8601)", "absolute")], "elapsed")
        self.window_start = self._text("Window start (blank = experiment start)")
        self.window_end = self._text("Window end (blank = experiment end)")
        self.window_slider = w.FloatRangeSlider(value=(0, 1), min=0, max=1, step=.001,
            description="Window (hours)", disabled=True, continuous_update=False, readout_format=".3f",
            style={"description_width": "initial"}, layout=w.Layout(width="100%"))
        self.window_figures = self._check("Apply to time-series figures", True)
        self.window_panels = self._check("Apply to image panels / individual figures", True)
        self.window_videos = self._check("Apply to videos", True)
        self.csv_scope = self._choice("CSV range", [("Full experiment", "full"), ("Selected time window", "window")], "full")
        self.csv_comparison = self._check("Also export nearest-time comparison with electrochemistry", True)
        self.csv_tolerance = self._text("Maximum matching difference (seconds)", "30")
        self.echem = self._text("Electrochemistry file/folder (optional)")
        self.echem_timezone = self._text("Electrochemistry timezone (if timestamps have no offset)")
        self.status = w.HTML()
        self.advanced = w.Textarea(value="{}", description="Section settings (JSON)",
                                  style={"description_width": "initial"},
                                  layout=w.Layout(width="100%", height="200px"))
        self.apply_advanced = self._button("Apply advanced settings", lambda _: self.apply_settings())
        project = w.VBox([w.HTML("<b>Analysis &amp; visualization</b><br>Inputs are read-only. Outputs and settings belong to a named profile."),
            self.mode, self._browse(self.source, "either"), self.profile, self._browse(self.output_root, "directory"), self.export_layout,
            self._browse(self.config_path, "file", (".ini",)), self._row([self.load_btn, self.save_btn]),
            self.pattern, self.fits_hdu, self.pixel_size, self.units, self.timezone, self.start_time, self.interval,
            self._row([self.open_btn, self.tof_btn])])
        appearance = w.VBox([self.frame, self.preview_btn, self.indices, self.count, self.cmap,
                            self.vmin, self.vmax, self.individual_color_scale,
                            self._help("Colour scaling", "Global uses all loaded frames. Per image reveals contrast but colours are not comparable between figures. Explicit minimum/maximum override either mode. Panels and videos always share a scale."),
                            self.scale, self.dpi, self.columns, self.figure_format, self.fps,
                            self.render_3d, self.surface_elev, self.surface_azim, self.surface_stride,
                            self.surface_zmin, self.surface_zmax,
                            self._help("3D surfaces", "Height is the image value, not a measured physical surface. Stride subsamples display only. Spatial axes replace the 2D scale bar; units are pixels unless calibrated. This does not change synchronized EC videos."),
                            self.video_timing, self.video_dt, self.video_tolerance,
                            self._help("Elapsed-time playback", "Requires real timestamps. Each output frame advances the chosen experimental interval. Nearest images are shown only within the tolerance (automatic: 0.75 × median acquisition interval); otherwise a blank No acquisition data frame is shown. Review automatic timing for sparse or irregular series. No pixel interpolation. A timing CSV/JSON records the mapping.")])
        self.circle_controls = w.VBox([self.roi_x, self.roi_y, self.roi_r])
        measurements = w.VBox([self._help("Measurement methods", "ROI mean averages valid pixels. Electrode-area mean = sum(thickness × pixel area) / electrode area. "
            "Active-area mean includes only pixels above your threshold and exports coverage. Median and trimmed mean are robust summaries, not volume-conserving means. "
            "Negative values are retained. Coordinates refer to original image pixels, not a display crop."),
                               self.region, self.circle_controls, self.rectangle, self.polygon, self.boundary_confirmed,
                               self.electrode_diameter, self._button('Estimate centre from fixed electrode diameter',lambda _:self.estimate_center()),
                               self._button("Preview measurement region", lambda _: self.preview_region()),
                               self.baseline_indices,
                               self._button("Check image quality and save ROI diagnostic", lambda _: self.run("quality")),
                               self.methods, self.primary_method, self.electrode_area, self.active_threshold, self.trim_fraction,
                               self._button("Measure images and save table", lambda _: self.run("image_measurements"))])
        pattern = w.VBox([self._help("Plating pattern metrics", "Uses the same measurement region above. CV is standard deviation divided by mean; "
            "P90–P10 describes the spread. A circular region also gives edge minus centre and positive-signal centroid offset. "
            "Coverage and active CV require the explicit active threshold above. CV can be unstable near zero; "
            "set a reliable mean floor from independent baseline/noise evidence when available."),
            self.pattern_floor, self._button("Measure plating inhomogeneity", lambda _: self.run("inhomogeneity"))])
        windows = w.VBox([self._help("Time-window behavior", "Measurements and electrochemical conversion always use the full loaded record. "
            "Zoom does not reset elapsed time, thickness or cycle numbering. Image panels keep the original event letters. "
            "An empty selection requires selecting different frames. CSV range is chosen independently below."),
            self.window_enabled, self.window_mode, self.window_start, self.window_end, self.window_slider,
            self.window_figures, self.window_panels, self.window_videos,
            self._row([self._button("Apply time window", lambda _: self.run("time_window")),
                       self._button("Reset to full experiment", lambda _: self.reset_window())])])
        csv = w.VBox([self._help("CSV exports", "Image and electrochemistry tables retain their native sampling. "
            "Comparison uses the nearest measured electrochemistry point within the tolerance; no extrapolation. "
            "Unmatched values stay blank. Times are UTC plus elapsed hours from the first image."),
            self.csv_scope, self.csv_comparison, self.csv_tolerance,
            self._button("Export CSV tables", lambda _: self.run("csv_export"))])
        echem = w.VBox([self._browse(self.echem, "either"), self.echem_timezone,
            self._button("Load electrochemistry", lambda _: self.run("electrochemistry")),
            self._button("Synchronize and save table", lambda _: self.run("synchronization")),
            self._button("Detect events", lambda _: self.run("event_detection")),
            self._button("Summarize cycle endpoints and residual signal", lambda _: self.run("cycle_summary")),
            self.plot_methods, self.show_ec_thickness,
            self._button("Make combined figure", lambda _: self.run("combined_plot")),
            self._button("Make synchronized video", lambda _: self.run("combined_video"))])
        exports = w.VBox([self._button("Make image panel", lambda _: self.run("image_plot")),
                          self._button("Make individual figures", lambda _: self.run("individual_image_plots")),
                          self._button("Make image video", lambda _: self.run("image_video")),
                          self._button("Update overview (MD + PDF)", lambda _: self.report())])
        spectral = w.VBox([w.HTML("<small>Open a reconstructed cube above. Set axis_order, axis_start, axis_step, axis_label and bands in advanced tof_cube settings.</small>"),
                           self._button("Analyze ToF cube", lambda _: self.run("tof_cube"))])
        nci = w.VBox([self._help("Legacy plating metrics", "The older routine clips negative pixels to zero and, unless configured otherwise, uses 5% of each frame's maximum as its threshold. "
            "Use Plating inhomogeneity for a fixed, explicit threshold and the same measurement region."),
                      self._button("Calculate legacy NCI plating metrics", lambda _: self.run("nci_plating_metrics")),
                      self._button("Make NCI event outputs", lambda _: self.run("nci_event_outputs"))])
        self.panels = [("Dataset and profile", project, None), ("Preview and appearance", appearance, None),
                       ("Thickness / ROI measurements", measurements, None), ("Plating inhomogeneity", pattern, "nci"),
                       ("Time window / zoom", windows, None),
                       ("Figures and videos", exports, None), ("CSV exports", csv, None),
                       ("Electrochemistry", echem, "operando_nci"), ("NCI metrics", nci, "operando_nci"),
                       ("Time-of-flight", spectral, "tof"),
                       ("Advanced settings", w.VBox([self.advanced, self.apply_advanced]), None)]
        self.accordion = w.Accordion(layout=w.Layout(width="100%"))
        self.widget = w.VBox([self.browser.widget, self.accordion, self.status, self.output], layout=w.Layout(width="100%"))
        self.mode.observe(self._mode_changed, names="value")
        self.region.observe(self._region_changed, names="value")
        self.window_slider.observe(self._slider_changed, names="value")
        self.window_mode.observe(lambda _: self._update_slider(), names="value")
        self._region_changed()
        self._mode_changed()

    def _text(self, label, value=""):
        return self.w.Text(description=label, value=value, style={"description_width": "initial"},
                           layout=self.w.Layout(width="100%", min_width="300px"))

    def _button(self, label, callback):
        button = self.w.Button(description=label, layout=self.w.Layout(width="auto", min_width=f"{max(180, len(label)*8+35)}px"))
        button.on_click(callback)
        return button

    def _choice(self, label, options, value):
        return self.w.Dropdown(description=label, options=options, value=value,
            style={"description_width": "initial"}, layout=self.w.Layout(width="100%"))

    def _check(self, label, value):
        return self.w.Checkbox(description=label, value=value, indent=False, layout=self.w.Layout(width="100%"))

    def _help(self, title, explanation):
        return self.w.HTML(f"<details><summary>{escape(title)} ⓘ</summary>{escape(explanation)}</details>")

    def _region_changed(self, _=None):
        for control, kind in ((self.circle_controls, "circle"), (self.rectangle, "rectangle"), (self.polygon, "polygon")):
            control.layout.display = "" if self.region.value == kind else "none"

    def _slider_changed(self, change):
        if getattr(self, "_updating_slider", False) or self.window_slider.disabled:
            return
        self.window_start.value, self.window_end.value = [f"{hour:.12g}" for hour in change["new"]]
        self.window_enabled.value = True

    def _update_slider(self):
        times = self.session.artifacts.get("image_times", []) if self.session else []
        available = bool(times) and all(stamp is not None for stamp in times)
        span = (max(times)-min(times)).total_seconds()/3600 if available else 0
        self._updating_slider = True
        try:
            self.window_slider.disabled = span <= 0 or self.window_mode.value != "elapsed"
            self.window_slider.max = max(span, .001)
            self.window_slider.step = max(span/1000, .000001)
            start, end = 0., span
            if self.window_mode.value == "elapsed":
                try:
                    start = float(self.window_start.value or 0)
                    end = float(self.window_end.value or span)
                except ValueError:
                    pass
            self.window_slider.value = (max(0, min(start, span)), max(0, min(max(start, end), span)))
        finally:
            self._updating_slider = False

    def reset_window(self):
        self.window_enabled.value = False
        self.window_start.value = self.window_end.value = ""
        self._update_slider()
        if self.session:
            self.run("time_window")

    def _measurement_settings(self):
        roi = {}
        if self.region.value == "circle":
            coordinates = [self.roi_x.value, self.roi_y.value, self.roi_r.value]
            if not all(coordinates):
                raise ValueError("Set the circle centre x, y and radius.")
            roi = dict(zip(("x0", "y0", "r0"), map(float, coordinates)))
        elif self.region.value == "rectangle":
            coordinates = [float(value) for value in self.rectangle.value.split(",")]
            if len(coordinates) != 4:
                raise ValueError("Rectangle needs x_min, x_max, y_min, y_max.")
            roi = dict(zip(("x_min", "x_max", "y_min", "y_max"), coordinates))
        elif self.region.value == "polygon":
            roi = {"vertices": json.loads(self.polygon.value)}
        return dict(region=self.region.value, roi=roi, methods=list(self.methods.value) or ["mean"],
            primary_method=self.primary_method.value, electrode_boundary_confirmed=self.boundary_confirmed.value,
            electrode_diameter_mm=float(self.electrode_diameter.value) if self.electrode_diameter.value.strip() else None,
            electrode_area_mm2=float(self.electrode_area.value) if self.electrode_area.value.strip() else None,
            active_threshold=float(self.active_threshold.value) if self.active_threshold.value.strip() else None,
            trim_fraction=float(self.trim_fraction.value))

    def _row(self, controls):
        return self.w.HBox(controls, layout=self.w.Layout(flex_flow="row wrap", width="100%"))

    def _browse(self, field, mode, extensions=()):
        button = self._button("Browse", lambda _: self.browser.show(field, mode=mode, extensions=extensions))
        return self._row([field, button])

    def _mode_changed(self, _=None):
        panels = [item for item in self.panels if item[2] in (None, self.mode.value)
                  or (item[2] == 'nci' and self.mode.value in {'exsitu', 'operando_nci'})]
        self.accordion.children = tuple(item[1] for item in panels)
        for i, (title, _, _) in enumerate(panels):
            self.accordion.set_title(i, title)
        self.tof_btn.layout.display = "" if self.mode.value == "tof" else "none"
        if self.session is not None:
            self.session.preset = self.mode.value

    def _show_error(self, exc):
        with self.output:
            self.output.clear_output(wait=True)
            print(f"{type(exc).__name__}: {exc}")

    def _input_settings(self):
        return {"pattern": self.pattern.value, "fits_hdu": self.fits_hdu.value,
                "pixel_size_um": float(self.pixel_size.value) if self.pixel_size.value.strip() else None,
                "unit": self.units.value, "timestamp_timezone": self.timezone.value,
                "start_time": self.start_time.value,
                "interval_seconds": float(self.interval.value) if self.interval.value.strip() else None}

    def open(self, cube=False):
        try:
            values = deepcopy(self.session.config) if self.session else {}
            values["input"] = self._input_settings()
            values.setdefault("visualization", {})["export_layout"] = self.export_layout.value
            self.session = open_visualization(self.source.value, profile=self.profile.value,
                output_root=self.output_root.value or None, preset=self.mode.value, config=values, cube=cube)
            self._loaded_input = self._input_settings()
            self._loaded_source = self.source.value
            self._loaded_profile = (self.profile.value, self.output_root.value)
            self.frame.max = max(0, len(self.session.artifacts.get("image_stack", []))-1)
            self._update_slider()
            self._sync()
            self.save()
            self.refresh()
        except Exception as exc:
            self._show_error(exc)

    def _sync(self):
        if self.session is None:
            raise ValueError("Open a processed dataset or image folder first.")
        if (self._input_settings() != self._loaded_input or self.source.value != self._loaded_source
                or (self.profile.value, self.output_root.value) != self._loaded_profile):
            raise ValueError("Dataset/profile controls changed. Click Open images (or Open ToF cube) before running or saving.")
        low = float(self.vmin.value) if self.vmin.value.strip() else None
        self.session.config["visualization"]["export_layout"] = self.export_layout.value
        high = float(self.vmax.value) if self.vmax.value.strip() else None
        style = {"cmap": self.cmap.value, "vmin": low, "vmax": high,
                 "scalebar_length_um": float(self.scale.value), "dpi": int(self.dpi.value)}
        for section in ("image_plot", "individual_image_plots", "image_video", "combined_video", "nci_event_outputs"):
            self.session.configure(section, **style, enabled=True)
        self.session.configure("image_plot", columns=int(self.columns.value), format=self.figure_format.value)
        self.session.configure("individual_image_plots", indices=self.indices.value, format=self.figure_format.value)
        if bool(self.surface_zmin.value.strip()) != bool(self.surface_zmax.value.strip()):
            raise ValueError("Supply both 3D height limits or leave both blank.")
        zlim = [float(self.surface_zmin.value), float(self.surface_zmax.value)] if self.surface_zmin.value.strip() else None
        for section in ("image_plot", "individual_image_plots", "image_video", "nci_event_outputs"):
            self.session.configure(section, render_3d=self.render_3d.value,
                surface_elev=float(self.surface_elev.value), surface_azim=float(self.surface_azim.value),
                surface_stride=int(self.surface_stride.value), surface_zlim=zlim)
        self.session.configure("individual_image_plots", color_scale=self.individual_color_scale.value)
        self.session.configure("image_video", fps=float(self.fps.value), timing_mode=self.video_timing.value,
            dt_seconds=float(self.video_dt.value) if self.video_dt.value.strip() else None,
            image_tolerance_seconds=float(self.video_tolerance.value) if self.video_tolerance.value.strip() else None)
        self.session.configure("combined_video", fps=float(self.fps.value))
        self.session.configure("selection", indices=self.indices.value, representative_count=int(self.count.value))
        measurement = self._measurement_settings()
        self.session.configure("image_measurements", **measurement)
        self.session.configure("inhomogeneity", active_threshold=measurement['active_threshold'],
            min_mean_signal=float(self.pattern_floor.value) if self.pattern_floor.value.strip() else None)
        if self.region.value in {"whole", "circle"}:
            self.session.configure("nci_plating_metrics", roi=measurement["roi"])
        self.session.configure("combined_plot", image_methods=list(self.plot_methods.value),
                               thickness_traces="both" if self.show_ec_thickness.value else "nci")
        self.session.configure("time_window", enabled=self.window_enabled.value, mode=self.window_mode.value,
            start=self.window_start.value, end=self.window_end.value, apply_figures=self.window_figures.value,
            apply_panels=self.window_panels.value, apply_videos=self.window_videos.value)
        self.session.configure("csv_export", scope=self.csv_scope.value, include_comparison=self.csv_comparison.value,
                               tolerance_seconds=float(self.csv_tolerance.value))
        self.session.configure("electrochemistry", path=self.echem.value, enabled=bool(self.echem.value.strip()),
                               timestamp_timezone=self.echem_timezone.value)
        self.session.configure('quality', baseline_indices=[int(i.strip()) for i in self.baseline_indices.value.split(',') if i.strip()])

    def run(self, section):
        try:
            self._sync()
            if section not in visible_sections(self.mode.value, self.session.sections):
                raise ValueError("This action is not part of the selected mode.")
            with self.output:
                self.output.clear_output(wait=True)
                if section in {"image_plot", "combined_plot", "nci_event_outputs"}:
                    selected = self.session.run("selection")
                    if selected.state.value != "completed":
                        raise ValueError(selected.message)
                if section != "time_window":
                    self.session.configure(section, enabled=True)
                if section == "nci_plating_metrics" and self.region.value not in {"whole", "circle"}:
                    raise ValueError("Legacy NCI plating metrics require a circle; use Measure images for other regions.")
                outcome = self.session.run(section)
                print(f"{outcome.state.value}: {outcome.message}")
            self._update_slider()
            self.refresh()
        except Exception as exc:
            self._show_error(exc)

    def preview(self):
        try:
            import matplotlib.pyplot as plt
            from .visualization import _draw_frame, _limits
            self._sync()
            images = self.session.artifacts.require("image_stack")
            index = self.frame.value
            config = self.session.section_config("image_plot")
            figure, axis = plt.subplots(subplot_kw={"projection": "3d"} if config.get("render_3d", False) else {})
            try:
                artist = _draw_frame(axis, images[index], self.session, config, index, _limits(images, config))
                figure.colorbar(artist, ax=axis, label=self.session.artifacts.get("image_unit") or "Image intensity")
                with self.output:
                    self.output.clear_output(wait=True)
                    self.display(figure)
            finally:
                plt.close(figure)
        except Exception as exc:
            self._show_error(exc)

    def preview_region(self):
        try:
            import numpy as np
            import matplotlib.pyplot as plt
            self._sync()
            image = self.session.artifacts.require("image_stack")[self.frame.value]
            mask = region_mask(image.shape, self._measurement_settings())
            figure, axis = plt.subplots()
            try:
                axis.imshow(image, origin=self.session.section_config("image_plot").get("image_origin", "upper"), cmap=self.cmap.value)
                axis.imshow(np.ma.masked_where(~mask, np.ones(mask.shape)), cmap="summer", vmin=0, vmax=1,
                            alpha=.3, origin=self.session.section_config("image_plot").get("image_origin", "upper"))
                axis.set(title=f"Measurement region: {int(mask.sum())} pixels (full original image)", xlabel="x (pixel)", ylabel="y (pixel)")
                with self.output:
                    self.output.clear_output(wait=True)
                    self.display(figure)
            finally:
                plt.close(figure)
        except Exception as exc:
            self._show_error(exc)

    def estimate_center(self):
        try:
            from .geometry import estimate_electrode_center
            self._sync()
            if self.region.value != 'circle': raise ValueError('Choose a circular measurement region first.')
            fit=estimate_electrode_center(self.session.artifacts.require('image_stack'),self._measurement_settings()['roi'],
                pixel_size_um=self.session.artifacts.require('pixel_size_um'),diameter_mm=float(self.electrode_diameter.value))
            self.roi_x.value,self.roi_y.value,self.roi_r.value=[str(fit['roi'][key]) for key in ('x0','y0','r0')]
            self.boundary_confirmed.value=False
            self._sync();self.session.configure('quality',centre_estimation=fit)
            save_visualization_settings(self.session)
            with self.output:
                self.output.clear_output(wait=True)
                print(fit['reason']);print(f"ROI: {fit['roi']}; supported arc: {fit['arc_support_fraction']:.0%}. Remeasure images after changing the ROI.")
        except Exception as exc:
            self._show_error(exc)

    def report(self):
        """Use recorded results, not unsaved GUI edits or rerun sections."""
        try:
            if self.session is None:
                raise ValueError("Open a saved visualization profile first.")
            from .reports import create_report
            paths = create_report(self.session.artifacts.require("results_dir"), style=self.session.config.get('reports',{}).get('style','concise'))
            with self.output:
                self.output.clear_output(wait=True)
                print("Overview of recorded results (no processing rerun):")
                for path in paths.values():
                    print(path)
        except Exception as exc:
            self._show_error(exc)

    def save(self):
        try:
            self._sync()
            self.config_path.value = str(save_visualization_settings(self.session))
            self.refresh()
        except Exception as exc:
            self._show_error(exc)

    def _populate(self):
        values = self.session.config
        inputs = values["input"]
        self.mode.value = self.session.preset
        self.source.value = inputs["path"]
        self.profile.value = values["visualization"]["profile"]
        self.output_root.value = values["visualization"]["output_root"]
        self.export_layout.value = values["visualization"].get("export_layout", "versioned")
        for control, key in ((self.pattern, "pattern"), (self.fits_hdu, "fits_hdu"),
                             (self.pixel_size, "pixel_size_um"), (self.units, "unit"),
                             (self.timezone, "timestamp_timezone"), (self.start_time, "start_time"), (self.interval, "interval_seconds")):
            value = inputs.get(key)
            control.value = "" if value in (None, "") else str(value)
        figures = values["image_plot"]
        for control, key, default in ((self.cmap, "cmap", "inferno"), (self.vmin, "vmin", ""),
            (self.vmax, "vmax", ""), (self.scale, "scalebar_length_um", 0), (self.dpi, "dpi", 150),
            (self.columns, "columns", 0), (self.figure_format, "format", "png")):
            value = figures.get(key, default)
            control.value = "" if value is None else str(value)
        self.fps.value = str(values["image_video"].get("fps", 10))
        self.render_3d.value = bool(figures.get("render_3d", False))
        self.surface_elev.value = str(figures.get("surface_elev", 35))
        self.surface_azim.value = str(figures.get("surface_azim", -60))
        self.surface_stride.value = str(figures.get("surface_stride", 2))
        zlim = figures.get("surface_zlim") or None
        self.surface_zmin.value = str(zlim[0]) if zlim is not None else ""
        self.surface_zmax.value = str(zlim[1]) if zlim is not None else ""
        self.individual_color_scale.value = values["individual_image_plots"].get("color_scale", "global")
        self.video_timing.value = values["image_video"].get("timing_mode", "image_sequence")
        for control, key in ((self.video_dt, "dt_seconds"), (self.video_tolerance, "image_tolerance_seconds")):
            value = values["image_video"].get(key)
            control.value = "" if value in (None, "") else str(value)
        self.indices.value = str(values["selection"].get("indices", ""))
        self.count.value = str(values["selection"].get("representative_count", 4))
        self.echem.value = str(values["electrochemistry"].get("path", ""))
        self.echem_timezone.value = str(values["electrochemistry"].get("timestamp_timezone", ""))
        for control, key in ((self.roi_x, "x0"), (self.roi_y, "y0"), (self.roi_r, "r0")):
            control.value = str(values["image_measurements"].get("roi", {}).get(key, ""))
        measurement = values["image_measurements"]
        self.boundary_confirmed.value = bool(measurement.get('electrode_boundary_confirmed',False))
        diameter=measurement.get('electrode_diameter_mm')
        self.electrode_diameter.value='' if diameter in (None,'') else str(diameter)
        self.baseline_indices.value = ', '.join(map(str,values.get('quality',{}).get('baseline_indices') or []))
        roi = measurement.get("roi") or {}
        self.region.value = measurement.get("region") or ("circle" if all(k in roi for k in ("x0", "y0", "r0")) else "whole")
        self.rectangle.value = ", ".join(str(roi.get(key, "")) for key in ("x_min", "x_max", "y_min", "y_max"))
        self.polygon.value = json.dumps(roi.get("vertices", []))
        self.methods.value = tuple(method_names(measurement.get("methods")))
        self.primary_method.value = measurement.get("primary_method") or self.methods.value[0]
        self.electrode_area.value = str(measurement.get("electrode_area_mm2") or "")
        self.active_threshold.value = "" if measurement.get("active_threshold") in (None, "") else str(measurement["active_threshold"])
        pattern_threshold = values.get('inhomogeneity', {}).get('active_threshold')
        if pattern_threshold not in (None, ''):
            self.active_threshold.value = str(pattern_threshold)
        self.trim_fraction.value = str(measurement.get("trim_fraction", .1))
        floor = values.get('inhomogeneity', {}).get('min_mean_signal')
        self.pattern_floor.value = '' if floor in (None, '') else str(floor)
        self.plot_methods.value = tuple(method_names(values["combined_plot"]["image_methods"])) if values["combined_plot"].get("image_methods") else ()
        from . import backends
        self.show_ec_thickness.value = "echem" in backends.nci_plots().normalize_thickness_traces(values["combined_plot"].get("thickness_traces", "both"))
        window = values.get("time_window", {})
        self.window_enabled.value = bool(window.get("enabled", False))
        self.window_mode.value = window.get("mode", "elapsed")
        self.window_start.value = "" if window.get("start") in (None, "") else str(window["start"])
        self.window_end.value = "" if window.get("end") in (None, "") else str(window["end"])
        for control, key in ((self.window_figures, "apply_figures"), (self.window_panels, "apply_panels"), (self.window_videos, "apply_videos")):
            control.value = bool(window.get(key, True))
        csv = values.get("csv_export", {})
        self.csv_scope.value = csv.get("scope", "full")
        self.csv_comparison.value = bool(csv.get("include_comparison", True))
        self.csv_tolerance.value = str(csv.get("tolerance_seconds", 30))
        self._loaded_input, self._loaded_source = self._input_settings(), self.source.value
        self._loaded_profile = (self.profile.value, self.output_root.value)
        self.frame.max = max(0, len(self.session.artifacts.get("image_stack", []))-1)
        self._update_slider()

    def load(self):
        try:
            self.session = load_visualization_settings(self.config_path.value)
            self._populate()
            self.refresh()
        except Exception as exc:
            self._show_error(exc)

    def apply_settings(self):
        try:
            self._sync()
            updates = json.loads(self.advanced.value)
            if not isinstance(updates, dict):
                raise ValueError("Advanced settings must be a JSON object of section objects.")
            for section, values in updates.items():
                if section not in self.session.sections or not isinstance(values, dict):
                    raise ValueError(f"Unknown analysis section or invalid settings: {section}")
            for section, values in updates.items():
                self.session.configure(section, **values)
            self._populate()
            self.save()
        except Exception as exc:
            self._show_error(exc)

    def refresh(self):
        if self.session is None:
            return
        root = self.session.artifacts.require("results_dir")
        count = len(self.session.artifacts.get("image_stack", []))
        rows = "".join(f"<li>{escape(row['title'])}: {escape(row['state'])} {escape(row['message'])}</li>"
                       for row in self.session.summary() if row["section"] in visible_sections(self.mode.value, self.session.sections))
        self.status.value = (f"<p>{count} frame(s). Real timestamps: {self.session.artifacts.get('timing_available', False)}. "
                             f"Profile: <code>{escape(str(root))}</code></p><details><summary>Section status</summary><ul>{rows}</ul></details>")

    def show(self):
        self.display(self.widget)
        return self


def launch_visualization_gui(preset="exsitu"):
    return VisualizationGUI(preset).show()

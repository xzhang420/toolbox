"""Notebook-native GUI controller for the operando workflow.

This module intentionally uses ``ipywidgets`` because the current environment
already supports it, while heavier local web GUI packages are not installed.
The GUI calls the same backend functions used by ``operando_pipeline.ipynb`` and
prints/displays notebook-style outputs in each step.
"""

import configparser
import importlib.util
import ast
import json
import os
import shutil
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tifffile


TOOLBOX_DIR = Path(__file__).resolve().parents[1]
# Always load sibling modules from this development copy.  Using
# ``TOOLBOX_DIR / 'operando'`` would silently execute the production backend.
OPERANDO_DIR = Path(__file__).resolve().parent
PLOTS_DIR = TOOLBOX_DIR / "plots"


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


operando_data = _load_module("operando_data_gui", OPERANDO_DIR / "operando_data.py")
operando_plots = _load_module("operando_plots_gui", OPERANDO_DIR / "operando_plots.py")
operando_plating_metrics = _load_module(
    "operando_plating_metrics_gui",
    OPERANDO_DIR / "operando_plating_metrics.py",
)
operando_interactive = _load_module("operando_interactive_gui", OPERANDO_DIR / "operando_interactive.py")
plots_mod = _load_module("plots_mod_gui", PLOTS_DIR / "plots.py")


class OperandoWorkflowGUI:
    """Interactive notebook GUI for the operando pipeline."""

    def __init__(self, campaign_dir=None, history_path=None):
        try:
            import ipywidgets as widgets
            from IPython.display import Image, Video, display
        except Exception as exc:
            raise RuntimeError(
                "ipywidgets and IPython are required for the notebook GUI."
            ) from exc

        self.widgets = widgets
        self.display = display
        self.Image = Image
        self.Video = Video
        self.campaign_dir = Path(campaign_dir or os.getcwd()).resolve()
        self.history_path = Path(history_path or self.campaign_dir / ".operando_gui_history.json")
        self.history = self._load_history()
        self.state = {}
        self.config = configparser.ConfigParser()

        self._build_widgets()
        self._wire_events()
        self.load_config()

    # ------------------------------------------------------------------
    # Persistence and path helpers
    # ------------------------------------------------------------------
    def _load_history(self):
        if not self.history_path.exists():
            return {
                "campaign_dirs": [],
                "exp_dirs": [],
                "reference_dirs": [],
                "echem_folders": [],
                "result_roots": [],
                "beamhistory_paths": [],
                "reference_beamhistory_paths": [],
                "settings": {},
            }
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _history_settings(self):
        settings = self.history.get("settings")
        if not isinstance(settings, dict):
            settings = {}
            self.history["settings"] = settings
        return settings

    def _history_value(self, key, default):
        return self._history_settings().get(key, default)

    def _remember_setting(self, key, value):
        self._history_settings()[key] = value
        self._save_history()

    def _setting_widget_map(self):
        names = [
            "reference_bin_y",
            "reference_bin_x",
            "reference_final_subfolder",
            "reference_thickness_um",
            "file_filter",
            "tpx_time",
            "n_tpx_per_image",
            "skip_correction_if_exists",
            "recompute_obint",
            "clip_ob_to_small_roi_p5",
            "ob_clip_percentile",
            "background_n_images",
            "background_correction",
            "recompute_background",
            "recompute_final",
            "clip_negatives",
            "conv_thickness",
            "thickness_capacity_mode",
            "time_column",
            "frames_per_merge",
            "spatial_pixel_merge",
            "merge_method",
            "pause_gap_minutes",
            "ec_plot_ymin",
            "ec_plot_ymax",
            "roi_large_r0",
            "roi_large_x0",
            "roi_large_y0",
            "roi_small_r0",
            "roi_small_x0",
            "roi_small_y0",
            "plot_thickness",
            "thickness_traces",
            "thickness_layout",
            "overlay_offset_fraction",
            "nci_metric",
            "lithium_density_g_cm3",
            "plot_cmap",
            "plot_vmin",
            "plot_vmax",
            "plot_cbar_mode",
            "plot_cbar_ticks",
            "video_plot_thickness",
            "video_thickness_traces",
            "video_thickness_layout",
            "video_overlay_offset_fraction",
            "video_cmap",
            "video_vmin",
            "video_vmax",
            "video_cbar_ticks",
            "preview_morphology_3d",
            "pixel_size",
            "scalebar_length",
            "scale_label",
            "plating_metrics_enabled",
            "plating_threshold_fraction",
            "plating_threshold_abs",
            "surface_stride",
            "surface_elev",
            "surface_azim",
            "manual_selected_indices",
            "manual_cycle_numbers",
            "show_cycle_numbers",
            "show_nci_thickness",
            "snap_markers_to_tips",
            "tip_snap_tolerance_seconds",
            "fps",
            "dt_seconds",
            "video_dpi",
        ]
        return {name: getattr(self, name) for name in names if hasattr(self, name)}

    def _remember_all_settings(self):
        settings = self._history_settings()
        for name, widget in self._setting_widget_map().items():
            settings[name] = widget.value
        self._save_history()

    def _apply_history_settings(self):
        settings = self._history_settings()
        for name, widget in self._setting_widget_map().items():
            if name not in settings:
                continue
            try:
                widget.value = settings[name]
            except Exception:
                pass

    def _save_history(self):
        self.history_path.write_text(
            json.dumps(self.history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _remember(self, key, value):
        value = str(value or "").strip()
        if not value:
            return
        values = list(self.history.get(key, []))
        if value in values:
            values.remove(value)
        values.insert(0, value)
        self.history[key] = values[:20]
        self._save_history()

    def _resolve(self, path_value):
        path_value = str(path_value or "").strip()
        if not path_value:
            return ""
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return str(path)
        return str((Path(self.base_dir.value).expanduser() / path).resolve())

    def _abs_or_none(self, path_value):
        resolved = self._resolve(path_value)
        return os.path.abspath(resolved) if resolved else None

    def _relative_or_abs(self, path_value):
        path = Path(str(path_value)).expanduser()
        try:
            return str(path.resolve().relative_to(Path(self.base_dir.value).resolve()))
        except Exception:
            return str(path)

    def _relative_to_base_if_possible(self, path_value):
        path_value = str(path_value or "").strip()
        if not path_value:
            return ""
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            return path_value
        try:
            return str(path.resolve().relative_to(Path(self.base_dir.value).expanduser().resolve()))
        except Exception:
            return str(path)

    def _display_path_value(self, label, path_value):
        if label == "base_dir":
            return str(path_value)
        return self._relative_or_abs(path_value)

    def _base_relative_subdirs(self, *parts, include_self=False, recursive=False):
        base = Path(self.base_dir.value).expanduser()
        root = base.joinpath(*parts)
        options = []
        if include_self and root.exists() and root.is_dir():
            try:
                options.append(str(root.resolve().relative_to(base.resolve())))
            except Exception:
                pass
        if not root.exists() or not root.is_dir():
            return options
        iterator = root.rglob("*") if recursive else root.iterdir()
        for path in iterator:
            if not path.is_dir():
                continue
            try:
                rel = str(path.resolve().relative_to(base.resolve()))
            except Exception:
                continue
            options.append(rel)
        return sorted(dict.fromkeys(options))

    def _refresh_base_dir_choices(self):
        base = Path(self.base_dir.value).expanduser()
        if not base.exists():
            return
        discovered = []
        try:
            discovered = operando_data.list_nci_experiment_dirs(str(base))
        except Exception:
            discovered = []
        relative_discovered = [self._relative_to_base_if_possible(path) for path in discovered]
        legacy_experiments = self._base_relative_subdirs("experiment")
        legacy_references = self._base_relative_subdirs("reference")
        self.exp_dir.options = sorted(dict.fromkeys(relative_discovered + legacy_experiments))
        self.reference_dir.options = sorted(dict.fromkeys(relative_discovered + legacy_references))
        top_level_dirs = self._base_relative_subdirs()
        self.echem_folder.options = top_level_dirs
        self.result_root.options = top_level_dirs

    def _derived_paths(self):
        exp_dir = Path(self._resolve(self.exp_dir.value))
        try:
            exp_name = operando_data.resolve_nci_experiment(str(exp_dir))["experiment"]
        except Exception:
            exp_name = exp_dir.name
        result_root = Path(self._resolve(self.result_root.value))
        results_dir = result_root / f"results_{exp_name}"
        return {
            "exp_dir": str(exp_dir),
            "exp_name": exp_name,
            "result_root": str(result_root),
            "results_dir": str(results_dir),
            "results_images_dir": str(results_dir / "images"),
            "results_processing_dir": str(results_dir / "processing"),
            "results_output_dir": str(results_dir / "output"),
            "corrected_dir": str(results_dir / "images" / f"obint_corrected_{exp_name}"),
            "uncorrected_dir": str(results_dir / "images" / f"uncorrected_{exp_name}"),
            "final_dir": str(results_dir / "images" / f"final_{exp_name}"),
            "bg_path": str(results_dir / "processing" / f"background_{exp_name}.tif"),
            "metadata_path": str(results_dir / "processing" / "processing_metadata.json"),
            "obint_beam_log_path": str(results_dir / "processing" / "obint_beam_log.csv"),
            "obint_preview_path": str(results_dir / "processing" / "obint_corrected_preview.png"),
            "output_plot_dir": str(results_dir / "output"),
            "video_output_path": str(results_dir / "output" / f"{exp_name}.mp4"),
            "selected_summary_path": str(results_dir / "output" / f"selected_image_summary_{exp_name}.txt"),
            "plating_metrics_csv_path": str(results_dir / "processing" / f"plating_metrics_{exp_name}.csv"),
            "plating_metrics_plot_path": str(results_dir / "output" / f"plating_metrics_{exp_name}.png"),
        }

    def _analysis_image_dir(self, derived=None):
        derived = derived or self._derived_paths()
        return derived["final_dir"] if self.background_correction.value else derived["corrected_dir"]

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _combobox(self, description, value="", history_key=None, width="760px"):
        options = self.history.get(history_key, []) if history_key else []
        return self.widgets.Combobox(
            description=description,
            value=str(value or ""),
            options=options,
            ensure_option=False,
            layout=self.widgets.Layout(width=width),
            style={"description_width": "170px"},
        )

    def _button(self, label, style=""):
        return self.widgets.Button(
            description=label,
            button_style=style,
            layout=self.widgets.Layout(width="180px"),
        )

    def _compact_step5_widgets(self):
        compact_width = self.widgets.Layout(width="250px")
        small_width = self.widgets.Layout(width="210px")
        compact_style = {"description_width": "115px"}
        small_style = {"description_width": "105px"}
        for widget in [
            self.conv_thickness,
            self.time_column,
            self.frames_per_merge,
            self.spatial_pixel_merge,
            self.merge_method,
            self.pause_gap_minutes,
            self.ec_plot_ymin,
            self.ec_plot_ymax,
            self.roi_large_r0,
            self.roi_large_x0,
            self.roi_large_y0,
            self.roi_small_r0,
            self.roi_small_x0,
            self.roi_small_y0,
            self.overlay_offset_fraction,
            self.lithium_density_g_cm3,
            self.plot_cmap,
            self.plot_vmin,
            self.plot_vmax,
            self.plot_cbar_ticks,
            self.fps,
            self.dt_seconds,
            self.video_dpi,
            self.video_overlay_offset_fraction,
            self.video_cmap,
            self.video_vmin,
            self.video_vmax,
            self.video_cbar_ticks,
            self.pixel_size,
            self.scalebar_length,
            self.scale_label,
            self.plating_threshold_fraction,
            self.surface_stride,
            self.surface_elev,
            self.surface_azim,
        ]:
            widget.layout = small_width
            widget.style = small_style
        for widget in [
            self.thickness_capacity_mode,
            self.thickness_traces,
            self.thickness_layout,
            self.nci_metric,
            self.plot_cbar_mode,
        ]:
            widget.layout = compact_width
            widget.style = compact_style
        video_wide_width = self.widgets.Layout(width="360px")
        video_wide_style = {"description_width": "190px"}
        for widget in [
            self.video_thickness_traces,
            self.video_thickness_layout,
            self.video_overlay_offset_fraction,
        ]:
            widget.layout = video_wide_width
            widget.style = video_wide_style

    def _build_widgets(self):
        w = self.widgets
        self.config_path = self._combobox(
            "Config path",
            str(self.campaign_dir / "operando_config.ini"),
            width="860px",
        )
        self.load_config_btn = self._button("Load config", "info")
        self.save_config_btn = self._button("Save config", "warning")

        self.base_dir = self._combobox("Base dir", str(self.campaign_dir), "campaign_dirs")
        self.exp_dir = self._combobox("Experiment dir", "", "exp_dirs")
        self.reference_dir = self._combobox("Reference dir", "", "reference_dirs")
        self.echem_folder = self._combobox("Echem folder", "", "echem_folders")
        self.result_root = self._combobox("Result root", "result", "result_roots")

        self.beamhistory_path = self._combobox("Exp beamhistory", "", "beamhistory_paths")
        self.reference_beamhistory_path = self._combobox(
            "Ref beamhistory",
            "",
            "reference_beamhistory_paths",
        )

        self.reference_bin_y = w.IntText(description="Reference bin y", value=2, style={"description_width": "170px"})
        self.reference_bin_x = w.IntText(description="Reference bin x", value=2, style={"description_width": "170px"})
        self.reference_final_subfolder = w.Text(
            description="Ref final subfolder",
            value="final",
            layout=w.Layout(width="420px"),
            style={"description_width": "170px"},
        )
        self.reference_thickness_um = w.FloatText(
            description=f"Ref thickness {operando_plots.MICRON_UNIT}",
            value=1.0,
            style={"description_width": "170px"},
        )
        self.file_filter = w.Text(
            description="File filter",
            value="",
            layout=w.Layout(width="420px"),
            style={"description_width": "170px"},
        )
        self.tpx_time = w.IntText(description="TPX time", value=2, style={"description_width": "170px"})
        self.n_tpx_per_image = w.IntText(description="TPX per image", value=30, style={"description_width": "170px"})

        self.skip_correction_if_exists = w.Checkbox(
            description="Reuse corrected files if present",
            value=True,
            indent=False,
        )
        self.recompute_obint = w.Checkbox(
            description="Recompute OBINT correction",
            value=False,
            indent=False,
        )
        self.clip_ob_to_small_roi_p5 = w.Checkbox(
            description="Clip OB to small ROI percentile",
            value=False,
            indent=False,
        )
        self.ob_clip_percentile = w.BoundedFloatText(
            description="OB clip percentile",
            value=5.0,
            min=0.0,
            max=100.0,
            step=0.5,
            style={"description_width": "170px"},
            layout=w.Layout(width="250px"),
        )
        self.background_n_images = w.IntText(
            description="Background images",
            value=int(self._history_value("background_n_images", 5)),
            style={"description_width": "170px"},
        )
        self.background_correction = w.Checkbox(
            description="Apply background correction",
            value=False,
            indent=False,
        )
        self.recompute_background = w.Checkbox(
            description="Recompute background even if file exists",
            value=False,
            indent=False,
        )
        self.recompute_final = w.Checkbox(
            description="Recompute final background-subtracted images",
            value=False,
            indent=False,
        )
        self.clip_negatives = w.Checkbox(description="Clip negatives", value=True, indent=False)

        self.conv_thickness = w.FloatText(
            description="conv_thickness",
            value=3.66,
            style={"description_width": "170px"},
        )
        self.thickness_capacity_mode = w.Dropdown(
            description="EC thickness mode",
            options=[
                ("current sign half-cycle", "current_sign_half_cycle"),
                ("potential sign half-cycle", "potential_sign_half_cycle"),
                ("raw capacity", "raw"),
            ],
            value="current_sign_half_cycle",
            style={"description_width": "170px"},
            layout=w.Layout(width="460px"),
        )
        self.time_column = w.Text(description="Time column", value="time/s", style={"description_width": "170px"})
        self.frames_per_merge = w.IntText(description="Frames per merge", value=1, style={"description_width": "170px"})
        self.spatial_pixel_merge = w.IntText(description="Pixel merge", value=1, style={"description_width": "170px"})
        self.merge_method = w.Dropdown(
            description="Merge method",
            options=["mean", "sum"],
            value="mean",
            style={"description_width": "170px"},
        )
        self.pause_gap_minutes = w.FloatText(
            description="Pause gap min",
            value=5.5,
            style={"description_width": "170px"},
        )
        self.ec_plot_ymin = w.FloatText(
            description="EC y min",
            value=-0.4,
            style={"description_width": "170px"},
        )
        self.ec_plot_ymax = w.FloatText(
            description="EC y max",
            value=1.0,
            style={"description_width": "170px"},
        )

        self.roi_large_r0 = w.IntText(description="Large r0", value=160, style={"description_width": "110px"})
        self.roi_large_x0 = w.IntText(description="Large x0", value=221, style={"description_width": "110px"})
        self.roi_large_y0 = w.IntText(description="Large y0", value=222, style={"description_width": "110px"})
        self.roi_small_r0 = w.IntText(description="Small r0", value=111, style={"description_width": "110px"})
        self.roi_small_x0 = w.IntText(description="Small x0", value=221, style={"description_width": "110px"})
        self.roi_small_y0 = w.IntText(description="Small y0", value=222, style={"description_width": "110px"})

        self.plot_thickness = w.Checkbox(description="Show thickness in plots", value=False, indent=False)
        self.thickness_traces = w.Dropdown(
            description="Thickness traces",
            options=["nci", "echem", "both", "none"],
            value="nci",
            style={"description_width": "170px"},
        )
        self.thickness_layout = w.Dropdown(
            description="Thickness layout",
            options=["overlay", "separate"],
            value="overlay",
            style={"description_width": "170px"},
        )
        self.overlay_offset_fraction = w.FloatText(
            description="Thickness offset",
            value=0.0,
            style={"description_width": "170px"},
        )
        self.nci_metric = w.Dropdown(
            description="NCI metric",
            options=[
                ("average thickness", "average_thickness"),
                ("total Li mass", "total_mass"),
            ],
            value="average_thickness",
            style={"description_width": "170px"},
        )
        self.lithium_density_g_cm3 = w.FloatText(
            description="Li density g/cm3",
            value=0.534,
            style={"description_width": "170px"},
        )
        self.plot_cmap = w.Text(description="Plot cmap", value="inferno", style={"description_width": "170px"})
        self.plot_vmin = w.FloatText(description="Plot vmin", value=0.0, style={"description_width": "170px"})
        self.plot_vmax = w.FloatText(description="Plot vmax", value=3.0, style={"description_width": "170px"})
        self.plot_cbar_mode = w.Dropdown(
            description="Plot cbar mode",
            options=["shared left", "shared right", "individual bottom", "individual top"],
            value="shared left",
            style={"description_width": "170px"},
        )
        self.plot_cbar_ticks = w.IntText(description="Plot cbar ticks", value=4, style={"description_width": "170px"})
        self.video_plot_thickness = w.Checkbox(description="Show thickness in video", value=False, indent=False)
        self.video_thickness_traces = w.Dropdown(
            description="Video thickness traces",
            options=["nci", "echem", "both", "none"],
            value="nci",
            style={"description_width": "170px"},
        )
        self.video_thickness_layout = w.Dropdown(
            description="Video thickness layout",
            options=["overlay", "separate"],
            value="overlay",
            style={"description_width": "170px"},
        )
        self.video_overlay_offset_fraction = w.FloatText(
            description="Video thickness offset",
            value=0.0,
            style={"description_width": "170px"},
        )
        self.video_cmap = w.Text(description="Video cmap", value="inferno", style={"description_width": "170px"})
        self.video_vmin = w.FloatText(description="Video vmin", value=0.0, style={"description_width": "170px"})
        self.video_vmax = w.FloatText(description="Video vmax", value=1.5, style={"description_width": "170px"})
        self.video_cbar_ticks = w.IntText(description="Video cbar ticks", value=5, style={"description_width": "170px"})

        self.preview_morphology_3d = w.Checkbox(description="Preview 3D", value=False, indent=False)
        self.pixel_size = w.FloatText(description="Pixel size", value=55, style={"description_width": "170px"})
        self.scalebar_length = w.FloatText(description="Scale bar length", value=5000, style={"description_width": "170px"})
        self.scale_label = w.Text(description="Scale label", value="5 mm", style={"description_width": "170px"})
        self.plating_metrics_enabled = w.Checkbox(
            description="Enable plating metrics",
            value=True,
            indent=False,
        )
        self.plating_threshold_fraction = w.BoundedFloatText(
            description="Active threshold frac",
            value=0.05,
            min=0.0,
            max=1.0,
            step=0.01,
            style={"description_width": "170px"},
        )
        self.plating_threshold_abs = w.Text(
            description="Active threshold abs",
            value="",
            placeholder="optional; overrides fraction",
            layout=w.Layout(width="360px"),
            style={"description_width": "170px"},
        )
        self.surface_stride = w.IntText(description="Surface stride", value=2, style={"description_width": "170px"})
        self.surface_elev = w.IntText(description="Surface elev", value=35, style={"description_width": "170px"})
        self.surface_azim = w.IntText(description="Surface azim", value=-60, style={"description_width": "170px"})

        self.manual_selected_indices = w.Text(
            description="Selected image indices",
            value="",
            placeholder="e.g. 0, 10, 20 or leave empty for detected events",
            layout=w.Layout(width="760px"),
            style={"description_width": "170px"},
        )
        self.manual_cycle_numbers = w.Text(
            description="Manual cycles",
            value="",
            placeholder="optional, one per selected image",
            layout=w.Layout(width="760px"),
            style={"description_width": "170px"},
        )
        self.show_cycle_numbers = w.Checkbox(description="Show cycle labels", value=True, indent=False)
        self.show_nci_thickness = w.Checkbox(description="Show NCI thickness labels", value=True, indent=False)
        self.snap_markers_to_tips = w.Checkbox(description="Snap markers to tips", value=True, indent=False)
        self.tip_snap_tolerance_seconds = w.IntText(
            description="Tip snap tol sec",
            value=300,
            style={"description_width": "170px"},
        )

        self.fps = w.IntText(description="Video FPS", value=10, style={"description_width": "170px"})
        self.dt_seconds = w.IntText(description="Video dt sec", value=120, style={"description_width": "170px"})
        self.video_dpi = w.IntText(description="Video DPI", value=150, style={"description_width": "170px"})

        self.validate_paths_btn = self._button("Validate paths", "info")
        self.tpx_ranges_btn = self._button("Show TPX ranges", "info")
        self.validate_beam_btn = self._button("Validate beam CSVs", "info")
        self.inspect_ob_btn = self._button("Inspect OB", "info")
        self.run_correction_btn = self._button("Run OBINT correction", "danger")
        self.inspect_obint_btn = self._button("Inspect OBINT correction", "info")
        self.run_background_btn = self._button("Run background correction", "warning")
        self.load_ec_preview_btn = self._button("Load EC preview", "info")
        self.load_nci_preview_btn = self._button("Load NCI preview", "info")
        self.calculate_plating_metrics_btn = self._button("Calculate metrics", "success")
        self.detect_events_btn = self._button("Detect events", "info")
        self.launch_selector_btn = self._button("Launch selector", "success")
        self.save_outputs_btn = self._button("Save final outputs", "success")
        self.export_video_btn = self._button("Export video", "success")
        self._compact_step5_widgets()

        self.outputs = {name: w.Output() for name in [
            "paths", "tpx", "beam", "processing", "metrics", "load", "events", "outputs", "video"
        ]}
        self.statuses = {
            name: w.HTML(value="")
            for name in ["paths", "tpx", "beam", "processing", "metrics", "load", "events", "outputs", "video"]
        }
        self.stop_requested = {name: False for name in self.statuses}
        self.stop_buttons = {}
        self.status_rows = {}
        for name in self.statuses:
            stop_button = self._button("Stop", "warning")
            stop_button.disabled = True
            stop_button.layout.display = "none"
            self.stop_buttons[name] = stop_button
            self.status_rows[name] = w.HBox([self.statuses[name], stop_button])

        self.path_step = w.VBox([
            w.HTML("<b>Step 1. Select paths and keep history</b>"),
            w.HBox([self.config_path, self.load_config_btn, self.save_config_btn]),
            self.base_dir,
            self.exp_dir,
            self.reference_dir,
            self.echem_folder,
            self.result_root,
            w.HBox([self.validate_paths_btn]),
            self.status_rows["paths"],
            self.outputs["paths"],
        ])
        self.tpx_step = w.VBox([
            w.HTML("<b>Step 2. TPX time ranges</b>"),
            w.HBox([self.tpx_ranges_btn]),
            self.status_rows["tpx"],
            self.outputs["tpx"],
        ])
        self.beam_step = w.VBox([
            w.HTML("<b>Step 3. Beam-history paths and coverage checks</b>"),
            self.beamhistory_path,
            self.reference_beamhistory_path,
            w.HBox([self.validate_beam_btn]),
            self.status_rows["beam"],
            self.outputs["beam"],
        ])
        self.processing_step = w.VBox([
            w.HTML("<b>Step 4. Optional intensity/OB/background correction</b>"),
            w.HBox([self.reference_bin_y, self.reference_bin_x, self.reference_thickness_um]),
            w.HBox([self.reference_final_subfolder, self.file_filter]),
            w.HBox([self.tpx_time, self.n_tpx_per_image]),
            w.HBox([self.skip_correction_if_exists, self.recompute_obint, self.clip_ob_to_small_roi_p5]),
            self.ob_clip_percentile,
            w.HTML("<b>Large crop ROI</b>"),
            w.HBox([self.roi_large_r0, self.roi_large_x0, self.roi_large_y0]),
            w.HTML("<b>Small NCI average ROI</b>"),
            w.HBox([self.roi_small_r0, self.roi_small_x0, self.roi_small_y0]),
            w.HBox([self.background_correction, self.background_n_images]),
            w.HBox([self.recompute_background, self.recompute_final, self.clip_negatives]),
            w.HBox([self.inspect_ob_btn, self.run_correction_btn, self.inspect_obint_btn, self.run_background_btn]),
            self.status_rows["processing"],
            self.outputs["processing"],
        ])
        self.metrics_step = w.VBox([
            w.HTML("<b>Step 5. Optional plating distribution metrics</b>"),
            w.HTML("<b>Large crop ROI</b>"),
            w.HBox([self.roi_large_r0, self.roi_large_x0, self.roi_large_y0]),
            w.HTML("<b>Nominal anode ROI</b>"),
            w.HBox([self.roi_small_r0, self.roi_small_x0, self.roi_small_y0]),
            w.HBox([self.pixel_size, self.plating_threshold_fraction]),
            w.HBox([self.plating_metrics_enabled, self.plating_threshold_abs]),
            w.HBox([self.calculate_plating_metrics_btn]),
            self.status_rows["metrics"],
            self.outputs["metrics"],
        ])
        self.load_step = w.VBox([
            w.HTML("<b>Step 6. Load images/echem and show the notebook previews</b>"),
            w.HBox([self.conv_thickness, self.time_column]),
            w.HBox([self.thickness_capacity_mode]),
            w.HBox([self.frames_per_merge, self.spatial_pixel_merge, self.merge_method]),
            w.HBox([self.pause_gap_minutes]),
            w.HBox([self.ec_plot_ymin, self.ec_plot_ymax]),
            w.HBox([self.plot_thickness, self.thickness_traces, self.thickness_layout]),
            w.HBox([self.overlay_offset_fraction, self.nci_metric, self.lithium_density_g_cm3]),
            w.HBox([self.pixel_size, self.scalebar_length, self.scale_label]),
            w.HBox([self.plot_cmap, self.plot_vmin, self.plot_vmax]),
            w.HBox([self.plot_cbar_mode, self.plot_cbar_ticks]),
            w.HBox([self.preview_morphology_3d]),
            w.HBox([self.surface_stride, self.surface_elev, self.surface_azim]),
            w.HBox([self.load_ec_preview_btn, self.load_nci_preview_btn]),
            self.status_rows["load"],
            self.outputs["load"],
        ])
        self.event_step = w.VBox([
            w.HTML("<b>Step 7. Detect events and select images</b>"),
            self.manual_selected_indices,
            w.HBox([self.snap_markers_to_tips, self.tip_snap_tolerance_seconds]),
            w.HBox([self.detect_events_btn, self.launch_selector_btn]),
            self.status_rows["events"],
            self.outputs["events"],
        ])
        self.output_step = w.VBox([
            w.HTML("<b>Step 8. Save final notebook outputs</b>"),
            self.manual_cycle_numbers,
            w.HBox([self.show_cycle_numbers, self.show_nci_thickness]),
            w.HBox([self.save_outputs_btn]),
            self.status_rows["outputs"],
            self.outputs["outputs"],
        ])
        self.video_step = w.VBox([
            w.HTML("<b>Step 9. Export synchronized video</b>"),
            w.HBox([self.fps, self.dt_seconds, self.video_dpi]),
            w.HBox([self.video_plot_thickness]),
            w.HBox([self.video_thickness_traces, self.video_thickness_layout]),
            w.HBox([self.video_overlay_offset_fraction]),
            w.HBox([self.video_cmap, self.video_vmin, self.video_vmax]),
            w.HBox([self.video_cbar_ticks]),
            w.HBox([self.export_video_btn]),
            self.status_rows["video"],
            self.outputs["video"],
        ])

        self.accordion = w.Accordion(children=[
            self.path_step,
            self.tpx_step,
            self.beam_step,
            self.processing_step,
            self.metrics_step,
            self.load_step,
            self.event_step,
            self.output_step,
            self.video_step,
        ])
        titles = [
            "1 Paths",
            "2 TPX Ranges",
            "3 Beam Histories",
            "4 Processing",
            "5 Plating Metrics",
            "6 Load Preview",
            "7 Events + Selection",
            "8 Final Outputs",
            "9 Video",
        ]
        for i, title in enumerate(titles):
            self.accordion.set_title(i, title)
        self.accordion.selected_index = 0

    def _wire_events(self):
        if getattr(self, "_events_wired", False):
            return
        self._events_wired = True
        for name, stop_button in self.stop_buttons.items():
            stop_button.on_click(lambda _, status_key=name: self._request_stop(status_key))
        self.load_config_btn.on_click(lambda _: self._run_with_status("paths", self.load_config_btn, self.load_config, "Loading config", show=True))
        self.save_config_btn.on_click(lambda _: self._run_with_status("paths", self.save_config_btn, self.save_config, "Saving config"))
        self.validate_paths_btn.on_click(lambda _: self._run_with_status("paths", self.validate_paths_btn, self.validate_paths, "Validating paths"))
        self.tpx_ranges_btn.on_click(lambda _: self._run_with_status("tpx", self.tpx_ranges_btn, self.show_tpx_ranges, "Reading TPX ranges"))
        self.validate_beam_btn.on_click(lambda _: self._run_with_status("beam", self.validate_beam_btn, self.validate_beamhistories, "Validating beam histories"))
        self.inspect_ob_btn.on_click(lambda _: self._run_with_status("processing", self.inspect_ob_btn, self.inspect_ob, "Inspecting OB image"))
        self.run_correction_btn.on_click(lambda _: self._run_with_status("processing", self.run_correction_btn, self.run_correction, "Running OBINT correction"))
        self.inspect_obint_btn.on_click(lambda _: self._run_with_status("processing", self.inspect_obint_btn, self.inspect_obint_correction, "Inspecting OBINT correction"))
        self.run_background_btn.on_click(lambda _: self._run_with_status("processing", self.run_background_btn, self.run_background_processing, "Running background processing"))
        self.calculate_plating_metrics_btn.on_click(lambda _: self._run_with_status("metrics", self.calculate_plating_metrics_btn, self.calculate_plating_metrics, "Calculating plating metrics"))
        self.load_ec_preview_btn.on_click(lambda _: self._run_with_status("load", self.load_ec_preview_btn, self.load_ec_preview, "Loading EC preview"))
        self.load_nci_preview_btn.on_click(lambda _: self._run_with_status("load", self.load_nci_preview_btn, self.load_nci_preview, "Loading NCI preview"))
        self.detect_events_btn.on_click(lambda _: self._run_with_status("events", self.detect_events_btn, self.detect_events, "Detecting events"))
        self.launch_selector_btn.on_click(lambda _: self._run_with_status("events", self.launch_selector_btn, self.launch_selector, "Launching selector"))
        self.save_outputs_btn.on_click(lambda _: self._run_with_status("outputs", self.save_outputs_btn, self.save_final_outputs, "Saving final outputs"))
        self.export_video_btn.on_click(lambda _: self._run_with_status("video", self.export_video_btn, self.export_video, "Exporting video"))
        self.base_dir.observe(lambda change: self._refresh_base_dir_choices(), names="value")
        for name, widget in self._setting_widget_map().items():
            widget.observe(
                lambda change, setting_name=name: self._remember_setting(setting_name, change["new"]),
                names="value",
            )
            if name in self._preview_state_setting_names():
                widget.observe(
                    lambda change: self._invalidate_preview_state(),
                    names="value",
                )

    def display_gui(self):
        self.display(self.accordion)
        return self

    def close_gui(self):
        """Close this GUI's widgets so stale notebook instances stop responding."""
        visited = set()

        def close_widget(widget):
            widget_id = id(widget)
            if widget_id in visited:
                return
            visited.add(widget_id)
            for child in getattr(widget, "children", ()):
                close_widget(child)
            try:
                widget.close()
            except Exception:
                pass

        close_widget(self.accordion)
        for output in self.outputs.values():
            try:
                output.clear_output()
                output.close()
            except Exception:
                pass
        return None

    def _run_with_status(self, status_key, button, func, message, *args, **kwargs):
        status = self.statuses.get(status_key)
        stop_button = self.stop_buttons.get(status_key)
        old_description = getattr(button, "description", "")
        old_disabled = getattr(button, "disabled", False)
        self.stop_requested[status_key] = False
        if status is not None:
            status.value = f"<span style='color:#b45309; font-weight:600;'>Running: {message}...</span>"
        if stop_button is not None:
            stop_button.disabled = False
            stop_button.layout.display = ""
        button.disabled = True
        button.description = "Running..."
        try:
            result = func(*args, **kwargs)
        except KeyboardInterrupt:
            if status is not None:
                status.value = f"<span style='color:#b45309; font-weight:600;'>Stopped: {message}.</span>"
        except Exception as exc:
            if status is not None:
                status.value = f"<span style='color:#b91c1c; font-weight:600;'>Failed: {message}. {type(exc).__name__}: {exc}</span>"
            raise
        else:
            if status is not None:
                status.value = f"<span style='color:#166534; font-weight:600;'>Done: {message}.</span>"
            return result
        finally:
            button.description = old_description
            button.disabled = old_disabled
            if stop_button is not None:
                stop_button.disabled = True
                stop_button.layout.display = "none"
            self.stop_requested[status_key] = False

    def _request_stop(self, status_key):
        self.stop_requested[status_key] = True
        status = self.statuses.get(status_key)
        if status is not None:
            status.value = (
                "<span style='color:#b45309; font-weight:600;'>Stop requested. "
                "The current backend call will stop at the next checkpoint; use the notebook kernel interrupt for an immediate hard stop.</span>"
            )

    def _check_stop(self, status_key):
        if self.stop_requested.get(status_key):
            raise KeyboardInterrupt

    def _preview_state_setting_names(self):
        return {
            "conv_thickness",
            "thickness_capacity_mode",
            "time_column",
            "frames_per_merge",
            "spatial_pixel_merge",
            "merge_method",
            "nci_metric",
            "lithium_density_g_cm3",
            "roi_large_r0",
            "roi_large_x0",
            "roi_large_y0",
            "roi_small_r0",
            "roi_small_x0",
            "roi_small_y0",
        }

    def _invalidate_preview_state(self):
        for key in [
            "derived",
            "roi_large",
            "roi_small",
            "img_datetimes",
            "img_series",
            "img_paths",
            "echem_datetimes",
            "potential_series",
            "echem_thickness_series",
            "echem_cycle_series",
            "avg_thickness_series",
            "event_context",
            "interactive_selection",
            "selected_outputs",
        ]:
            self.state.pop(key, None)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def load_config(self, show=False):
        path = Path(self.config_path.value).expanduser()
        self.config = configparser.ConfigParser()
        self.config.read(path, encoding="utf-8")
        if "general" in self.config:
            general = self.config["general"]
            self.base_dir.value = general.get("base_dir", fallback=self.base_dir.value)
            self._refresh_base_dir_choices()
            self.echem_folder.value = self._relative_to_base_if_possible(general.get("echem_folder", ""))
            self.result_root.value = self._relative_to_base_if_possible(general.get("result_root", "result"))
            self.skip_correction_if_exists.value = general.getboolean(
                "skip_pipeline_if_corrected_exists",
                fallback=True,
            )
        if "obint_correction.parameters" in self.config:
            obint = self.config["obint_correction.parameters"]
            self.exp_dir.value = self._relative_to_base_if_possible(obint.get("exp_dir", ""))
            self.beamhistory_path.value = obint.get("beamhistory_path", "")
            self.reference_dir.value = self._relative_to_base_if_possible(obint.get("reference_dir", ""))
            self.reference_beamhistory_path.value = obint.get("reference_beamhistory_path", "")
            self.reference_bin_y.value = obint.getint("reference_bin_y", fallback=2)
            self.reference_bin_x.value = obint.getint("reference_bin_x", fallback=2)
            self.reference_final_subfolder.value = obint.get("reference_final_subfolder", fallback="final")
            self.reference_thickness_um.value = obint.getfloat("reference_thickness_um", fallback=1.0)
            self.file_filter.value = obint.get("file_filter", fallback="")
            self.tpx_time.value = obint.getint("tpx_time", fallback=2)
            self.n_tpx_per_image.value = obint.getint("n_tpx_per_image", fallback=30)
            self.recompute_obint.value = obint.getboolean(
                "recompute_obint",
                fallback=self.recompute_obint.value,
            )
            self.clip_ob_to_small_roi_p5.value = obint.getboolean(
                "clip_ob_to_small_roi_p5",
                fallback=self.clip_ob_to_small_roi_p5.value,
            )
            self.ob_clip_percentile.value = obint.getfloat(
                "ob_clip_percentile",
                fallback=self.ob_clip_percentile.value,
            )
        if "background_processing.parameters" in self.config:
            background = self.config["background_processing.parameters"]
            self.background_correction.value = background.getboolean(
                "enabled",
                fallback=self.background_correction.value,
            )
            self.background_n_images.value = background.getint(
                "background_n_images",
                fallback=self.background_n_images.value,
            )
            self.recompute_background.value = background.getboolean(
                "recompute_background",
                fallback=self.recompute_background.value,
            )
            self.recompute_final.value = background.getboolean(
                "recompute_final",
                fallback=self.recompute_final.value,
            )
            self.clip_negatives.value = background.getboolean(
                "clip_negatives",
                fallback=self.clip_negatives.value,
            )
        if "data_preview.parameters" in self.config:
            preview = self.config["data_preview.parameters"]
            self.conv_thickness.value = preview.getfloat("conv_thickness", fallback=self.conv_thickness.value)
            self.thickness_capacity_mode.value = preview.get("thickness_capacity_mode", fallback=self.thickness_capacity_mode.value)
            self.time_column.value = preview.get("time_column", fallback=self.time_column.value)
            self.frames_per_merge.value = preview.getint(
                "frames_per_merge",
                fallback=preview.getint("merge_minutes", fallback=self.frames_per_merge.value),
            )
            self.spatial_pixel_merge.value = preview.getint("spatial_pixel_merge", fallback=self.spatial_pixel_merge.value)
            self.merge_method.value = preview.get("merge_method", fallback=self.merge_method.value)
            self.pause_gap_minutes.value = preview.getfloat("pause_gap_minutes", fallback=self.pause_gap_minutes.value)
            self.ec_plot_ymin.value = preview.getfloat("ec_plot_ymin", fallback=self.ec_plot_ymin.value)
            self.ec_plot_ymax.value = preview.getfloat("ec_plot_ymax", fallback=self.ec_plot_ymax.value)
            self.roi_large_r0.value = preview.getint("roi_large_r0", fallback=self.roi_large_r0.value)
            self.roi_large_x0.value = preview.getint("roi_large_x0", fallback=self.roi_large_x0.value)
            self.roi_large_y0.value = preview.getint("roi_large_y0", fallback=self.roi_large_y0.value)
            self.roi_small_r0.value = preview.getint("roi_small_r0", fallback=self.roi_small_r0.value)
            self.roi_small_x0.value = preview.getint("roi_small_x0", fallback=self.roi_small_x0.value)
            self.roi_small_y0.value = preview.getint("roi_small_y0", fallback=self.roi_small_y0.value)
            self.plot_thickness.value = preview.getboolean("plot_thickness", fallback=self.plot_thickness.value)
            self.thickness_traces.value = preview.get("thickness_traces", fallback=self.thickness_traces.value)
            self.thickness_layout.value = preview.get("thickness_layout", fallback=self.thickness_layout.value)
            self.overlay_offset_fraction.value = preview.getfloat("overlay_offset_fraction", fallback=self.overlay_offset_fraction.value)
            if "nci_metric" in preview:
                self.nci_metric.value = preview.get("nci_metric", fallback=self.nci_metric.value)
            elif preview.getboolean("show_total_mass", fallback=False):
                self.nci_metric.value = "total_mass"
            self.lithium_density_g_cm3.value = preview.getfloat("lithium_density_g_cm3", fallback=self.lithium_density_g_cm3.value)
            self.plot_cmap.value = preview.get("plot_cmap", fallback=self.plot_cmap.value)
            self.plot_vmin.value = preview.getfloat("plot_vmin", fallback=self.plot_vmin.value)
            self.plot_vmax.value = preview.getfloat("plot_vmax", fallback=self.plot_vmax.value)
            self.plot_cbar_mode.value = preview.get("plot_cbar_mode", fallback=self.plot_cbar_mode.value)
            self.plot_cbar_ticks.value = preview.getint("plot_cbar_ticks", fallback=self.plot_cbar_ticks.value)
            self.preview_morphology_3d.value = preview.getboolean("preview_morphology_3d", fallback=self.preview_morphology_3d.value)
            self.pixel_size.value = preview.getfloat("pixel_size", fallback=self.pixel_size.value)
            self.scalebar_length.value = preview.getfloat("scalebar_length", fallback=self.scalebar_length.value)
            self.scale_label.value = preview.get("scale_label", fallback=self.scale_label.value)
            self.surface_stride.value = preview.getint("surface_stride", fallback=self.surface_stride.value)
            self.surface_elev.value = preview.getint("surface_elev", fallback=self.surface_elev.value)
            self.surface_azim.value = preview.getint("surface_azim", fallback=self.surface_azim.value)
        if "plating_metrics.parameters" in self.config:
            metrics = self.config["plating_metrics.parameters"]
            self.plating_metrics_enabled.value = metrics.getboolean(
                "enabled",
                fallback=self.plating_metrics_enabled.value,
            )
            self.plating_threshold_fraction.value = metrics.getfloat(
                "threshold_fraction",
                fallback=self.plating_threshold_fraction.value,
            )
            self.plating_threshold_abs.value = metrics.get(
                "threshold_abs",
                fallback=self.plating_threshold_abs.value,
            )
        if "selection.parameters" in self.config:
            selection = self.config["selection.parameters"]
            self.manual_selected_indices.value = selection.get("manual_selected_indices", fallback=self.manual_selected_indices.value)
            self.manual_cycle_numbers.value = selection.get("manual_cycle_numbers", fallback=self.manual_cycle_numbers.value)
            self.show_cycle_numbers.value = selection.getboolean("show_cycle_numbers", fallback=self.show_cycle_numbers.value)
            self.show_nci_thickness.value = selection.getboolean("show_nci_thickness", fallback=self.show_nci_thickness.value)
            self.snap_markers_to_tips.value = selection.getboolean("snap_markers_to_tips", fallback=self.snap_markers_to_tips.value)
            self.tip_snap_tolerance_seconds.value = selection.getint("tip_snap_tolerance_seconds", fallback=self.tip_snap_tolerance_seconds.value)
        if "video.parameters" in self.config:
            video = self.config["video.parameters"]
            self.fps.value = video.getint("fps", fallback=self.fps.value)
            self.dt_seconds.value = video.getint("dt_seconds", fallback=self.dt_seconds.value)
            self.video_dpi.value = video.getint("video_dpi", fallback=self.video_dpi.value)
            self.video_plot_thickness.value = video.getboolean("video_plot_thickness", fallback=self.video_plot_thickness.value)
            self.video_thickness_traces.value = video.get("video_thickness_traces", fallback=self.video_thickness_traces.value)
            self.video_thickness_layout.value = video.get("video_thickness_layout", fallback=self.video_thickness_layout.value)
            self.video_overlay_offset_fraction.value = video.getfloat("video_overlay_offset_fraction", fallback=self.video_overlay_offset_fraction.value)
            self.video_cmap.value = video.get("video_cmap", fallback=self.video_cmap.value)
            self.video_vmin.value = video.getfloat("video_vmin", fallback=self.video_vmin.value)
            self.video_vmax.value = video.getfloat("video_vmax", fallback=self.video_vmax.value)
            self.video_cbar_ticks.value = video.getint("video_cbar_ticks", fallback=self.video_cbar_ticks.value)
        self._apply_history_settings()
        self._refresh_base_dir_choices()
        if show:
            self.validate_paths(clear=True)

    def save_config(self):
        with self.outputs["paths"]:
            self.outputs["paths"].clear_output()
            config = self._current_config_snapshot()
            path = Path(self.config_path.value).expanduser()
            with open(path, "w", encoding="utf-8") as handle:
                config.write(handle)
            print(f"Saved config to: {path}")
            self._remember_all_settings()
            self._update_history()

    def _current_config_snapshot(self, absolute_paths=False):
        config = configparser.ConfigParser()
        if absolute_paths:
            base_dir = self.base_dir.value
            echem_folder = self._resolve(self.echem_folder.value)
            result_root = self._resolve(self.result_root.value)
            exp_dir = self._resolve(self.exp_dir.value)
            beamhistory_path = self._resolve(self.beamhistory_path.value)
            reference_dir = self._resolve(self.reference_dir.value)
            reference_beamhistory_path = self._resolve(self.reference_beamhistory_path.value)
        else:
            base_dir = self.base_dir.value
            echem_folder = self.echem_folder.value
            result_root = self.result_root.value
            exp_dir = self.exp_dir.value
            beamhistory_path = self.beamhistory_path.value
            reference_dir = self.reference_dir.value
            reference_beamhistory_path = self.reference_beamhistory_path.value
        config["general"] = {
            "base_dir": base_dir,
            "echem_folder": echem_folder,
            "result_root": result_root,
            "skip_pipeline_if_corrected_exists": str(self.skip_correction_if_exists.value).lower(),
        }
        config["obint_correction.parameters"] = {
            "exp_dir": exp_dir,
            "beamhistory_path": beamhistory_path,
            "reference_dir": reference_dir,
            "reference_beamhistory_path": reference_beamhistory_path,
            "reference_bin_y": str(self.reference_bin_y.value),
            "reference_bin_x": str(self.reference_bin_x.value),
            "reference_final_subfolder": self.reference_final_subfolder.value,
            "reference_thickness_um": str(self.reference_thickness_um.value),
            "file_filter": self.file_filter.value,
            "tpx_time": str(self.tpx_time.value),
            "n_tpx_per_image": str(self.n_tpx_per_image.value),
            "recompute_obint": str(self.recompute_obint.value).lower(),
            "clip_ob_to_small_roi_p5": str(self.clip_ob_to_small_roi_p5.value).lower(),
            "ob_clip_percentile": str(self.ob_clip_percentile.value),
        }
        config["background_processing.parameters"] = {
            "enabled": str(self.background_correction.value).lower(),
            "background_n_images": str(self.background_n_images.value),
            "recompute_background": str(self.recompute_background.value).lower(),
            "recompute_final": str(self.recompute_final.value).lower(),
            "clip_negatives": str(self.clip_negatives.value).lower(),
        }
        config["data_preview.parameters"] = {
            "conv_thickness": str(self.conv_thickness.value),
            "thickness_capacity_mode": self.thickness_capacity_mode.value,
            "time_column": self.time_column.value,
            "frames_per_merge": str(self.frames_per_merge.value),
            "spatial_pixel_merge": str(self.spatial_pixel_merge.value),
            "merge_method": self.merge_method.value,
            "pause_gap_minutes": str(self.pause_gap_minutes.value),
            "ec_plot_ymin": str(self.ec_plot_ymin.value),
            "ec_plot_ymax": str(self.ec_plot_ymax.value),
            "roi_large_r0": str(self.roi_large_r0.value),
            "roi_large_x0": str(self.roi_large_x0.value),
            "roi_large_y0": str(self.roi_large_y0.value),
            "roi_small_r0": str(self.roi_small_r0.value),
            "roi_small_x0": str(self.roi_small_x0.value),
            "roi_small_y0": str(self.roi_small_y0.value),
            "plot_thickness": str(self.plot_thickness.value).lower(),
            "thickness_traces": self.thickness_traces.value,
            "thickness_layout": self.thickness_layout.value,
            "overlay_offset_fraction": str(self.overlay_offset_fraction.value),
            "nci_metric": self.nci_metric.value,
            "lithium_density_g_cm3": str(self.lithium_density_g_cm3.value),
            "plot_cmap": self.plot_cmap.value,
            "plot_vmin": str(self.plot_vmin.value),
            "plot_vmax": str(self.plot_vmax.value),
            "plot_cbar_mode": self.plot_cbar_mode.value,
            "plot_cbar_ticks": str(self.plot_cbar_ticks.value),
            "preview_morphology_3d": str(self.preview_morphology_3d.value).lower(),
            "pixel_size": str(self.pixel_size.value),
            "scalebar_length": str(self.scalebar_length.value),
            "scale_label": self.scale_label.value,
            "surface_stride": str(self.surface_stride.value),
            "surface_elev": str(self.surface_elev.value),
            "surface_azim": str(self.surface_azim.value),
        }
        config["plating_metrics.parameters"] = {
            "enabled": str(self.plating_metrics_enabled.value).lower(),
            "threshold_fraction": str(self.plating_threshold_fraction.value),
            "threshold_abs": self.plating_threshold_abs.value,
        }
        config["selection.parameters"] = {
            "manual_selected_indices": self.manual_selected_indices.value,
            "manual_cycle_numbers": self.manual_cycle_numbers.value,
            "show_cycle_numbers": str(self.show_cycle_numbers.value).lower(),
            "show_nci_thickness": str(self.show_nci_thickness.value).lower(),
            "snap_markers_to_tips": str(self.snap_markers_to_tips.value).lower(),
            "tip_snap_tolerance_seconds": str(self.tip_snap_tolerance_seconds.value),
        }
        config["video.parameters"] = {
            "fps": str(self.fps.value),
            "dt_seconds": str(self.dt_seconds.value),
            "video_dpi": str(self.video_dpi.value),
            "video_plot_thickness": str(self.video_plot_thickness.value).lower(),
            "video_thickness_traces": self.video_thickness_traces.value,
            "video_thickness_layout": self.video_thickness_layout.value,
            "video_overlay_offset_fraction": str(self.video_overlay_offset_fraction.value),
            "video_cmap": self.video_cmap.value,
            "video_vmin": str(self.video_vmin.value),
            "video_vmax": str(self.video_vmax.value),
            "video_cbar_ticks": str(self.video_cbar_ticks.value),
        }
        return config

    def _update_history(self):
        self._remember("campaign_dirs", self.base_dir.value)
        self._remember("exp_dirs", self.exp_dir.value)
        self._remember("reference_dirs", self.reference_dir.value)
        self._remember("echem_folders", self.echem_folder.value)
        self._remember("result_roots", self.result_root.value)
        self._remember("beamhistory_paths", self.beamhistory_path.value)
        self._remember("reference_beamhistory_paths", self.reference_beamhistory_path.value)

    # ------------------------------------------------------------------
    # Step actions
    # ------------------------------------------------------------------
    def validate_paths(self, clear=True):
        out = self.outputs["paths"]
        with out:
            if clear:
                out.clear_output()
            self._display_path_table()

    def _display_path_table(self):
        self._update_history()
        derived = self._derived_paths()
        rows = []
        for label, value, optional in [
            ("base_dir", self.base_dir.value, False),
            ("exp_dir", derived["exp_dir"], False),
            ("reference_dir", self._resolve(self.reference_dir.value), True),
            ("echem_folder", self._resolve(self.echem_folder.value), True),
            ("result_root", derived["result_root"], False),
            ("results_dir", derived["results_dir"], False),
            ("corrected_dir", derived["corrected_dir"], False),
            ("analysis_image_dir", self._analysis_image_dir(derived), False),
            ("background", derived["bg_path"] if self.background_correction.value else "", True),
            ("metadata", derived["metadata_path"], False),
        ]:
            exists = os.path.exists(value) if value else None
            rows.append({
                "path": label,
                "value": self._display_path_value(label, value) if value else "not set (optional)",
                "exists": exists,
                "optional": optional,
            })
        df = pd.DataFrame(rows)
        styled = df.style.set_properties(
            subset=["value"],
            **{
                "min-width": "688px",
                "max-width": "1168px",
                "white-space": "normal",
                "word-break": "break-word",
                "text-align": "left",
            },
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("text-align", "left")],
                }
            ]
        )
        self.display(styled)

    def show_tpx_ranges(self):
        out = self.outputs["tpx"]
        with out:
            out.clear_output()
            display_columns = [
                "exp_name",
                "layout_type",
                "run_count",
                "tpx_file_count",
                "timing_window_count",
                "timing_sources",
                "earliest_datetime",
                "latest_datetime",
                "duration_hours",
            ]
            rows = []
            for role, path_value in (
                ("Experiment", self.exp_dir.value),
                ("Reference", self.reference_dir.value),
            ):
                path = self._resolve(path_value)
                if not path:
                    print(f"{role}: not set (optional)")
                    continue
                try:
                    summary = operando_data.summarize_experiment_tpx_time_range(path)
                except Exception as exc:
                    print(f"{role}: could not inspect {path}: {exc}")
                    continue
                summary["role"] = role
                rows.append(summary)
            if rows:
                frame = pd.DataFrame(rows)
                self.display(frame[["role", *display_columns]])
            print("\nBeam history is optional. Leave both beam-history paths blank to skip intensity correction.")

    def validate_beamhistories(self):
        out = self.outputs["beam"]
        with out:
            out.clear_output()
            for label, path_value in [
                ("Experiment beamhistory", self.beamhistory_path.value),
                ("Reference beamhistory", self.reference_beamhistory_path.value),
            ]:
                path = self._resolve(path_value)
                print()
                print(f"{label}: {path or 'not set (optional)'}")
                if not path:
                    print("  SKIPPED")
                    continue
                if not os.path.isfile(path):
                    print("  MISSING")
                    continue
                df = operando_data.load_beamhistory(path)
                print(
                    f"  rows={len(df)} | range={df['timestamp'].min()} to {df['timestamp'].max()}"
                )
                print("  First 3 rows:")
                self.display(df.head(3))
                print("  Last 3 rows:")
                self.display(df.tail(3))
                plot_df = df.dropna(subset=["timestamp", "beam_count"]).copy()
                plot_times = [ts.to_pydatetime() for ts in pd.to_datetime(plot_df["timestamp"])]
                plot_beam = pd.to_numeric(plot_df["beam_count"], errors="coerce").to_numpy(dtype=float)
                fig, ax = plt.subplots(figsize=(10, 3.2))
                ax.plot(
                    plot_times,
                    plot_beam,
                    "-",
                    linewidth=1.0,
                    color="tab:blue",
                )
                ax.set_xlabel("Time")
                ax.set_ylabel("Beam count")
                ax.set_title(label)
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                self.display(fig)
                plt.close(fig)
            self._update_history()

    def _metadata_check(self, stage_name, expected_parameters, section_name=None):
        metadata_path = self._derived_paths()["metadata_path"]
        stored = operando_data.read_stage_metadata(
            metadata_path,
            stage_name=stage_name,
            section_name=section_name,
        )
        differences = operando_data.compare_stage_metadata(expected_parameters, stored)
        if differences:
            operando_data.print_stage_metadata_warnings(stage_name, metadata_path, differences)
        else:
            print(f"{stage_name} metadata check passed.")
        return differences

    def _display_obint_outputs(self, result):
        print("")
        print("Corrected directory  :", result["corrected_dir"])
        print("Uncorrected directory:", result["uncorrected_dir"])
        if result.get("beam_log_path"):
            print("Beam log path        :", result["beam_log_path"])

        beam_log = result.get("beam_log") or []
        if not beam_log:
            print("Beam log not available. Run OBINT correction once to create the saved inspection file.")
            return

        df = pd.DataFrame(beam_log)
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")
        print("Beam log preview:")
        self.display(pd.concat([df.head(3), df.tail(3)]).drop_duplicates().reset_index(drop=True))

        first_time = df["time"].iloc[0]
        last_time = df["time"].iloc[-1]
        print("First image time :", first_time.strftime("%Y-%m-%d %H:%M:%S"))
        print("Last image time  :", last_time.strftime("%Y-%m-%d %H:%M:%S"))

        time_values = [ts.to_pydatetime() for ts in df["time"]]
        relative_values = df["relative_factor"].tolist()

        time_diffs = df["time"].diff()
        expected_cadence = time_diffs.dropna().median()
        gap_threshold = max(pd.Timedelta(minutes=5), expected_cadence * 3)
        gap_rows = df.loc[time_diffs > gap_threshold, ["time"]].copy()
        gap_rows["gap_start"] = df["time"].shift(1)[time_diffs > gap_threshold].values
        gap_rows["gap_duration"] = time_diffs[time_diffs > gap_threshold].values

        if len(gap_rows) == 0:
            print("No image-acquisition gaps detected.")
        else:
            print("Image-acquisition gaps detected:")
            for _, row in gap_rows.iterrows():
                print(
                    f"  {row['gap_start'].strftime('%Y-%m-%d %H:%M:%S')} to "
                    f"{row['time'].strftime('%Y-%m-%d %H:%M:%S')} "
                    f"({row['gap_duration']})"
                )

        fig = plt.figure(figsize=(10, 4))
        plt.plot(time_values, relative_values, ".", markersize=3)
        for j, (_, row) in enumerate(gap_rows.iterrows()):
            plt.axvspan(
                row["gap_start"].to_pydatetime(),
                row["time"].to_pydatetime(),
                color="tab:red",
                alpha=0.18,
                label="No images acquired" if j == 0 else None,
            )
        plt.xlabel("Time")
        plt.ylabel("Beam / Reference (relative factor)")
        plt.title("Beam intensity over time")
        plt.grid(True)
        if len(gap_rows) > 0:
            plt.legend()
        plt.tight_layout()
        self.display(fig)
        plt.close(fig)

    def _display_image_directory_preview(
        self,
        image_dir,
        preview_path=None,
        title=None,
        missing_label="TIFF",
        saved_label="Preview path",
    ):
        import math

        image_preview_files = [
            entry.path
            for entry in sorted(os.scandir(image_dir), key=lambda e: e.name)
            if entry.is_file() and entry.name.lower().endswith((".tif", ".tiff"))
        ]

        n_preview = min(20, len(image_preview_files))
        if n_preview == 0:
            print(f"No {missing_label} files found in {image_dir}")
            return

        preview_files = image_preview_files[:n_preview]
        preview_images = [tifffile.imread(path) for path in preview_files]
        preview_means = [float(np.mean(image)) for image in preview_images]
        all_preview_pixels = np.concatenate([image.ravel() for image in preview_images])
        display_vmin = np.percentile(all_preview_pixels, 2)
        display_vmax = np.percentile(all_preview_pixels, 99.8)

        ncols = 5
        nrows = math.ceil(n_preview / ncols)
        fig = plt.figure(figsize=(3.2 * ncols, 3.0 * nrows + 2.8), constrained_layout=True)
        gs = fig.add_gridspec(nrows + 1, ncols, height_ratios=[1] * nrows + [0.9])
        axes = [fig.add_subplot(gs[row, col]) for row in range(nrows) for col in range(ncols)]

        for ax, path, image in zip(axes, preview_files, preview_images):
            ax.imshow(image, cmap="gray", vmin=display_vmin, vmax=display_vmax)
            ax.set_title(os.path.basename(path), fontsize=8)
            ax.axis("off")

        for ax in axes[n_preview:]:
            ax.axis("off")

        trend_ax = fig.add_subplot(gs[-1, :])
        trend_ax.plot(range(n_preview), preview_means, marker="o", linewidth=1.5, markersize=4)
        trend_ax.set_xlabel("Preview image index")
        trend_ax.set_xticks([x for x in [0, 5, 10, 15, 20] if x < n_preview])
        trend_ax.set_ylabel("Average pixel value")
        trend_ax.set_title("Average intensity evolution across preview images", fontsize=11)
        trend_ax.grid(True, alpha=0.3)

        fig.suptitle(
            title or f"First {n_preview} images from {os.path.basename(image_dir)}",
            fontsize=12,
        )
        if preview_path:
            os.makedirs(os.path.dirname(os.path.abspath(preview_path)), exist_ok=True)
            fig.savefig(preview_path, dpi=150)
            print(f"{saved_label}:", preview_path)
        self.display(fig)
        plt.close(fig)

    def _display_corrected_preview(self, corrected_dir, preview_path=None):
        self._display_image_directory_preview(
            corrected_dir,
            preview_path=preview_path,
            title="First obint-corrected images for background selection",
            missing_label="corrected TIFF",
            saved_label="Corrected preview path",
        )

    def _count_tiff_files(self, directory):
        if not os.path.isdir(directory):
            return 0
        return sum(
            1
            for entry in os.scandir(directory)
            if entry.is_file() and operando_data.is_tiff_file(entry.path)
        )

    def _expected_obint_output_count(self, derived):
        reference_beam_integral_per_minute = None
        if (
            self._resolve(self.beamhistory_path.value)
            and self._resolve(self.reference_dir.value)
            and self._resolve(self.reference_beamhistory_path.value)
        ):
            reference_beam_integral_per_minute = operando_data.compute_reference_beam_integral_per_minute(
                self._resolve(self.reference_dir.value),
                self._resolve(self.reference_beamhistory_path.value),
                self.tpx_time.value,
            )
        beam_log = operando_data.compute_obint_beam_log(
            exp_dir=self._resolve(self.exp_dir.value),
            beamhistory_path=self._resolve(self.beamhistory_path.value),
            reference_beam_integral_per_minute=reference_beam_integral_per_minute,
            tpx_time=self.tpx_time.value,
            n_tpx_per_image=self.n_tpx_per_image.value,
            file_filter=self.file_filter.value or None,
        )
        os.makedirs(os.path.dirname(os.path.abspath(derived["obint_beam_log_path"])), exist_ok=True)
        operando_data.save_obint_beam_log(beam_log, derived["obint_beam_log_path"])
        return len(beam_log), beam_log

    def _output_count_status(self, directory, expected_count):
        actual_count = self._count_tiff_files(directory)
        complete = expected_count > 0 and actual_count == expected_count
        return complete, actual_count

    def _load_or_create_obint_beam_log(self, derived):
        beam_log_path = derived["obint_beam_log_path"]
        beam_log = operando_data.load_obint_beam_log(beam_log_path)
        if beam_log:
            return beam_log

        print("Saved correction log not found. Reconstructing it from the selected image inputs...")
        reference_beam_integral_per_minute = None
        if (
            self._resolve(self.beamhistory_path.value)
            and self._resolve(self.reference_dir.value)
            and self._resolve(self.reference_beamhistory_path.value)
        ):
            reference_beam_integral_per_minute = operando_data.compute_reference_beam_integral_per_minute(
                self._resolve(self.reference_dir.value),
                self._resolve(self.reference_beamhistory_path.value),
                self.tpx_time.value,
            )
        beam_log = operando_data.compute_obint_beam_log(
            exp_dir=self._resolve(self.exp_dir.value),
            beamhistory_path=self._resolve(self.beamhistory_path.value),
            reference_beam_integral_per_minute=reference_beam_integral_per_minute,
            tpx_time=self.tpx_time.value,
            n_tpx_per_image=self.n_tpx_per_image.value,
            file_filter=self.file_filter.value or None,
        )
        operando_data.save_obint_beam_log(beam_log, beam_log_path)
        print(f"Saved OBINT beam log: {beam_log_path}")
        return beam_log

    def run_correction(self):
        out = self.outputs["processing"]
        with out:
            out.clear_output()
            self._check_stop("processing")
            derived = self._derived_paths()
            corrected_dir = derived["corrected_dir"]
            expected = {
                "exp_dir": os.path.abspath(self._resolve(self.exp_dir.value)),
                "beamhistory_path": self._abs_or_none(self.beamhistory_path.value),
                "output_dir": os.path.abspath(corrected_dir),
                "uncorrected_dir": os.path.abspath(derived["uncorrected_dir"]),
                "tpx_time": self.tpx_time.value,
                "n_tpx_per_image": self.n_tpx_per_image.value,
                "reference_dir": self._abs_or_none(self.reference_dir.value),
                "reference_beamhistory_path": self._abs_or_none(self.reference_beamhistory_path.value),
                "reference_bin_y": self.reference_bin_y.value,
                "reference_bin_x": self.reference_bin_x.value,
                "reference_final_subfolder": self.reference_final_subfolder.value,
                "reference_thickness_um": self.reference_thickness_um.value,
                "compute_dtype": np.float32,
                "corrected_output_dtype": np.float32,
                "file_filter": self.file_filter.value or None,
                "clip_ob_to_small_roi_p5": self.clip_ob_to_small_roi_p5.value,
            }
            if self.clip_ob_to_small_roi_p5.value:
                expected["ob_clip_roi"] = self._roi_small()
                expected["ob_clip_percentile"] = float(self.ob_clip_percentile.value)
            expected_count, beam_log = self._expected_obint_output_count(derived)
            corrected_complete, corrected_count = self._output_count_status(corrected_dir, expected_count)
            uncorrected_complete, uncorrected_count = self._output_count_status(derived["uncorrected_dir"], expected_count)
            uncorrected_has_expected_outputs = expected_count > 0 and uncorrected_count >= expected_count
            print(
                "OBINT output file check: "
                f"expected={expected_count}, corrected={corrected_count}, uncorrected={uncorrected_count}"
            )
            if uncorrected_count > expected_count:
                print(
                    "Uncorrected directory contains extra provenance TIFF files. "
                    "This can happen when a run has image files but no TPX files; "
                    "extra uncorrected files do not force OBINT recomputation."
                )
            if (
                self.skip_correction_if_exists.value
                and not self.recompute_obint.value
                and corrected_complete
                and uncorrected_has_expected_outputs
            ):
                print("CORRECTED DIRECTORY ALREADY EXISTS")
                print(f"Skipping correction: {corrected_dir}")
                self._metadata_check("obint_correction", expected)
                self.state["obint_result"] = {
                    "corrected_dir": corrected_dir,
                    "uncorrected_dir": derived["uncorrected_dir"],
                    "beam_log": beam_log,
                    "beam_log_path": derived["obint_beam_log_path"],
                }
                return
            if self.recompute_obint.value:
                print("Recompute OBINT correction is enabled; running correction even if existing files are present.")
            elif self.skip_correction_if_exists.value and (corrected_count or uncorrected_count):
                print(
                    "Existing OBINT outputs are incomplete or have extra TIFF files; "
                    "running correction instead of skipping."
                )
            self._check_stop("processing")
            result = operando_data.run_operando_pipeline(
                exp_dir=self._resolve(self.exp_dir.value),
                beamhistory_path=self._resolve(self.beamhistory_path.value),
                tpx_time=self.tpx_time.value,
                n_tpx_per_image=self.n_tpx_per_image.value,
                reference_dir=self._resolve(self.reference_dir.value),
                reference_beamhistory_path=self._resolve(self.reference_beamhistory_path.value),
                reference_bin_y=self.reference_bin_y.value,
                reference_bin_x=self.reference_bin_x.value,
                reference_final_subfolder=self.reference_final_subfolder.value,
                reference_thickness_um=self.reference_thickness_um.value,
                clip_ob_to_small_roi_p5=self.clip_ob_to_small_roi_p5.value,
                ob_clip_roi=self._roi_small(),
                ob_clip_percentile=self.ob_clip_percentile.value,
                file_filter=self.file_filter.value or None,
                metadata_path=derived["metadata_path"],
                results_dir=derived["results_dir"],
                corrected_dir=derived["corrected_dir"],
                uncorrected_dir=derived["uncorrected_dir"],
            )
            print("Correction summary:")
            for key, value in result.items():
                if key != "beam_log":
                    print(f"  {key}: {value}")
            self.state["obint_result"] = result

    def inspect_obint_correction(self):
        out = self.outputs["processing"]
        with out:
            out.clear_output()
            result = self.state.get("obint_result")
            if result is None:
                derived = self._derived_paths()
                corrected_dir = derived["corrected_dir"]
                if os.path.isdir(corrected_dir):
                    result = {
                        "corrected_dir": corrected_dir,
                        "uncorrected_dir": derived["uncorrected_dir"],
                        "beam_log": self._load_or_create_obint_beam_log(derived),
                        "beam_log_path": derived["obint_beam_log_path"],
                    }
                else:
                    print("Run OBINT correction before inspecting the correction output.")
                    return
            self._display_obint_outputs(result)
            derived = self._derived_paths()
            self._display_corrected_preview(result["corrected_dir"], derived["obint_preview_path"])

    def inspect_ob(self):
        out = self.outputs["processing"]
        with out:
            out.clear_output()
            self._check_stop("processing")
            reference_dir = self._resolve(self.reference_dir.value)
            if not reference_dir:
                print("No reference image path is set; OB/reference correction is disabled.")
                return
            ref_series = operando_data.load_reference_image_series(
                reference_dir,
                final_subfolder=self.reference_final_subfolder.value,
                dtype=np.float32,
                file_filter=self.file_filter.value or None,
            )
            if len(ref_series) == 0:
                print(
                    "No reference TIFF images found. "
                    "Check reference_dir, reference_final_subfolder, and file_filter."
                )
                return

            ob_image = operando_data.process_reference_images(
                ref_series,
                bin_y=self.reference_bin_y.value,
                bin_x=self.reference_bin_x.value,
                compute_dtype=np.float32,
            )
            roi_small = self._roi_small()
            clip_threshold = None
            clip_percentile = float(self.ob_clip_percentile.value)
            roi_values_before_clip = operando_data.circular_roi_values(
                ob_image,
                roi_small["r0"],
                roi_small["x0"],
                roi_small["y0"],
            )
            if roi_values_before_clip.size:
                roi_clip_percentile_before = float(np.percentile(roi_values_before_clip, clip_percentile))
                roi_mean_before = float(np.mean(roi_values_before_clip))
            else:
                roi_clip_percentile_before = np.nan
                roi_mean_before = np.nan
            if self.clip_ob_to_small_roi_p5.value:
                ob_image, clip_threshold = operando_data.clip_image_to_circular_roi_percentile(
                    ob_image,
                    roi_small,
                    percentile=clip_percentile,
                    compute_dtype=np.float32,
                )

            roi_values_after_clip = operando_data.circular_roi_values(
                ob_image,
                roi_small["r0"],
                roi_small["x0"],
                roi_small["y0"],
            )
            if roi_values_after_clip.size:
                roi_mean_after = float(np.mean(roi_values_after_clip))
            else:
                roi_mean_after = np.nan

            print(f"Reference image count: {len(ref_series)}")
            print(f"Final OB image shape: {ob_image.shape}")
            print(
                "Small ROI: "
                f"center=({roi_small['x0']}, {roi_small['y0']}), radius={roi_small['r0']} px"
            )
            print(
                "Small ROI before OB clipping: "
                f"mean={roi_mean_before:.6g}, p{clip_percentile:g}={roi_clip_percentile_before:.6g}"
            )
            if clip_threshold is not None:
                print(
                    "Applied OB floor to entire image: "
                    f"all pixels below {clip_threshold:.6g} were set to {clip_threshold:.6g}"
                )
                print(f"Small ROI after OB clipping: mean={roi_mean_after:.6g}")

            finite_pixels = ob_image[np.isfinite(ob_image)]
            if finite_pixels.size == 0:
                print("Final OB image contains no finite pixels.")
                return
            display_vmin = float(np.percentile(finite_pixels, 2.0))
            display_vmax = float(np.percentile(finite_pixels, 99.8))
            if display_vmin == display_vmax:
                display_vmin = float(np.min(finite_pixels))
                display_vmax = float(np.max(finite_pixels))

            fig, ax = plt.subplots(figsize=(7.2, 7.2))
            im = ax.imshow(ob_image, cmap="gray", vmin=display_vmin, vmax=display_vmax)
            circle = plt.Circle(
                (roi_small["x0"], roi_small["y0"]),
                roi_small["r0"],
                fill=False,
                edgecolor="cyan",
                linewidth=1.8,
                linestyle="-",
                alpha=0.95,
                zorder=20,
            )
            ax.add_patch(circle)
            ax.set_title("Final OB image with small ROI", fontsize=12)
            ax.set_xlabel("x pixel")
            ax.set_ylabel("y pixel")
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("OB intensity")
            fig.tight_layout()
            self.display(fig)
            plt.close(fig)

    def run_background_processing(self):
        out = self.outputs["processing"]
        with out:
            out.clear_output()
            self._check_stop("processing")
            derived = self._derived_paths()
            if not self.background_correction.value:
                analysis_dir = self._analysis_image_dir(derived)
                if not os.path.isdir(analysis_dir):
                    raise RuntimeError(
                        f"Corrected image directory does not exist: {analysis_dir}. "
                        "Run the correction/copy step first."
                    )
                self.state["analysis_image_dir"] = analysis_dir
                print("Background correction is disabled; using corrected images directly.")
                print(f"Analysis image directory: {analysis_dir}")
                return
            self._remember_setting("background_n_images", self.background_n_images.value)
            os.makedirs(derived["results_processing_dir"], exist_ok=True)
            os.makedirs(derived["final_dir"], exist_ok=True)
            bg_params = {
                "image_dir": os.path.abspath(derived["corrected_dir"]),
                "n_images": self.background_n_images.value,
                "compute_dtype": np.float32,
            }
            if os.path.exists(derived["bg_path"]) and not self.recompute_background.value:
                print("BACKGROUND IMAGE ALREADY EXISTS")
                print(f"Skipping background image computation: {derived['bg_path']}")
                self._metadata_check("background_image", bg_params, section_name="background_processing")
            else:
                bg = operando_data.compute_background_image(
                    derived["corrected_dir"],
                    n_images=self.background_n_images.value,
                    metadata_path=derived["metadata_path"],
                )
                tifffile.imwrite(derived["bg_path"], bg.astype(np.float32))
                print(f"Background path: {derived['bg_path']}")

            print("")
            self._check_stop("processing")
            expected_final_count = self._count_tiff_files(derived["corrected_dir"])
            final_complete, final_count = self._output_count_status(derived["final_dir"], expected_final_count)
            print(
                "Background-subtraction output file check: "
                f"expected={expected_final_count}, final={final_count}"
            )
            sub_params = {
                "input_dir": os.path.abspath(derived["corrected_dir"]),
                "output_dir": os.path.abspath(derived["final_dir"]),
                "background_path": os.path.abspath(derived["bg_path"]),
                "clip_negatives": self.clip_negatives.value,
                "compute_dtype": np.float32,
                "output_dtype": np.float32,
            }
            if final_complete and not self.recompute_final.value:
                print("FINAL BACKGROUND-SUBTRACTED DIRECTORY ALREADY EXISTS")
                print(f"Skipping background subtraction: {derived['final_dir']}")
                self._metadata_check("background_subtraction", sub_params, section_name="background_processing")
            else:
                if final_count and not self.recompute_final.value:
                    print(
                        "Existing final outputs are incomplete or have extra TIFF files; "
                        "running background subtraction instead of skipping."
                    )
                self._check_stop("processing")
                summary = operando_data.background_subtract_directory(
                    derived["corrected_dir"],
                    derived["final_dir"],
                    derived["bg_path"],
                    clip_negatives=self.clip_negatives.value,
                    skip_existing=not self.recompute_final.value,
                    metadata_path=derived["metadata_path"],
                )
                print("Background subtraction summary:")
                self.display(pd.DataFrame([summary]))

    def _roi_mask_center_radius_for_image(self, image, roi):
        arr = np.asarray(image)
        ny, nx = arr.shape
        crop_x0, crop_y0 = getattr(image, "crop_origin_xy", (0, 0))
        bin_factor = max(1, int(getattr(image, "spatial_bin_factor", 1)))
        cx = float(roi["x0"]) / bin_factor - float(crop_x0)
        cy = float(roi["y0"]) / bin_factor - float(crop_y0)
        radius = max(1.0, float(roi["r0"]) / bin_factor)
        yy, xx = np.ogrid[:ny, :nx]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        return mask, (cx, cy), radius

    def _plating_threshold_abs_value(self):
        text = str(self.plating_threshold_abs.value or "").strip()
        if not text:
            return None
        return float(text)

    def calculate_plating_metrics(self):
        out = self.outputs["metrics"]
        with out:
            out.clear_output()
            self._check_stop("metrics")
            if not self.plating_metrics_enabled.value:
                print("Plating metrics are disabled in this GUI/config.")
                return
            if not self._nci_is_thickness_calibrated():
                print(
                    "Plating metrics require reference-normalized thickness maps. "
                    "No reference correction is recorded, so this step was skipped."
                )
                return

            derived = self._derived_paths()
            os.makedirs(derived["output_plot_dir"], exist_ok=True)
            roi_large = self._roi_large()
            roi_small = self._roi_small()
            analysis_dir = self._analysis_image_dir(derived)
            print("Loading processed Li maps for optional plating metrics.")
            print(f"Analysis image directory: {analysis_dir}")
            print(
                "Nominal anode ROI: "
                f"center=({roi_small['x0']}, {roi_small['y0']}), radius={roi_small['r0']} px"
            )

            img_datetimes, img_series, img_paths = operando_plots.load_corrected_images_with_time(
                analysis_dir,
                tag_id=65002,
                **roi_large,
            )
            if not img_series:
                raise RuntimeError(f"No processed images were loaded from {analysis_dir}.")

            threshold_abs = self._plating_threshold_abs_value()
            threshold_fraction = float(self.plating_threshold_fraction.value)
            rows = []
            for frame_index, (img_dt, image) in enumerate(zip(img_datetimes, img_series)):
                self._check_stop("metrics")
                roi_mask, center, radius = self._roi_mask_center_radius_for_image(image, roi_small)
                metrics = operando_plots.calculate_plating_metrics(
                    image,
                    roi_mask=roi_mask,
                    pixel_size=self.pixel_size.value,
                    center=center,
                    threshold=threshold_abs,
                    threshold_fraction=threshold_fraction,
                    anode_radius=radius,
                )
                source_path = ""
                if frame_index < len(img_paths):
                    source_path = img_paths[frame_index]
                rows.append({
                    "frame_index": frame_index,
                    "datetime": img_dt,
                    "source_image_path": source_path,
                    "roi_center_x_px": center[0],
                    "roi_center_y_px": center[1],
                    "roi_radius_px": radius,
                    "pixel_size_um": self.pixel_size.value,
                    "threshold_fraction": threshold_fraction,
                    "threshold_abs": threshold_abs if threshold_abs is not None else "",
                    **metrics,
                })

            df = pd.DataFrame(rows)
            df = operando_plating_metrics.add_elapsed_time_columns(df)
            os.makedirs(os.path.dirname(os.path.abspath(derived["plating_metrics_csv_path"])), exist_ok=True)
            df.to_csv(derived["plating_metrics_csv_path"], index=False)
            self.state["plating_metrics"] = df
            print(f"Saved plating metrics CSV: {derived['plating_metrics_csv_path']}")
            print(f"Calculated metrics for {len(df)} processed frame(s).")
            self.display(df.head(12))

            fig, _, _ = operando_plating_metrics.plot_plating_metric_timeseries(
                df,
                metrics=("f_plate", "CV_active", "CV_global", "rho_Li", "d_COM_norm"),
                x="elapsed_time_h",
                dataset_col=None,
                style="screen",
                title="Lithium plating distribution metrics",
                save_path=derived["plating_metrics_plot_path"],
                dpi=220,
            )
            self.display(fig)
            plt.close(fig)
            print(f"Saved plating metrics plot: {derived['plating_metrics_plot_path']}")

    def _load_preview_state(self, report_summary=True):
        derived = self._derived_paths()
        roi_large = self._roi_large()
        roi_small = self._roi_small()
        analysis_dir = self._analysis_image_dir(derived)
        print("Loading processed images and optional electrochemical data for downstream plots.")
        img_datetimes, img_series, img_paths = operando_plots.load_corrected_images_with_time(
            analysis_dir,
            tag_id=65002,
            **roi_large,
        )
        if not img_datetimes:
            raise RuntimeError(f"No timestamped TIFF images were loaded from {analysis_dir}.")
        echem_path = self._resolve(self.echem_folder.value)
        if echem_path and os.path.isdir(echem_path):
            echem_datetimes, potential_series, echem_thickness_series, echem_cycle_series = operando_plots.load_echem_data(
                echem_path,
                conv_thickness=self.conv_thickness.value,
                thickness_capacity_mode=self.thickness_capacity_mode.value,
                time_column=self.time_column.value,
                report_summary=report_summary,
                include_cycle=True,
                window_start=min(img_datetimes),
                window_end=max(img_datetimes),
            )
        else:
            echem_datetimes = []
            potential_series = np.asarray([], dtype=float)
            echem_thickness_series = np.asarray([], dtype=float)
            echem_cycle_series = np.asarray([], dtype=float)
            print("No electrochemistry folder supplied; loaded NCI images only.")
        img_datetimes, img_series = operando_plots.merge_images_by_frame_count(
            img_datetimes,
            img_series,
            frames_per_merge=self.frames_per_merge.value,
            method=self.merge_method.value,
        )
        img_series = operando_plots.spatial_average_images(
            img_series,
            factor=self.spatial_pixel_merge.value,
        )
        avg_thickness_series = operando_plots.circular_average_thickness_series(img_series, **roi_small)
        total_mass_series = operando_plots.total_lithium_mass_series(
            img_series,
            pixel_size=self.pixel_size.value,
            lithium_density_g_cm3=self.lithium_density_g_cm3.value,
        )
        total_mass_mg_series = total_mass_series * 1000.0
        nci_is_thickness_calibrated = self._nci_is_thickness_calibrated()
        if self.nci_metric.value == "total_mass" and not nci_is_thickness_calibrated:
            print(
                "No reference normalization is recorded; total-Li-mass mode is not "
                "physically calibrated. Showing mean NCI intensity instead."
            )
        nci_metric_series = total_mass_mg_series if self._show_total_mass() else avg_thickness_series
        self.state.update({
            "derived": derived,
            "analysis_image_dir": analysis_dir,
            "roi_large": roi_large,
            "roi_small": roi_small,
            "img_datetimes": img_datetimes,
            "img_series": img_series,
            "img_paths": img_paths,
            "echem_datetimes": echem_datetimes,
            "potential_series": potential_series,
            "echem_thickness_series": echem_thickness_series,
            "echem_cycle_series": echem_cycle_series,
            "avg_thickness_series": avg_thickness_series,
            "total_mass_series": total_mass_series,
            "total_mass_mg_series": total_mass_mg_series,
            "nci_is_thickness_calibrated": nci_is_thickness_calibrated,
            "nci_metric_series": nci_metric_series,
        })
        return derived, img_datetimes, img_series, img_paths, echem_datetimes, potential_series, echem_thickness_series, echem_cycle_series, avg_thickness_series

    def load_ec_preview(self):
        out = self.outputs["load"]
        with out:
            out.clear_output()
            self._check_stop("load")
            (
                derived,
                img_datetimes,
                img_series,
                img_paths,
                echem_datetimes,
                potential_series,
                echem_thickness_series,
                echem_cycle_series,
                avg_thickness_series,
            ) = self._load_preview_state(report_summary=True)
            echem_path = self._resolve(self.echem_folder.value)
            used_echem_files = []
            if len(echem_datetimes) > 0:
                used_echem_files = operando_plots.summarize_echem_files_in_window(
                    echem_path,
                    window_start=min(img_datetimes),
                    window_end=max(img_datetimes),
                    time_column=self.time_column.value,
                )
            print(f"Loaded {len(img_series)} images from: {self.state['analysis_image_dir']}")
            if len(echem_datetimes) > 0:
                print(f"Loaded {len(echem_datetimes)} electrochemical points from: {echem_path}")
            else:
                print("Electrochemistry: not set; continuing in NCI-only mode.")
            print(f"Image time window: {min(img_datetimes)} to {max(img_datetimes)}")
            if used_echem_files:
                print("EC files contributing points in the current image window:")
                self.display(pd.DataFrame(used_echem_files))
            else:
                print("No EC files overlap the current image window.")

            self._check_stop("load")
            if len(echem_datetimes) > 0:
                pauses = operando_plots.find_paused_intervals(
                    echem_datetimes,
                    min_gap_minutes=self.pause_gap_minutes.value,
                )
                print(f"Detected EC pause intervals using pause_gap_minutes={self.pause_gap_minutes.value}: {len(pauses)}")
                for start, end in pauses:
                    print(f"  {start} -> {end} | gap={end - start}")
                fig = operando_plots.plot_potential_and_thickness(
                    echem_datetime_series=echem_datetimes,
                    potential_series=potential_series,
                    echem_thickness_series=echem_thickness_series,
                    img_datetime_series=img_datetimes,
                    avg_thickness_series=self.state["nci_metric_series"],
                    plot_thickness=self.plot_thickness.value,
                    pause_gap_minutes=self.pause_gap_minutes.value,
                    thickness_traces=self.thickness_traces.value,
                    thickness_layout=self.thickness_layout.value,
                    overlay_offset_fraction=self.overlay_offset_fraction.value,
                    **self._nci_metric_plot_kwargs(),
                    potential_ylim=self._ec_plot_ylim(),
                    show=False,
                )
            else:
                elapsed_hours = np.asarray(
                    [(value - img_datetimes[0]).total_seconds() / 3600.0 for value in img_datetimes]
                )
                fig, ax = plt.subplots(figsize=(8, 4.5))
                ax.plot(elapsed_hours, self.state["nci_metric_series"], "o-", color="tab:orange")
                ax.set_xlabel("Time elapsed (h)")
                ax.set_ylabel(self._nci_metric_plot_kwargs()["thickness_axis_label"])
                ax.set_title("NCI-only evolution")
                ax.grid(True, alpha=0.25)
                fig.tight_layout()
            self.display(fig)
            plt.close(fig)

    def _auto_preview_image_indices(self, count=4):
        thickness = np.asarray(self.state.get("avg_thickness_series", []), dtype=float)
        n_images = len(self.state.get("img_series", []))
        if n_images == 0:
            return [], []

        finite_indices = np.flatnonzero(np.isfinite(thickness))
        if finite_indices.size:
            sorted_indices = finite_indices[np.argsort(thickness[finite_indices])]
            quantile_positions = np.linspace(0, len(sorted_indices) - 1, min(int(count), len(sorted_indices)))
            preview_indices = []
            for pos in quantile_positions:
                idx = int(sorted_indices[int(round(pos))])
                if idx not in preview_indices:
                    preview_indices.append(idx)
            for idx in sorted_indices:
                idx = int(idx)
                if len(preview_indices) >= int(count):
                    break
                if idx not in preview_indices:
                    preview_indices.append(idx)
            preview_values = [float(thickness[idx]) for idx in preview_indices]
            return preview_indices, preview_values

        fallback_count = min(int(count), n_images)
        preview_indices = sorted(dict.fromkeys(
            int(round(pos)) for pos in np.linspace(0, n_images - 1, fallback_count)
        ))
        return preview_indices, [np.nan] * len(preview_indices)

    def load_nci_preview(self):
        out = self.outputs["load"]
        with out:
            out.clear_output()
            self._check_stop("load")
            if "img_datetimes" not in self.state:
                self._load_preview_state(report_summary=False)
            preview_image_indices, preview_thickness_values = self._auto_preview_image_indices(count=4)
            if not preview_image_indices:
                print("No images available for preview.")
                return
            print("Preview image indices selected from low-to-high average Li thickness:")
            for idx, value in zip(preview_image_indices, preview_thickness_values):
                if np.isfinite(value):
                    print(f"  image {idx}: avg thickness={value:.4g} {operando_plots.MICRON_UNIT}")
                else:
                    print(f"  image {idx}: avg thickness unavailable")
            self._check_stop("load")
            image_preview_kwargs = self._event_plot_kwargs()
            for key in [
                "show_cycle_numbers",
                "show_nci_thickness",
                "nci_thickness_unit",
                "nci_thickness_decimals",
                "image_annotation_fontsize",
            ]:
                image_preview_kwargs.pop(key, None)
            image_preview_kwargs.update({
                "figsize": (8.6, 8.4),
                "spacing": (0.008, 0.02),
                "render_3d": self.preview_morphology_3d.value,
                "lower_right_texts": [
                    f"{value:.3g} {operando_plots.MICRON_UNIT}" if np.isfinite(value) else ""
                    for value in preview_thickness_values
                ],
            })
            fig, axes = plots_mod.plot_images(
                img_series=self.state["img_series"],
                img_indices=preview_image_indices,
                label_start=0,
                nrows=2,
                ncols=2,
                **image_preview_kwargs,
            )
            if not self.preview_morphology_3d.value:
                for ax, image_idx in zip(np.ravel(axes), preview_image_indices):
                    if not hasattr(ax, "add_patch"):
                        continue
                    image = self.state["img_series"][image_idx]
                    crop_x0, crop_y0 = getattr(image, "crop_origin_xy", (0, 0))
                    bin_factor = max(1, int(getattr(image, "spatial_bin_factor", 1)))
                    roi_center_x = (self.roi_small_x0.value / bin_factor - crop_x0) * self._effective_pixel_size()
                    roi_center_y = (self.roi_small_y0.value / bin_factor - crop_y0) * self._effective_pixel_size()
                    roi_radius = self.roi_small_r0.value / bin_factor * self._effective_pixel_size()
                    circle = plt.Circle(
                        (roi_center_x, roi_center_y),
                        roi_radius,
                        fill=False,
                        edgecolor="cyan",
                        linewidth=1.8,
                        linestyle="-",
                        alpha=0.95,
                        zorder=20,
                    )
                    ax.add_patch(circle)
            self.display(fig)
            plt.close(fig)

    def load_preview_data(self):
        self.load_ec_preview()

    def detect_events(self):
        out = self.outputs["events"]
        with out:
            out.clear_output()
            self._require_loaded()
            ctx = self._build_event_context()
            self.state["event_context"] = ctx
            operando_plots.print_event_context_table(ctx, self.state["img_datetimes"])
            preview_text = operando_plots.format_event_context_preview_text(
                ctx,
                self.state["img_datetimes"],
                label_start=0,
            )
            print("\nPreview text shown above interactive selection:")
            print(preview_text)

    def launch_selector(self):
        out = self.outputs["events"]
        with out:
            out.clear_output()
            self._require_loaded()
            if "event_context" not in self.state:
                self.state["event_context"] = self._build_event_context()
            initial = self._initial_selector_indices()
            if len(self.state.get("echem_datetimes", [])) == 0:
                self.state["interactive_selection"] = {
                    "selected_image_indices": list(initial),
                }
                self.manual_selected_indices.value = ", ".join(str(i) for i in initial)
                print(
                    "No electrochemistry data is loaded. Using NCI-only image selections: "
                    f"{initial}. Edit the manual image-index field if desired, then save outputs."
                )
                return
            self._enable_widget_backend_for_selector()
            selector_output = self.widgets.Output()
            self.display(selector_output)
            selection = None
            with selector_output:
                selection = operando_interactive.launch_draggable_event_selector(
                    operando_plots=operando_plots,
                    event_context=self.state["event_context"],
                    echem_datetimes=self.state["echem_datetimes"],
                    potential_series=self.state["potential_series"],
                    img_datetimes=self.state["img_datetimes"],
                    initial_selected_image_indices=initial or None,
                    echem_thickness_series=self.state["echem_thickness_series"],
                    avg_thickness_series=self.state["nci_metric_series"],
                    plot_thickness=self.plot_thickness.value,
                    potential_plot_kwargs={**self._potential_plot_kwargs(), **self._nci_metric_plot_kwargs()},
                    side_text=operando_plots.format_event_context_preview_text(
                        self.state["event_context"],
                        self.state["img_datetimes"],
                    ),
                    snap_markers_to_tips=self.snap_markers_to_tips.value,
                    tip_snap_tolerance_seconds=self.tip_snap_tolerance_seconds.value,
                    display_func=self.display,
                )
            if selection is None:
                raise RuntimeError("Interactive selector did not return a selection.")
            self.state["interactive_selection"] = selection
            self.state["selector_output"] = selector_output
            print("Interactive selector launched. Use the widget output to edit selections.")
            print("When done, run Step 8. It will read the live selected_image_indices list.")

    def save_final_outputs(self):
        out = self.outputs["outputs"]
        with out:
            out.clear_output()
            self._check_stop("outputs")
            self._require_loaded()
            if "event_context" not in self.state:
                self.detect_events()
            derived = self._derived_paths()
            os.makedirs(derived["output_plot_dir"], exist_ok=True)
            selected = self._current_selected_indices()
            manual_cycles = self._parse_text_list(self.manual_cycle_numbers.value)
            self._archive_run_snapshot(derived)
            event_plot_kwargs = self._event_plot_kwargs()
            event_plot_kwargs["render_3d"] = False
            if manual_cycles:
                event_plot_kwargs["manual_cycle_numbers"] = manual_cycles
            metric_label, metric_unit = self._nci_metric_label_and_unit()
            outputs = operando_plots.save_selected_event_outputs(
                event_context=self.state["event_context"],
                selected_image_indices=selected,
                img_datetimes=self.state["img_datetimes"],
                img_series=self.state["img_series"],
                avg_thickness_series=self.state["nci_metric_series"],
                echem_datetimes=self.state["echem_datetimes"],
                potential_series=self.state["potential_series"],
                echem_thickness_series=self.state["echem_thickness_series"],
                output_plot_dir=derived["output_plot_dir"],
                exp_name=derived["exp_name"],
                roi_small=self.state["roi_small"],
                event_plot_kwargs=event_plot_kwargs,
                plot_images_func=plots_mod.plot_images,
                potential_plot_kwargs={**self._potential_plot_kwargs(), **self._nci_metric_plot_kwargs()},
                label_start=0,
                event_marker_size=6,
                event_label_fontsize=18,
                snap_plot_markers_to_tips=(
                    self.snap_markers_to_tips.value and len(self.state["echem_datetimes"]) > 0
                ),
                tip_snap_tolerance_seconds=self.tip_snap_tolerance_seconds.value,
                tip_snap_mode="upper",
                echem_cycle_series=self.state["echem_cycle_series"],
                metric_label=metric_label,
                metric_unit=metric_unit,
                save_potential=len(self.state["echem_datetimes"]) > 0,
            )
            self._check_stop("outputs")
            event_plot_kwargs_3d = self._event_plot_kwargs()
            event_plot_kwargs_3d["render_3d"] = True
            if manual_cycles:
                event_plot_kwargs_3d["manual_cycle_numbers"] = manual_cycles
            outputs_3d = operando_plots.save_selected_event_outputs(
                event_context=self.state["event_context"],
                selected_image_indices=selected,
                img_datetimes=self.state["img_datetimes"],
                img_series=self.state["img_series"],
                avg_thickness_series=self.state["nci_metric_series"],
                echem_datetimes=self.state["echem_datetimes"],
                potential_series=self.state["potential_series"],
                echem_thickness_series=self.state["echem_thickness_series"],
                output_plot_dir=derived["output_plot_dir"],
                exp_name=derived["exp_name"],
                roi_small=self.state["roi_small"],
                event_plot_kwargs=event_plot_kwargs_3d,
                plot_images_func=plots_mod.plot_images,
                potential_plot_kwargs=self._potential_plot_kwargs(),
                label_start=0,
                event_marker_size=6,
                event_label_fontsize=18,
                snap_plot_markers_to_tips=(
                    self.snap_markers_to_tips.value and len(self.state["echem_datetimes"]) > 0
                ),
                tip_snap_tolerance_seconds=self.tip_snap_tolerance_seconds.value,
                tip_snap_mode="upper",
                echem_cycle_series=self.state["echem_cycle_series"],
                metric_label=metric_label,
                metric_unit=metric_unit,
                filename_suffix="_3D",
                save_summary=False,
                save_potential=False,
            )
            outputs["saved_paths"]["images_3d"] = outputs_3d.get("saved_paths", {}).get("images")
            self.state["selected_outputs"] = outputs
            saved_paths = outputs.get("saved_paths", {})
            for key in ("potential", "images"):
                saved_path = saved_paths.get(key)
                if saved_path and Path(saved_path).exists():
                    self.display(self.Image(filename=str(saved_path), width=720))
            images_3d_path = saved_paths.get("images_3d")
            if images_3d_path:
                print(f"3D NCI image saved to: {images_3d_path}")
                if Path(images_3d_path).exists():
                    self.display(self.Image(filename=str(images_3d_path), width=720))

    def export_video(self):
        out = self.outputs["video"]
        with out:
            out.clear_output()
            self._check_stop("video")
            self._require_loaded()
            derived = self._derived_paths()
            self._archive_run_snapshot(derived)
            print("Make the synchronized potential plus image video.")
            calibrated = self._nci_is_thickness_calibrated()
            video_kwargs = {
                "echem_datetimes": self.state["echem_datetimes"],
                "potential_series": self.state["potential_series"],
                "img_datetimes": self.state["img_datetimes"],
                "img_series": self.state["img_series"],
                "echem_thickness_series": self.state["echem_thickness_series"],
                "avg_thickness_series": self.state["nci_metric_series"],
                "fps": self.fps.value,
                "dt_seconds": self.dt_seconds.value,
                "dpi": self.video_dpi.value,
                "vmin": self.video_vmin.value,
                "vmax": self.video_vmax.value,
                "cmap": self.video_cmap.value,
                "pixel_size": self._effective_pixel_size(),
                "scalebar_length": self.scalebar_length.value,
                "scale_label": self.scale_label.value,
                "colorbar_label": (
                    f"Li thickness ({operando_plots.MICRON_UNIT})"
                    if calibrated
                    else "NCI intensity (a.u.)"
                ),
                "cbar_label": (
                    f"Li thickness ({operando_plots.MICRON_UNIT})"
                    if calibrated
                    else "NCI intensity (a.u.)"
                ),
                "cbar_n_ticks": self.video_cbar_ticks.value,
                "scalebar_color": "white",
                "scalebar_position": "lower left",
                "scalebar_margin": 0.05,
                "plot_thickness": self.video_plot_thickness.value,
                "pause_gap_minutes": self.pause_gap_minutes.value,
                "thickness_traces": self.video_thickness_traces.value,
                "thickness_layout": self.video_thickness_layout.value,
                "overlay_offset_fraction": self.video_overlay_offset_fraction.value,
                "potential_ylim": self._ec_plot_ylim(),
                "surface_stride": self.surface_stride.value,
                "surface_elev": self.surface_elev.value,
                "surface_azim": self.surface_azim.value,
                "surface_zlim": (0, 3) if calibrated else None,
                "surface_zlabel": operando_plots.MICRON_UNIT if calibrated else "a.u.",
                "electrode_roi": self._roi_small(),
                "show_electrode_roi": True,
                **self._nci_metric_plot_kwargs(),
            }
            video_function = (
                operando_plots.make_potential_image_video_uniform_time
                if len(self.state["echem_datetimes"]) > 0
                else operando_plots.make_nci_image_video_uniform_time
            )
            if len(self.state["echem_datetimes"]) == 0:
                print("No electrochemistry data loaded; exporting an NCI-only video.")
            video_function(
                output_path=derived["video_output_path"],
                render_3d=False,
                **video_kwargs,
            )
            print(f"Video saved to: {derived['video_output_path']}")
            if Path(derived["video_output_path"]).exists():
                self.display(self.Video(derived["video_output_path"], embed=False, width=720))
            self._check_stop("video")
            video_path = Path(derived["video_output_path"])
            video_path_3d = video_path.with_name(f"{video_path.stem}_3D{video_path.suffix}")
            video_function(
                output_path=str(video_path_3d),
                render_3d=True,
                **video_kwargs,
            )
            print(f"3D video saved to: {video_path_3d}")
            if video_path_3d.exists():
                self.display(self.Video(str(video_path_3d), embed=False, width=720))
            self._check_stop("video")

    # ------------------------------------------------------------------
    # Common settings
    # ------------------------------------------------------------------
    def _roi_large(self):
        return {"r0": self.roi_large_r0.value, "x0": self.roi_large_x0.value, "y0": self.roi_large_y0.value}

    def _copy_notebook_archive(self, derived):
        archive_path = Path(derived["results_processing_dir"]) / "operando_pipeline.ipynb"
        notebook_path = self.campaign_dir / "operando_pipeline.ipynb"
        if notebook_path.exists():
            os.makedirs(archive_path.parent, exist_ok=True)
            try:
                if notebook_path.resolve() == archive_path.resolve():
                    print(f"Notebook already archived at: {archive_path}")
                    return
            except Exception:
                pass
            if archive_path.exists():
                print(f"Overwriting archived notebook: {archive_path}")
            shutil.copy2(notebook_path, archive_path)
            print(f"Notebook copied to: {archive_path}")
        else:
            print(f"Notebook copy skipped; not found: {notebook_path}")

    def _write_config_archive(self, derived):
        archive_dir = Path(derived["results_processing_dir"])
        os.makedirs(archive_dir, exist_ok=True)

        archive_path = archive_dir / "operando_config.ini"
        config = self._current_config_snapshot(absolute_paths=True)
        if archive_path.exists():
            print(f"Overwriting archived config: {archive_path}")
        with open(archive_path, "w", encoding="utf-8") as handle:
            config.write(handle)
        print(f"Runnable config copied to: {archive_path}")
        return archive_path

    def _archive_run_snapshot(self, derived):
        self._write_config_archive(derived)
        self._copy_notebook_archive(derived)

    def _roi_small(self):
        return {"r0": self.roi_small_r0.value, "x0": self.roi_small_x0.value, "y0": self.roi_small_y0.value}

    def _effective_pixel_size(self):
        return self.pixel_size.value * max(1, int(self.spatial_pixel_merge.value or 1))

    def _show_total_mass(self):
        return self.nci_metric.value == "total_mass" and self._nci_is_thickness_calibrated()

    def _nci_is_thickness_calibrated(self):
        """Return whether reference normalization produced thickness maps."""
        if "nci_is_thickness_calibrated" in self.state:
            return bool(self.state["nci_is_thickness_calibrated"])
        result = self.state.get("obint_result") or {}
        if "reference_corrected" in result:
            return bool(result["reference_corrected"])
        derived = self._derived_paths()
        stored = operando_data.read_stage_metadata(
            derived["metadata_path"],
            stage_name="obint_correction",
        )
        extra = (stored or {}).get("extra", {})
        return bool(extra.get("reference_correction_applied", False))

    def _nci_metric_label_and_unit(self):
        if not self._nci_is_thickness_calibrated():
            return "Mean NCI intensity", "a.u."
        if self._show_total_mass():
            return "Total Li mass", "mg"
        return "Avg thickness", operando_plots.MICRON_UNIT

    def _ec_plot_ylim(self):
        ymin = float(self.ec_plot_ymin.value)
        ymax = float(self.ec_plot_ymax.value)
        if ymin >= ymax:
            raise ValueError(f"EC y min must be smaller than EC y max: {ymin} >= {ymax}")
        return (ymin, ymax)

    def _nci_metric_plot_kwargs(self):
        if not self._nci_is_thickness_calibrated():
            return {
                "thickness_unit": "a.u.",
                "nci_trace_label": "Mean NCI intensity",
                "thickness_axis_label": "Mean NCI intensity (a.u.)",
            }
        if self._show_total_mass():
            return {
                "thickness_unit": "mg",
                "nci_trace_label": "NCI total Li mass",
                "thickness_axis_label": "Total Li mass (mg)",
            }
        return {
            "thickness_unit": operando_plots.MICRON_UNIT,
            "nci_trace_label": "NCI thickness",
            "thickness_axis_label": f"Li thickness ({operando_plots.MICRON_UNIT})",
        }

    def _potential_plot_kwargs(self):
        return {
            "potential_ylim": self._ec_plot_ylim(),
            "pause_gap_minutes": self.pause_gap_minutes.value,
            "plot_thickness": self.plot_thickness.value,
            "thickness_traces": self.thickness_traces.value,
            "thickness_layout": self.thickness_layout.value,
            "overlay_offset_fraction": self.overlay_offset_fraction.value,
            "potential_color": "tab:blue",
            "ec_thickness_color": "tab:green",
            "nci_thickness_color": "tab:orange",
            "thickness_unit": operando_plots.MICRON_UNIT,
        }

    def _event_plot_kwargs(self):
        calibrated = self._nci_is_thickness_calibrated()
        _, metric_unit = self._nci_metric_label_and_unit()
        return {
            "cmap": self.plot_cmap.value,
            "pixel_size": self._effective_pixel_size(),
            "scalebar_length": self.scalebar_length.value,
            "scale_label": self.scale_label.value,
            "cbar_n_ticks": self.plot_cbar_ticks.value,
            "scale_fontsize": 16,
            "scalebar_width_frac": 0.02,
            "scalebar_y_frac": 0.16,
            "show_labels": True,
            "cbar_ranges": [(self.plot_vmin.value, self.plot_vmax.value)],
            "cbar_label": "Li thickness" if calibrated else "NCI intensity",
            "cbar_shared_length": 0.6,
            "cbar_mode": self.plot_cbar_mode.value,
            "cbar_fontsize": 16,
            "cbar_unit": [operando_plots.MICRON_UNIT if calibrated else "a.u."],
            "cbar_width_frac": 0.015,
            "cbar_show_ticklines": True,
            "cbar_pad": 0.008,
            "scalebar_mode": "inside",
            "spacing": (0.006, 0.055),
            "figsize": None,
            "render_3d": False,
            "surface_stride": self.surface_stride.value,
            "surface_elev": self.surface_elev.value,
            "surface_azim": self.surface_azim.value,
            "surface_zlim": (0, 3) if calibrated else None,
            "surface_zlabel": operando_plots.MICRON_UNIT if calibrated else "a.u.",
            "electrode_roi": self._roi_small(),
            "show_electrode_roi": True,
            # Automatically generated cycle labels are meaningful only when
            # electrochemistry is loaded. Manual cycle labels remain available.
            "show_cycle_numbers": (
                self.show_cycle_numbers.value
                and len(self.state.get("echem_datetimes", [])) > 0
            ),
            "show_nci_thickness": self.show_nci_thickness.value,
            "nci_thickness_unit": metric_unit,
            "nci_thickness_decimals": 2,
            "image_annotation_fontsize": 12,
        }

    def _require_loaded(self):
        if "img_datetimes" not in self.state:
            raise RuntimeError("Run Step 6: Load data preview first.")

    def _parse_int_list(self, text):
        if isinstance(text, (list, tuple, np.ndarray)):
            return [int(value) for value in text]
        values = []
        for part in str(text or "").replace(";", ",").split(","):
            part = part.strip()
            if not part:
                continue
            values.append(int(part))
        return values

    def _parse_text_list(self, text):
        return [part.strip() for part in str(text or "").replace(";", ",").split(",") if part.strip()]

    def _sanitize_selected_indices(self, indices, source_label, *, update_manual_field=False):
        valid_count = len(self.state.get("img_datetimes", []))
        if valid_count <= 0:
            return list(indices)
        cleaned = []
        dropped = []
        for value in indices:
            idx = int(value)
            if 0 <= idx < valid_count:
                if idx not in cleaned:
                    cleaned.append(idx)
            else:
                dropped.append(idx)
        if dropped:
            print(
                f"Ignoring out-of-range {source_label} image indices {dropped}. "
                f"Valid image indices for this experiment are 0 to {valid_count - 1}."
            )
        cleaned = sorted(cleaned, key=lambda idx: self.state["img_datetimes"][idx])
        if update_manual_field:
            self.manual_selected_indices.value = ", ".join(str(i) for i in cleaned)
        return cleaned

    def _build_event_context(self):
        self._require_loaded()
        return operando_plots.build_turning_point_event_context(
            echem_datetimes=self.state["echem_datetimes"],
            echem_thickness_series=self.state["echem_thickness_series"],
            img_datetimes=self.state["img_datetimes"],
            event_finder=operando_data.find_capacity_turning_points,
            image_index_finder=operando_data.find_nearest_image_indices,
            conv_thickness=self.conv_thickness.value,
            n_tpx_per_image=self.n_tpx_per_image.value,
            tpx_time=self.tpx_time.value,
            initial_ocv_image_count=(
                self.background_n_images.value if self.background_correction.value else None
            ),
        )

    def _load_previous_selected_indices(self):
        summary_path = self._derived_paths()["selected_summary_path"]
        if not os.path.isfile(summary_path):
            return None
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if "Selected image indices:" not in line:
                        continue
                    indices = ast.literal_eval(line.split("Selected image indices:", 1)[1].strip())
                    return self._parse_int_list(indices)
        except Exception as exc:
            print(f"Could not load previous selected image indices from {summary_path}: {exc}")
        return None

    def _initial_selector_indices(self):
        manual = self._parse_int_list(self.manual_selected_indices.value)
        if manual:
            manual = self._sanitize_selected_indices(
                manual,
                "manually entered",
                update_manual_field=True,
            )
            if manual:
                print(f"Using manually entered selected image indices: {manual}")
                return manual
        existing_selection = self.state.get("interactive_selection")
        if existing_selection and "selected_image_indices" in existing_selection:
            live_indices = self._sanitize_selected_indices(
                list(existing_selection["selected_image_indices"]),
                "live selected",
            )
            if live_indices:
                print(f"Using current live selected image indices: {live_indices}")
                return live_indices
        previous = self._load_previous_selected_indices()
        if previous:
            previous = self._sanitize_selected_indices(previous, "previously saved")
        if previous:
            print(f"Loaded previous selected image indices: {previous}")
            return previous
        if len(self.state.get("echem_datetimes", [])) == 0:
            nci_indices, _ = self._auto_preview_image_indices(count=4)
            print(f"Using NCI-only representative image indices: {nci_indices}")
            return nci_indices
        auto_indices = list(self.state.get("event_context", {}).get("img_indices", []))
        auto_indices = self._sanitize_selected_indices(auto_indices, "auto-detected")
        print(f"Using auto-detected selected image indices: {auto_indices}")
        return auto_indices

    def _enable_widget_backend_for_selector(self):
        try:
            import matplotlib
            import ipympl  # noqa: F401
            from IPython import get_ipython
            import matplotlib.pyplot as plt

            ipython = get_ipython()
            if ipython is not None:
                ipython.run_line_magic("matplotlib", "widget")
            matplotlib.use("module://ipympl.backend_nbagg", force=True)
            plt.switch_backend("module://ipympl.backend_nbagg")
            plt.close("all")
        except Exception as exc:
            print(f"Could not enable the interactive matplotlib widget backend automatically: {exc}")

    def _current_selected_indices(self):
        selection = self.state.get("interactive_selection")
        if selection and "selected_image_indices" in selection:
            indices = self._sanitize_selected_indices(
                list(selection["selected_image_indices"]),
                "live selected",
            )
            self.manual_selected_indices.value = ", ".join(str(i) for i in indices)
            return indices
        manual = self._parse_int_list(self.manual_selected_indices.value)
        if manual:
            manual = self._sanitize_selected_indices(
                manual,
                "manually entered",
                update_manual_field=True,
            )
            return manual
        detected = self._sanitize_selected_indices(
            list(self.state.get("event_context", {}).get("img_indices", [])),
            "auto-detected",
        )
        if detected:
            return detected
        nci_indices, _ = self._auto_preview_image_indices(count=4)
        return self._sanitize_selected_indices(nci_indices, "NCI-only representative")


def launch_operando_gui(campaign_dir=None, history_path=None):
    """Create and display the notebook-native operando GUI."""
    try:
        from IPython import get_ipython
        from IPython.display import clear_output

        ipython = get_ipython()
        if ipython is not None:
            old_gui = ipython.user_ns.get("operando_app")
            if old_gui is not None and hasattr(old_gui, "close_gui"):
                old_gui.close_gui()

        clear_output(wait=True)
    except Exception:
        pass
    gui = OperandoWorkflowGUI(campaign_dir=campaign_dir, history_path=history_path)
    return gui.display_gui()

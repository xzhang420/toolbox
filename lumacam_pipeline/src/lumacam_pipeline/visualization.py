"""Analysis and rendering of read-only image datasets in independent profiles."""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid

import numpy as np

from .config import load_config, save_config
from .handover import atomic_json, file_digest, load_processed_images
from .image_io import parse_timestamp
from .event_selection import selected_records
from .presets import preset_config
from .section import PipelineSession, PipelineSection, SectionOutcome, SectionState
from .sections.images import SelectionSection
from .measurements import SpatialMeasurementsSection
from .time_window import TimeWindowSection, image_indices
from .csv_export import CsvExportSection
from .inhomogeneity import InhomogeneitySection
from .quality import ImageQualitySection
from .cycles import CycleSummarySection
from .sections.electrochemistry import ElectrochemistrySection, SynchronizationSection
from .sections.events import EventDetectionSection
from .sections.nci import PlatingMetricsSection
from .sections.outputs import CombinedPlotSection, CombinedVideoSection, NciEventOutputsSection, _category_dir
from .sections.tof import TimeOfFlightCubeSection

TIMED_SECTIONS = {"synchronization", "event_detection", "combined_plot", "combined_video", "nci_event_outputs"}


class TimedElectrochemistrySection(ElectrochemistrySection):
    def run(self, session, config):
        outcome = super().run(session, config)
        times = [parse_timestamp(time, config.get("timestamp_timezone", "")) for time in outcome.outputs["echem_times"]]
        if any(time is None for time in times):
            raise ValueError("Electrochemistry timestamps need an explicit timestamp_timezone when no UTC offset is present.")
        outcome.outputs["echem_times"] = times
        return outcome


def _profile_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", str(name)):
        raise ValueError("Profile name must contain letters, digits, '.', '_' or '-' and start with a letter/digit.")
    return str(name)


def _limits(images, config):
    low, high = config.get("vmin"), config.get("vmax")
    if low not in (None, "") or high not in (None, ""):
        if low in (None, "") or high in (None, "") or not np.isfinite([float(low), float(high)]).all() or float(low) >= float(high):
            raise ValueError("Supply both finite colour limits with vmin < vmax, or leave both blank.")
        return float(low), float(high)
    finite = [np.asarray(image)[np.isfinite(image)] for image in images]
    finite = [values for values in finite if values.size]
    if not finite:
        return 0., 1.
    low, high = min(float(a.min()) for a in finite), max(float(a.max()) for a in finite)
    return low, high if high > low else low + 1.


def _display_image(image, config):
    roi = config.get("display_roi") or {}
    if not roi:
        return image
    x, y, radius = (int(roi[key]) for key in ("x0", "y0", "r0"))
    cropped = image[max(0, y-radius):y+radius, max(0, x-radius):x+radius]
    if radius <= 0 or not cropped.size:
        raise ValueError("Display crop must have positive radius and overlap the image.")
    return cropped


def _frame_title(session, index, clean):
    times = session.artifacts.require("image_times")
    stamp = times[index]
    if clean and stamp is not None:
        label=f"{(stamp-times[0]).total_seconds()/3600:.2f} h"
        if not np.isfinite(np.asarray(session.artifacts.require('image_stack')[index])).any(): label+=' | invalid image'
        return label
    return f"Frame {index}" + (f" | {stamp.isoformat()}" if stamp is not None else "")


def _draw_frame(axis, image, session, config, index, limits):
    image = _display_image(image, config)
    if config.get("render_3d", False):
        return _draw_surface(axis, image, session, config, index, limits)
    artist = axis.imshow(image, origin=config.get("image_origin", "upper"), cmap=config.get("cmap", "inferno"),
                         vmin=limits[0], vmax=limits[1], interpolation="nearest")
    title = _frame_title(session, index, config.get("clean_layout", True))
    axis.set_title(title, fontsize=9)
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    if config.get("clean_layout", True):
        axis.set_axis_off()
    length = float(config.get("scalebar_length_um") or 0)
    size = session.artifacts.get("pixel_size_um")
    if length:
        if size is None:
            raise ValueError("Set a known pixel size before drawing a physical scale bar.")
        width = length / float(size)
        height, image_width = np.shape(image)
        if width > .85 * image_width:
            raise ValueError("Scale bar exceeds image width; choose a shorter bar.")
        x, y = .08 * image_width, .9 * height
        axis.plot([x, x + width], [y, y], color="white", linewidth=3)
        axis.text(x, y - .04 * height, str(config.get("scalebar_label") or f"{length:g} μm"), color="white")
    return artist


def _draw_surface(axis, image, session, config, index, limits):
    """Height is measured pixel value, not inferred electrode topography."""
    if min(image.shape) < 2:
        raise ValueError("3D rendering requires at least two rows and columns after cropping.")
    stride = int(config.get("surface_stride", 2))
    if stride < 1:
        raise ValueError("Surface stride must be a positive integer.")
    size = session.artifacts.get("pixel_size_um")
    height, width = image.shape
    ix = np.unique(np.r_[np.arange(0, width, stride), width-1])
    iy = np.unique(np.r_[np.arange(0, height, stride), height-1])
    xx, yy = np.meshgrid(ix * (size or 1), iy * (size or 1))
    artist = axis.plot_surface(xx, yy, np.asarray(image)[np.ix_(iy, ix)],
        cmap=config.get("cmap", "inferno"), vmin=limits[0], vmax=limits[1],
        rcount=len(iy), ccount=len(ix), linewidth=0, antialiased=False)
    zlim = config.get("surface_zlim") or None
    zlim = _limits([], {"vmin": zlim[0], "vmax": zlim[1]}) if zlim is not None else limits
    axis.set_zlim(*zlim)
    axis.set_xlim(0, (width-1)*(size or 1))
    ymax = (height-1)*(size or 1)
    # Explicit 3D limits avoid older Matplotlib invert_yaxis alias behaviour.
    axis.set_ylim((ymax, 0) if config.get("image_origin", "upper") == "upper" else (0, ymax))
    axis.view_init(elev=float(config.get("surface_elev", 35)), azim=float(config.get("surface_azim", -60)))
    axis.set_xlabel("x (µm)" if size else "x (pixel)")
    axis.set_ylabel("y (µm)" if size else "y (pixel)")
    axis.set_zlabel(str(config.get("surface_zlabel") or session.artifacts.get("image_unit") or "Image value (a.u.)"))
    from matplotlib.ticker import MaxNLocator
    axis.xaxis.set_major_locator(MaxNLocator(4))
    axis.yaxis.set_major_locator(MaxNLocator(4))
    axis.zaxis.set_major_locator(MaxNLocator(4))
    axis.set_title(_frame_title(session, index, config.get("clean_layout", True)), fontsize=9)
    if not np.isfinite(image).any():
        axis.text2D(.5, .5, "Invalid image", ha="center", transform=axis.transAxes)
    return artist


class FrameFigureSection(PipelineSection):
    id = "image_plot"
    title = "Image figures"
    requires = ("image_stack", "selected_image_indices", "results_dir", "image_times")
    optional_inputs = ("event_context", "image_metric_values", "view_window")
    produces = ("image_plot_path",)

    def output_stem(self, session):
        return "images"

    def run(self, session, config):
        import matplotlib.pyplot as plt
        images = session.artifacts.require("image_stack")
        indices = image_indices(session, session.artifacts.require("selected_image_indices"), "panels")
        if not indices:
            raise ValueError("No selected images in this window. Choose frame indices inside it, or reset the window.")
        if config.get("clean_layout", True) and not config.get("render_3d", False):
            return self.clean_panel(session, config, images, indices)
        columns = max(1, min(int(config.get("columns") or 2), len(indices)))
        rows = (len(indices) + columns - 1) // columns
        surface = bool(config.get("render_3d", False))
        figure, axes = plt.subplots(rows, columns, squeeze=False,
                                    subplot_kw={"projection": "3d"} if surface else {},
                                    figsize=tuple(config.get("figsize", (5 * columns, 4.5 * rows) if surface else (4 * columns, 4 * rows))),
                                    constrained_layout=not surface)
        if surface:
            figure.subplots_adjust(left=.02, right=.83, bottom=.1, top=.9, wspace=.35, hspace=.25)
        try:
            limits = _limits(images, config)
            for j, (axis, index) in enumerate(zip(axes.flat, indices)):
                style = config if j == 0 else {**config, "scalebar_length_um": 0}
                artist = _draw_frame(axis, images[index], session, style, index, limits)
            for axis in list(axes.flat)[len(indices):]:
                axis.set_visible(False)
            label = str(config.get("colorbar_label") or session.artifacts.get("image_unit") or "Image intensity (a.u.)")
            if surface:
                figure.colorbar(artist, cax=figure.add_axes([.9, .2, .02, .6]), label=label)
            else:
                figure.colorbar(artist, ax=list(axes.flat)[:len(indices)], label=label)
            extension = str(config.get("format", "png")).lower()
            if extension not in {"png", "pdf", "svg", "tiff"}:
                raise ValueError("Figure format must be png, pdf, svg or tiff.")
            suffix = "_3d" if config.get("render_3d", False) else ""
            path = _category_dir(session, "figures") / f"{self.output_stem(session)}{suffix}.{extension}"
            figure.savefig(path, dpi=int(config.get("dpi", 150)))
        finally:
            plt.close(figure)
        return SectionOutcome(self.id, SectionState.COMPLETED, f"Saved {path}", {"image_plot_path": path})

    def clean_panel(self, session, config, images, indices):
        from . import backends
        import matplotlib.pyplot as plt
        records = selected_records(session, category="panels")
        limits = _limits(images, config)
        crop = config.get("display_roi") or {}
        if crop and (set(crop) != {"x0", "y0", "r0"} or float(crop["r0"]) <= 0):
            raise ValueError("display_roi needs x0, y0 and a positive r0 (pixels); it changes display only.")
        crop = {key: int(value) for key, value in crop.items()}
        size = session.artifacts.get("pixel_size_um")
        length = float(config.get("scalebar_length_um") or 0)
        if length and not size:
            raise ValueError("Set a known pixel size before drawing a physical scale bar.")
        image_width = np.shape(images[0])[1]
        width = min(image_width, crop.get("x0", 0) + crop["r0"]) - max(0, crop["x0"]-crop["r0"]) if crop else image_width
        if width <= 0 or (length and length / size > .85 * width):
            raise ValueError("Display crop is empty or the scale bar exceeds its width.")
        columns = int(config.get("columns") or (2 if len(indices) <= 4 else 3 if len(indices) <= 9 else 4))
        columns = max(1, min(columns, len(indices)))
        rows = (len(indices)+columns-1)//columns
        means = session.artifacts.get("image_metric_values")
        unit = session.artifacts.get("image_unit") or "a.u."
        unit = "µm" if unit in {"um", "μm", "µm"} else unit
        fig, _ = backends.image_plots().plot_images(
            [images[i] for i in indices], list(range(len(indices))), pixel_size=size or 1,
            **crop, cbar_ranges=[limits] * len(indices), cmap=config.get("cmap", "inferno"),
            cbar_mode="shared right", cbar_label=config.get("colorbar_label") or f"Image value ({unit})",
            cbar_fontname="DejaVu Sans", cbar_fontsize=10, cbar_n_ticks=5,
            scalebar_mode="inside" if length else "none", scalebar_length=length,
            scale_label=config.get("scalebar_label") or f"{length:g} µm", scale_fontname="DejaVu Sans",
            scale_fontsize=11, scalebar_y_frac=.17, label_fontname="DejaVu Sans", label_fontsize=18,
            lower_left_texts=[row["caption"] for row in records],
            panel_labels=[row["label"] for row in records],
            lower_right_texts=([f"{means[i]:.2f} {unit}" for i in indices] if means is not None else None),
            annotation_fontname="DejaVu Sans", annotation_fontsize=10,
            annotation_bbox={"facecolor": "black", "alpha": .5, "edgecolor": "none", "pad": 2},
            ncols=columns, nrows=rows, figsize=tuple(config.get("figsize") or (3.2*columns+1.1, 3.2*rows+.4)),
            spacing=(.04, .04), image_origin=str(config.get("image_origin", "upper")))
        extension = str(config.get("format", "png")).lower()
        try:
            if extension not in {"png", "pdf", "svg", "tiff"}:
                raise ValueError("Figure format must be png, pdf, svg or tiff.")
            path = _category_dir(session, "figures") / f"{self.output_stem(session)}.{extension}"
            fig.savefig(path, dpi=int(config.get("dpi", 180)), bbox_inches="tight")
        finally:
            plt.close(fig)
        return SectionOutcome(self.id, SectionState.COMPLETED, f"Saved {path}", {"image_plot_path": path})


class FrameIndividualSection(FrameFigureSection):
    id = "individual_image_plots"
    title = "Individual image figures"
    requires = ("image_stack", "results_dir", "image_times")
    produces = ("individual_image_plot_paths",)

    def output_stem(self, session):
        index = session.artifacts.require("selected_image_indices")[0]
        return f"frame_{index:05d}_{Path(session.artifacts.require('image_paths')[index]).stem}"

    def run(self, session, config):
        images = session.artifacts.require("image_stack")
        raw = config.get("indices") or []
        if isinstance(raw, str):
            raw = [i.strip() for i in raw.split(",") if i.strip()]
        indices = [int(i) for i in raw] or list(range(len(images)))
        if any(index < 0 or index >= len(images) for index in indices):
            raise ValueError("Individual image index out of range.")
        indices = image_indices(session, indices, "panels")
        if not indices:
            raise ValueError("No individual images in the selected time window.")
        previous = session.artifacts.get("selected_image_indices")
        scale = config.get("color_scale", "global")
        if scale not in {"global", "individual"}:
            raise ValueError("Individual color_scale must be global or individual.")
        global_limits = _limits(images, config)
        paths = []
        try:
            for index in indices:
                session.artifacts.put("selected_image_indices", [index], "selection")
                low, high = _limits([images[index]], config) if scale == "individual" else global_limits
                outcome = super().run(session, {**config, "vmin": low, "vmax": high})
                paths.append(outcome.outputs["image_plot_path"])
        finally:
            if previous is not None:
                session.artifacts.put("selected_image_indices", previous, "selection")
            else:
                session.artifacts.discard_by_producer("selection")
        return SectionOutcome(self.id, SectionState.COMPLETED, f"Saved {len(paths)} individual figures.",
                              {"individual_image_plot_paths": paths})


class FrameVideoSection(PipelineSection):
    id = "image_video"
    title = "Image video"
    requires = ("image_stack", "image_times", "results_dir")
    optional_inputs = ("view_window",)
    produces = ("image_video_path", "image_video_timing_path", "image_video_timing_settings_path")

    def run(self, session, config):
        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from .video_timing import video_plan
        import pandas as pd
        if not FFMpegWriter.isAvailable():
            raise RuntimeError("Video export requires an FFmpeg executable on PATH; run the doctor command.")
        plan, resolved = video_plan(session, config)
        fps = resolved["fps"]
        images = session.artifacts.require("image_stack")
        limits = _limits(images, config)
        surface = bool(config.get("render_3d", False))
        figure, axis = plt.subplots(figsize=tuple(config.get("figsize", (6, 6))), constrained_layout=True,
                                   subplot_kw={"projection": "3d"} if surface else {})
        stem = "images_3d" if surface else "images"
        path = _category_dir(session, "videos") / f"{stem}.mp4"
        try:
            figure.colorbar(ScalarMappable(norm=Normalize(*limits), cmap=config.get("cmap", "inferno")),
                ax=axis, label=str(config.get("colorbar_label") or session.artifacts.get("image_unit") or "Image intensity (a.u.)"))
            writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
            with writer.saving(figure, str(path), dpi=int(config.get("dpi", 100))):
                for row in plan:
                    axis.clear()
                    index = row["image_index"]
                    if index is None:
                        axis.set_axis_off()
                        text = axis.text2D if surface else axis.text
                        text(.5, .5, "No acquisition data", ha="center", va="center", transform=axis.transAxes, fontsize=16)
                    else:
                        axis.set_axis_on()
                        _draw_frame(axis, images[index], session, config, index, limits)
                    if resolved["timing_mode"] == "elapsed_time":
                        title = f"Elapsed {row['elapsed_seconds']/3600:.3f} h"
                        if index is not None:
                            times = session.artifacts.require("image_times")
                            acquired = (times[index]-times[0]).total_seconds()/3600
                            title += f" | acquired {acquired:.3f} h"
                        axis.set_title(title, fontsize=9)
                    writer.grab_frame()
        finally:
            plt.close(figure)
        table = _category_dir(session, "tables") / f"{stem}_video_timing.csv"
        pd.DataFrame(plan).to_csv(table, index=False)
        settings = table.with_suffix(".json")
        atomic_json(settings, resolved)
        message = f"Saved {path}; {resolved['timing_mode']}, {len(plan)} frames, {resolved['gap_frame_count']} gap frames."
        return SectionOutcome(self.id, SectionState.COMPLETED, message,
                              {"image_video_path": path, "image_video_timing_path": table,
                               "image_video_timing_settings_path": settings})


class VisualizationSession(PipelineSession):
    """Analysis-only capabilities; no correction or processing-load section exists."""

    def configure(self, section_id, **values):
        if section_id == 'reports':
            self.config.setdefault('reports',{}).update(values)
            return
        return super().configure(section_id,**values)

    def run(self, section_id):
        if section_id not in self.sections:
            raise KeyError(f"Unknown analysis section: {section_id}")
        if section_id == 'inhomogeneity' and not self.artifacts.has('frame_quality_table'):
            quality = self.run('quality')
            if quality.state != SectionState.COMPLETED:
                return SectionOutcome(section_id, SectionState.FAILED, 'Image-quality check failed: '+quality.message)
        if section_id == "nci_plating_metrics":
            if self.artifacts.get("pixel_size_um") is None:
                outcome = SectionOutcome(section_id, SectionState.FAILED, "NCI physical metrics require an explicit pixel size.")
                self.outcomes[section_id] = outcome
                return outcome
            self.section_config(section_id)["pixel_size_um"] = self.artifacts.get("pixel_size_um")
        if section_id in TIMED_SECTIONS and not self.artifacts.get("timing_available", False):
            outcome = SectionOutcome(section_id, SectionState.FAILED,
                                     "This action needs real acquisition timestamps. Supply timing metadata or explicit start/interval/timezone.")
            self.outcomes[section_id] = outcome
            return outcome
        if section_id in {"synchronization", "combined_plot", "combined_video"}:
            image_times = self.artifacts.get("image_times", [])
            echem_times = self.artifacts.get("echem_times", [])
            if image_times and echem_times and (max(image_times) < min(echem_times) or max(echem_times) < min(image_times)):
                outcome = SectionOutcome(section_id, SectionState.FAILED,
                                         "Image and electrochemistry dates do not overlap. Check the selected file and timezones.")
                self.outcomes[section_id] = outcome
                return outcome
        try:
            save_visualization_settings(self)
        except (ValueError, OSError) as exc:
            return SectionOutcome(section_id, SectionState.FAILED, str(exc))
        self.artifacts.put("export_id", datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%f") + "_" + uuid.uuid4().hex[:6], "profile")
        return super().run(section_id)


class ProfileEventOutputsSection(NciEventOutputsSection):
    """Use the same clean panel and marked curve as the separate GUI actions."""
    optional_inputs = NciEventOutputsSection.optional_inputs + ("view_window", "image_metrics_table")
    def run(self, session, config):
        style = {**session.section_config("image_plot"), **config}
        panel = FrameFigureSection().run(session, style)
        paths = {"images": panel.outputs["image_plot_path"]}
        if session.artifacts.get("echem_times"):
            curve = CombinedPlotSection().run(session, {**session.section_config("combined_plot"), **config})
            paths["potential"] = curve.outputs["combined_plot_path"]
        return SectionOutcome(self.id, SectionState.COMPLETED, "Saved linked NCI panel and electrochemistry figure.",
                              {"nci_event_output_paths": paths})


def open_visualization(source, *, profile="overview", output_root=None, preset="exsitu", config=None, cube=False):
    values = preset_config(preset)
    for section, options in (config or {}).items():
        values.setdefault(section, {}).update(deepcopy(options))
    values["workflow"] = {"role": "visualization"}
    # The old common preset's 60-second video interval is not a dataset fact.
    video = (config or {}).get("image_video", {})
    values["image_video"].update(timing_mode=video.get("timing_mode", "image_sequence"),
        dt_seconds=video.get("dt_seconds"), image_tolerance_seconds=video.get("image_tolerance_seconds"))
    values["synchronization"].setdefault("allow_extrapolation", False)
    values["combined_video"].setdefault("limit_to_overlap", True)
    values["combined_video"].setdefault("image_origin", "upper")
    inputs = values.setdefault("input", {})
    inputs["path"] = str(Path(source).expanduser().resolve())
    if cube:
        source_path = Path(source).expanduser().resolve()
        from .sections.tof import _load_cube
        _load_cube(source_path, str(values.get("tof_cube", {}).get("npz_key", "cube")))
        dataset = {"root": source_path.parent, "paths": [source_path], "images": [], "times": [],
                   "experiment": source_path.stem, "unit": "", "pixel_size_um": None,
                   "nci_calibrated": False, "timing_available": False,
                   "dataset_id": file_digest(source_path), "records": [{"path": source_path.name, "sha256": file_digest(source_path)}]}
        values.setdefault("tof_cube", {}).update(path=str(source_path), enabled=True)
        inputs["cube"] = True
    else:
        options = {key: inputs[key] for key in ("pattern", "fits_hdu", "timestamp_timezone", "pixel_size_um",
                                               "unit", "start_time", "interval_seconds") if key in inputs}
        dataset = load_processed_images(source, **options)
        inputs["cube"] = False
    profile = _profile_name(profile)
    base = Path(output_root).expanduser().resolve() if output_root else dataset["root"] / "visualization"
    target = (base / profile).resolve()
    if any(path.resolve().is_relative_to(target) for path in dataset["paths"]):
        raise ValueError("Visualization output cannot contain source images; choose a separate output folder.")
    identity = target / "dataset.json"
    if identity.exists() and json.loads(identity.read_text()).get("dataset_id") != dataset["dataset_id"]:
        raise ValueError("This profile belongs to a different/changed dataset. Choose another profile name.")
    values.setdefault("visualization", {}).update(profile=profile, output_root=str(base), dataset_id=dataset["dataset_id"])
    if values["visualization"].get("export_layout", "versioned") not in {"flat", "versioned"}:
        raise ValueError("visualization.export_layout must be flat or versioned.")
    # Never inherit assumed 55-µm scales or enable science calibration from a
    # generic preset. Explicit input metadata is the authority.
    for section in ("image_plot", "individual_image_plots", "image_video", "combined_video", "nci_event_outputs"):
        explicit = (config or {}).get(section, {})
        values[section].update({"enabled": True, "scalebar_length_um": explicit.get("scalebar_length_um", 0),
                                "scalebar_label": explicit.get("scalebar_label", ""),
                                "pixel_size_um": dataset["pixel_size_um"] or 1})
    values["image_measurements"]["calibrated"] = dataset["nci_calibrated"]
    values["image_measurements"]["label"] = ((config or {}).get("image_measurements", {}).get("label") or
        "Mean image intensity" + (f" ({dataset['unit']})" if dataset["unit"] else " (a.u.)"))
    # A generic TIFF has no known thickness calibration. Do not put arbitrary
    # image counts and electrochemical micrometres on one shared value axis.
    micrometres = dataset["unit"].strip().casefold() in {"um", "μm", "µm"}
    for section in ("combined_plot", "combined_video"):
        if "thickness_traces" not in (config or {}).get(section, {}):
            values[section]["thickness_traces"] = "both" if micrometres or dataset["nci_calibrated"] else "nci"
    classes = [SpatialMeasurementsSection, ImageQualitySection, InhomogeneitySection, TimeWindowSection, CsvExportSection, SelectionSection, TimedElectrochemistrySection, SynchronizationSection,
               EventDetectionSection, CycleSummarySection, PlatingMetricsSection, FrameFigureSection, FrameIndividualSection,
               FrameVideoSection, CombinedPlotSection, CombinedVideoSection, ProfileEventOutputsSection, TimeOfFlightCubeSection]
    sections = [cls() for cls in classes]
    allowed = {section.id for section in sections} | {"workflow", "input", "visualization", "reports"}
    values = {key: value for key, value in values.items() if key in allowed}
    session = VisualizationSession(sections, values, preset=preset)
    artifacts = {"results_dir": target, "experiment": dataset["experiment"],
                 "image_unit": dataset["unit"], "pixel_size_um": dataset["pixel_size_um"],
                 "nci_calibrated": dataset["nci_calibrated"], "timing_available": dataset["timing_available"],
                 "source_dataset": dataset}
    if not cube:
        artifacts.update(image_stack=dataset["images"], image_paths=[str(p) for p in dataset["paths"]], image_times=dataset["times"],
                         image_exposure_bounds=dataset.get('exposure_bounds'))
    for name, value in artifacts.items():
        session.artifacts.put(name, value, "dataset")
    return session


def save_visualization_settings(session):
    root = Path(session.artifacts.require("results_dir"))
    dataset = session.artifacts.require("source_dataset")
    marker = root / "dataset.json"
    if marker.exists() and json.loads(marker.read_text()).get("dataset_id") != dataset["dataset_id"]:
        raise ValueError("Dataset changed; use a new visualization profile.")
    values = deepcopy(session.config)
    # Relative inputs survive copying an experiment including its profiles.
    import os
    for section, key in (("input", "path"), ("electrochemistry", "path"), ("tof_cube", "path")):
        value = values.get(section, {}).get(key)
        if value:
            try:
                values[section][key] = Path(os.path.relpath(Path(value).resolve(), root)).as_posix()
            except ValueError:  # Different Windows drive: retain explicit absolute path.
                pass
    values["visualization"]["output_root"] = ".."
    root.mkdir(parents=True, exist_ok=True)
    save_config(root / "settings.ini", session.preset, values)
    atomic_json(marker, {"dataset_id": dataset["dataset_id"], "inputs": dataset["records"]})
    return root / "settings.ini"


def load_visualization_settings(path):
    path = Path(path).expanduser().resolve()
    preset, values = load_config(path)
    if values.get("workflow", {}).get("role") != "visualization":
        raise ValueError("Select visualization settings.ini, not the proposal processing config.")
    for section, key in (("input", "path"), ("electrochemistry", "path"), ("tof_cube", "path"), ("visualization", "output_root")):
        value = values.get(section, {}).get(key)
        if value and not Path(value).expanduser().is_absolute():
            values[section][key] = str((path.parent / value).resolve())
    return open_visualization(values["input"]["path"], preset=preset, config=values,
                              profile=values["visualization"]["profile"],
                              output_root=values["visualization"]["output_root"], cube=values["input"].get("cube", False))


def record_visualization(session, section):
    root = Path(session.artifacts.require("results_dir"))
    run_id = session.artifacts.require("export_id")
    record_dir = root / "records" / run_id
    record_dir.mkdir(parents=True, exist_ok=True)
    settings = save_visualization_settings(session)
    # Keep the per-export settings relative to their own location.
    preset, values = load_config(settings)
    for name, key in (("input", "path"), ("electrochemistry", "path"), ("tof_cube", "path"), ("visualization", "output_root")):
        value = values.get(name, {}).get(key)
        if value and not Path(value).is_absolute():
            values[name][key] = (Path("../..") / value).as_posix()
    save_config(record_dir / "settings.ini", preset, values)
    table_paths = []
    if section == "image_measurements":
        table = session.artifacts.require("image_metrics_table")
        target = _category_dir(session, "tables") / "image_measurements.csv"
        table.to_csv(target, index=False)
        table_paths.append(target)
    elif section == "synchronization":
        target = _category_dir(session, "tables") / "synchronized.csv"
        session.artifacts.require("synchronized_table").to_csv(target, index=False)
        table_paths.append(target)
    elif section == "event_detection":
        target = _category_dir(session, "tables") / "detected_events.json"
        # Datetimes remain explicit ISO strings, including UTC offsets.
        target.write_text(json.dumps(session.artifacts.require("event_context"), indent=2, default=str), encoding="utf-8")
        table_paths.append(target)
    if section in {"selection", "image_plot", "combined_plot", "nci_event_outputs"}:
        import pandas as pd
        target = _category_dir(session, "tables") / f"{section}_points.csv"
        category = {"image_plot": "panels", "combined_plot": "figures", "nci_event_outputs": "panels"}.get(section)
        pd.DataFrame(selected_records(session, category=category)).to_csv(target, index=False)
        table_paths.append(target)
    outputs = []
    for category in ("figures", "videos", "tables"):
        directory = root / category / run_id
        if directory.exists():
            outputs.extend({"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
                           for path in directory.rglob("*") if path.is_file())
    # Include established analysis products whose writers do not use the common
    # figure/video category helper (for example reconstructed ToF maps).
    def paths_in(value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from paths_in(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                yield from paths_in(nested)
        elif isinstance(value, Path):
            yield value
    for artifact_name in session.sections[section].produces:
        artifact = session.artifacts.get(artifact_name)
        for candidate in paths_in(artifact):
            if isinstance(candidate, Path) and candidate.is_file() and candidate.is_relative_to(root):
                row = {"path": candidate.relative_to(root).as_posix(), "sha256": file_digest(candidate)}
                if row not in outputs:
                    outputs.append(row)
    for path in table_paths:
        row = {"path": path.relative_to(root).as_posix(), "sha256": file_digest(path)}
        if row not in outputs:
            outputs.append(row)
    from . import __version__
    atomic_json(record_dir / "export.json", {"section": section, "software_version": __version__,
                "dataset_id": session.artifacts.require("source_dataset")["dataset_id"], "outputs": outputs})

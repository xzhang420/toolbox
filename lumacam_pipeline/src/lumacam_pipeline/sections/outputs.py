"""Independent plotting and video export sections."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

from .. import backends
from ..section import PipelineSection, SectionOutcome, SectionState


def _category_dir(session, category):
    output = Path(session.artifacts.require("results_dir")) / category
    if (session.config.get("workflow", {}).get("role") == "visualization"
            and session.config.get("visualization", {}).get("export_layout", "versioned") != "flat"):
        output = output / session.artifacts.require("export_id")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _safe_stem(value, fallback):
    stem = Path(str(value)).stem.strip()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or fallback


def _finite_stack_range(images):
    minima = []
    maxima = []
    for image in images:
        array = np.asarray(image, dtype=float)
        finite = array[np.isfinite(array)]
        if finite.size:
            minima.append(float(finite.min()))
            maxima.append(float(finite.max()))
    if not minima:
        return 0.0, 1.0
    low, high = min(minima), max(maxima)
    if low == high:
        high = low + 1.0
    return low, high


class ImagePlotSection(PipelineSection):
    id = "image_plot"
    title = "Image figures"
    requires = ("image_stack", "selected_image_indices", "results_dir", "experiment")
    optional_inputs = ("image_metric_label",)
    produces = ("image_plot_path",)

    def run(self, session, config):
        images = session.artifacts.require("image_stack")
        indices = session.artifacts.require("selected_image_indices")
        metric_label = session.artifacts.get("image_metric_label", "Image intensity")
        figure, _ = backends.image_plots().plot_images(
            images,
            indices,
            pixel_size=float(config.get("pixel_size_um", 55)),
            cmap=str(config.get("cmap", "inferno")),
            cbar_mode=str(config.get("colorbar_mode", "shared left")),
            cbar_label=str(config.get("colorbar_label") or metric_label),
            cbar_unit=str(config.get("colorbar_unit", "")),
            cbar_n_ticks=int(config.get("colorbar_ticks", 4)),
            scalebar_length=float(config.get("scalebar_length_um", 5000)),
            scale_label=str(config.get("scalebar_label", "5 mm")),
            figsize=tuple(config.get("figsize", (8, 6))),
            render_3d=bool(config.get("render_3d", False)),
        )
        suffix = "_3D" if config.get("render_3d", False) else ""
        path = _category_dir(session, "figures") / f"images_{session.artifacts.require('experiment')}{suffix}.png"
        figure.savefig(path, dpi=int(config.get("dpi", 300)), bbox_inches="tight")
        return SectionOutcome(self.id, SectionState.COMPLETED, f"Saved {path}", {"image_plot_path": path})


class IndividualImagePlotSection(PipelineSection):
    id = "individual_image_plots"
    title = "Individual image figures"
    requires = ("image_stack", "image_paths", "results_dir", "experiment")
    optional_inputs = ("image_times", "image_metric_label")
    produces = ("individual_image_plot_paths",)

    def is_configured(self, config):
        return bool(config.get("enabled", False))

    def run(self, session, config):
        images = list(session.artifacts.require("image_stack"))
        source_paths = list(session.artifacts.require("image_paths"))
        image_times = list(session.artifacts.get("image_times", []))
        experiment = str(session.artifacts.require("experiment"))
        if len(source_paths) != len(images):
            source_paths = [f"image_{index:05d}" for index in range(len(images))]

        raw_indices = config.get("indices", "")
        if isinstance(raw_indices, str):
            raw_indices = [
                part.strip()
                for part in raw_indices.replace(";", ",").split(",")
                if part.strip()
            ]
        indices = list(range(len(images))) if not raw_indices else [int(value) for value in raw_indices]
        invalid = [index for index in indices if index < 0 or index >= len(images)]
        if invalid:
            raise ValueError(f"Individual image indices are out of range: {invalid}")

        extension = str(config.get("format", "png")).strip().lower().lstrip(".")
        if extension not in {"png", "pdf", "svg", "tif", "tiff"}:
            raise ValueError("Individual image format must be png, pdf, svg, tif, or tiff.")
        subfolder = _safe_stem(config.get("subfolder", "individual_images"), "individual_images")
        output_dir = _category_dir(session, "figures") / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        cbar_ranges = None
        if str(config.get("color_scale", "global")).lower() == "global":
            cbar_ranges = [_finite_stack_range(images)]
        metric_label = str(session.artifacts.get("image_metric_label", "Image intensity"))
        saved = []
        for index in indices:
            source_stem = _safe_stem(source_paths[index], f"image_{index:05d}")
            filename = f"{experiment}_image_{index:05d}_{source_stem}.{extension}"
            timestamp = image_times[index] if index < len(image_times) else None
            annotation = ""
            if timestamp is not None:
                annotation = timestamp.isoformat(sep=" ", timespec="seconds") if hasattr(timestamp, "isoformat") else str(timestamp)
            figure, _ = backends.image_plots().plot_images(
                images,
                [index],
                pixel_size=float(config.get("pixel_size_um", 55)),
                cmap=str(config.get("cmap", "inferno")),
                cbar_ranges=cbar_ranges,
                cbar_mode=str(config.get("colorbar_mode", "shared right")),
                cbar_label=str(config.get("colorbar_label") or metric_label),
                cbar_unit=str(config.get("colorbar_unit", "")),
                cbar_n_ticks=int(config.get("colorbar_ticks", 4)),
                scalebar_length=float(config.get("scalebar_length_um", 5000)),
                scale_label=str(config.get("scalebar_label", "5 mm")),
                figsize=tuple(config.get("figsize", (6, 6))),
                show_labels=False,
                lower_right_texts=[annotation] if annotation else None,
                render_3d=bool(config.get("render_3d", False)),
            )
            path = output_dir / filename
            figure.savefig(path, dpi=int(config.get("dpi", 300)), bbox_inches="tight")
            plt.close(figure)
            saved.append(path)
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Saved {len(saved)} individual image figure(s) to {output_dir}",
            {"individual_image_plot_paths": saved},
        )


def _check_thickness_units(session, config):
    """Do not place arbitrary image counts and EC micrometres on one value axis."""
    traces = backends.nci_plots().normalize_thickness_traces(config.get("thickness_traces", "both"))
    if (session.config.get("workflow", {}).get("role") == "visualization"
            and {"nci", "echem"}.issubset(traces)
            and not session.artifacts.get("nci_calibrated")
            and str(session.artifacts.get("image_unit", "")).strip().casefold() not in {"um", "µm", "μm"}):
        raise ValueError("Comparing image and EC thickness needs calibrated image values in µm. "
                         "Do not relabel counts as thickness; deselect electrochemical thickness for an intensity plot.")


class CombinedPlotSection(PipelineSection):
    id = "combined_plot"
    title = "Electrochemistry and image plot"
    requires = (
        "echem_times",
        "echem_potential",
        "image_times",
        "image_metric_values",
        "results_dir",
        "experiment",
    )
    produces = ("combined_plot_path",)
    optional_inputs = ("selected_image_indices", "event_context", "echem_thickness", "image_metric_label",
                       "image_metrics_table", "measurement_columns", "measurement_labels", "view_window", "echem_current")

    def is_configured(self, config):
        return bool(config.get("enabled", True))

    def run(self, session, config):
        _check_thickness_units(session, config)
        metric_label = session.artifacts.get("image_metric_label", "Image intensity")
        from ..event_selection import selected_records
        from ..time_window import bounds
        window = bounds(session, "figures")
        rows = selected_records(session, category="figures")
        from ..measurements import method_names
        requested = config.get("image_methods")
        methods = method_names(requested) if requested else []
        columns = session.artifacts.get("measurement_columns", {})
        labels = session.artifacts.get("measurement_labels", {})
        table = session.artifacts.get("image_metrics_table")
        if any(method not in columns for method in methods):
            raise ValueError("Calculate all requested plot methods with Measure images first.")
        metric_values = session.artifacts.require("image_metric_values")
        if methods:
            metric_values = table[columns[methods[0]]].to_numpy()
            metric_label = labels[methods[0]]
        event_times = []
        ec_times = session.artifacts.require("echem_times")
        tolerance = float(config.get("marker_tolerance_seconds", 90))
        for row in rows:
            stamp = row["time"]
            matched = stamp is not None and min(ec_times) <= stamp <= max(ec_times)
            if matched:
                matched = min(abs((time-stamp).total_seconds()) for time in ec_times) <= tolerance
            event_times.append(stamp if matched else None)
        figure = backends.nci_plots().plot_potential_and_thickness(
            session.artifacts.require("echem_times"),
            session.artifacts.require("echem_potential"),
            echem_thickness_series=session.artifacts.get("echem_thickness"),
            img_datetime_series=session.artifacts.require("image_times"),
            avg_thickness_series=metric_values,
            plot_thickness=True,
            potential_ylim=config.get("potential_ylim") or None,
            pause_gap_minutes=float(config.get("pause_gap_minutes", 5)),
            thickness_traces=str(config.get("thickness_traces", "both")),
            thickness_layout=str(config.get("thickness_layout", "separate")),
            thickness_axis_label=metric_label,
            nci_trace_label=metric_label if methods else str(config.get("image_trace_label", metric_label)),
            event_times=event_times,
            event_labels=[row["label"] for row in rows],
            event_marker_size=5,
            event_label_fontsize=12,
            show=False,
            current_series=session.artifacts.get("echem_current") if config.get("show_current", True) else None,
            image_gap_minutes=float(config.get("image_gap_minutes", 5)),
        )
        path = _category_dir(session, "figures") / f"electrochemistry_images_{session.artifacts.require('experiment')}.png"
        import matplotlib.pyplot as plt
        try:
            if config.get("clean_layout", True):
                # The potential axis already labels this single trace. Its
                # automatic legend can otherwise cover the first event label.
                figure.axes[0].set_title("")
                legend = figure.axes[0].get_legend()
                if legend is not None:
                    legend.remove()
            # Same letter on the actual image measurement; no interpolated NCI value.
            traces = backends.nci_plots().normalize_thickness_traces(config.get("thickness_traces", "both"))
            times = session.artifacts.require("image_times")
            if "nci" in traces and len(methods) > 1:
                axis = figure.axes[1]
                pauses = backends.nci_plots().find_paused_intervals(times,
                    min_gap_minutes=float(config.get("image_gap_minutes", 5)))
                for number, method in enumerate(methods[1:]):
                    x, y = backends.nci_plots().build_datetime_series_with_pauses(
                        times, table[columns[method]].to_numpy(), pauses, min(times))
                    axis.plot(x, y, label=labels[method], lw=1.2,
                              color=("#9467bd", "#d62728", "#17becf", "#8c564b")[number])
                axis.set_ylabel("Thickness (µm)" if session.artifacts.get("nci_calibrated") or
                    session.artifacts.get("image_unit") in {"um", "µm", "μm"} else "Image value")
                # Keep a multi-method legend outside the data area: automatic
                # placement can obscure event markers near the curve minima.
                axis.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
            if window:
                left, right = [(stamp-min(times)).total_seconds()/3600 for stamp in window]
                if right <= left:
                    raise ValueError("A curve zoom needs a nonzero time range.")
                for axis in figure.axes:
                    axis.set_xlim(left, right)
                    # Pause labels were computed on the full record. Do not let
                    # off-window text expand the saved figure's bounding box.
                    for text in axis.texts:
                        text.set_clip_on(True)
            if rows and "nci" in traces and str(config.get("thickness_layout", "separate")) == "separate":
                times = session.artifacts.require("image_times")
                values = metric_values
                axis = figure.axes[1]
                previous_hour, previous_offset = None, 6
                span = max((max(times)-min(times)).total_seconds()/3600, 1e-9)
                for row in rows:
                    index = row["image_index"]
                    hour = (times[index] - min(times)).total_seconds()/3600
                    if np.isfinite(values[index]):
                        close = previous_hour is not None and abs(hour-previous_hour) < .04*span
                        offset = -17 if close and previous_offset > 0 else 6
                        axis.plot(hour, values[index], "ko", ms=4)
                        axis.annotate(row["label"], (hour, values[index]), xytext=(4, offset),
                                      textcoords="offset points", fontsize=10, fontweight="bold")
                        previous_hour, previous_offset = hour, offset
            figure.savefig(path, dpi=int(config.get("dpi", 300)), bbox_inches="tight")
        finally:
            plt.close(figure)
        return SectionOutcome(self.id, SectionState.COMPLETED, f"Saved {path}", {"combined_plot_path": path})


class NciEventOutputsSection(PipelineSection):
    id = "nci_event_outputs"
    title = "NCI and electrochemistry event outputs"
    requires = (
        "selected_image_indices",
        "image_times",
        "image_stack",
        "image_metric_values",
        "results_dir",
        "experiment",
    )
    optional_inputs = (
        "event_context", "echem_times", "echem_potential", "echem_thickness",
        "echem_cycles", "measurement_roi", "image_metric_label",
    )
    produces = ("nci_event_output_paths",)

    def is_configured(self, config):
        return bool(config.get("enabled", True))

    def run(self, session, config):
        image_label = str(session.artifacts.get("image_metric_label", "Mean image intensity (a.u.)"))
        if "μm" in image_label or "thickness" in image_label.lower():
            metric_label, metric_unit = "Avg thickness", "μm"
            cbar_label, cbar_unit = "Li thickness", "μm"
        else:
            metric_label, metric_unit = "Mean image intensity", "a.u."
            cbar_label, cbar_unit = "NCI intensity", "a.u."
        event_plot_kwargs = {
            "pixel_size": float(config.get("pixel_size_um", 55)),
            "scalebar_length": float(config.get("scalebar_length_um", 5000)),
            "scale_label": str(config.get("scalebar_label", "5 mm")),
            "cmap": str(config.get("cmap", "inferno")),
            "cbar_mode": str(config.get("colorbar_mode", "shared left")),
            "cbar_label": [cbar_label],
            "cbar_unit": [cbar_unit],
            "cbar_n_ticks": int(config.get("colorbar_ticks", 4)),
            "render_3d": bool(config.get("render_3d", False)),
        }
        echem_times = session.artifacts.get("echem_times", [])
        echem_potential = session.artifacts.get("echem_potential", [])
        echem_thickness = session.artifacts.get("echem_thickness", [])
        save_potential = bool(len(echem_times) and len(echem_potential))
        roi = dict(session.artifacts.get("measurement_roi", {}) or {})
        if not {"x0", "y0", "r0"}.issubset(roi):
            shape = np.asarray(session.artifacts.require("image_stack")[0]).shape
            roi = {"x0": shape[1] // 2, "y0": shape[0] // 2, "r0": min(shape) // 2}
        result = backends.nci_plots().save_selected_event_outputs(
            event_context=session.artifacts.get(
                "event_context",
                {"event_times": [], "img_indices": [], "matched_count": 0, "total_events_found": 0},
            ),
            selected_image_indices=session.artifacts.require("selected_image_indices"),
            img_datetimes=session.artifacts.require("image_times"),
            img_series=session.artifacts.require("image_stack"),
            avg_thickness_series=session.artifacts.require("image_metric_values"),
            echem_datetimes=echem_times,
            potential_series=echem_potential,
            echem_thickness_series=echem_thickness,
            output_plot_dir=str(_category_dir(session, "figures")),
            exp_name=session.artifacts.require("experiment"),
            roi_small=roi,
            event_plot_kwargs=event_plot_kwargs,
            plot_images_func=backends.image_plots().plot_images,
            potential_plot_kwargs={
                "plot_thickness": bool(config.get("plot_thickness", True)),
                "thickness_axis_label": image_label,
                "nci_trace_label": image_label,
                "thickness_traces": str(config.get("thickness_traces", "both")),
                "thickness_layout": str(config.get("thickness_layout", "separate")),
            },
            echem_cycle_series=session.artifacts.get("echem_cycles"),
            metric_label=metric_label,
            metric_unit=metric_unit,
            save_potential=save_potential,
        )
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            "Saved established NCI event-panel outputs.",
            {"nci_event_output_paths": result.get("saved_paths", result)},
        )


class ImageVideoSection(PipelineSection):
    id = "image_video"
    title = "Image video"
    requires = ("image_times", "image_stack", "results_dir", "experiment")
    optional_inputs = ("image_metric_label",)
    produces = ("image_video_path",)

    def is_configured(self, config):
        return bool(config.get("enabled", False))

    def run(self, session, config):
        path = _category_dir(session, "videos") / f"{session.artifacts.require('experiment')}_images.mp4"
        backends.nci_plots().make_nci_image_video_uniform_time(
            session.artifacts.require("image_times"),
            session.artifacts.require("image_stack"),
            str(path),
            fps=int(config.get("fps", 10)),
            dt_seconds=config.get("dt_seconds"),
            dpi=int(config.get("dpi", 150)),
            cmap=str(config.get("cmap", "inferno")),
            vmin=config.get("vmin"),
            vmax=config.get("vmax"),
            pixel_size=float(config.get("pixel_size_um", 55)),
            scalebar_length=float(config.get("scalebar_length_um", 5000)),
            scale_label=str(config.get("scalebar_label", "5 mm")),
            cbar_label=str(config.get("colorbar_label") or session.artifacts.get("image_metric_label", "Image intensity")),
            cbar_n_ticks=int(config.get("colorbar_ticks", 5)),
            render_3d=bool(config.get("render_3d", False)),
        )
        return SectionOutcome(self.id, SectionState.COMPLETED, f"Saved {path}", {"image_video_path": path})


class CombinedVideoSection(PipelineSection):
    id = "combined_video"
    title = "Synchronized electrochemistry and image video"
    requires = (
        "echem_times",
        "echem_potential",
        "image_times",
        "image_stack",
        "results_dir",
        "experiment",
    )
    optional_inputs = ("image_metric_values", "image_metric_label", "echem_thickness", "view_window")
    produces = ("combined_video_path",)

    def is_configured(self, config):
        return bool(config.get("enabled", False))

    def run(self, session, config):
        if session.artifacts.has("image_metric_values"):
            _check_thickness_units(session, config)
        path = _category_dir(session, "videos") / f"{session.artifacts.require('experiment')}_synchronized.mp4"
        from ..visualization import _display_image
        from ..time_window import bounds
        display_images = [_display_image(image, config) for image in session.artifacts.require("image_stack")]
        backends.nci_plots().make_potential_image_video_uniform_time(
            session.artifacts.require("echem_times"),
            session.artifacts.require("echem_potential"),
            session.artifacts.require("image_times"),
            display_images,
            str(path),
            echem_thickness_series=session.artifacts.get("echem_thickness"),
            avg_thickness_series=session.artifacts.get("image_metric_values"),
            plot_thickness=session.artifacts.has("image_metric_values"),
            fps=int(config.get("fps", 10)),
            dt_seconds=config.get("dt_seconds"),
            dpi=int(config.get("dpi", 150)),
            cmap=str(config.get("cmap", "inferno")),
            vmin=config.get("vmin"),
            vmax=config.get("vmax"),
            pixel_size=float(config.get("pixel_size_um", 55)),
            scalebar_length=float(config.get("scalebar_length_um", 5000)),
            scale_label=str(config.get("scalebar_label", "5 mm")),
            colorbar_label=str(config.get("colorbar_label") or session.artifacts.get("image_metric_label", "Image intensity")),
            cbar_n_ticks=int(config.get("colorbar_ticks", 5)),
            thickness_layout=str(config.get("thickness_layout", "overlay")),
            thickness_traces=str(config.get("thickness_traces", "both")),
            thickness_axis_label=session.artifacts.get("image_metric_label", "Image intensity"),
            nci_trace_label=session.artifacts.get("image_metric_label", "Image intensity"),
            limit_to_overlap=bool(config.get("limit_to_overlap", False)),
            image_origin=str(config.get("image_origin", "lower")),
            time_window=bounds(session, "videos"),
            pause_gap_minutes=float(config.get('pause_gap_minutes',5)),
            image_gap_minutes=float(config.get('image_gap_minutes',5)),
            image_tolerance_seconds=float(config.get('image_tolerance_seconds',90)),
        )
        return SectionOutcome(self.id, SectionState.COMPLETED, f"Saved {path}", {"combined_video_path": path})

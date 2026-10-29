"""Optional electrochemistry event detection for image selection."""

from __future__ import annotations

from .. import backends
from ..section import PipelineSection, SectionOutcome, SectionState
import numpy as np
from ..event_selection import current_events


class EventDetectionSection(PipelineSection):
    id = "event_detection"
    title = "Electrochemistry events"
    requires = ("echem_times", "echem_thickness", "image_times")
    optional_inputs = ("echem_current", "image_metric_values", "image_exposure_bounds")
    produces = ("event_context", "event_image_indices")

    def is_configured(self, config):
        return bool(config.get("enabled", True))

    def run(self, session, config):
        method = str(config.get("method", "auto"))
        if method not in {"auto", "current", "capacity"}:
            raise ValueError("Event method must be auto, current or capacity.")
        current = session.artifacts.get("echem_current")
        available = current is not None and np.isfinite(current).any()
        if method == "current" and not available:
            raise ValueError("Current-based selection needs a current column in the electrochemistry file.")
        if method != "capacity" and available:
            values = session.artifacts.get("image_metric_values")
            valid = None if values is None else np.isfinite(values)
            if valid is not None and config.get("exclude_zero_mean_images", True):
                valid &= np.asarray(values) != 0
            sign = config.get("plating_current_sign")
            context = current_events(session.artifacts.require("echem_times"), current,
                session.artifacts.require("image_times"),
                tolerance_seconds=float(config.get("match_tolerance_seconds", 90)),
                zero_current_ma=float(config.get("zero_current_ma", .01)),
                min_phase_seconds=float(config.get("min_phase_seconds", 30)),
                plating_current_sign=None if sign in (None, "") else int(sign), valid_images=valid,
                image_exposure_bounds=session.artifacts.get('image_exposure_bounds'))
            indices = context["img_indices"]
            return SectionOutcome(self.id, SectionState.COMPLETED,
                f"Selected {len(indices)} OCV/half-cycle images; {len(context['skipped_events'])} events without a usable image. "
                "Zero/nonfinite means excluded only from representative selection; measurements are unchanged.",
                {"event_context": context, "event_image_indices": indices})
        initial_ocv_image_count = config.get("initial_ocv_image_count")
        if initial_ocv_image_count in ("", None):
            initial_ocv_image_count = None
        else:
            initial_ocv_image_count = int(initial_ocv_image_count)
        context = backends.nci_plots().build_turning_point_event_context(
            session.artifacts.require("echem_times"),
            session.artifacts.require("echem_thickness"),
            session.artifacts.require("image_times"),
            event_finder=backends.nci_data().find_capacity_turning_points,
            image_index_finder=backends.nci_data().find_nearest_image_indices,
            conv_thickness=float(config.get("conversion", 1.0)),
            n_tpx_per_image=int(config.get("n_tpx_per_image", 30)),
            tpx_time=float(config.get("tpx_time", 2)),
            initial_ocv_image_count=initial_ocv_image_count,
        )
        indices = list(context.get("img_indices", []))
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Matched {len(indices)} electrochemistry events to images.",
            {"event_context": context, "event_image_indices": indices},
        )

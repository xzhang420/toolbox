"""Optional electrochemistry event detection for image selection."""

from __future__ import annotations

from .. import backends
from ..section import PipelineSection, SectionOutcome, SectionState


class EventDetectionSection(PipelineSection):
    id = "event_detection"
    title = "Electrochemistry events"
    requires = ("echem_times", "echem_thickness", "image_times")
    produces = ("event_context", "event_image_indices")

    def is_configured(self, config):
        return bool(config.get("enabled", True))

    def run(self, session, config):
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

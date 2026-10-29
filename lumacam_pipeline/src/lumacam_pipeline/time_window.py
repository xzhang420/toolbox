"""Presentation windows only: never crop data before measurement or EC conversion."""
from datetime import timedelta
import numpy as np
from .image_io import parse_timestamp
from .section import PipelineSection, SectionOutcome, SectionState


def bounds(session, category=None, *, force=False):
    config = session.config.get("time_window", {})
    if not config.get("enabled", False):
        if force:
            raise ValueError("Enable a time window before exporting the selected window.")
        return None
    if category and not config.get(f"apply_{category}", True):
        return None
    times = session.artifacts.require("image_times")
    if not times or any(t is None for t in times):
        raise ValueError("A time window requires real image timestamps.")
    first, last = min(times), max(times)
    mode = config.get("mode", "elapsed")
    if mode not in {"elapsed", "absolute"}:
        raise ValueError("Window mode must be elapsed or absolute.")
    def parse(value, default):
        if value in (None, ""):
            return default
        if mode == "elapsed":
            hour = float(value)
            if not np.isfinite(hour):
                raise ValueError("Window hours must be finite.")
            return first+timedelta(hours=hour)
        if mode != "absolute":
            raise ValueError("Window mode must be elapsed or absolute.")
        stamp = parse_timestamp(value, session.config.get("input", {}).get("timestamp_timezone", ""))
        if stamp is None:
            raise ValueError("Use ISO date/time with UTC offset, or set the input timezone.")
        return stamp
    start, end = parse(config.get("start"), first), parse(config.get("end"), last)
    if start > end:
        raise ValueError("Time-window start must not be after its end.")
    if end < first or start > last:
        raise ValueError("Time window does not overlap the image experiment.")
    return max(start, first), min(end, last)


def time_mask(times, window):
    if window is None:
        return np.ones(len(times), dtype=bool)
    return np.array([t is not None and window[0] <= t <= window[1] for t in times], dtype=bool)


def image_indices(session, indices, category):
    window = bounds(session, category)
    keep = time_mask(session.artifacts.require("image_times"), window)
    return [int(i) for i in indices if keep[i]]


class TimeWindowSection(PipelineSection):
    id = "time_window"
    title = "Time window"
    produces = ("view_window",)

    def is_configured(self, config):
        return True

    def run(self, session, config):
        window = bounds(session)
        return SectionOutcome(self.id, SectionState.COMPLETED,
            f"Window: {window[0].isoformat()} to {window[1].isoformat()}. Full-data measurements retained."
            if window else "Full experiment shown; measurements retained.", {"view_window": window})

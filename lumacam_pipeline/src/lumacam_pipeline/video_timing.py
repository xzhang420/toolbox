"""Auditable image-only playback; gaps never interpolate image values."""
from datetime import timedelta

import numpy as np

from .time_window import bounds, image_indices


def video_plan(session, config):
    """Return output-frame mappings and resolved settings, without writing files."""
    mode = config.get("timing_mode", "image_sequence")
    if mode not in {"image_sequence", "elapsed_time"}:
        raise ValueError("Video timing_mode must be image_sequence or elapsed_time.")
    fps = float(config.get("fps", 10))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Video fps must be positive and finite.")
    times = session.artifacts.require("image_times")
    rows = []
    resolved = {"timing_mode": mode, "fps": fps, "dt_seconds": None,
                "image_tolerance_seconds": None}

    def row(number, index, stamp):
        acquired = times[index] if index is not None else None
        return {"video_frame": number, "playback_seconds": number / fps,
                "timeline_utc": stamp.isoformat() if stamp is not None else "",
                "elapsed_seconds": (stamp-times[0]).total_seconds() if stamp is not None and times[0] is not None else None,
                "image_index": index, "acquisition_utc": acquired.isoformat() if acquired is not None else "",
                "time_offset_seconds": (stamp-acquired).total_seconds() if stamp is not None and acquired is not None else None,
                "status": "no_acquisition_data" if index is None else "acquired_image"}

    if mode == "image_sequence":
        indices = image_indices(session, range(len(times)), "videos")
        if not indices:
            raise ValueError("No image frames in the selected video window.")
        rows = [row(j, i, times[i]) for j, i in enumerate(indices)]
    else:
        if not times or any(t is None for t in times):
            raise ValueError("Elapsed-time playback requires real acquisition timestamps for every image.")
        seconds = np.array([(t-times[0]).total_seconds() for t in times])
        intervals = np.diff(seconds)
        if np.any(intervals <= 0):
            raise ValueError("Elapsed-time playback needs strictly increasing, unique image timestamps.")
        cadence = float(np.median(intervals)) if intervals.size else 1.0
        dt = config.get("dt_seconds")
        dt = cadence if dt in (None, "") else float(dt)
        tolerance = config.get("image_tolerance_seconds")
        tolerance = .75 * cadence if tolerance in (None, "") else float(tolerance)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Video dt_seconds must be positive and finite.")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Image time tolerance must be finite and nonnegative.")
        start, end = bounds(session, "videos") or (times[0], times[-1])
        duration = (end-start).total_seconds()
        if not np.isfinite(duration / dt) or duration / dt >= 100000:
            raise ValueError("Video would exceed 100,000 frames; increase dt_seconds or narrow the time window.")
        count = int(np.floor(duration / dt + 1e-9)) + 1
        grid = (start-times[0]).total_seconds() + np.arange(count) * dt
        # O(output frames log input frames), not a full image scan per frame.
        right = np.clip(np.searchsorted(seconds, grid), 0, len(seconds)-1)
        left = np.maximum(0, right-1)
        nearest = np.where(np.abs(grid-seconds[left]) <= np.abs(grid-seconds[right]), left, right)
        for j, (t, i) in enumerate(zip(grid, nearest)):
            index = int(i) if abs(t-seconds[i]) <= tolerance + 1e-9 else None
            rows.append(row(j, index, times[0]+timedelta(seconds=float(t))))
        resolved.update(dt_seconds=dt, image_tolerance_seconds=tolerance,
                        inferred_cadence_seconds=cadence,
                        experimental_seconds_per_playback_second=fps * dt)
    resolved.update(frame_count=len(rows), gap_frame_count=sum(r["image_index"] is None for r in rows))
    return rows, resolved

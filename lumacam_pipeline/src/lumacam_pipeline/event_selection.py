"""Auditable representative images: observed OCV and current half-cycle ends.

Selection never changes measurements. A frame must be before an endpoint and
within the configured tolerance; gaps are not filled by distant images.
"""

import numpy as np


def panel_label(index):
    """a..z, aa..az (also used for curve markers)."""
    label = ""
    while index >= 0:
        label = chr(97 + index % 26) + label
        index = index // 26 - 1
    return label


def current_events(times, current, image_times, *, tolerance_seconds=90,
                   zero_current_ma=0.01, min_phase_seconds=30,
                   plating_current_sign=None, valid_images=None, image_exposure_bounds=None):
    times, image_times = list(times), list(image_times)
    current = np.asarray(current, dtype=float)
    if not times or not image_times or len(current) != len(times):
        raise ValueError("Current event selection requires aligned, nonempty time/current arrays.")
    if any(b < a for a, b in zip(times, times[1:])):
        raise ValueError("Electrochemistry must be time-sorted.")
    if tolerance_seconds <= 0 or zero_current_ma < 0 or min_phase_seconds < 0:
        raise ValueError("Event tolerance must be positive; thresholds must be nonnegative.")
    if plating_current_sign not in (None, -1, 1):
        raise ValueError("plating_current_sign must be -1, 1 or blank (generic sign labels).")
    valid = np.ones(len(image_times), dtype=bool) if valid_images is None else np.asarray(valid_images, dtype=bool)
    if len(valid) != len(image_times):
        raise ValueError("One validity flag is required per image.")
    exposures = image_exposure_bounds or [None]*len(image_times)
    if len(exposures) != len(image_times):
        raise ValueError('Exposure bounds must align with images.')
    sign = np.where(np.isfinite(current), np.where(np.abs(current) > zero_current_ma, np.sign(current), 0), 2)
    starts = np.r_[0, np.flatnonzero(sign[1:] != sign[:-1]) + 1]
    stops = np.r_[starts[1:] - 1, len(sign)-1]
    phases, ignored = [], []
    for start, end in zip(starts, stops):
        if sign[start] not in (-1, 1):
            continue
        duration = (times[end] - times[start]).total_seconds()
        if duration < min_phase_seconds:
            ignored.append({"start": times[start], "end": times[end], "reason": "short current segment", "seconds": duration})
            continue
        # Pauses/restarts at the same polarity do not create extra half cycles.
        if (phases and phases[-1]["sign"] == sign[start]
                and not np.any(sign[phases[-1]["end"]+1:start] == -sign[start])):
            phases[-1].update(end=int(end), interrupted=True)
        else:
            phases.append({"start": int(start), "end": int(end), "sign": int(sign[start]), "interrupted": False})
    candidates = []
    if phases:
        first = phases[0]["start"]
        rest = np.flatnonzero(sign[:first] == 0)
        # Never call a frame OCV if the file begins under load.
        if len(rest) and not np.any(np.isin(sign[:rest[-1]], (-1, 1))):
            candidates.append({"time": times[int(rest[-1])], "kind": "ocv", "caption": "OCV", "status": "observed zero current"})
    irregular = bool(ignored) or any(a["sign"] == b["sign"] for a, b in zip(phases, phases[1:]))
    for number, phase in enumerate(phases):
        sign_value = phase["sign"]
        kind = ("plating" if sign_value == plating_current_sign else "stripping") if plating_current_sign else ("negative current" if sign_value < 0 else "positive current")
        end = phase["end"]
        status = "last recorded; completion unconfirmed" if end == len(times)-1 else "current boundary"
        contains_gap = any((b-a).total_seconds()>300 for a,b in zip(times[phase['start']:end], times[phase['start']+1:end+1]))
        if phase['interrupted'] or contains_gap:
            status = 'interrupted phase; ' + status
        candidates.append({"time": times[end], "start": times[phase["start"]], "kind": kind,
                           "caption": kind.capitalize() if irregular else f"{kind.capitalize()} {(number // 2) + 1}", "half_cycle": number+1,
                           "status": status, "interrupted": phase["interrupted"] or contains_gap,
                           "contains_ec_gap": contains_gap})
    selected, skipped, used = [], [], set()
    acquisition_start = min(exposure[0] if exposure else t for exposure,t in zip(exposures,image_times))
    acquisition_end = max(exposure[1] if exposure else t for exposure,t in zip(exposures,image_times))
    for event in candidates:
        if not acquisition_start <= event["time"] <= acquisition_end:
            skipped.append({**event, "reason": "event outside image acquisition interval"})
            continue
        offsets = np.array([(event["time"] - t).total_seconds() for t in image_times])
        eligible = valid & (offsets >= 0) & (offsets <= tolerance_seconds)
        if event.get("start"):
            eligible &= np.array([t >= event["start"] for t in image_times])
        for i, exposure in enumerate(exposures):
            if exposure is not None:
                start, end = exposure
                if start > end: raise ValueError('Exposure starts after its end.')
                eligible[i] &= (end <= event['time'] and (event['time']-end).total_seconds() <= tolerance_seconds
                                and (not event.get('start') or start >= event['start']))
        choices = np.flatnonzero(eligible)
        if not len(choices):
            skipped.append({**event, "reason": "no valid preceding image within tolerance and phase"})
            continue
        index = int(choices[np.argmin(offsets[choices])])
        if index in used:
            skipped.append({**event, "reason": "image already selected for another event"})
            continue
        used.add(index)
        selected.append({**event, "image_index": index, "image_time": image_times[index],
                         "image_offset_seconds": -float(offsets[index]), "label": panel_label(len(selected)),
                         "timing_basis": 'full exposure bounds' if exposures[index] else 'representative timestamp; exposure unavailable'})
    return {"method": "current_half_cycles", "events": selected,
            "event_times": [row["time"] for row in selected],
            "img_indices": [row["image_index"] for row in selected],
            "event_times_for_plot": [row["time"] for row in selected],
            "matched_count": len(selected), "total_events_found": len(candidates),
            "skipped_events": skipped, "ignored_segments": ignored,
            "match_warning_threshold_seconds": tolerance_seconds}


def selected_records(session, category=None):
    """One stable label per selected image, shared by panel, curves and CSV."""
    context = session.artifacts.get("event_context", {})
    events = {row["image_index"]: row for row in context.get("events", []) if "image_index" in row}
    times = session.artifacts.require("image_times")
    rows = []
    for number, index in enumerate(session.artifacts.get("selected_image_indices", [])):
        row = dict(events.get(index, {}))
        row.update(label=panel_label(number), image_index=index, image_time=times[index])
        row.setdefault("time", times[index])
        row.setdefault("caption", f"Frame {index}")
        rows.append(row)
    if category:
        from .time_window import image_indices
        keep = set(image_indices(session, [row["image_index"] for row in rows], category))
        rows = [row for row in rows if row["image_index"] in keep]
    return rows

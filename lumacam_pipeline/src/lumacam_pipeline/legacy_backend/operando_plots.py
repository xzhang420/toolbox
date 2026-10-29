"""Established NCI/electrochemistry plotting and video calculations."""

import os
import io
import re
from collections import deque
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as patches
from bisect import bisect_right
from matplotlib_scalebar.scalebar import ScaleBar
from ..products import is_all_runs_sum


class ImageArray(np.ndarray):
    """np.ndarray subclass that can carry lightweight per-image metadata."""
    pass


MICRON_UNIT = "\u03bcm"


def normalize_scalebar_length_um(scalebar_length, scale_label=None):
    """Return a scale-bar length in micrometers, accepting legacy mm-style values."""
    length = float(scalebar_length)
    label = str(scale_label or "").lower()
    if "mm" in label and length < 100:
        return length * 1000.0
    return length


def generated_cycle_labels(count):
    """Return default labels: OCV, cycle 1, cycle 1, cycle 2, cycle 2, ..."""
    labels = []
    for pos in range(int(count)):
        if pos == 0:
            labels.append("OCV")
        else:
            labels.append(f"cycle {(pos + 1) // 2}")
    return labels


def spatial_average_images(img_series, factor=1):
    """Average neighboring spatial pixels by an integer factor."""
    factor = int(factor or 1)
    if factor <= 1:
        return img_series

    averaged_images = []
    for img in img_series:
        arr = np.asarray(img, dtype=float)
        ny, nx = arr.shape
        trim_y = ny - (ny % factor)
        trim_x = nx - (nx % factor)
        if trim_y <= 0 or trim_x <= 0:
            raise ValueError(f"spatial merge factor {factor} is larger than the image shape {arr.shape}.")
        block = arr[:trim_y, :trim_x].reshape(trim_y // factor, factor, trim_x // factor, factor)
        averaged = np.nanmean(block, axis=(1, 3)).view(ImageArray)
        crop_x0, crop_y0 = getattr(img, "crop_origin_xy", (0, 0))
        averaged.crop_origin_xy = (float(crop_x0) / factor, float(crop_y0) / factor)
        averaged.spatial_bin_factor = factor * int(getattr(img, "spatial_bin_factor", 1))
        averaged_images.append(averaged)
    return averaged_images


def merge_images_by_frame_count(img_datetimes, img_series, frames_per_merge=1, method="mean"):
    """Merge a fixed number of consecutive image frames into one averaged frame."""
    frames_per_merge = int(frames_per_merge or 1)
    if frames_per_merge <= 1:
        return img_datetimes, img_series

    merged_datetimes = []
    merged_images = []
    for start in range(0, len(img_series), frames_per_merge):
        frame_images = list(img_series[start:start + frames_per_merge])
        frame_times = list(img_datetimes[start:start + frames_per_merge])
        if not frame_images:
            continue
        stack = np.stack(frame_images, axis=0)
        if method == "sum":
            merged = np.nansum(stack, axis=0)
        else:
            merged = np.nanmean(stack, axis=0)
        merged = np.asarray(merged).view(ImageArray)
        merged.crop_origin_xy = getattr(frame_images[0], "crop_origin_xy", (0, 0))
        merged.spatial_bin_factor = int(getattr(frame_images[0], "spatial_bin_factor", 1))
        merged_images.append(merged)
        merged_datetimes.append(frame_times[len(frame_times) // 2])
    return merged_datetimes, merged_images


def total_lithium_mass_series(img_series, pixel_size=55, lithium_density_g_cm3=0.534):
    """
    Calculate requested total lithium mass-like value from each image.

    The formula follows the GUI request: sum(thickness) * pixel_size^2 * 1e-12
    divided by the specified density.
    """
    density = float(lithium_density_g_cm3)
    if density <= 0:
        raise ValueError("lithium_density_g_cm3 must be positive.")
    masses = []
    for img in img_series:
        arr = np.asarray(img, dtype=float)
        values = arr[np.isfinite(arr)]
        effective_pixel_size = float(pixel_size) * int(getattr(img, "spatial_bin_factor", 1))
        masses.append(np.nansum(values) * effective_pixel_size**2 * 1e-12 / density)
    return np.asarray(masses, dtype=float)


def calculate_plating_metrics(
    image,
    roi_mask,
    pixel_size=None,
    center=None,
    threshold=None,
    threshold_fraction=0.05,
    anode_radius=None,
):
    """
    Calculate per-frame lithium plating distribution metrics inside a nominal anode ROI.

    Physically, ``f_plate`` is the plated footprint fraction, ``CV_active`` and
    ``CV_global`` describe local and whole-anode Li-signal homogeneity,
    ``rho_Li`` is the Li-weighted mean deposition radius normalized by the
    nominal anode radius, and ``d_COM_norm`` is the normalized offset of the
    Li-weighted center of mass from the nominal anode center.
    """
    arr = np.asarray(image, dtype=float)
    mask = np.asarray(roi_mask, dtype=bool)
    if arr.shape != mask.shape:
        raise ValueError(f"roi_mask shape {mask.shape} does not match image shape {arr.shape}.")

    valid_mask = mask & np.isfinite(arr)
    nominal_count = int(np.count_nonzero(valid_mask))
    empty = {
        "f_plate": np.nan,
        "CV_active": np.nan,
        "CV_global": np.nan,
        "rho_Li": np.nan,
        "d_COM_norm": np.nan,
        "active_plating_area_px": 0,
        "nominal_anode_area_px": nominal_count,
        "active_plating_area_mm2": np.nan,
        "nominal_anode_area_mm2": np.nan,
        "threshold_value": np.nan,
        "mean_active": np.nan,
        "mean_nominal": np.nan,
        "sum_nominal": np.nan,
    }
    if nominal_count == 0:
        return empty

    h_nominal = arr[valid_mask].astype(float)
    h_nominal = np.where(h_nominal > 0, h_nominal, 0.0)

    if threshold is None:
        finite_max = np.nanmax(h_nominal) if h_nominal.size else np.nan
        threshold_value = float(threshold_fraction) * float(finite_max) if np.isfinite(finite_max) else np.nan
    else:
        threshold_value = float(threshold)

    if np.isfinite(threshold_value):
        active_mask = valid_mask & (np.where(arr > 0, arr, 0.0) > threshold_value)
    else:
        active_mask = np.zeros_like(valid_mask, dtype=bool)
    active_count = int(np.count_nonzero(active_mask))

    h_active = np.where(arr[active_mask] > 0, arr[active_mask], 0.0).astype(float)
    mean_nominal = float(np.nanmean(h_nominal)) if h_nominal.size else np.nan
    mean_active = float(np.nanmean(h_active)) if h_active.size else np.nan
    cv_global = float(np.nanstd(h_nominal) / mean_nominal) if np.isfinite(mean_nominal) and mean_nominal > 0 else np.nan
    cv_active = float(np.nanstd(h_active) / mean_active) if np.isfinite(mean_active) and mean_active > 0 else np.nan

    yy, xx = np.indices(arr.shape)
    if center is None:
        xs = xx[valid_mask].astype(float)
        ys = yy[valid_mask].astype(float)
        cx = float(np.nanmean(xs))
        cy = float(np.nanmean(ys))
    else:
        cx, cy = center
        cx = float(cx)
        cy = float(cy)

    if anode_radius is None:
        rr_nominal = np.sqrt((xx[valid_mask] - cx) ** 2 + (yy[valid_mask] - cy) ** 2)
        anode_radius = float(np.nanmax(rr_nominal)) if rr_nominal.size else np.nan
    else:
        anode_radius = float(anode_radius)

    weights = np.zeros(arr.shape, dtype=float)
    weights[valid_mask] = h_nominal
    sum_h = float(np.nansum(weights[valid_mask]))
    if np.isfinite(sum_h) and sum_h > 0 and np.isfinite(anode_radius) and anode_radius > 0:
        rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        rho_li = float(np.nansum(rr[valid_mask] * weights[valid_mask]) / (anode_radius * sum_h))
        com_x = float(np.nansum(xx[valid_mask] * weights[valid_mask]) / sum_h)
        com_y = float(np.nansum(yy[valid_mask] * weights[valid_mask]) / sum_h)
        d_com_norm = float(np.sqrt((com_x - cx) ** 2 + (com_y - cy) ** 2) / anode_radius)
    else:
        rho_li = np.nan
        d_com_norm = np.nan

    pixel_area_mm2 = np.nan
    if pixel_size is not None and np.isfinite(float(pixel_size)):
        pixel_area_mm2 = (float(pixel_size) / 1000.0) ** 2

    return {
        "f_plate": float(active_count / nominal_count) if nominal_count else np.nan,
        "CV_active": cv_active,
        "CV_global": cv_global,
        "rho_Li": rho_li,
        "d_COM_norm": d_com_norm,
        "active_plating_area_px": active_count,
        "nominal_anode_area_px": nominal_count,
        "active_plating_area_mm2": active_count * pixel_area_mm2 if np.isfinite(pixel_area_mm2) else np.nan,
        "nominal_anode_area_mm2": nominal_count * pixel_area_mm2 if np.isfinite(pixel_area_mm2) else np.nan,
        "threshold_value": threshold_value,
        "mean_active": mean_active,
        "mean_nominal": mean_nominal,
        "sum_nominal": sum_h,
    }


def find_paused_intervals(datetime_series, min_gap_minutes=5):
    """
    Detect long gaps in a datetime series that likely correspond to paused acquisition.

    Returns
    -------
    list of tuple(datetime, datetime)
        Start and end datetimes of each detected pause.
    """
    if datetime_series is None or len(datetime_series) < 2:
        return []

    dt_series = pd.to_datetime(list(datetime_series))
    time_diffs = dt_series.to_series().diff().iloc[1:]
    if time_diffs.empty:
        return []

    gap_threshold = pd.Timedelta(minutes=min_gap_minutes)

    pauses = []
    for i, delta in enumerate(time_diffs, start=1):
        if delta > gap_threshold:
            pauses.append((dt_series[i - 1].to_pydatetime(), dt_series[i].to_pydatetime()))

    return pauses


def build_series_with_pauses(x_values, y_values, pauses, time_origin):
    """
    Insert NaN separators into a time series so plotted lines break over pauses.
    """
    if len(x_values) == 0:
        return np.asarray(x_values), np.asarray(y_values)

    pause_starts = {
        (pause_start - time_origin).total_seconds() / 3600.0
        for pause_start, _ in pauses
    }

    x_plot = []
    y_plot = []
    for i, (x_val, y_val) in enumerate(zip(x_values, y_values)):
        x_plot.append(x_val)
        y_plot.append(y_val)
        if i < len(x_values) - 1 and x_val in pause_starts:
            x_plot.append(np.nan)
            y_plot.append(np.nan)

    return np.asarray(x_plot), np.asarray(y_plot)


def build_datetime_series_with_pauses(datetime_values, y_values, pauses, time_origin):
    """
    Insert NaN separators into a datetime-aligned series when segments cross pauses.
    """
    if len(datetime_values) == 0:
        return np.array([]), np.asarray(y_values)

    dt_values = list(pd.to_datetime(list(datetime_values)).to_pydatetime())
    x_values = np.array([(t - time_origin).total_seconds() for t in dt_values]) / 3600.0
    y_values = np.asarray(y_values)

    x_plot = []
    y_plot = []
    for i, (x_val, y_val) in enumerate(zip(x_values, y_values)):
        x_plot.append(x_val)
        y_plot.append(y_val)
        if i >= len(x_values) - 1:
            continue

        seg_start = dt_values[i]
        seg_end = dt_values[i + 1]
        crosses_pause = any(
            seg_start < pause_end and seg_end > pause_start
            for pause_start, pause_end in pauses
        )
        if crosses_pause:
            x_plot.append(np.nan)
            y_plot.append(np.nan)

    return np.asarray(x_plot), np.asarray(y_plot)


def normalize_thickness_traces(thickness_traces, plot_thickness=True):
    """
    Normalize thickness trace selection to a stable set of trace keys.
    """
    if not plot_thickness:
        return set()

    if thickness_traces is None:
        thickness_traces = "both"

    if isinstance(thickness_traces, str):
        trace_key = thickness_traces.strip().lower()
        aliases = {
            "both": {"echem", "nci"},
            "all": {"echem", "nci"},
            "ec": {"echem"},
            "echem": {"echem"},
            "electrochemical": {"echem"},
            "nci": {"nci"},
            "neutron": {"nci"},
            "none": set(),
            "off": set(),
            "false": set(),
        }
        if trace_key not in aliases:
            raise ValueError(
                "thickness_traces must be one of 'both', 'echem', 'nci', or 'none'."
            )
        return aliases[trace_key]

    traces = {str(trace).strip().lower() for trace in thickness_traces}
    normalized = set()
    for trace in traces:
        if trace in ("ec", "echem", "electrochemical"):
            normalized.add("echem")
        elif trace in ("nci", "neutron"):
            normalized.add("nci")
        else:
            raise ValueError(
                "thickness_traces entries must be 'echem'/'ec' or 'nci'."
            )
    return normalized


def apply_vertical_offsets_to_traces(trace_series, offset_fraction=0.08):
    """
    Apply small vertical offsets so overlaid traces remain visually separable.
    """
    trace_series = list(trace_series)
    if len(trace_series) <= 1:
        return [(label, values, 0.0) for label, values in trace_series]

    finite_values = []
    for _, values in trace_series:
        values = np.asarray(values, dtype=float)
        finite_values.extend(values[np.isfinite(values)])

    if finite_values:
        data_range = float(np.nanmax(finite_values) - np.nanmin(finite_values))
    else:
        data_range = 0.0
    if data_range == 0.0:
        data_range = 1.0

    offset_step = data_range * float(offset_fraction)
    center = (len(trace_series) - 1) / 2.0
    return [
        (label, np.asarray(values, dtype=float) + (idx - center) * offset_step, (idx - center) * offset_step)
        for idx, (label, values) in enumerate(trace_series)
    ]


def capacity_from_sign_half_cycles(capacity_series, sign_series, *, zero_current=0,
                                   source_ids=None):
    """
    Convert raw capacity into a continuous half-cycle capacity from a sign series.

    Accumulate capacity increments, adding at negative current and subtracting
    at positive current. Counter resets, polarity changes and independent file
    boundaries start a new local counter without resetting accumulated charge.
    Rest holds the accumulated value even if the instrument clears its counter.
    The result is net transferred charge relative to the beginning of the record,
    not an absolute inventory of lithium. Negative values are meaningful.
    """
    capacity_series = np.asarray(capacity_series, dtype=float)
    sign_series = np.asarray(sign_series, dtype=float)
    if capacity_series.shape != sign_series.shape:
        raise ValueError("capacity_series and sign_series must have the same shape.")

    adjusted_capacity = np.full_like(capacity_series, np.nan, dtype=float)
    if len(capacity_series) == 0:
        return adjusted_capacity

    if not np.isfinite(zero_current) or zero_current < 0:
        raise ValueError("zero_current must be finite and nonnegative.")
    sources = [None] * len(capacity_series) if source_ids is None else list(source_ids)
    if len(sources) != len(capacity_series):
        raise ValueError("source_ids must align with capacity samples.")
    total = 0.0
    previous_sign = 0.0
    previous_capacity = np.nan
    previous_source = None
    for idx, (capacity, sign_value) in enumerate(zip(capacity_series, sign_series)):
        if not np.isfinite(capacity) or not np.isfinite(sign_value):
            continue
        current_sign = np.sign(sign_value) if abs(sign_value) > zero_current else 0
        counter = abs(float(capacity))
        new_counter = (not np.isfinite(previous_capacity) or sources[idx] != previous_source
                       or (current_sign and previous_sign and current_sign != previous_sign)
                       or counter < previous_capacity - 1e-10)
        if current_sign:
            increment = counter if new_counter else max(0., counter - previous_capacity)
            total -= current_sign * increment
            previous_sign = current_sign
        adjusted_capacity[idx] = total
        previous_capacity, previous_source = counter, sources[idx]
    return adjusted_capacity


def capacity_from_potential_sign_half_cycles(capacity_series, potential_series):
    """Backward-compatible wrapper for the previous potential-sign behavior."""
    return capacity_from_sign_half_cycles(capacity_series, potential_series)


def capacity_from_current_sign_half_cycles(capacity_series, current_series, **kwargs):
    """Convert raw capacity using current sign: negative adds, positive subtracts."""
    return capacity_from_sign_half_cycles(capacity_series, current_series, **kwargs)


def parse_echem_datetime_value(value):
    """Parse EC timestamp strings while preserving optional fractional seconds."""
    value_str = str(value).strip()
    for fmt in ("%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(value_str, fmt)
        except ValueError:
            pass
    raise ValueError(f"Unsupported electrochemical timestamp format: {value!r}")


ECHEM_EXTENSIONS = (".txt", ".csv")


ECHEM_COLUMN_ALIASES = {
    "time": (
        "time/s",
        "time",
        "absolute time",
        "date/time",
        "date / time",
        "date",
    ),
    "potential": (
        "Ewe/V",
        "potential / v",
        "potential/v",
        "potential",
    ),
    "current": (
        "<I>/mA",
        "current / ma",
        "current/ma",
        "current",
    ),
    "capacity": (
        "Capacity/mA.h",
        "capacity / mah",
        "capacity/mah",
        "capacity",
    ),
    "cycle": (
        "cycle number",
        "cycle",
    ),
}


ECHEM_COLUMN_KEYWORDS = {
    "time": {
        "required_any": (("time",), ("date",), ("timestamp",)),
        "preferred_any": (("time", "s"), ("date", "time"), ("absolute", "time")),
        "reject_any": (("cycle",), ("capacity",), ("current",), ("potential",), ("ewe",)),
    },
    "potential": {
        "required_any": (("ewe",), ("voltage",), ("potential",)),
        "preferred_any": (("ewe", "v"), ("potential", "v"), ("voltage", "v")),
        "reject_any": (("capacity",), ("current",), ("cycle",), ("time",)),
    },
    "current": {
        "required_any": (("i",), ("current",)),
        "preferred_any": (("i", "ma"), ("current", "ma")),
        "reject_any": (("capacity",), ("voltage",), ("potential",), ("cycle",), ("time",)),
    },
    "capacity": {
        "required_any": (("capacity",), ("cap",), ("q",)),
        "preferred_any": (("capacity", "mah"), ("capacity", "ma", "h"), ("q", "mah")),
        "reject_any": (("current",), ("potential",), ("voltage",), ("cycle",), ("time",)),
    },
    "cycle": {
        "required_any": (("cycle",),),
        "preferred_any": (("cycle", "number"), ("cycle", "index")),
        "reject_any": (("time",), ("capacity",), ("current",), ("potential",), ("voltage",)),
    },
}


def _clean_echem_column_name(name):
    return str(name).strip().strip(",").strip()


def _echem_column_key(name):
    return _clean_echem_column_name(name).lower()


def _echem_column_tokens(name):
    key = _echem_column_key(name)
    normalized = (
        key.replace("<", " ")
        .replace(">", " ")
        .replace("/", " ")
        .replace("\\", " ")
        .replace(".", " ")
        .replace("_", " ")
        .replace("-", " ")
    )
    return tuple(tok for tok in re.split(r"[^a-z0-9]+", normalized) if tok)


def _column_has_all(tokens, wanted):
    token_set = set(tokens)
    return all(tok in token_set for tok in wanted)


def _score_echem_column(column, role):
    tokens = _echem_column_tokens(column)
    if not tokens:
        return -1

    rules = ECHEM_COLUMN_KEYWORDS[role]
    if any(_column_has_all(tokens, rejected) for rejected in rules["reject_any"]):
        return -1
    if not any(_column_has_all(tokens, required) for required in rules["required_any"]):
        return -1

    score = 10
    for preferred in rules["preferred_any"]:
        if _column_has_all(tokens, preferred):
            score += 10 + len(preferred)
    score -= max(0, len(tokens) - 3)
    return score


def _find_echem_column(columns, aliases, preferred=None, role=None):
    by_key = {_echem_column_key(col): col for col in columns}
    if preferred is not None:
        preferred_key = _echem_column_key(preferred)
        if preferred_key in by_key:
            return by_key[preferred_key]
    for alias in aliases:
        key = _echem_column_key(alias)
        if key in by_key:
            return by_key[key]
    if role is not None:
        scored = [
            (_score_echem_column(col, role), idx, col)
            for idx, col in enumerate(columns)
        ]
        scored = [item for item in scored if item[0] >= 0]
        if scored:
            scored.sort(key=lambda item: (-item[0], item[1]))
            return scored[0][2]
    return None


def _echem_separator_for_path(path):
    return "," if str(path).lower().endswith(".csv") else "\t"


def _read_echem_table(path, nrows=None):
    sep = _echem_separator_for_path(path)
    df = pd.read_csv(path, sep=sep, engine="python", nrows=nrows)
    df.columns = [_clean_echem_column_name(col) for col in df.columns]
    unnamed_cols = [
        col for col in df.columns
        if not col or str(col).lower().startswith("unnamed")
    ]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def _read_echem_tail_table(path, nrows=20):
    sep = _echem_separator_for_path(path)
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        header = handle.readline()
        tail_lines = deque(maxlen=nrows)
        for line in handle:
            if line.strip():
                tail_lines.append(line)
    if not header or not tail_lines:
        return pd.DataFrame()
    text = header + "".join(tail_lines)
    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    df.columns = [_clean_echem_column_name(col) for col in df.columns]
    unnamed_cols = [
        col for col in df.columns
        if not col or str(col).lower().startswith("unnamed")
    ]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
    return df


def _normalize_echem_table(
    df,
    path,
    time_column="time/s",
    include_cycle=False,
    cycle_column="cycle number",
    require_current=True,
    require_capacity=True,
    require_cycle=True,
):
    time_col = _find_echem_column(df.columns, ECHEM_COLUMN_ALIASES["time"], preferred=time_column, role="time")
    potential_col = _find_echem_column(df.columns, ECHEM_COLUMN_ALIASES["potential"], preferred="Ewe/V", role="potential")
    current_col = _find_echem_column(df.columns, ECHEM_COLUMN_ALIASES["current"], preferred="<I>/mA", role="current")
    capacity_col = _find_echem_column(df.columns, ECHEM_COLUMN_ALIASES["capacity"], preferred="Capacity/mA.h", role="capacity")

    required = {
        time_column: time_col,
        "Ewe/V": potential_col,
    }
    if require_capacity:
        required["Capacity/mA.h"] = capacity_col
    if require_current:
        required["<I>/mA"] = current_col
    missing = [out_col for out_col, in_col in required.items() if in_col is None]
    if missing:
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Missing required columns {missing} in {path}. Available columns: {available}")

    source_cols = [time_col, potential_col]
    output_cols = [time_column, "Ewe/V"]
    if capacity_col is not None:
        source_cols.append(capacity_col)
        output_cols.append("Capacity/mA.h")
    if current_col is not None:
        source_cols.append(current_col)
        output_cols.append("<I>/mA")

    cycle_col = None
    if include_cycle:
        cycle_col = _find_echem_column(df.columns, ECHEM_COLUMN_ALIASES["cycle"], preferred=cycle_column, role="cycle")
        if cycle_col is None and require_cycle:
            raise ValueError(f"Missing required column {cycle_column!r} in {path}")
        if cycle_col is not None:
            source_cols.append(cycle_col)
            output_cols.append(cycle_column)

    subset = df[source_cols].copy()
    subset.columns = output_cols
    if capacity_col is None and not require_capacity:
        subset["Capacity/mA.h"] = np.nan
    if include_cycle and cycle_col is None and not require_cycle:
        subset[cycle_column] = np.nan
    subset = subset.dropna(subset=[time_column])
    return subset


def _echem_datetimes_from_values(values):
    values = list(values)
    if len(values) == 0:
        return []
    sample_val = next((val for val in values if pd.notna(val)), None)
    if sample_val is None:
        return []
    if isinstance(sample_val, str):
        return [parse_echem_datetime_value(val) for val in values if pd.notna(val)]
    start_time = datetime(2000, 1, 1)
    return [
        start_time + timedelta(seconds=float(t))
        for t in values
        if pd.notna(t)
    ]


def _echem_file_time_range(path, time_column="time/s", probe_rows=20):
    head_df = _normalize_echem_table(
        _read_echem_table(path, nrows=probe_rows),
        path,
        time_column=time_column,
        include_cycle=False,
        require_current=False,
        require_capacity=False,
    )
    tail_df = _normalize_echem_table(
        _read_echem_tail_table(path, nrows=probe_rows),
        path,
        time_column=time_column,
        include_cycle=False,
        require_current=False,
        require_capacity=False,
    )
    probe = pd.concat([head_df, tail_df], ignore_index=True)
    file_datetimes = _echem_datetimes_from_values(probe[time_column])
    if not file_datetimes:
        raise ValueError(f"No parseable timestamps found in {path}")
    return min(file_datetimes), max(file_datetimes)


def _iter_echem_files(root_folder):
    if os.path.isfile(root_folder):
        if str(root_folder).lower().endswith(ECHEM_EXTENSIONS):
            yield os.fspath(root_folder)
        else:
            raise ValueError(f"Electrochemistry input must be a TXT or CSV file: {root_folder}")
        return
    for root, directories, files in os.walk(root_folder):
        directories.sort()
        for fname in sorted(files):
            if fname.lower().endswith(ECHEM_EXTENSIONS):
                yield os.path.join(root, fname)


def load_images(
    root_folder,
    datetime_formats=("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"),
    r0=None,
    x0=None,
    y0=None
):
    """
    Recursively load all TIFF and FITS images in a folder tree,
    extract datetimes from metadata, optionally crop to a square region,
    sort them, and return images + datetimes.

    Parameters
    ----------
    root_folder : str
        Root directory containing images (including subfolders)
    datetime_formats : tuple of str
        Accepted datetime string formats
    r0 : int, optional
        Half side length of the square crop region
    x0 : int, optional
        X index of the crop center
    y0 : int, optional
        Y index of the crop center

    Returns
    -------
    datetime_series : list of datetime.datetime
        Corresponding datetimes
    img_series : list of np.ndarray
        2D image arrays (cropped if r0, x0, y0 are provided)
    paths : list of str
        File paths (useful for debugging / provenance)
    """

    records = []

    for root, _, files in os.walk(root_folder):
        for fname in files:
            path = os.path.join(root, fname)
            ext = fname.lower().split(".")[-1]

            img = None
            dt = None

            # ---- TIFF files ----
            if ext in ("tif", "tiff"):
                try:
                    img = tifffile.imread(path)
                except Exception:
                    img = np.array(Image.open(path))

                # Extract datetime from TIFF metadata
                try:
                    with tifffile.TiffFile(path) as tif:
                        tags = tif.pages[0].tags
                        for key in ("DateTime", "DateTimeOriginal"):
                            if key in tags:
                                dt_str = tags[key].value
                                for fmt in datetime_formats:
                                    try:
                                        dt = datetime.strptime(dt_str, fmt)
                                        break
                                    except ValueError:
                                        pass
                            if dt is not None:
                                break
                except Exception:
                    pass

                # Fallback: PIL EXIF
                if dt is None:
                    try:
                        with Image.open(path) as im:
                            exif = im.getexif()
                            if exif:
                                dt_str = exif.get(306)  # DateTime
                                if dt_str:
                                    for fmt in datetime_formats:
                                        try:
                                            dt = datetime.strptime(dt_str, fmt)
                                            break
                                        except ValueError:
                                            pass
                    except Exception:
                        pass

            # ---- FITS files ----
            elif ext == "fits":
                try:
                    with fits.open(path) as hdul:
                        img = hdul[0].data
                        header = hdul[0].header
                        # Try common datetime header keywords
                        for key in ("DATE-OBS", "DATE"):
                            if key in header:
                                dt_str = header[key]
                                for fmt in datetime_formats:
                                    try:
                                        dt = datetime.strptime(dt_str, fmt)
                                        break
                                    except ValueError:
                                        pass
                            if dt is not None:
                                break
                except Exception:
                    raise ValueError(f"Failed to read FITS file: {path}")

            else:
                # Skip unsupported file types
                continue

            if img is None or dt is None:
                raise ValueError(f"No valid image or datetime found in: {path}")

            # ---- Crop image if requested ----
            if r0 is not None and x0 is not None and y0 is not None:
                y_start = max(0, y0 - r0)
                y_end = min(img.shape[0], y0 + r0)
                x_start = max(0, x0 - r0)
                x_end = min(img.shape[1], x0 + r0)
                img = img[y_start:y_end, x_start:x_end]

            records.append((dt, img, path))

    # ---- Sort by datetime ----
    records.sort(key=lambda x: x[0])

    datetime_series = [r[0] for r in records]
    img_series = [r[1] for r in records]
    paths = [r[2] for r in records]

    return datetime_series, img_series, paths


def load_echem_data(
    root_folder,
    conv_thickness=3.66,
    time_column="time/s",
    report_summary=True,
    include_cycle=False,
    cycle_column="cycle number",
    window_start=None,
    window_end=None,
    thickness_capacity_mode="current_sign_half_cycle",
    include_current=False,
    include_capacity=False,
    include_provenance=False,
    zero_current_ma=0.01,
):
    """
    Load one electrochemical text file or recursively load a folder tree.

    By default, the electrochemical thickness series is computed from a
    continuous net transferred capacity. Counter increments at negative current
    add charge, positive current subtracts charge and rest holds charge. Local
    capacity-counter resets never reset the accumulated charge.
    Files without a capacity column can still be loaded for potential-vs-time
    plots; their electrochemical thickness values are returned as NaN.

    Supports time either as seconds ('time/s') or as datetime strings
    with optional sub-second precision.

    Parameters
    ----------
    root_folder : str
        A TXT/CSV file or directory containing electrochemical text files.
    conv_thickness : float
        Factor used to convert ``Capacity/mA.h`` into electrochemical thickness.
    time_column : str
        Column name containing the time (seconds or datetime string)
    report_summary : bool, optional
        If True, print a lightweight validation summary including file ranges,
        duplicate timestamps, and overlaps/gaps between files.
    window_start, window_end : datetime.datetime, optional
        When supplied, only files whose time range overlaps this window are
        loaded. This is useful when ``root_folder`` contains multiple
        experiments and cycle offsets should start from the first file used in
        the current image window.
    thickness_capacity_mode : {"current_sign_half_cycle", "potential_sign_half_cycle", "raw"}, optional
        Capacity series used for electrochemical thickness conversion. ``"raw"``
        keeps the historical behavior: ``Capacity/mA.h * conv_thickness``.

    Returns
    -------
    datetime_series : list of datetime.datetime
        Time series as datetime objects
    potential_series : np.ndarray
        Ewe/V series (potential)
    echem_thickness_series : np.ndarray
        Electrochemical thickness series calculated from capacity.
    cycle_series : np.ndarray, optional
        Returned only when ``include_cycle=True``. Values are read from
        ``cycle_column`` when that column is present. Files are ordered by
        timestamp, and each later file's displayed cycle values are offset by
        the previous file's last displayed cycle value.
    """

    file_records = []
    file_summaries = []
    requested_thickness_capacity_mode = str(thickness_capacity_mode or "current_sign_half_cycle").lower()

    skipped_by_window = 0
    for path in _iter_echem_files(root_folder):
        try:
            file_start, file_end = _echem_file_time_range(path, time_column=time_column)
            if window_start is not None and file_end < window_start:
                skipped_by_window += 1
                continue
            if window_end is not None and file_start > window_end:
                skipped_by_window += 1
                continue

            df = _read_echem_table(path)
            subset = _normalize_echem_table(
                df,
                path,
                time_column=time_column,
                include_cycle=include_cycle,
                cycle_column=cycle_column,
                require_current=False,
                require_capacity=False,
                require_cycle=False,
            )
            if len(subset) == 0:
                raise ValueError(f"No data rows in {path}")
            subset["__source_file"] = os.path.basename(path)

            file_datetimes = _echem_datetimes_from_values(subset[time_column])
            if len(file_datetimes) == 0:
                raise ValueError(f"No parseable timestamps found in {path}")

            capacity_values = subset["Capacity/mA.h"].to_numpy(dtype=float)
            file_start = min(file_datetimes)
            file_end = max(file_datetimes)

            file_record = {
                "file": path,
                "rows": len(subset),
                "start": file_start,
                "end": file_end,
                "capacity_min": float(np.nanmin(capacity_values)) if np.isfinite(capacity_values).any() else np.nan,
                "capacity_max": float(np.nanmax(capacity_values)) if np.isfinite(capacity_values).any() else np.nan,
                "data": subset,
                "cycle_offset": 0.0,
                "cycle_min": None,
                "cycle_max": None,
                "display_cycle_min": None,
                "display_cycle_max": None,
            }
            if include_cycle:
                cycle_values = subset[cycle_column].to_numpy(dtype=float)
                file_record["cycle_min"] = float(np.nanmin(cycle_values))
                file_record["cycle_max"] = float(np.nanmax(cycle_values))

            file_records.append(file_record)
        except Exception as e:
            print(f"Skipping {path} due to error: {e}")

    if not file_records:
        raise ValueError("No valid electrochemical data found in the folder tree.")

    file_records.sort(key=lambda x: (x["start"], x["file"]))

    if include_cycle:
        last_display_cycle = 0.0
        for record in file_records:
            subset = record["data"].copy()
            raw_cycle_values = subset[cycle_column].to_numpy(dtype=float)
            finite_raw_cycles = raw_cycle_values[np.isfinite(raw_cycle_values)]
            cycle_offset = last_display_cycle if len(finite_raw_cycles) else 0.0
            display_cycle_values = raw_cycle_values + cycle_offset
            subset[cycle_column] = display_cycle_values
            record["data"] = subset
            record["cycle_offset"] = float(cycle_offset)
            finite_display_cycles = display_cycle_values[np.isfinite(display_cycle_values)]
            if len(finite_display_cycles):
                record["display_cycle_min"] = float(np.nanmin(finite_display_cycles))
                record["display_cycle_max"] = float(np.nanmax(finite_display_cycles))
            else:
                record["display_cycle_min"] = np.nan
                record["display_cycle_max"] = np.nan
            finite_display_cycles = display_cycle_values[np.isfinite(display_cycle_values)]
            if len(finite_display_cycles):
                last_display_cycle = float(finite_display_cycles[-1])

    all_data = [record["data"] for record in file_records]
    file_summaries = [
        {
            key: record[key]
            for key in (
                "file",
                "rows",
                "start",
                "end",
                "capacity_min",
                "capacity_max",
                "cycle_offset",
                "cycle_min",
                "cycle_max",
                "display_cycle_min",
                "display_cycle_max",
            )
        }
        for record in file_records
    ]

    # Concatenate all data
    combined = pd.concat(all_data, ignore_index=True)

    # ---- Parse time column ----
    sample_val = combined[time_column].iloc[0]

    datetime_series = []
    if isinstance(sample_val, str):
        # Datetime string with optional sub-second precision (DD/MM/YYYY)
        for val in combined[time_column]:
            datetime_series.append(parse_echem_datetime_value(val))
    else:
        # Numeric seconds
        start_time = datetime(2000, 1, 1)  # arbitrary reference
        datetime_series = [
            start_time + timedelta(seconds=float(t))
            for t in combined[time_column]
        ]

    # ---- Sort by datetime ----
    sorted_idx = np.argsort(datetime_series, kind="stable")

    datetime_series = [datetime_series[i] for i in sorted_idx]
    potential_series = combined["Ewe/V"].to_numpy(dtype=float)[sorted_idx]
    current_series = None
    if "<I>/mA" in combined.columns:
        current_series = combined["<I>/mA"].to_numpy(dtype=float)[sorted_idx]
    raw_capacity_series = combined["Capacity/mA.h"].to_numpy(dtype=float)[sorted_idx]
    source_series = combined["__source_file"].to_numpy()[sorted_idx]
    thickness_capacity_mode = requested_thickness_capacity_mode
    if thickness_capacity_mode in {"raw", "direct"}:
        thickness_capacity_series = raw_capacity_series
    elif thickness_capacity_mode in {"current_sign_half_cycle", "current_half_cycle", "current_sign"}:
        if current_series is None:
            print(
                "EC current column was not found; using potential_sign_half_cycle "
                "for electrochemical thickness instead of current_sign_half_cycle."
            )
            thickness_capacity_mode = "potential_sign_half_cycle"
            thickness_capacity_series = capacity_from_potential_sign_half_cycles(
                raw_capacity_series,
                potential_series,
            )
        else:
            thickness_capacity_series = capacity_from_current_sign_half_cycles(
                raw_capacity_series,
                current_series,
                zero_current=zero_current_ma, source_ids=source_series,
            )
    elif thickness_capacity_mode in {"potential_sign_half_cycle", "sign_half_cycle", "half_cycle"}:
        thickness_capacity_series = capacity_from_potential_sign_half_cycles(
            raw_capacity_series,
            potential_series,
        )
    else:
        raise ValueError(
            "thickness_capacity_mode must be 'current_sign_half_cycle', "
            "'potential_sign_half_cycle', or 'raw'."
        )
    echem_thickness_series = thickness_capacity_series * conv_thickness
    if include_cycle:
        cycle_series = combined[cycle_column].to_numpy(dtype=float)[sorted_idx]

    if report_summary:
        print("Electrochemical data summary:")
        print(f"  Loaded files: {len(file_summaries)}")
        if window_start is not None or window_end is not None:
            print(f"  Files skipped outside image window: {skipped_by_window}")
        for summary in sorted(file_summaries, key=lambda x: x["start"]):
            cycle_text = ""
            if include_cycle:
                cycle_text = (
                    f" | raw cycles {summary['cycle_min']:g} to {summary['cycle_max']:g}"
                    f" | displayed cycles {summary['display_cycle_min']:g} to {summary['display_cycle_max']:g}"
                    f" | offset {summary['cycle_offset']:g}"
                )
            print(
                "  - "
                f"{os.path.basename(summary['file'])}: "
                f"{summary['start']} to {summary['end']} | "
                f"rows={summary['rows']} | "
                f"Capacity range {summary['capacity_min']:.6f} to {summary['capacity_max']:.6f} mA.h"
                f"{cycle_text}"
            )

        duplicate_timestamp_count = int(
            pd.Series(pd.to_datetime(datetime_series)).duplicated().sum()
        )
        print(
            f"  Combined range: {min(datetime_series)} to {max(datetime_series)} | "
            f"points={len(datetime_series)}"
        )
        print(f"  Duplicate timestamps: {duplicate_timestamp_count}")
        print(
            f"  Thickness capacity mode: {thickness_capacity_mode} | "
            f"conv_thickness={conv_thickness:g}"
        )

        sorted_summaries = sorted(file_summaries, key=lambda x: x["start"])
        overlaps = []
        gaps = []
        for prev, curr in zip(sorted_summaries[:-1], sorted_summaries[1:]):
            if curr["start"] <= prev["end"]:
                overlaps.append((prev, curr))
            else:
                gap = curr["start"] - prev["end"]
                gaps.append((prev, curr, gap))

        if overlaps:
            print("  Overlapping file ranges detected:")
            for prev, curr in overlaps:
                print(
                    "    "
                    f"{os.path.basename(prev['file'])} overlaps "
                    f"{os.path.basename(curr['file'])}"
                )
        else:
            print("  Overlapping file ranges: none")

        if gaps:
            print("  Gaps between files:")
            for prev, curr, gap in gaps:
                print(
                    "    "
                    f"{os.path.basename(prev['file'])} -> "
                    f"{os.path.basename(curr['file'])}: {gap}"
                )
        else:
            print("  Gaps between files: none")

    values = (datetime_series, potential_series, echem_thickness_series)
    if include_cycle:
        values += (cycle_series,)
    if include_current:
        values += (current_series,)
    if include_capacity:
        values += (raw_capacity_series,)
    if include_provenance:
        values += ({"source_files": source_series, "capacity_mode": thickness_capacity_mode,
                    "net_capacity_mAh": thickness_capacity_series,
                    "zero_current_ma": zero_current_ma},)
    return values



def summarize_echem_files_in_window(root_folder, window_start, window_end, time_column="time/s"):
    """
    List electrochemical files whose timestamp ranges overlap a given time window.

    Parameters
    ----------
    root_folder : str
        Root directory containing electrochemical txt files.
    window_start : datetime.datetime
        Start of the time window of interest.
    window_end : datetime.datetime
        End of the time window of interest.
    time_column : str
        Column name containing the time values.

    Returns
    -------
    list of dict
        One summary dict per overlapping file.
    """
    used_files = []

    for path in _iter_echem_files(root_folder):
        try:
            file_start, file_end = _echem_file_time_range(path, time_column=time_column)
            overlaps_window = file_start <= window_end and file_end >= window_start
            if not overlaps_window:
                continue

            df = _read_echem_table(path)
            subset = _normalize_echem_table(
                df,
                path,
                time_column=time_column,
                include_cycle=False,
                require_current=False,
                require_capacity=False,
            )
            file_datetimes = _echem_datetimes_from_values(subset[time_column])
            if len(file_datetimes) == 0:
                continue

            used_files.append(
                {
                    "file": path,
                    "rows": len(subset),
                    "start": min(file_datetimes),
                    "end": max(file_datetimes),
                    "points_in_window": sum(window_start <= dt <= window_end for dt in file_datetimes),
                }
            )
        except Exception:
            continue

    used_files.sort(key=lambda x: x["start"])
    return used_files



def make_nci_image_video_uniform_time(
    img_datetimes,
    img_series,
    output_path,
    fps=10,
    dt_seconds=None,
    dpi=150,
    cmap="inferno",
    vmin=None,
    vmax=None,
    pixel_size=55,
    scalebar_length=5000,
    scale_label="5 mm",
    cbar_label="Li thickness",
    cbar_n_ticks=5,
    scalebar_color="white",
    render_3d=False,
    surface_stride=2,
    surface_elev=35,
    surface_azim=-60,
    surface_zlim=None,
    surface_zlabel=None,
    **_unused,
):
    """Create a timestamp-driven NCI-only video without electrochemistry."""
    if len(img_datetimes) == 0 or len(img_series) == 0:
        raise ValueError("No image timestamps/images were loaded.")
    scalebar_length = normalize_scalebar_length_um(scalebar_length, scale_label)
    t0 = min(img_datetimes)
    img_t = np.asarray([(value - t0).total_seconds() for value in img_datetimes], dtype=float)
    if dt_seconds is None:
        positive_steps = np.diff(np.unique(img_t))
        dt_seconds = float(np.median(positive_steps)) if positive_steps.size else 1.0
    dt_seconds = float(dt_seconds)
    if dt_seconds <= 0:
        raise ValueError("dt_seconds must be positive.")
    end_time = float(np.max(img_t))
    anim_t = np.arange(0.0, end_time + 0.5 * dt_seconds, dt_seconds)
    if anim_t.size == 0:
        anim_t = np.asarray([0.0])

    first = np.asarray(img_series[0])
    ny, nx = first.shape
    x_max = nx * float(pixel_size)
    y_max = ny * float(pixel_size)
    fig = plt.figure(figsize=(6.2, 5.4))
    if render_3d:
        ax = fig.add_subplot(111, projection="3d")
        stride = max(1, int(surface_stride))
        x = np.linspace(0, x_max, nx)[::stride]
        y = np.linspace(0, y_max, ny)[::stride]
        xx, yy = np.meshgrid(x, y)
        surface = {"artist": None}

        def draw(image):
            if surface["artist"] is not None:
                surface["artist"].remove()
            surface["artist"] = ax.plot_surface(
                xx,
                yy,
                np.asarray(image)[::stride, ::stride],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidth=0,
                antialiased=True,
            )
            return surface["artist"]

        artist = draw(first)
        ax.view_init(elev=surface_elev, azim=surface_azim)
        if surface_zlim is not None:
            ax.set_zlim(surface_zlim)
        ax.set_zlabel(surface_zlabel or "")
    else:
        ax = fig.add_subplot(111)
        artist = ax.imshow(
            first,
            cmap=cmap,
            origin="lower",
            extent=[0, x_max, 0, y_max],
            vmin=vmin,
            vmax=vmax,
        )
        scale_x = max(0.0, x_max - scalebar_length - 0.06 * x_max)
        scale_y = 0.07 * y_max
        ax.plot(
            [scale_x, scale_x + scalebar_length],
            [scale_y, scale_y],
            color=scalebar_color,
            linewidth=4,
        )
        ax.text(
            scale_x + scalebar_length / 2,
            scale_y + 0.025 * y_max,
            scale_label,
            color=scalebar_color,
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    colorbar = fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(cbar_label)
    if vmin is not None and vmax is not None:
        colorbar.set_ticks(np.linspace(vmin, vmax, int(cbar_n_ticks)))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, dpi=dpi):
        for current in anim_t:
            image_index = int(np.argmin(np.abs(img_t - current)))
            if render_3d:
                draw(img_series[image_index])
            else:
                artist.set_data(img_series[image_index])
            ax.set_title(f"NCI image @ {img_datetimes[image_index].strftime('%Y-%m-%d %H:%M:%S')}")
            writer.grab_frame()
    plt.close(fig)
    return output_path


def make_potential_image_video_uniform_time(
    echem_datetimes,
    potential_series,
    img_datetimes,
    img_series,
    output_path,
    echem_thickness_series=None,
    avg_thickness_series=None,
    plot_thickness=False,
    fps=10,
    dt_seconds=None,
    dpi=150,
    cmap="inferno",
    vmin=None,
    vmax=None,
    potential_ylim=None,
    pixel_size=55,  # Pixel size in micrometers
    scalebar_length=5000,  # Length of scale bar in micrometers
    scale_label="5 mm",  # Label for scale bar
    colorbar_label=None,  # Backward-compatible alias for cbar_label
    cbar_label="Li thickness",  # Label for colorbar
    cbar_n_ticks=5,
    # Additional scale bar customization options
    scalebar_color='white',
    scale_fontsize=12,
    scalebar_width_frac=0.015,  # Height of scale bar as fraction of image height
    scalebar_position='lower right',  # 'lower right', 'lower left', 'upper right', 'upper left'
    scalebar_margin=0.1,  # Margin from edge as fraction of image dimensions
    scalebar_x_frac=None,  # Accepted for compatibility with still-image plot kwargs
    scalebar_y_frac=None,  # Accepted for compatibility with still-image plot kwargs
    pause_gap_minutes=5,
    potential_color="tab:blue",
    ec_thickness_color="tab:green",
    nci_thickness_color="tab:orange",
    thickness_unit="μm",
    nci_trace_label="NCI thickness",
    thickness_axis_label=None,
    thickness_traces="both",
    thickness_layout="overlay",
    overlay_offset_fraction=0.0,
    render_3d=False,
    surface_stride=2,
    surface_elev=35,
    surface_azim=-60,
    surface_zlim=None,
    surface_zlabel=None,
    electrode_roi=None,
    show_electrode_roi=False,
    electrode_roi_label="electrode area",
    electrode_roi_color="cyan",
    electrode_roi_linewidth=1.0,
    limit_to_overlap=False,
    image_origin="lower",
    time_window=None,
    image_gap_minutes=5,
    image_tolerance_seconds=None,
):
    """
    Create a synchronized potential-plus-image video on a uniform time grid.

    The electrochemical trace is sampled by zero-order hold between measured
    points. Each video frame displays the nearest image in time. Optional EC
    thickness and NCI thickness traces are drawn on the same left-hand panel.
    """

    if colorbar_label is not None:
        cbar_label = colorbar_label
    scalebar_length = normalize_scalebar_length_um(scalebar_length, scale_label)

    if len(img_datetimes) == 0 or len(img_series) == 0:
        raise ValueError(
            "No image timestamps/images were loaded. Check the input directory and TIFF metadata."
        )

    # ------------------------------------------------------------------
    # Limit potential series to the time range covered by images
    # ------------------------------------------------------------------
    img_start = min(img_datetimes)
    img_end = max(img_datetimes)
    echem_mask = [(t >= img_start) and (t <= img_end) for t in echem_datetimes]
    echem_datetimes = [t for t, m in zip(echem_datetimes, echem_mask) if m]
    echem_mask_arr = np.array(echem_mask, dtype=bool)
    potential_series = np.asarray(potential_series)[echem_mask_arr]
    selected_thickness_traces = normalize_thickness_traces(thickness_traces, plot_thickness)
    plot_thickness = bool(selected_thickness_traces)
    if plot_thickness:
        if "echem" in selected_thickness_traces:
            if echem_thickness_series is None:
                raise ValueError("EC thickness plotting in the video requires echem_thickness_series.")
            echem_thickness_series = np.asarray(echem_thickness_series)[echem_mask_arr]
        if "nci" in selected_thickness_traces:
            if avg_thickness_series is None:
                raise ValueError("NCI thickness plotting in the video requires avg_thickness_series.")
            avg_thickness_series = np.asarray(avg_thickness_series)

    if len(echem_datetimes) == 0:
        raise ValueError("No electrochemical data within image time range.")

    # ------------------------------------------------------------------
    # Convert datetimes to seconds (for speed & simplicity)
    # ------------------------------------------------------------------
    t0 = min(min(echem_datetimes), min(img_datetimes))

    echem_t = np.array([(t - t0).total_seconds() for t in echem_datetimes])
    img_t = np.array([(t - t0).total_seconds() for t in img_datetimes])
    
    # Define echem_t_h HERE before it's used
    echem_t_h = echem_t / 3600.0

    paused_intervals = find_paused_intervals(
        echem_datetimes,
        min_gap_minutes=pause_gap_minutes,
    )

    # ------------------------------------------------------------------
    # Define uniform animation time grid
    # ------------------------------------------------------------------
    t_start = min(echem_t.min(), img_t.min())
    t_end = max(echem_t.max(), img_t.max())
    if limit_to_overlap:
        t_start = max(echem_t.min(), img_t.min())
        t_end = min(echem_t.max(), img_t.max())
        if t_end <= t_start:
            raise ValueError("Combined video needs a nonzero shared image/electrochemistry time range.")
    window_indices = np.arange(len(img_t))
    if time_window is not None:
        t_start = max(t_start, (time_window[0]-t0).total_seconds())
        t_end = min(t_end, (time_window[1]-t0).total_seconds())
        window_indices = np.flatnonzero((img_t >= t_start) & (img_t <= t_end))
        if t_end <= t_start or not len(window_indices):
            raise ValueError("Video window needs a nonzero overlap and at least one image frame.")

    if dt_seconds is None:
        # sensible default: median image spacing
        positive_steps = np.diff(img_t)
        positive_steps = positive_steps[positive_steps > 0]
        dt_seconds = np.median(positive_steps) if positive_steps.size else 1.0
    dt_seconds = float(dt_seconds)
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError("Combined video dt_seconds must be positive and finite.")

    anim_t = np.arange(t_start, t_end, dt_seconds)

    # ------------------------------------------------------------------
    # Helper lookups
    # ------------------------------------------------------------------
    def is_paused_time(t):
        current_dt = t0 + timedelta(seconds=float(t))
        return any(start < current_dt < end for start, end in paused_intervals)

    def potential_at(t):
        """Zero-order hold potential outside paused intervals."""
        if is_paused_time(t):
            return np.nan
        i = bisect_right(echem_t, t) - 1
        return potential_series[i] if i >= 0 else potential_series[0]

    def nearest_image_index(t):
        if time_window is not None:
            return int(window_indices[np.argmin(abs(img_t[window_indices]-t))])
        i = bisect_right(img_t, t)
        if i == 0:
            return 0
        if i == len(img_t):
            return len(img_t) - 1
        before = img_t[i - 1]
        after = img_t[i]
        return i - 1 if abs(t - before) <= abs(after - t) else i

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    potential_panel_aspect = 8.0 / 4.5
    image_panel_aspect = 1.0
    fig_height = 4.5
    fig_width = fig_height * (potential_panel_aspect + image_panel_aspect)

    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[potential_panel_aspect, image_panel_aspect],
    )
    ax_v = fig.add_subplot(gs[0, 0])
    if render_3d:
        ax_img = fig.add_subplot(gs[0, 1], projection="3d")
    else:
        ax_img = fig.add_subplot(gs[0, 1])

    # Potential curve (static), broken across pauses.
    echem_t_plot, potential_plot = build_series_with_pauses(
        echem_t_h,
        potential_series,
        paused_intervals,
        t0,
    )
    ax_v.plot(echem_t_plot, potential_plot, lw=1.5, color=potential_color, label="Potential (Ewe/V)")
    marker, = ax_v.plot([], [], "ro", ms=6)
    paused_text_func = ax_img.text2D if render_3d else ax_img.text
    paused_text = paused_text_func(
        0.5,
        0.5,
        "paused",
        transform=ax_img.transAxes,
        color="white",
        fontsize=max(scale_fontsize, 14),
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=6),
        visible=False,
        zorder=11,
    )

    ax_v.set_xlabel("Time (h)")
    if time_window is not None:
        ax_v.set_xlim(t_start/3600., t_end/3600.)
    ax_v.set_ylabel("Potential (V)")
    ax_v.set_title("Potential evolution")
    if potential_ylim is not None:
        ax_v.set_ylim(potential_ylim)

    if plot_thickness:
        ax_v2 = ax_v.twinx()
        thickness_plot_data = []
        if "echem" in selected_thickness_traces:
            t_echem_thickness_plot, echem_thickness_plot = build_series_with_pauses(
                echem_t_h,
                echem_thickness_series,
                paused_intervals,
                t0,
            )
            thickness_plot_data.append(
                ("EC thickness", t_echem_thickness_plot, echem_thickness_plot, ec_thickness_color)
            )
        if "nci" in selected_thickness_traces:
            t_img_plot, nci_thickness_plot = build_datetime_series_with_pauses(
                img_datetimes,
                avg_thickness_series,
                find_paused_intervals(img_datetimes,min_gap_minutes=image_gap_minutes),
                t0,
            )
            thickness_plot_data.append(
                (nci_trace_label, t_img_plot, nci_thickness_plot, nci_thickness_color)
            )

        offset_data = apply_vertical_offsets_to_traces(
            [(label, values) for label, _, values, _ in thickness_plot_data],
            offset_fraction=overlay_offset_fraction,
        )
        offsets_by_label = {label: offset for label, _, offset in offset_data}
        values_by_label = {label: values for label, values, _ in offset_data}

        for label, x_values, _y_values, color in thickness_plot_data:
            offset = offsets_by_label[label]
            plot_label = label
            if abs(offset) > 0:
                plot_label = f"{label} ({offset:+.3g} {thickness_unit} offset)"
            ax_v2.plot(
                x_values,
                values_by_label[label],
                lw=1.2,
                color=color,
                label=plot_label,
            )
        axis_label = thickness_axis_label or f"Thickness ({thickness_unit})"
        ax_v2.set_ylabel(axis_label)
        lines_1, labels_1 = ax_v.get_legend_handles_labels()
        lines_2, labels_2 = ax_v2.get_legend_handles_labels()
        ax_v.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax_v.legend(loc="best")

    for pause_start, pause_end in paused_intervals:
        pause_start_h = (pause_start - t0).total_seconds() / 3600.0
        pause_end_h = (pause_end - t0).total_seconds() / 3600.0
        if time_window is not None and (pause_end_h < t_start/3600. or pause_start_h > t_end/3600.):
            continue
        ax_v.axvspan(pause_start_h, pause_end_h, color="tab:red", alpha=0.18)
        ylim = ax_v.get_ylim()
        y_text = ylim[1] - 0.08 * (ylim[1] - ylim[0])
        ax_v.text(
            0.5 * (pause_start_h + pause_end_h),
            y_text,
            "EC data gap",
            color="tab:red",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="top",
            clip_on=True,
        )

    # Get image dimensions for physical extent
    first_img = img_series[0]
    ny, nx = first_img.shape
    
    # Calculate physical extent in micrometers
    x_min = 0
    x_max = nx * pixel_size
    y_min = 0
    y_max = ny * pixel_size

    stride = max(1, int(surface_stride))
    x_plot = np.linspace(x_min, x_max, nx)[::stride]
    y_plot = np.linspace(y_min, y_max, ny)[::stride]
    x_grid, y_grid = np.meshgrid(x_plot, y_plot)
    surface_artist = {"artist": None}

    def draw_video_image(image):
        if render_3d:
            if surface_artist["artist"] is not None:
                surface_artist["artist"].remove()
            image_plot = np.asarray(image)[::stride, ::stride]
            surface_artist["artist"] = ax_img.plot_surface(
                x_grid,
                y_grid,
                image_plot,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidth=0,
                antialiased=True,
                shade=True,
            )
            return surface_artist["artist"]
        return ax_img.imshow(
            image,
            cmap=cmap,
            extent=[x_min, x_max, y_min, y_max],
            origin=image_origin,
            vmin=vmin,
            vmax=vmax,
        )

    im = draw_video_image(img_series[0])

    ax_img.set_xticks([])
    ax_img.set_yticks([])
    if render_3d:
        ax_img.set_zticks([])
        ax_img.set_xlim(x_min, x_max)
        ax_img.set_ylim(y_min, y_max)
        if surface_zlim is not None:
            ax_img.set_zlim(surface_zlim)
        elif vmin is not None and vmax is not None:
            ax_img.set_zlim(vmin, vmax)
        ax_img.view_init(elev=surface_elev, azim=surface_azim)
        ax_img.set_xlabel("")
        ax_img.set_ylabel("")
        ax_img.set_zlabel(surface_zlabel or "")
        ax_img.grid(False)
        ax_img.set_box_aspect((1, 1, 0.45))
    else:
        ax_img.set_xticklabels([])
        ax_img.set_yticklabels([])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    
    # Set ticks - exactly like in plot_images()!
    if vmin is not None and vmax is not None:
        ticks = np.linspace(vmin, vmax, cbar_n_ticks)
        cbar.set_ticks(ticks)
        
        # Simple tick labels without units (units are in the colorbar label)
        labels = []
        for t in ticks:
            if np.isclose(t, 0):
                labels.append("0")
            else:
                labels.append(f"{t:g}")
        cbar.set_ticklabels(labels)
    
    # ------------------------------------------------------------------
    # Create scale bar parameters (but don't add it yet)
    # ------------------------------------------------------------------
    
    # Determine scale bar position based on user preference
    if scalebar_position == 'lower right':
        scale_x = x_max - scalebar_length - scalebar_margin * (x_max - x_min)
        scale_y = y_min + scalebar_margin * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    elif scalebar_position == 'lower left':
        scale_x = x_min + scalebar_margin * (x_max - x_min)
        scale_y = y_min + scalebar_margin * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    elif scalebar_position == 'upper right':
        scale_x = x_max - scalebar_length - scalebar_margin * (x_max - x_min)
        scale_y = y_max - scalebar_margin * (y_max - y_min) - scalebar_width_frac * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    elif scalebar_position == 'upper left':
        scale_x = x_min + scalebar_margin * (x_max - x_min)
        scale_y = y_max - scalebar_margin * (y_max - y_min) - scalebar_width_frac * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    else:
        # Default to lower right
        scale_x = x_max - scalebar_length - scalebar_margin * (x_max - x_min)
        scale_y = y_min + scalebar_margin * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    
    # Calculate scale bar height in physical units (micrometers)
    scalebar_height = scalebar_width_frac * (y_max - y_min)
    
    if render_3d:
        z_base = surface_zlim[0] if surface_zlim is not None else (vmin if vmin is not None else 0.0)
        ax_img.plot(
            [scale_x, scale_x + scalebar_length],
            [scale_y, scale_y],
            [z_base, z_base],
            color=scalebar_color,
            linewidth=3,
            zorder=10,
        )
        ax_img.text2D(
            0.12,
            0.10,
            scale_label,
            transform=ax_img.transAxes,
            color=scalebar_color,
            ha="left",
            va="bottom",
            fontsize=scale_fontsize,
            fontweight="bold",
            zorder=10,
        )
        scalebar_rect = None
        if show_electrode_roi and electrode_roi:
            roi_x0 = float(electrode_roi.get("x0", nx / 2))
            roi_y0 = float(electrode_roi.get("y0", ny / 2))
            roi_r0 = float(electrode_roi.get("r0", min(nx, ny) / 4))
            source_img = img_series[0]
            crop_x0, crop_y0 = getattr(source_img, "crop_origin_xy", (0, 0))
            bin_factor = max(1, int(getattr(source_img, "spatial_bin_factor", 1)))
            center_x = (roi_x0 / bin_factor - float(crop_x0)) * pixel_size
            center_y = (roi_y0 / bin_factor - float(crop_y0)) * pixel_size
            radius = roi_r0 / bin_factor * pixel_size
            theta = np.linspace(0, 2 * np.pi, 181)
            x_circle = center_x + radius * np.cos(theta)
            y_circle = center_y + radius * np.sin(theta)
            if surface_zlim is not None:
                z_plane = surface_zlim[0]
            else:
                z_plane = vmin if vmin is not None else 0.0
            z_circle = np.full_like(theta, z_plane, dtype=float)
            ax_img.plot(
                x_circle,
                y_circle,
                z_circle,
                color=electrode_roi_color,
                linewidth=electrode_roi_linewidth,
                zorder=30,
            )
    else:
        # Create scale bar rectangle (we'll add it in the animation loop)
        scalebar_rect = patches.Rectangle(
            (scale_x, scale_y),
            scalebar_length,
            scalebar_height,
            facecolor=scalebar_color,
            edgecolor='none',
            alpha=0.9,
            zorder=10  # High zorder to appear on top of image
        )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=dpi):
        for t in anim_t:

            # Potential (ZOH)
            v = potential_at(t)
            if np.isfinite(v):
                marker.set_data([t / 3600.0], [v])
            else:
                marker.set_data([], [])

            # Image (nearest)
            j = nearest_image_index(t)
            unavailable = (image_tolerance_seconds is not None and abs(img_t[j]-t)>float(image_tolerance_seconds))
            unavailable = unavailable or not np.isfinite(np.asarray(img_series[j])).any()
            shown = np.full_like(img_series[j],np.nan,dtype=float) if unavailable else img_series[j]
            if render_3d:
                draw_video_image(shown)
            else:
                im.set_data(shown)
            paused_text.set_text('Image unavailable' if unavailable else '')
            paused_text.set_visible(unavailable)

            if not render_3d:
                # Remove previous scale bar rectangles and text
                for artist in ax_img.patches[:]:
                    if isinstance(artist, patches.Rectangle) and hasattr(artist, 'get_width') and artist.get_width() == scalebar_length:
                        artist.remove()
                for artist in ax_img.texts[:]:
                    if scale_label in artist.get_text():
                        artist.remove()

                # Add fresh scale bar rectangle
                ax_img.add_patch(scalebar_rect)

                # Add fresh scale bar label
                ax_img.text(
                    scale_x + text_x_offset,
                    scale_y + scalebar_height * 1.5,  # Position above the scale bar
                    scale_label,
                    color=scalebar_color,
                    ha=text_ha,
                    va='bottom',
                    fontsize=scale_fontsize,
                    fontweight='bold',
                    zorder=10  # High zorder to appear on top of image
                )

            ax_img.set_title(
                f"Image @ {img_datetimes[j].strftime('%Y-%m-%d %H:%M:%S')}"
            )

            writer.grab_frame()

    plt.close(fig)


    
def merge_images_by_time(
    img_datetimes,
    img_series,
    merge_minutes=1,
    method="mean",
):
    """
    Merge images into longer effective acquisition windows.

    Parameters
    ----------
    img_datetimes : list of datetime.datetime
    img_series : list of 2D np.ndarray
    merge_minutes : float
        Duration of each merged frame in minutes
    method : str
        'mean' or 'sum'

    Returns
    -------
    merged_datetimes : list of datetime.datetime
        Timestamp at center of each merged window
    merged_images : list of 2D np.ndarray
    """

    if merge_minutes <= 1:
        return img_datetimes, img_series

    dt = timedelta(minutes=merge_minutes)

    merged_images = []
    merged_datetimes = []

    window_start = img_datetimes[0]
    buffer = []

    for t, img in zip(img_datetimes, img_series):

        if t < window_start + dt:
            buffer.append(img)
        else:
            stack = np.stack(buffer, axis=0)
            merged = stack.mean(axis=0) if method == "mean" else stack.sum(axis=0)
            merged = np.asarray(merged).view(ImageArray)
            merged.crop_origin_xy = getattr(buffer[0], 'crop_origin_xy', (0, 0))

            merged_images.append(merged)
            merged_datetimes.append(window_start + dt / 2)

            # reset window
            window_start = t
            buffer = [img]

    # flush remainder
    if buffer:
        stack = np.stack(buffer, axis=0)
        merged = stack.mean(axis=0) if method == "mean" else stack.sum(axis=0)
        merged = np.asarray(merged).view(ImageArray)
        merged.crop_origin_xy = getattr(buffer[0], 'crop_origin_xy', (0, 0))
        merged_images.append(merged)
        merged_datetimes.append(window_start + dt / 2)

    return merged_datetimes, merged_images




def circular_average_thickness_series(img_series, x0=None, y0=None, r0=None):
    """
    Compute the average pixel value within a circular region
    for each image in an image series. Silently rejects infinite and NaN values.

    If an image carries ``img.attrs['crop_origin_xy'] = (x_start, y_start)``, the
    supplied ROI center is interpreted in the original uncropped-image coordinate
    system and translated to the cropped-image coordinate system automatically.

    Parameters
    ----------
    img_series : list or np.ndarray
        List or array of 2D images
    x0 : int, optional
        X index of circle center in uncropped-image coordinates
    y0 : int, optional
        Y index of circle center in uncropped-image coordinates
    r0 : int, optional
        Radius of the circular region (in pixels)

    Returns
    -------
    avg_thickness_series : np.ndarray
        Average value inside the circular region for each image
        (returns NaN for images with no valid pixels in ROI)
    """

    avg_thickness_series = []

    for img in img_series:
        ny, nx = img.shape
        crop_origin_xy = getattr(img, 'crop_origin_xy', (0, 0))
        crop_x0, crop_y0 = crop_origin_xy

        # ---- Default ROI: center of image, maximal inscribed circle ----
        if x0 is None or y0 is None or r0 is None:
            cx = nx // 2
            cy = ny // 2
            radius = min(nx, ny) // 2
        else:
            bin_factor = max(1, int(getattr(img, "spatial_bin_factor", 1)))
            cx = int(round(float(x0) / bin_factor - float(crop_x0)))
            cy = int(round(float(y0) / bin_factor - float(crop_y0)))
            radius = max(1, int(round(float(r0) / bin_factor)))

        # ---- Create circular mask ----
        y, x = np.ogrid[:ny, :nx]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2

        # ---- Extract values in ROI and handle invalid values ----
        roi_values = img[mask].astype(float)

        # Replace inf with nan, then compute the mean only if any valid pixels remain.
        roi_values[~np.isfinite(roi_values)] = np.nan

        if np.any(np.isfinite(roi_values)):
            avg = np.nanmean(roi_values)
        else:
            avg = np.nan
        avg_thickness_series.append(avg)

    return np.asarray(avg_thickness_series)



def load_corrected_images_with_time(corrected_dir, tag_id=65002, r0=None, x0=None, y0=None):
    """
    Load corrected or final TIFF images and recover timestamps from TIFF metadata.

    When cropping is requested, each returned image keeps lightweight crop-origin
    metadata in ``ndarray.attrs`` so downstream ROI-based analysis can still use
    coordinates from the original uncropped image.

    Parameters
    ----------
    corrected_dir : str
        Directory containing corrected TIFF images.
    tag_id : int, optional
        Preferred TIFF tag ID containing the timestamp string. If it is missing,
        the loader falls back to tag 65000 and then to the TIFF description text.
    r0 : int, optional
        Half side length of the square crop region.
    x0 : int, optional
        X index of the crop center.
    y0 : int, optional
        Y index of the crop center.

    Returns
    -------
    img_datetimes : list of datetime.datetime
        Image timestamps sorted in ascending order.
    img_series : list of np.ndarray
        Loaded image arrays, cropped if ROI arguments are provided.
    paths : list of str
        Source file paths corresponding to each loaded image.
    """
    records = []
    for fname in sorted(os.listdir(corrected_dir)):
        path = os.path.join(corrected_dir, fname)
        if not os.path.isfile(path) or is_all_runs_sum(path):
            continue
        # Try to open as TIFF (works even if no extension)
        try:
            with tifffile.TiffFile(path) as tif:
                img = tif.asarray()
                crop_origin_xy = (0, 0)
                if r0 is not None and x0 is not None and y0 is not None:
                    y_start = max(0, y0 - r0)
                    y_end = min(img.shape[0], y0 + r0)
                    x_start = max(0, x0 - r0)
                    x_end = min(img.shape[1], x0 + r0)
                    img = img[y_start:y_end, x_start:x_end]
                    crop_origin_xy = (int(x_start), int(y_start))
                img = np.asarray(img).view(ImageArray)
                img.crop_origin_xy = crop_origin_xy
                tags = tif.pages[0].tags
                dt_str = None

                for candidate_tag in (tag_id, 65000):
                    if candidate_tag in tags:
                        dt_str = str(tags[candidate_tag].value)
                        break

                if dt_str is None and 270 in tags:
                    description = str(tags[270].value)
                    for line in description.splitlines():
                        if line.startswith("MeanTPXModifiedTime:"):
                            dt_str = line.split(":", 1)[1].strip()
                            break
                        if line.startswith("AcquisitionTime:"):
                            dt_str = line.split(":", 1)[1].strip()
                            break

                if dt_str is None:
                    continue
        except Exception:
            continue

        # Parse datetime
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        records.append((dt, img, path))

    records.sort(key=lambda x: x[0])
    img_datetimes = [r[0] for r in records]
    img_series = [r[1] for r in records]
    paths = [r[2] for r in records]
    return img_datetimes, img_series, paths


def build_turning_point_event_context(
    echem_datetimes,
    echem_thickness_series,
    img_datetimes,
    event_finder,
    image_index_finder,
    conv_thickness=1.0,
    n_tpx_per_image=30,
    tpx_time=2,
    initial_ocv_image_count=None,
):
    """
    Build a matched event context between electrochemical turning points and images.

    Parameters
    ----------
    echem_datetimes : sequence of datetime.datetime
    echem_thickness_series : sequence of float
    img_datetimes : sequence of datetime.datetime
    event_finder : callable
        Function like ``find_capacity_turning_points(times, capacity_series=values)``.
    image_index_finder : callable
        Function like ``find_nearest_image_indices(event_times, img_datetimes)``.
    conv_thickness : float, optional
        Divisor applied to the electrochemical thickness series before event detection.
    n_tpx_per_image : int, optional
        Number of TPX files grouped into one image. Used for far-match warnings.
    tpx_time : int or float, optional
        Duration of each TPX exposure in seconds. Used for far-match warnings.
    initial_ocv_image_count : int, optional
        If provided and positive, prepend an ``ocv_end`` event corresponding to the
        last image used to build the background image, i.e. ``initial_ocv_image_count - 1``.

    Returns
    -------
    dict
        Event context with keys ``events``, ``event_times``, ``img_indices``,
        ``event_times_for_plot``, ``matched_count``, ``total_events_found``,
        ``img_start``, ``img_end``, ``match_warning_threshold_seconds``, and
        ``far_match_warnings``.
    """
    if len(img_datetimes) == 0:
        raise ValueError("No image timestamps found. Check corrected image metadata.")

    img_start = min(img_datetimes)
    img_end = max(img_datetimes)

    cap_times = []
    cap_values = []
    for t, c in zip(echem_datetimes, echem_thickness_series):
        if img_start <= t <= img_end:
            cap_times.append(t)
            cap_values.append(c / conv_thickness)

    event_context = {
        "events": [],
        "event_times": [],
        "img_indices": [],
        "event_times_for_plot": [],
        "matched_count": 0,
        "total_events_found": 0,
        "match_warning_threshold_seconds": float(2 * n_tpx_per_image * tpx_time),
        "far_match_warnings": [],
        "img_start": img_start,
        "img_end": img_end,
    }

    if len(cap_times) == 0:
        return event_context

    extra_events = []
    if initial_ocv_image_count is not None and initial_ocv_image_count > 0:
        ocv_img_idx = min(int(initial_ocv_image_count) - 1, len(img_datetimes) - 1)
        ocv_img_time = img_datetimes[ocv_img_idx]
        if len(cap_times) > 0:
            echem_time_diffs = np.array([abs((t - ocv_img_time).total_seconds()) for t in cap_times])
            nearest_echem_idx = int(np.argmin(echem_time_diffs))
            extra_events.append({
                "index": nearest_echem_idx,
                "time": cap_times[nearest_echem_idx],
                "labels": ["ocv_end"],
            })
        else:
            extra_events.append({
                "index": None,
                "time": ocv_img_time,
                "labels": ["ocv_end"],
            })

    try:
        all_events = event_finder(cap_times, capacity_series=cap_values, extra_events=extra_events)
    except TypeError:
        all_events = event_finder(cap_times, capacity_series=cap_values)

    all_event_times = [e["time"] for e in all_events]
    matched_img_indices = image_index_finder(all_event_times, img_datetimes)

    matched_count = min(len(all_events), len(matched_img_indices))
    matched_events = all_events[:matched_count]
    matched_img_indices = matched_img_indices[:matched_count]
    match_warning_threshold_seconds = float(2 * n_tpx_per_image * tpx_time)

    far_match_warnings = []
    for i, (event, idx) in enumerate(zip(matched_events, matched_img_indices)):
        image_time = img_datetimes[idx]
        time_diff_seconds = abs((image_time - event["time"]).total_seconds())
        if time_diff_seconds > match_warning_threshold_seconds:
            label = chr(97 + i) if i < 26 else str(i)
            far_match_warnings.append(
                {
                    "event_index": i,
                    "event_label": label,
                    "event_time": event["time"],
                    "matched_image_index": idx,
                    "matched_image_time": image_time,
                    "time_diff_seconds": time_diff_seconds,
                }
            )

    if far_match_warnings:
        print(
            "WARNING: Some turning-point/image matches are farther apart than the "
            f"warning threshold of {match_warning_threshold_seconds:.1f} s "
            f"(2 * n_tpx_per_image * tpx_time = 2 * {n_tpx_per_image} * {tpx_time})."
        )
        for warning in far_match_warnings:
            print(
                "  "
                f"{warning['event_label']} (event {warning['event_index']}): "
                f"turning point {warning['event_time'].strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"image idx {warning['matched_image_index']} at "
                f"{warning['matched_image_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                f"({warning['time_diff_seconds']:.1f} s)"
            )

    deduped_events = []
    deduped_img_indices = []
    deduped_event_times = []
    deduped_event_times_for_plot = []
    img_position_by_idx = {}

    for event, idx in zip(matched_events, matched_img_indices):
        if idx in img_position_by_idx:
            pos = img_position_by_idx[idx]
            existing_labels = set(deduped_events[pos].get("labels", []))
            new_labels = set(event.get("labels", []))
            merged_labels = existing_labels | new_labels
            if "ocv_end" in merged_labels and "turning" in merged_labels:
                ordered_labels = ["ocv_end", "turning"] + sorted(
                    label for label in merged_labels if label not in {"ocv_end", "turning"}
                )
            elif "ocv_end" in merged_labels:
                ordered_labels = ["ocv_end"] + sorted(label for label in merged_labels if label != "ocv_end")
            else:
                ordered_labels = sorted(merged_labels)
            deduped_events[pos]["labels"] = ordered_labels
        else:
            img_position_by_idx[idx] = len(deduped_events)
            deduped_events.append(event)
            deduped_img_indices.append(idx)
            deduped_event_times.append(event["time"])
            deduped_event_times_for_plot.append(img_datetimes[idx])

    event_context["events"] = deduped_events
    event_context["event_times"] = deduped_event_times
    event_context["img_indices"] = deduped_img_indices
    event_context["event_times_for_plot"] = deduped_event_times_for_plot
    event_context["matched_count"] = matched_count
    event_context["total_events_found"] = len(all_events)
    event_context["match_warning_threshold_seconds"] = match_warning_threshold_seconds
    event_context["far_match_warnings"] = far_match_warnings

    return event_context


def print_event_context_table(event_context, img_datetimes):
    """
    Print a compact table of detected turning points and matched image times.
    """
    if not event_context["events"]:
        print("No capacity events found.")
        return

    if event_context["matched_count"] < event_context["total_events_found"]:
        print(
            f"Matched {event_context['matched_count']} image-linked events out of "
            f"{event_context['total_events_found']} turning points. "
            "Only matched events will be used below."
        )

    print("All detected turning points and matched images:")
    print("idx | turning point time       | image idx | matched image time")
    print("-" * 72)
    for i, (event, idx) in enumerate(zip(event_context["events"], event_context["img_indices"])):
        image_time = img_datetimes[idx].strftime("%Y-%m-%d %H:%M:%S")
        event_time = event["time"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:>3} | {event_time} | {idx:>9} | {image_time}")

    print(
        f"Detected {len(event_context['img_indices'])} event-linked images "
        "from all detected turning points."
    )


def format_event_context_preview_text(event_context, img_datetimes, label_start=0):
    """
    Format detected turning-point/image matches for display below the EC preview.
    """
    if not event_context["events"]:
        return "No detected turning points available for preview."

    lines = []
    if event_context["matched_count"] < event_context["total_events_found"]:
        lines.append(
            f"Matched {event_context['matched_count']} image-linked events out of "
            f"{event_context['total_events_found']} turning points."
        )
        lines.append("Only matched events are shown below.")
        lines.append("")

    lines.append(f"Detected image indices: {event_context['img_indices']}")
    lines.append("")
    lines.append("All detected turning points and matched images:")
    lines.append("lbl | turning point time   | image idx | matched image time")
    lines.append("-" * 62)

    for i, (event, idx) in enumerate(zip(event_context["events"], event_context["img_indices"])):
        image_time = img_datetimes[idx].strftime("%Y-%m-%d %H:%M:%S")
        event_time = event["time"].strftime("%Y-%m-%d %H:%M:%S")
        label = chr(97 + int(label_start) + i)
        lines.append(f" {label}  | {event_time} | {idx:>9} | {image_time}")

    lines.append("")
    lines.append(
        f"Detected {len(event_context['img_indices'])} event-linked images from all detected turning points."
    )
    return "\n".join(lines)


def print_detected_image_indices_for_copy(event_context):
    """
    Print detected image indices in a copy/paste-friendly format.
    """
    detected_image_indices = event_context["img_indices"]
    if not detected_image_indices:
        print("No detected turning points available for preview.")
        return

    print(f"Detected image indices for copy/paste: {detected_image_indices}")


def normalize_selected_image_indices(selected_image_indices, valid_image_count):
    """
    Validate and deduplicate selected image indices while preserving order.
    """
    selected_image_indices = list(selected_image_indices)
    invalid_indices = [i for i in selected_image_indices if i < 0 or i >= valid_image_count]
    if invalid_indices:
        raise ValueError(
            f"selected_image_indices contains out-of-range values {invalid_indices}. "
            f"Valid image indices are 0 to {valid_image_count - 1}."
        )

    deduped_selected_image_indices = []
    for i in selected_image_indices:
        if i not in deduped_selected_image_indices:
            deduped_selected_image_indices.append(i)

    return deduped_selected_image_indices


def potential_tip_time_near_time(
    center_time,
    echem_datetime_series,
    potential_series,
    max_delta_seconds,
    tip_snap_mode="upper",
):
    """
    Find a local potential tip near a target time.

    The nearest neutron image can sit a little before or after a sharp EC
    extremum. This helper keeps image selection tied to image indices while
    letting plot markers sit on the closest prominent peak or valley.
    """
    echem_datetime_series = np.asarray(echem_datetime_series)
    potential_series = np.asarray(potential_series, dtype=float)
    if len(echem_datetime_series) == 0 or len(potential_series) == 0:
        return None, None

    deltas = np.asarray(
        [abs((dt - center_time).total_seconds()) for dt in echem_datetime_series],
        dtype=float,
    )
    window_indices = np.flatnonzero(deltas <= float(max_delta_seconds))
    if len(window_indices) == 0:
        return None, None

    tip_snap_mode = str(tip_snap_mode or "upper").lower()

    candidate_indices = []
    for idx in window_indices:
        if idx <= 0 or idx >= len(potential_series) - 1:
            continue
        dy_before = potential_series[idx] - potential_series[idx - 1]
        dy_after = potential_series[idx + 1] - potential_series[idx]
        is_peak = dy_before >= 0 and dy_after <= 0
        is_valley = dy_before <= 0 and dy_after >= 0
        if is_peak or is_valley:
            candidate_indices.append(int(idx))

    if tip_snap_mode in {"upper", "max", "maximum", "top"}:
        if candidate_indices:
            candidate_values = np.asarray([potential_series[idx] for idx in candidate_indices], dtype=float)
            candidate_deltas = np.asarray(
                [abs((echem_datetime_series[idx] - center_time).total_seconds()) for idx in candidate_indices],
                dtype=float,
            )
            sort_order = np.lexsort((candidate_deltas, -candidate_values))
            best_pos = int(sort_order[0])
            best_idx = int(candidate_indices[best_pos])
        else:
            best_idx = int(window_indices[int(np.nanargmax(potential_series[window_indices]))])
        best_delta = abs((echem_datetime_series[best_idx] - center_time).total_seconds())
        return echem_datetime_series[best_idx], float(best_delta)

    if tip_snap_mode in {"lower", "min", "minimum", "bottom"}:
        if candidate_indices:
            candidate_values = np.asarray([potential_series[idx] for idx in candidate_indices], dtype=float)
            candidate_deltas = np.asarray(
                [abs((echem_datetime_series[idx] - center_time).total_seconds()) for idx in candidate_indices],
                dtype=float,
            )
            sort_order = np.lexsort((candidate_deltas, candidate_values))
            best_pos = int(sort_order[0])
            best_idx = int(candidate_indices[best_pos])
        else:
            best_idx = int(window_indices[int(np.nanargmin(potential_series[window_indices]))])
        best_delta = abs((echem_datetime_series[best_idx] - center_time).total_seconds())
        return echem_datetime_series[best_idx], float(best_delta)

    if len(window_indices) >= 3:
        first_idx = int(window_indices[0])
        last_idx = int(window_indices[-1])
        x_window = np.asarray(
            [(echem_datetime_series[idx] - echem_datetime_series[first_idx]).total_seconds() for idx in window_indices],
            dtype=float,
        )
        y_window = potential_series[window_indices]
        if x_window[-1] > x_window[0]:
            baseline = np.interp(
                x_window,
                [x_window[0], x_window[-1]],
                [potential_series[first_idx], potential_series[last_idx]],
            )
            if candidate_indices:
                candidate_pos = np.asarray(
                    [int(np.where(window_indices == idx)[0][0]) for idx in candidate_indices],
                    dtype=int,
                )
                candidate_prominence = np.abs(y_window[candidate_pos] - baseline[candidate_pos])
                candidate_deltas = np.asarray(
                    [abs((echem_datetime_series[idx] - center_time).total_seconds()) for idx in candidate_indices],
                    dtype=float,
                )
                sort_order = np.lexsort((candidate_deltas, -candidate_prominence))
                best_pos = int(sort_order[0])
                best_idx = int(candidate_indices[best_pos])
                return echem_datetime_series[best_idx], float(candidate_deltas[best_pos])

            residual = np.abs(y_window - baseline)
            best_idx = int(window_indices[int(np.nanargmax(residual))])
            best_delta = abs((echem_datetime_series[best_idx] - center_time).total_seconds())
            return echem_datetime_series[best_idx], float(best_delta)

    if candidate_indices:
        center_idx = int(np.nanargmin(deltas))
        center_value = potential_series[center_idx]
        candidate_values = np.asarray([potential_series[idx] for idx in candidate_indices], dtype=float)
        candidate_prominence = np.abs(candidate_values - center_value)
        candidate_deltas = np.asarray(
            [abs((echem_datetime_series[idx] - center_time).total_seconds()) for idx in candidate_indices],
            dtype=float,
        )
        sort_order = np.lexsort((candidate_deltas, -candidate_prominence))
        best_pos = int(sort_order[0])
        best_idx = int(candidate_indices[best_pos])
        return echem_datetime_series[best_idx], float(candidate_deltas[best_pos])

    nearest_idx = int(window_indices[int(np.nanargmin(deltas[window_indices]))])
    nearest_delta = abs((echem_datetime_series[nearest_idx] - center_time).total_seconds())
    return echem_datetime_series[nearest_idx], float(nearest_delta)


def snap_image_indices_to_potential_tips(
    selected_image_indices,
    img_datetimes,
    echem_datetime_series,
    potential_series,
    event_context=None,
    tip_times=None,
    tip_snap_tolerance_seconds=None,
    tip_snap_mode="upper",
):
    """
    Return plot times for image indices, snapped to nearby potential tips.

    The selected image indices are not changed. Automatic events are searched
    around their matched turning-point times first; manual selections fall back
    to the image timestamp and then to nearby event times.
    """
    selected_image_indices = list(selected_image_indices)
    img_datetimes = np.asarray(img_datetimes)
    echem_datetime_series = np.asarray(echem_datetime_series)
    potential_series = np.asarray(potential_series)
    event_context = dict(event_context or {})

    if tip_times is None:
        tip_times = list(event_context.get("event_times", []))
    else:
        tip_times = list(tip_times)

    if tip_snap_tolerance_seconds is None:
        tip_snap_tolerance_seconds = event_context.get("match_warning_threshold_seconds", 120.0)
    tip_snap_tolerance_seconds = float(tip_snap_tolerance_seconds)

    tip_time_by_image_idx = {
        image_idx: event_time
        for image_idx, event_time in zip(
            event_context.get("img_indices", []),
            event_context.get("event_times", []),
        )
    }

    def image_is_one_of_two_nearest_to_time(image_idx, target_time):
        deltas = np.asarray(
            [abs((dt - target_time).total_seconds()) for dt in img_datetimes],
            dtype=float,
        )
        nearest_count = min(2, len(deltas))
        nearest_image_indices = set(np.argsort(deltas)[:nearest_count].astype(int))
        return int(image_idx) in nearest_image_indices

    plot_times = []
    snap_details = []
    for image_idx in selected_image_indices:
        image_time = img_datetimes[image_idx]
        center_times = []
        if image_idx in tip_time_by_image_idx:
            matched_tip_time = tip_time_by_image_idx[image_idx]
            if abs((matched_tip_time - image_time).total_seconds()) <= tip_snap_tolerance_seconds:
                center_times.append(matched_tip_time)
        center_times.append(image_time)

        snapped_time = None
        snap_delta = None
        for center_time in center_times:
            candidate_time, candidate_delta = potential_tip_time_near_time(
                center_time,
                echem_datetime_series,
                potential_series,
                tip_snap_tolerance_seconds,
                tip_snap_mode=tip_snap_mode,
            )
            if (
                candidate_time is not None
                and image_is_one_of_two_nearest_to_time(image_idx, candidate_time)
            ):
                snapped_time = candidate_time
                snap_delta = candidate_delta
                break

        if snapped_time is None and tip_times:
            event_deltas = np.asarray(
                [abs((event_time - image_time).total_seconds()) for event_time in tip_times],
                dtype=float,
            )
            nearest_tip_pos = int(np.nanargmin(event_deltas))
            nearest_delta = float(event_deltas[nearest_tip_pos])
            if nearest_delta <= tip_snap_tolerance_seconds:
                candidate_time, candidate_delta = potential_tip_time_near_time(
                    tip_times[nearest_tip_pos],
                    echem_datetime_series,
                    potential_series,
                    tip_snap_tolerance_seconds,
                    tip_snap_mode=tip_snap_mode,
                )
                if (
                    candidate_time is not None
                    and image_is_one_of_two_nearest_to_time(image_idx, candidate_time)
                ):
                    snapped_time = candidate_time
                    snap_delta = candidate_delta

        is_snapped = snapped_time is not None
        plot_time = snapped_time if is_snapped else image_time
        plot_times.append(plot_time)
        snap_details.append(
            {
                "image_index": image_idx,
                "image_time": image_time,
                "plot_time": plot_time,
                "snapped": is_snapped,
                "delta_seconds": snap_delta,
            }
        )

    return plot_times, snap_details


def auto_event_panel_layout(selected_count, cbar_mode=None):
    """
    Choose a sensible image-panel figure size from the number of selected images.

    This is only used when the caller does not provide an explicit ``figsize``.
    """
    if selected_count <= 0:
        return 1, 1, (8.0, 6.0)

    if selected_count <= 2:
        ncols = selected_count
    elif selected_count <= 4:
        ncols = 2
    elif selected_count <= 9:
        ncols = 3
    elif selected_count <= 12:
        ncols = 4
    else:
        ncols = 5

    nrows = int(np.ceil(selected_count / ncols))

    panel_width = 3.25
    panel_height = 3.35
    width = 0.8 + ncols * panel_width
    height = 0.9 + nrows * panel_height

    if cbar_mode and "shared" in str(cbar_mode):
        width += 1.2

    return ncols, nrows, (round(width, 2), round(height, 2))


def auto_event_panel_figsize(selected_count, cbar_mode=None):
    """Choose a figure size while keeping event image panels at a fixed size."""
    _ncols, _nrows, figsize = auto_event_panel_layout(selected_count, cbar_mode=cbar_mode)
    return figsize


def save_selected_event_outputs(
    event_context,
    selected_image_indices,
    img_datetimes,
    img_series,
    avg_thickness_series,
    echem_datetimes,
    potential_series,
    echem_thickness_series,
    output_plot_dir,
    exp_name,
    roi_small,
    event_plot_kwargs,
    plot_images_func,
    potential_plot_kwargs=None,
    label_start=0,
    event_marker_size=6,
    event_label_fontsize=18,
    snap_plot_markers_to_tips=False,
    tip_times=None,
    tip_snap_tolerance_seconds=None,
    tip_snap_mode="upper",
    echem_cycle_series=None,
    metric_label="Avg thickness",
    metric_unit=MICRON_UNIT,
    filename_suffix="",
    save_summary=True,
    save_potential=True,
):
    """
    Save final selected-image outputs: summary text, EC plot, and image panel.

    Returns
    -------
    dict
        Includes normalized image indices, selected plot times, and saved paths.
    """
    normalized_image_indices = normalize_selected_image_indices(
        selected_image_indices,
        len(img_series),
    )

    if len(normalized_image_indices) != len(list(selected_image_indices)):
        print(
            f"Removed duplicate selected image indices. Using {normalized_image_indices} "
            f"instead of {list(selected_image_indices)}."
        )

    if not normalized_image_indices:
        print("selected_image_indices is empty. No final plots or summary files were saved.")
        return {
            "selected_image_indices": [],
            "selected_plot_times": [],
            "snap_details": [],
            "saved_paths": {},
        }

    if snap_plot_markers_to_tips:
        selected_plot_times, snap_details = snap_image_indices_to_potential_tips(
            normalized_image_indices,
            img_datetimes,
            echem_datetimes,
            potential_series,
            event_context=event_context,
            tip_times=tip_times,
            tip_snap_tolerance_seconds=tip_snap_tolerance_seconds,
            tip_snap_mode=tip_snap_mode,
        )
    else:
        selected_plot_times = [img_datetimes[idx] for idx in normalized_image_indices]
        snap_details = [
            {
                "image_index": idx,
                "image_time": img_datetimes[idx],
                "plot_time": img_datetimes[idx],
                "snapped": False,
                "delta_seconds": None,
            }
            for idx in normalized_image_indices
        ]
    detected_event_pos_by_image_idx = {
        image_idx: pos for pos, image_idx in enumerate(event_context.get("img_indices", []))
    }

    print(f"Saving final outputs for selected image indices: {normalized_image_indices}")

    filename_suffix = str(filename_suffix or "")
    saved_paths = {}
    out_text_path = os.path.join(output_plot_dir, f"selected_image_summary_{exp_name}{filename_suffix}.txt")

    def print_and_save(message, file_handle):
        print(message)
        file_handle.write(message + "\n")

    if save_summary:
        with open(out_text_path, "w") as f:
            print_and_save("\n" + "=" * 60, f)
            print_and_save("SELECTED IMAGE SUMMARY", f)
            print_and_save("=" * 60, f)

            thickness_values = []
            for j, (image_idx, plot_time) in enumerate(
                zip(normalized_image_indices, selected_plot_times)
            ):
                thickness_value = avg_thickness_series[image_idx]
                thickness_values.append(thickness_value)
                event_label = chr(97 + int(label_start) + j)
                detected_event_pos = detected_event_pos_by_image_idx.get(image_idx)
                image_time = img_datetimes[image_idx]
                closest_time_str = image_time.strftime("%Y-%m-%d %H:%M:%S")
                plot_time_str = plot_time.strftime("%Y-%m-%d %H:%M:%S")

                print_and_save(f"Image {event_label} (image index {image_idx}):", f)
                if detected_event_pos is not None:
                    turning_time = event_context["event_times"][detected_event_pos]
                    turning_time_str = turning_time.strftime("%Y-%m-%d %H:%M:%S")
                    turning_to_image_seconds = abs((image_time - turning_time).total_seconds())
                    print_and_save(
                        f"  Detected event:  yes (detected image index list position {detected_event_pos})",
                        f,
                    )
                    print_and_save(f"  Turning point:   {turning_time_str}", f)
                    print_and_save(
                        f"  Turning point to image: {turning_to_image_seconds:.1f} seconds",
                        f,
                    )
                else:
                    print_and_save("  Detected event:  no (manually selected image)", f)
                    print_and_save("  Turning point:   n/a", f)
                snap_detail = snap_details[j]
                if snap_detail.get("snapped"):
                    snap_delta = snap_detail.get("delta_seconds")
                    snap_delta_text = "n/a" if snap_delta is None else f"{snap_delta:.1f} seconds"
                    print_and_save(f"  Plot time:       {plot_time_str} (snapped to potential tip)", f)
                    print_and_save(f"  Tip snap delta:  {snap_delta_text}", f)
                else:
                    print_and_save(f"  Plot time:       {plot_time_str}", f)
                print_and_save(f"  Image time:      {closest_time_str}", f)
                unit_text = f" {metric_unit}" if metric_unit else ""
                print_and_save(f"  {metric_label}:   {thickness_value:.4g}{unit_text}", f)
                print_and_save(f"  Image index:     {image_idx}", f)
                print_and_save("", f)

            print_and_save("-" * 60, f)
            print_and_save("SUMMARY:", f)
            print_and_save(
                f"  ROI: center=({roi_small['x0']}, {roi_small['y0']}), radius={roi_small['r0']} pixels",
                f,
            )
            print_and_save(f"  Total detected turning-point images: {len(event_context['img_indices'])}", f)
            print_and_save(f"  Selected image indices: {normalized_image_indices}", f)
            print_and_save(f"  Number of selected images: {len(normalized_image_indices)}", f)
            if thickness_values:
                print_and_save(
                    f"  {metric_label} range:  {np.nanmin(thickness_values):.4g} to {np.nanmax(thickness_values):.4g}{unit_text}",
                    f,
                )
                mean_label = (
                    metric_label
                    if str(metric_label).lower().startswith("mean ")
                    else f"Mean {str(metric_label).lower()}"
                )
                print_and_save(
                    f"  {mean_label}:   {np.nanmean(thickness_values):.4g}{unit_text}",
                    f,
                )
            print_and_save("=" * 60 + "\n", f)

        print(f"Text output saved to: {out_text_path}")
        saved_paths["summary"] = out_text_path

    potential_plot_kwargs = dict(potential_plot_kwargs or {})
    potential_plot_kwargs.setdefault("show", False)

    if save_potential:
        fig = plot_potential_and_thickness(
            echem_datetime_series=echem_datetimes,
            potential_series=potential_series,
            echem_thickness_series=echem_thickness_series,
            img_datetime_series=img_datetimes,
            avg_thickness_series=avg_thickness_series,
            event_times=selected_plot_times,
            label_start=label_start,
            event_marker_size=event_marker_size,
            event_label_fontsize=event_label_fontsize,
            **potential_plot_kwargs,
        )

        potential_plot_path = os.path.join(output_plot_dir, f"potential_{exp_name}{filename_suffix}.png")
        fig.savefig(potential_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Image saved to: {potential_plot_path}")
        saved_paths["potential"] = potential_plot_path

    event_plot_kwargs = dict(event_plot_kwargs or {})
    show_cycle_numbers = bool(event_plot_kwargs.pop("show_cycle_numbers", False))
    show_nci_thickness = bool(event_plot_kwargs.pop("show_nci_thickness", False))
    nci_thickness_unit = event_plot_kwargs.pop("nci_thickness_unit", "μm")
    nci_thickness_decimals = int(event_plot_kwargs.pop("nci_thickness_decimals", 2))
    manual_cycle_numbers = event_plot_kwargs.pop("manual_cycle_numbers", None)
    image_annotation_fontsize = event_plot_kwargs.pop("image_annotation_fontsize", 12)
    image_annotation_color = event_plot_kwargs.pop("image_annotation_color", "white")
    image_annotation_fontweight = event_plot_kwargs.pop("image_annotation_fontweight", "bold")
    image_annotation_bbox = event_plot_kwargs.pop(
        "image_annotation_bbox",
        {"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.5},
    )

    def format_cycle_label(cycle_value):
        if isinstance(cycle_value, str):
            cycle_value = cycle_value.strip()
            return cycle_value if cycle_value.lower().startswith("cycle") else f"cycle {cycle_value}"
        try:
            if np.isfinite(cycle_value) and float(cycle_value).is_integer():
                return f"cycle {int(cycle_value)}"
            return f"cycle {float(cycle_value):g}"
        except Exception:
            return f"cycle {cycle_value}"

    if manual_cycle_numbers is not None:
        if isinstance(manual_cycle_numbers, (str, int, float, np.integer, np.floating)):
            manual_cycle_numbers = [manual_cycle_numbers]
        if len(manual_cycle_numbers) != len(normalized_image_indices):
            raise ValueError(
                "manual_cycle_numbers must contain one value per selected image "
                f"({len(normalized_image_indices)} values required)."
            )
        lower_left_texts = [
            format_cycle_label(cycle_value)
            for cycle_value in manual_cycle_numbers
        ]
        event_plot_kwargs.setdefault("lower_left_texts", lower_left_texts)
    elif show_cycle_numbers:
        lower_left_texts = generated_cycle_labels(len(normalized_image_indices))
        event_plot_kwargs.setdefault("lower_left_texts", lower_left_texts)

    if show_nci_thickness:
        lower_right_texts = [
            f"{avg_thickness_series[image_idx]:.{nci_thickness_decimals}f} {nci_thickness_unit}".rstrip()
            for image_idx in normalized_image_indices
        ]
        event_plot_kwargs.setdefault("lower_right_texts", lower_right_texts)

    if manual_cycle_numbers is not None or show_cycle_numbers or show_nci_thickness:
        event_plot_kwargs.setdefault("annotation_fontsize", image_annotation_fontsize)
        event_plot_kwargs.setdefault("annotation_color", image_annotation_color)
        event_plot_kwargs.setdefault("annotation_fontweight", image_annotation_fontweight)
        event_plot_kwargs.setdefault("annotation_bbox", image_annotation_bbox)

    auto_ncols, auto_nrows, auto_figsize = auto_event_panel_layout(
        len(normalized_image_indices),
        cbar_mode=event_plot_kwargs.get("cbar_mode"),
    )
    event_plot_kwargs.setdefault("ncols", auto_ncols)
    event_plot_kwargs.setdefault("nrows", auto_nrows)
    if event_plot_kwargs.get("figsize") is None:
        event_plot_kwargs["figsize"] = auto_figsize

    fig, axes = plot_images_func(
        img_series=img_series,
        img_indices=normalized_image_indices,
        label_start=label_start,
        **event_plot_kwargs,
    )

    selected_event_plot_path = os.path.join(output_plot_dir, f"nci_{exp_name}{filename_suffix}.png")
    fig.savefig(selected_event_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {selected_event_plot_path}")
    saved_paths["images"] = selected_event_plot_path

    return {
        "selected_image_indices": normalized_image_indices,
        "selected_plot_times": selected_plot_times,
        "snap_details": snap_details,
        "saved_paths": saved_paths,
    }



def plot_potential_and_thickness(
    echem_datetime_series,
    potential_series,
    echem_thickness_series=None,
    img_datetime_series=None,
    avg_thickness_series=None,
    plot_thickness=False,
    potential_ylim=None,
    event_times=None,
    label_start=0,
    event_marker_size=3,
    event_label_fontsize=10,
    show=True,
    pause_gap_minutes=5,
    potential_color="tab:blue",
    ec_thickness_color="tab:green",
    nci_thickness_color="tab:orange",
    thickness_unit="μm",
    nci_trace_label="NCI thickness",
    thickness_axis_label=None,
    thickness_traces="both",
    thickness_layout="separate",
    overlay_offset_fraction=0.0,
    side_text=None,
    side_text_fontsize=8,
    return_axes=False,
    event_labels=None,
    current_series=None,
    image_gap_minutes=None,
):
    """
    Plot potential and, optionally, EC and NCI thickness on a common time axis.

    The plot is restricted to the image time window when image timestamps are
    supplied. Paused electrochemical intervals are detected from absolute gaps
    in the EC timestamps and shown as shaded regions. Optional event labels are
    added to the potential trace using the provided ``event_times``.
    """

    # ---- Convert to numpy arrays ----
    echem_datetime_series = np.asarray(echem_datetime_series)
    img_datetime_series = np.asarray(img_datetime_series)

    potential_series = np.asarray(potential_series)
    current_series = None if current_series is None else np.asarray(current_series)
    selected_thickness_traces = normalize_thickness_traces(thickness_traces, plot_thickness)
    plot_thickness = bool(selected_thickness_traces)
    thickness_layout = str(thickness_layout).strip().lower()
    if thickness_layout not in ("separate", "overlay"):
        raise ValueError("thickness_layout must be 'separate' or 'overlay'.")

    if plot_thickness:
        if "echem" in selected_thickness_traces and echem_thickness_series is None:
            raise ValueError("EC thickness plotting requires echem_thickness_series.")
        if "nci" in selected_thickness_traces and (
            img_datetime_series is None or avg_thickness_series is None
        ):
            raise ValueError("NCI thickness plotting requires img_datetime_series and avg_thickness_series.")
        if "echem" in selected_thickness_traces:
            echem_thickness_series = np.asarray(echem_thickness_series)
        if "nci" in selected_thickness_traces:
            avg_thickness_series = np.asarray(avg_thickness_series)

    # ---- Determine overlapping time window ----
    if img_datetime_series is not None and len(img_datetime_series) > 0:
        t_start = img_datetime_series.min()
        t_end = img_datetime_series.max()
        # ---- Mask electrochemical data to image time window ----
        echem_mask = (echem_datetime_series >= t_start) & (echem_datetime_series <= t_end)
        echem_datetime_series = echem_datetime_series[echem_mask]
        potential_series = potential_series[echem_mask]
        if current_series is not None:
            current_series = current_series[echem_mask]
        if plot_thickness and "echem" in selected_thickness_traces:
            echem_thickness_series = echem_thickness_series[echem_mask]
    else:
        t_start = echem_datetime_series.min()
        t_end = echem_datetime_series.max()

    # ---- Define common time origin ----
    t0 = t_start

    # ---- Convert datetimes to elapsed time (hours) ----
    t_echem = np.array([(t - t0).total_seconds() for t in echem_datetime_series]) / 3600.0
    t_img = np.array([(t - t0).total_seconds() for t in img_datetime_series]) / 3600.0
    paused_intervals = find_paused_intervals(
        echem_datetime_series,
        min_gap_minutes=pause_gap_minutes,
    )

    # ---- Plot ----
    if plot_thickness and thickness_layout == "separate":
        if side_text:
            fig = plt.figure(figsize=(16, 8.5))
            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=[2.2, 1.8],
                height_ratios=[1, 1],
                hspace=0.10,
                wspace=0.20,
            )
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax_text = fig.add_subplot(gs[:, 1])
            ax_text.axis("off")
            ax_text.text(
                0.0,
                1.0,
                side_text,
                transform=ax_text.transAxes,
                va="top",
                ha="left",
                fontsize=side_text_fontsize,
                family="monospace",
            )
        elif current_series is not None:
            fig = plt.figure(figsize=(8, 7.5))
            gs = fig.add_gridspec(3, 1, height_ratios=[1, .55, 1], hspace=.12)
            ax1 = fig.add_subplot(gs[0])
            ax2 = fig.add_subplot(gs[2], sharex=ax1)
            ax_current = fig.add_subplot(gs[1], sharex=ax1)
            x_current, y_current = build_series_with_pauses(t_echem, current_series, paused_intervals, t0)
            ax_current.plot(x_current, y_current, color='0.35', lw=1)
            ax_current.axhline(0, color='0.7', lw=.6)
            ax_current.set_ylabel('Current (mA)')
            plt.setp(ax_current.get_xticklabels(), visible=False)
        else:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(8, 6.2),
                sharex=True,
                gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
            )
    else:
        ax2 = None
        if side_text:
            fig, (ax1, ax_text) = plt.subplots(
                1,
                2,
                figsize=(16, 8.5),
                gridspec_kw={"width_ratios": [2.2, 1.8]},
            )
            ax_text.axis("off")
            ax_text.text(
                0.0,
                1.0,
                side_text,
                transform=ax_text.transAxes,
                va="top",
                ha="left",
                fontsize=side_text_fontsize,
                family="monospace",
            )
        else:
            fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Potential (left axis), broken across paused intervals.
    t_echem_plot, potential_plot = build_series_with_pauses(
        t_echem,
        potential_series,
        paused_intervals,
        t0,
    )
    ax1.plot(t_echem_plot, potential_plot, label="Potential (Ewe/V)", color=potential_color)
    ax1.set_ylabel("Potential (V)")
    ax1.tick_params(axis="y")
    ax1.set_title("Potential and Thickness Evolution" if plot_thickness else "Potential evolution")
    if potential_ylim is not None:
        ax1.set_ylim(potential_ylim)

    def visible_potential_marker_y(y_value):
        y_min, y_max = ax1.get_ylim()
        if not np.isfinite(y_value) or y_max <= y_min:
            return y_value
        margin = 0.025 * (y_max - y_min)
        return float(np.clip(y_value, y_min + margin, y_max - margin))

    def potential_label_offset(marker_y):
        y_min, y_max = ax1.get_ylim()
        if not np.isfinite(marker_y) or y_max <= y_min:
            return (4, 4), "left", "bottom"
        top_margin = 0.08 * (y_max - y_min)
        bottom_margin = 0.08 * (y_max - y_min)
        if marker_y >= y_max - top_margin:
            return (4, -4), "left", "top"
        if marker_y <= y_min + bottom_margin:
            return (4, 4), "left", "bottom"
        return (4, 4), "left", "bottom"

    for pause_start, pause_end in paused_intervals:
        pause_start_h = (pause_start - t0).total_seconds() / 3600.0
        pause_end_h = (pause_end - t0).total_seconds() / 3600.0
        ax1.axvspan(pause_start_h, pause_end_h, color="tab:red", alpha=0.18)
        ylim = ax1.get_ylim()
        y_text = ylim[1] - 0.08 * (ylim[1] - ylim[0])
        ax1.text(
            0.5 * (pause_start_h + pause_end_h),
            y_text,
            "EC data gap",
            color="tab:red",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="top",
        )
        if current_series is not None and plot_thickness and thickness_layout == "separate" and not side_text:
            ax_current.axvspan(pause_start_h, pause_end_h, color="tab:red", alpha=0.10)

    # Optional event markers/labels on the potential trace.
    if event_times is not None and len(event_times) > 0:
        for j, raw_time in enumerate(event_times):
            if raw_time is None:
                continue
            event_dt = raw_time
            if isinstance(raw_time, str):
                event_dt = pd.to_datetime(raw_time).to_pydatetime()

            dt_seconds = np.array([abs((t - event_dt).total_seconds()) for t in echem_datetime_series])
            idx = int(np.argmin(dt_seconds))

            from ..event_selection import panel_label
            label = event_labels[j] if event_labels is not None else panel_label(int(label_start) + j)
            marker_y = visible_potential_marker_y(potential_series[idx])
            label_offset, label_ha, label_va = potential_label_offset(marker_y)
            clipped = potential_series[idx] != marker_y
            marker = ("^" if potential_series[idx] > marker_y else "v") if clipped else "o"
            ax1.plot(t_echem[idx], marker_y, color="black", marker=marker, ls="none", ms=event_marker_size)
            ax1.annotate(
                label,
                (t_echem[idx], marker_y),
                textcoords="offset points",
                xytext=label_offset,
                fontsize=event_label_fontsize,
                fontweight="bold",
                ha=label_ha,
                va=label_va,
                clip_on=True,
            )


    # Thickness (bottom panel or overlaid right axis when enabled)
    if plot_thickness:
        thickness_axis = ax2 if thickness_layout == "separate" else ax1.twinx()
        thickness_plot_data = []
        if "echem" in selected_thickness_traces:
            t_echem_thickness_plot, echem_thickness_plot = build_series_with_pauses(
                t_echem,
                echem_thickness_series,
                paused_intervals,
                t0,
            )
            thickness_plot_data.append(
                ("EC thickness", t_echem_thickness_plot, echem_thickness_plot, ec_thickness_color)
            )
        if "nci" in selected_thickness_traces:
            image_pauses = (find_paused_intervals(img_datetime_series, min_gap_minutes=image_gap_minutes)
                            if image_gap_minutes is not None else paused_intervals)
            t_img_plot, nci_thickness_plot = build_datetime_series_with_pauses(
                img_datetime_series,
                avg_thickness_series,
                image_pauses,
                t0,
            )
            thickness_plot_data.append(
                (nci_trace_label, t_img_plot, nci_thickness_plot, nci_thickness_color)
            )

        if thickness_layout == "overlay":
            offset_data = apply_vertical_offsets_to_traces(
                [(label, values) for label, _, values, _ in thickness_plot_data],
                offset_fraction=overlay_offset_fraction,
            )
            offsets_by_label = {label: offset for label, _, offset in offset_data}
            values_by_label = {label: values for label, values, _ in offset_data}
        else:
            offsets_by_label = {label: 0.0 for label, _, _, _ in thickness_plot_data}
            values_by_label = {label: values for label, _, values, _ in thickness_plot_data}

        for label, x_values, _y_values, color in thickness_plot_data:
            offset = offsets_by_label[label]
            plot_label = label
            if thickness_layout == "overlay" and abs(offset) > 0:
                plot_label = f"{label} ({offset:+.3g} {thickness_unit} offset)"
            thickness_axis.plot(
                x_values,
                values_by_label[label],
                label=plot_label,
                color=color,
                lw=1.8,
            )

        axis_label = thickness_axis_label or f"Thickness ({thickness_unit})"
        thickness_axis.set_ylabel(axis_label)
        if thickness_layout == "separate":
            thickness_axis.set_xlabel("Time elapsed (h)")
            thickness_axis.legend(loc="best")
            ax1.legend(loc="best")
            plt.setp(ax1.get_xticklabels(), visible=False)
        else:
            ax1.set_xlabel("Time elapsed (h)")
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = thickness_axis.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
            ax2 = thickness_axis

    else:
        ax1.set_xlabel("Time elapsed (h)")
        ax1.legend(loc="best")
    
    if side_text:
        fig.subplots_adjust(left=0.07, right=0.98, bottom=0.10, top=0.92, wspace=0.22, hspace=0.12)
    elif plot_thickness and thickness_layout == "overlay":
        fig.subplots_adjust(left=0.12, right=0.86, bottom=0.14, top=0.90)
    else:
        fig.tight_layout()
    if show:
        plt.show()
    if return_axes:
        return fig, ax1, ax2 if plot_thickness else None
    return fig



def launch_interactive_event_selector(
    event_context,
    echem_datetimes,
    potential_series,
    img_datetimes,
    initial_selected_image_indices=None,
    echem_thickness_series=None,
    avg_thickness_series=None,
    plot_thickness=False,
    potential_plot_kwargs=None,
    side_text=None,
    label_start=0,
    event_marker_size=6,
    event_label_fontsize=18,
):
    """
    Show an interactive potential plot where clicking candidate points toggles
    which matched image indices will be used for downstream outputs.

    The returned dict contains a live ``selected_image_indices`` list that is
    mutated in place as the user clicks or uses the helper buttons.

    Notes
    -----
    Interactive clicking requires an interactive matplotlib backend such as
    ``widget`` or ``notebook``. With a static inline backend the plot will still
    render, but clicks will not update the selection.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:
        raise RuntimeError(
            'ipywidgets is required for interactive selection. Install/enable it, '
            'or fall back to manual selected_image_indices.'
        ) from exc

    candidate_img_indices = list(event_context.get('img_indices', []))
    if not candidate_img_indices:
        print('No code-identified images available for interactive selection.')
        return {
            'selected_image_indices': [],
            'candidate_image_indices': [],
            'figure': None,
            'widgets': {},
        }

    img_datetimes = np.asarray(img_datetimes)
    echem_datetimes = np.asarray(echem_datetimes)
    potential_series = np.asarray(potential_series)

    if initial_selected_image_indices is None:
        initial_selected_image_indices = candidate_img_indices
    selected_indices = normalize_selected_image_indices(
        initial_selected_image_indices,
        len(img_datetimes),
    )
    selected_image_indices = list(selected_indices)

    potential_plot_kwargs = dict(potential_plot_kwargs or {})
    fig, ax1, _ax2 = plot_potential_and_thickness(
        echem_datetime_series=echem_datetimes,
        potential_series=potential_series,
        echem_thickness_series=echem_thickness_series,
        img_datetime_series=img_datetimes,
        avg_thickness_series=avg_thickness_series,
        plot_thickness=plot_thickness,
        event_times=event_context.get('event_times_for_plot', []),
        label_start=label_start,
        event_marker_size=event_marker_size,
        event_label_fontsize=event_label_fontsize,
        side_text=side_text,
        show=False,
        return_axes=True,
        **potential_plot_kwargs,
    )

    if len(img_datetimes) > 0:
        t_start = img_datetimes.min()
        t_end = img_datetimes.max()
        echem_mask = (echem_datetimes >= t_start) & (echem_datetimes <= t_end)
        echem_plot_datetimes = echem_datetimes[echem_mask]
        potential_plot_series = potential_series[echem_mask]
        t0 = t_start
    else:
        echem_plot_datetimes = echem_datetimes
        potential_plot_series = potential_series
        t0 = echem_datetimes.min()

    candidate_labels = []
    candidate_x = []
    candidate_y = []
    candidate_times = []
    for pos, image_idx in enumerate(candidate_img_indices):
        image_time = img_datetimes[image_idx]
        candidate_times.append(image_time)
        candidate_labels.append(chr(97 + int(label_start) + pos))
        candidate_x.append((image_time - t0).total_seconds() / 3600.0)
        dt_seconds = np.array([abs((t - image_time).total_seconds()) for t in echem_plot_datetimes])
        nearest_echem_idx = int(np.argmin(dt_seconds))
        candidate_y.append(float(potential_plot_series[nearest_echem_idx]))

    candidate_x = np.asarray(candidate_x, dtype=float)
    candidate_y = np.asarray(candidate_y, dtype=float)

    def current_selected_mask():
        return np.array([idx in selected_image_indices for idx in candidate_img_indices], dtype=bool)

    def sort_image_indices_by_time(image_indices):
        return sorted(image_indices, key=lambda idx: img_datetimes[int(idx)])

    selectable_scatter = ax1.scatter(
        candidate_x,
        candidate_y,
        s=180,
        facecolors='none',
        edgecolors='tab:gray',
        linewidths=1.8,
        zorder=6,
    )

    status_html = widgets.HTML()
    help_html = widgets.HTML(
        '<b>Interactive selection:</b> click a labeled point on the EC plot to toggle it. '
        'Use the buttons to reset, clear, or restore all auto-detected points.'
    )
    backend_html = widgets.HTML()
    backend_name = str(plt.get_backend()).lower()
    if 'inline' in backend_name:
        backend_html.value = (
            '<span style="color:#b45309;">Current matplotlib backend looks static '
            f'({plt.get_backend()}). If clicks do not respond, switch to an interactive '
            'backend such as <code>%matplotlib widget</code>.</span>'
        )

    reset_button = widgets.Button(description='Reset Initial', button_style='')
    auto_button = widgets.Button(description='Use All Auto', button_style='')
    clear_button = widgets.Button(description='Clear', button_style='warning')

    def set_selected_indices(new_indices):
        normalized = normalize_selected_image_indices(new_indices, len(img_datetimes))
        normalized = sort_image_indices_by_time(normalized)
        selected_image_indices[:] = normalized
        selected_mask = current_selected_mask()
        facecolors = np.where(selected_mask, 'tab:orange', 'none')
        edgecolors = np.where(selected_mask, 'tab:orange', 'tab:gray')
        selectable_scatter.set_facecolors(facecolors)
        selectable_scatter.set_edgecolors(edgecolors)

        selected_pairs = [
            f"{chr(97 + int(label_start) + pos)}:{idx}"
            for pos, idx in enumerate(selected_image_indices)
        ]
        if selected_pairs:
            status_html.value = (
                '<b>Selected image indices:</b> '
                + ', '.join(selected_pairs)
            )
        else:
            status_html.value = '<b>Selected image indices:</b> none'
        fig.canvas.draw_idle()

    def toggle_candidate_at_position(position):
        image_idx = candidate_img_indices[position]
        if image_idx in selected_image_indices:
            selected_image_indices[:] = [idx for idx in selected_image_indices if idx != image_idx]
        else:
            selected_image_indices.append(image_idx)
        set_selected_indices(selected_image_indices)

    x_span = float(np.nanmax(candidate_x) - np.nanmin(candidate_x)) if len(candidate_x) > 1 else 0.0
    x_threshold = max(0.05, 0.03 * x_span)

    def on_click(event):
        if event.inaxes is not ax1 or event.xdata is None:
            return
        nearest = int(np.argmin(np.abs(candidate_x - float(event.xdata))))
        if abs(candidate_x[nearest] - float(event.xdata)) > x_threshold:
            return
        toggle_candidate_at_position(nearest)

    def on_reset(_button):
        set_selected_indices(initial_selected_image_indices)

    def on_auto(_button):
        set_selected_indices(candidate_img_indices)

    def on_clear(_button):
        set_selected_indices([])

    fig.canvas.mpl_connect('button_press_event', on_click)
    reset_button.on_click(on_reset)
    auto_button.on_click(on_auto)
    clear_button.on_click(on_clear)

    set_selected_indices(selected_image_indices)

    controls = widgets.HBox([reset_button, auto_button, clear_button])
    display(widgets.VBox([help_html, backend_html, status_html, controls]))
    plt.show()

    return {
        'selected_image_indices': selected_image_indices,
        'candidate_image_indices': candidate_img_indices,
        'candidate_labels': candidate_labels,
        'figure': fig,
        'widgets': {
            'status_html': status_html,
            'help_html': help_html,
            'backend_html': backend_html,
            'reset_button': reset_button,
            'auto_button': auto_button,
            'clear_button': clear_button,
        },
    }

    
    
    
    
    

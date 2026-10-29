"""Selective bright-outlier replacement inspired by pyerre's ro_filter_2D.

Reference: pyerre/proc_functions/filters.py (P. Boillat): local median,
absolute or sqrt(median) threshold, median replacement. This standalone
adaptation targets bright pixels only, uses reflected edges, preserves invalid
pixels, and adds a robust automatic noise threshold inspired by ro_auto_threshold.
It does not require pyerre or mix consecutive frames.
"""

from __future__ import annotations

import warnings
import numpy as np


DEFAULTS = {
    "white_spot_enabled": False,
    "white_spot_size": 3,
    "white_spot_method": "mad",
    "white_spot_threshold": 8.0,
}


def filter_settings(config):
    values = {key: config.get(key, default) for key, default in DEFAULTS.items()}
    if not isinstance(values["white_spot_enabled"], (bool, np.bool_)):
        raise ValueError("White-spot enabled must be True or False.")
    try:
        size = int(str(values["white_spot_size"]))
        threshold = float(values["white_spot_threshold"])
    except (TypeError, ValueError):
        raise ValueError("White-spot window must be an odd integer and threshold a positive number.") from None
    if size < 3 or size > 31 or size % 2 != 1:
        raise ValueError("White-spot window must be odd, from 3 to 31 pixels (start with 3).")
    method = str(values["white_spot_method"]).strip().lower()
    if method not in {"mad", "abs", "prop"}:
        raise ValueError("White-spot threshold method must be mad, abs, or prop.")
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("White-spot threshold must be finite and greater than zero.")
    return {"enabled": bool(values["white_spot_enabled"]), "filter_size": size,
            "method": method, "threshold": threshold}


def effective_image_settings(config):
    """Do not invalidate older results/resume records when the filter is off."""
    values = {key: value for key, value in config.items() if key not in DEFAULTS}
    if config.get("white_spot_enabled", False):
        settings = filter_settings(config)
        values.update(white_spot_enabled=True, white_spot_size=settings["filter_size"],
                      white_spot_method=settings["method"], white_spot_threshold=settings["threshold"])
    return values


def remove_white_spots(image, *, filter_size=3, method="mad", threshold=8.0):
    """Return (float32 image, boolean replacement mask, statistics).

    Only positive deviations are replaced. No temporal filtering, global
    smoothing, or replacement of NaN/Inf values is performed. Small real bright
    features can also be classified as outliers: preview and tune before use.
    """
    from scipy.ndimage import maximum_filter, median_filter

    settings = filter_settings({"white_spot_enabled": True, "white_spot_size": filter_size,
                                "white_spot_method": method, "white_spot_threshold": threshold})
    array = np.array(image, dtype=np.float32, copy=True)
    if array.ndim != 2 or not array.size:
        raise ValueError("White-spot removal expects a non-empty 2D image.")
    finite = np.isfinite(array)
    if finite.all():
        local = median_filter(array, size=settings["filter_size"], mode="reflect")
    elif finite.any():
        valid_array = np.where(finite, array, np.nan)
        size = settings["filter_size"]
        local = median_filter(np.where(finite, array, 0), size=size, mode="reflect")
        affected = finite & maximum_filter(~finite, size=size, mode="reflect")
        yy, xx = np.nonzero(affected)
        radius = size // 2
        windows = np.lib.stride_tricks.sliding_window_view(
            np.pad(valid_array, radius, mode="symmetric"), (size, size),
        )
        # Only recalculate finite pixels near invalid neighbours, in bounded
        # chunks. A Python callback on every pixel is prohibitively slow on stacks.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for start in range(0, len(yy), 2048):
                y, x = yy[start:start+2048], xx[start:start+2048]
                local[y, x] = np.nanmedian(windows[y, x], axis=(-2, -1))
    else:
        local = np.full_like(array, np.nan)
    valid = finite & np.isfinite(local)
    difference = np.zeros(array.shape, dtype=np.float64)
    np.subtract(array, local, out=difference, where=valid, dtype=np.float64)
    noise_sigma = None
    cutoff = np.full(array.shape, settings["threshold"], dtype=np.float64)
    if settings["method"] == "prop":
        # Pyerre's count-noise rule; nonnegative baseline avoids sqrt of negatives.
        cutoff = settings["threshold"] * np.sqrt(np.maximum(local, 0).astype(np.float64))
    elif settings["method"] == "mad":
        residual = difference[valid]
        center = float(np.median(residual)) if residual.size else 0.0
        noise_sigma = float(np.median(np.abs(residual - center)) / 0.6744897501960817) if residual.size else 0.0
        cutoff.fill(max(0.0, center) + settings["threshold"] * noise_sigma)
    # Suppress floating-point roundoff when a flat image has zero MAD.
    floor = 8 * np.finfo(np.float32).eps * np.maximum(1.0, np.abs(local))
    cutoff = np.maximum(cutoff, floor)
    mask = valid & (difference > cutoff)
    array[mask] = local[mask]
    count = int(mask.sum())
    valid_count = int(valid.sum())
    stats = {"replaced_pixels": count, "finite_pixels": int(finite.sum()),
             "total_pixels": int(array.size), "replaced_fraction": count / max(1, valid_count),
             "noise_sigma": noise_sigma,
             "threshold_min": float(cutoff[valid].min()) if valid_count else None,
             "threshold_max": float(cutoff[valid].max()) if valid_count else None}
    return array, mask, stats

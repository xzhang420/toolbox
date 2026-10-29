"""Read-only spatial measurements. Masks and normalizing areas are explicit."""

import numpy as np
import pandas as pd

from .section import PipelineSection, SectionOutcome, SectionState

METHODS = {
    "mean": "ROI mean", "area_normalized": "Electrode-area mean",
    "active_mean": "Active-area mean", "median": "ROI median", "trimmed_mean": "ROI trimmed mean",
}


def method_names(value):
    if isinstance(value, str):
        value = [part.strip() for part in value.split(",") if part.strip()]
    names = list(dict.fromkeys(value or ["mean"]))
    if set(names) - METHODS.keys():
        raise ValueError(f"Unknown measurement method; choose from {', '.join(METHODS)}.")
    return names


def region_mask(shape, config):
    """Pixel centres in original array coordinates; rectangle upper edges excluded."""
    if len(shape) != 2:
        raise ValueError("Measurements require 2D images.")
    roi = config.get("roi") or {}
    kind = config.get("region") or ("circle" if all(k in roi for k in ("x0", "y0", "r0")) else "whole")
    yy, xx = np.indices(shape)
    if kind == "whole":
        mask = np.ones(shape, bool)
    elif kind == "circle":
        x, y, radius = [float(roi[k]) for k in ("x0", "y0", "r0")]
        if not np.isfinite([x, y, radius]).all() or radius <= 0:
            raise ValueError("Circle coordinates must be finite; radius must be positive.")
        mask = (xx-x)**2 + (yy-y)**2 <= radius**2
    elif kind == "rectangle":
        x1, x2, y1, y2 = [float(roi[k]) for k in ("x_min", "x_max", "y_min", "y_max")]
        if not np.isfinite([x1, x2, y1, y2]).all() or x1 >= x2 or y1 >= y2:
            raise ValueError("Rectangle needs finite, increasing x/y bounds.")
        mask = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    elif kind == "polygon":
        from matplotlib.path import Path
        points = np.asarray(roi.get("vertices", []), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("Polygon needs at least three finite [x, y] vertices.")
        # Include boundary pixel centres consistently for either winding order.
        signed = np.sum(points[:, 0]*np.roll(points[:, 1], -1)-points[:, 1]*np.roll(points[:, 0], -1))
        if abs(signed) < 1e-12:
            raise ValueError("Polygon has zero area; check vertex order.")
        if signed < 0:
            points = points[::-1]
        mask = Path(points).contains_points(np.c_[xx.ravel(), yy.ravel()], radius=1e-9).reshape(shape)
    else:
        raise ValueError("Region must be whole, circle, rectangle or polygon.")
    if not mask.any():
        raise ValueError("The measurement region contains no image pixel centres.")
    return mask


def measure_stack(images, times, config, *, pixel_size_um=None, unit="", calibrated=False):
    if not len(images):
        raise ValueError("No images to measure.")
    names = method_names(config.get("methods"))
    primary = str(config.get("primary_method") or names[0])
    if primary not in names:
        raise ValueError("The primary method must be among the calculated methods.")
    mask = region_mask(np.shape(images[0]), config)
    if len(times) not in {0, len(images)}:
        raise ValueError("Image timestamps must match the image count.")
    roi = config.get("roi") or {}
    kind = config.get("region") or ("circle" if all(k in roi for k in ("x0", "y0", "r0")) else "whole")
    thickness = calibrated or str(unit).strip().casefold() in {"um", "µm", "μm"}
    area_px = np.nan
    if pixel_size_um is not None:
        size = float(pixel_size_um)
        if not np.isfinite(size) or size <= 0:
            raise ValueError("Pixel size must be positive and finite.")
        area_px = (size/1000)**2
    area = config.get("electrode_area_mm2")
    if "area_normalized" in names:
        if not thickness or not np.isfinite(area_px):
            raise ValueError("Electrode-area thickness requires known micrometre units and pixel size.")
        if area in (None, "") or not np.isfinite(float(area)) or float(area) <= 0:
            raise ValueError("Supply a positive electrode area in mm².")
        area = float(area)
    threshold = config.get("active_threshold")
    if "active_mean" in names and threshold in (None, ""):
        raise ValueError("Active-area mean requires an explicit threshold in the image units.")
    threshold = None if threshold in (None, "") else float(threshold)
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("Active threshold must be finite.")
    trim = float(config.get("trim_fraction", .1))
    if not np.isfinite(trim) or not 0 <= trim < .5:
        raise ValueError("Trim fraction must be between 0 and 0.5 (excluded), per tail.")
    columns = {name: f"nci_{name}_um" if thickness else f"image_{name}_value" for name in names}
    display_unit = "µm" if thickness else (unit or "a.u.")
    labels = {name: f"{METHODS[name]} ({display_unit})" for name in names}
    rows = []
    count = int(mask.sum())
    for index, image in enumerate(images):
        if np.shape(image) != mask.shape:
            raise ValueError("All measurement images must have the same shape.")
        samples = np.asarray(image, dtype=float)[mask]
        finite = samples[np.isfinite(samples)]
        bad = count-len(finite)
        active = finite[finite > threshold] if threshold is not None else np.array([])
        flags = []
        if bad:
            flags.append("nonfinite_pixels")
        if not len(finite):
            flags.append("no_valid_pixels")
        if len(finite) and np.all(finite == 0):
            flags.append("all_zero_region")
        if threshold is not None and not len(active):
            flags.append("empty_active_area")
        from .quality import pixel_diagnostics
        diagnostics, extra_flags = pixel_diagnostics(finite)
        flags.extend(extra_flags)
        row = {"frame": index, "time": times[index] if len(times) else None,
               "region": kind, "active_threshold": threshold,
               "trim_fraction_per_tail": trim if "trimmed_mean" in names else np.nan,
               "region_pixels": count, "valid_pixels": len(finite), "nonfinite_pixels": bad,
               "region_area_mm2": count*area_px, "valid_area_mm2": len(finite)*area_px,
               "active_pixels": len(active) if threshold is not None else np.nan,
               "active_area_mm2": len(active)*area_px if threshold is not None else np.nan,
               "active_fraction_of_valid": len(active)/len(finite) if threshold is not None and len(finite) else np.nan,
               "electrode_area_mm2": area if "area_normalized" in names else np.nan,
               "quality_flags": ";".join(flags)}
        row.update(diagnostics)
        for name in names:
            value = np.nan
            if len(finite):
                if name == "mean":
                    value = finite.mean()
                elif name == "median":
                    value = np.median(finite)
                elif name == "trimmed_mean":
                    ordered = np.sort(finite); k = int(len(ordered)*trim)
                    value = ordered[k:len(ordered)-k].mean()
                elif name == "active_mean" and len(active):
                    value = active.mean()
                elif name == "area_normalized" and not bad:
                    # Do not silently treat unmeasured/nonfinite pixels as zero.
                    value = finite.sum()*area_px/area
            row[columns[name]] = float(value)
        row.update(value=row[columns[primary]], label=labels[primary], primary_method=primary)
        rows.append(row)
    return pd.DataFrame(rows), mask, columns, labels


class SpatialMeasurementsSection(PipelineSection):
    id = "image_measurements"
    title = "Thickness / ROI measurements"
    requires = ("image_stack", "image_times")
    optional_inputs = ("nci_calibrated", "pixel_size_um", "image_unit")
    produces = ("image_metric_values", "image_metric_label", "measurement_roi", "measurement_mask",
                "image_metrics_table", "measurement_columns", "measurement_labels")

    def run(self, session, config):
        table, mask, columns, labels = measure_stack(session.artifacts.require("image_stack"),
            session.artifacts.require("image_times"), config,
            pixel_size_um=session.artifacts.get("pixel_size_um"), unit=session.artifacts.get("image_unit", ""),
            calibrated=bool(session.artifacts.get("nci_calibrated", False)))
        # Preserve explicitly saved legacy labels for the legacy default mean.
        label = labels[table.primary_method.iloc[0]]
        if not config.get("methods") and config.get("label"):
            label = config["label"]; table["label"] = label
        return SectionOutcome(self.id, SectionState.COMPLETED,
            f"Measured {len(table)} frames using {', '.join(columns)}. Nonfinite and empty regions are flagged; images are unchanged.",
            {"image_metric_values": table.value.tolist(), "image_metric_label": label,
             "measurement_roi": config.get("roi") or {}, "measurement_mask": mask,
             "image_metrics_table": table, "measurement_columns": columns, "measurement_labels": labels})

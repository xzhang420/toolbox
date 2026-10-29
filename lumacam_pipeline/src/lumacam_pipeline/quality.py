"""Read-only image quality diagnostics; flags never change scientific values."""
import numpy as np
import pandas as pd

from .measurements import region_mask
from .section import PipelineSection, SectionOutcome, SectionState


def pixel_diagnostics(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {}, []
    mean = values.mean()
    deviation = values - mean
    variance_sum = np.sum(deviation * deviation)
    largest = values.max()
    p10, median, p90 = np.percentile(values, [10, 50, 90])
    influence = float((largest-mean)**2 / variance_sum) if variance_sum > 0 else 0.
    suspect = (len(values) >= 100 and influence > .1
               and largest > median + 20 * max(p90-p10, np.finfo(float).eps))
    flags=['suspected_bright_outlier'] if suspect else []
    if mean<0 and np.mean(values<0)>.95: flags.append('background_dominated_roi')
    return {"zero_fraction": float(np.mean(values == 0)),
            "negative_fraction": float(np.mean(values < 0)),
            "maximum": float(largest), "top_pixel_variance_fraction": influence}, flags


class ImageQualitySection(PipelineSection):
    id = "quality"
    title = "Image quality and electrode region"
    requires = ("image_stack", "image_times", "results_dir")
    optional_inputs = ("image_metrics_table", "pixel_size_um", "image_unit")
    produces = ("frame_quality_table", "frame_quality_path", "quality_summary", "quality_summary_path", "roi_preview_path")

    def run(self, session, config):
        import matplotlib.pyplot as plt
        from .handover import atomic_json
        from .sections.outputs import _category_dir
        images = session.artifacts.require("image_stack")
        measurement = session.section_config("image_measurements")
        mask = region_mask(np.shape(images[0]), measurement)
        rows = []
        for i, image in enumerate(images):
            array = np.asarray(image)
            values = array[mask]
            stats, flags = pixel_diagnostics(values)
            if np.all(array == 0): flags.append("all_zero_image")
            if np.all(values == 0): flags.append("all_zero_roi")
            if not np.isfinite(values).all(): flags.append("nonfinite_pixels")
            if not np.isfinite(values).any(): flags.append('no_valid_pixels')
            rows.append({"frame": i, "time_utc": session.artifacts.require("image_times")[i],
                         "roi_mean": float(np.nanmean(values)) if np.isfinite(values).any() else np.nan,
                         **stats, "quality_flags": ";".join(flags)})
        table = pd.DataFrame(rows)
        baseline = config.get("baseline_indices") or []
        if isinstance(baseline, str): baseline = [int(i.strip()) for i in baseline.split(",") if i.strip()]
        baseline = list(dict.fromkeys(map(int, baseline)))
        if any(i < 0 or i >= len(images) for i in baseline):
            raise ValueError("Baseline frame indices are outside the image stack.")
        summary = {"region": measurement.get("region", "circle" if measurement.get("roi") else "whole"),
                   "roi": measurement.get("roi", {}), "region_pixels": int(mask.sum()),
                   "electrode_boundary_confirmed": bool(measurement.get("electrode_boundary_confirmed", False)),
                   "flagged_frames": table.loc[table.quality_flags.ne(""), "frame"].tolist(),
                   "suspected_outlier_frames": table.loc[table.quality_flags.str.contains("suspected_bright"), "frame"].tolist(),
                   "baseline_indices": baseline,
                   "baseline_is_independent": bool(config.get("baseline_is_independent", False)),
                   "outlier_rule": ">=100 pixels; brightest pixel >10% of variance and >median+20*(P90-P10); diagnostic only"}
        summary['electrode_diameter_mm'] = measurement.get('electrode_diameter_mm')
        fit=config.get('centre_estimation') or {}
        summary['centre_estimation'] = fit if fit.get('roi')==measurement.get('roi') else {}
        if baseline:
            values = table.roi_mean.iloc[baseline].to_numpy()
            values = values[np.isfinite(values)]
            summary.update(baseline_mean=float(values.mean()) if len(values) else None,
                           baseline_temporal_sd=float(values.std(ddof=1)) if len(values)>1 else None)
        folder = _category_dir(session, "tables")
        path = folder / "frame_quality.csv"; table.to_csv(path, index=False, float_format="%.12g")
        summary_path = folder / "image_quality.json"; atomic_json(summary_path, summary)
        peak = int(np.nanargmax(table.roi_mean.to_numpy())) if table.roi_mean.notna().any() else 0
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), constrained_layout=True)
        try:
            for axis, index in zip(axes, (0, peak)):
                image = np.asarray(images[index], dtype=float)
                finite = image[np.isfinite(image)]
                low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
                axis.imshow(image, cmap="inferno", origin=session.section_config("image_plot").get("image_origin", "upper"),
                            vmin=low, vmax=high if high>low else low+1)
                if mask.any() and not mask.all(): axis.contour(mask, levels=[.5], colors=["cyan"], linewidths=.8)
                axis.set_title(f"Frame {index}: measurement region", fontsize=9)
                axis.set_axis_off()
            fig.suptitle("Electrode boundary confirmed" if summary["electrode_boundary_confirmed"]
                         else "ROI requires electrode-boundary confirmation", fontsize=10)
            preview = _category_dir(session, "figures") / "measurement_region.png"
            fig.savefig(preview, dpi=int(config.get("dpi", 140)))
        finally:
            plt.close(fig)
        return SectionOutcome(self.id, SectionState.COMPLETED,
            f"Checked {len(images)} frames; {len(summary['flagged_frames'])} flagged. Values unchanged; no signal threshold inferred.",
            {"frame_quality_table": table, "frame_quality_path": path, "quality_summary": summary,
             "quality_summary_path": summary_path, "roi_preview_path": preview})

"""Load, combine, select, and measure image stacks."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile

from .. import backends
from ..section import PipelineSection, SectionOutcome, SectionState


def _source_product_time(dataset, product, tpx_time=2.0):
    module = backends.nci_data()
    run = dataset.run(product.run)
    if run is not None:
        windows = module.get_exposure_windows(
            tpx_folder=str(run.tpx3_dir) if run.tpx3_dir else None,
            metadata_dir=str(run.metadata_dir) if run.metadata_dir else None,
            tpx_time=tpx_time,
        )
        if windows:
            return module.mean_exposure_midpoint(windows)
    return module._source_image_datetime(str(product.path))


class ImageLoadSection(PipelineSection):
    id = "images"
    title = "Image products"
    requires = ("dataset", "source_products")
    optional_inputs = ("processed_image_dir", "analysis_image_dir")
    produces = ("image_times", "image_stack", "image_paths")

    def run(self, session, config):
        dataset = session.artifacts.require("dataset")
        chosen_dir = session.artifacts.get("analysis_image_dir") or session.artifacts.get("processed_image_dir")
        crop = config.get("crop_roi") or {}
        r0, x0, y0 = crop.get("r0"), crop.get("x0"), crop.get("y0")
        if chosen_dir is not None:
            times, images, paths = backends.nci_plots().load_corrected_images_with_time(
                str(chosen_dir), r0=r0, x0=x0, y0=y0
            )
        else:
            records = []
            for product in session.artifacts.require("source_products"):
                try:
                    image = np.asarray(tifffile.imread(product.path), dtype=np.float32)
                except Exception as exc:
                    if config.get("skip_unreadable", True):
                        continue
                    raise ValueError(f"Cannot read {product.path}: {exc}") from exc
                if image.ndim != 2:
                    raise ValueError(f"Expected a 2D TIFF, got {image.shape}: {product.path}")
                if None not in (r0, x0, y0):
                    r0_i, x0_i, y0_i = int(r0), int(x0), int(y0)
                    image = image[max(0, y0_i-r0_i):y0_i+r0_i, max(0, x0_i-r0_i):x0_i+r0_i]
                timestamp = _source_product_time(dataset, product, float(config.get("tpx_time", 2)))
                records.append((timestamp, image, str(product.path)))
            records.sort(key=lambda row: row[0] if isinstance(row[0], datetime) else datetime.min)
            times = [row[0] for row in records]
            images = [row[1] for row in records]
            paths = [row[2] for row in records]
        if not images:
            raise ValueError("No readable images were loaded.")
        frame_merge = int(config.get("frames_per_merge", 1))
        if frame_merge > 1:
            times, images = backends.nci_plots().merge_images_by_frame_count(
                times, images, frame_merge, method=str(config.get("merge_method", "mean"))
            )
            paths = [f"merged_{index:05d}" for index in range(len(images))]
        spatial_merge = int(config.get("spatial_pixel_merge", 1))
        if spatial_merge > 1:
            images = backends.nci_plots().spatial_average_images(images, spatial_merge)
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Loaded {len(images)} image(s).",
            {"image_times": list(times), "image_stack": list(images), "image_paths": list(paths)},
        )


class AveragingSection(PipelineSection):
    id = "average"
    title = "Average image"
    requires = ("image_stack", "results_dir", "experiment")
    produces = ("average_image", "average_image_path")

    def run(self, session, config):
        images = session.artifacts.require("image_stack")
        method = str(config.get("method", "mean")).lower()
        stack = np.stack([np.asarray(image, dtype=np.float32) for image in images], axis=0)
        average = np.nanmedian(stack, axis=0) if method == "median" else np.nanmean(stack, axis=0)
        output_dir = Path(session.artifacts.require("results_dir")) / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / str(
            config.get("filename") or f"average_{session.artifacts.require('experiment')}.tiff"
        )
        tifffile.imwrite(output_path, average.astype(np.float32))
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Wrote {method} image to {output_path}",
            {"average_image": average, "average_image_path": output_path},
        )


class ImageMeasurementSection(PipelineSection):
    id = "image_measurements"
    title = "Image measurements"
    requires = ("image_stack",)
    optional_inputs = ("nci_calibrated",)
    produces = ("image_metric_values", "image_metric_label", "measurement_roi")

    def run(self, session, config):
        images = session.artifacts.require("image_stack")
        roi = config.get("roi") or {}
        r0, x0, y0 = roi.get("r0"), roi.get("x0"), roi.get("y0")
        values = []
        for image in images:
            array = np.asarray(image, dtype=float)
            if None not in (r0, x0, y0):
                values.append(
                    float(backends.nci_data().circular_roi_values(array, r0, x0, y0).mean())
                )
            else:
                values.append(float(np.nanmean(array)))
        calibrated = bool(session.artifacts.get("nci_calibrated", config.get("calibrated", False)))
        label = "Li thickness (μm)" if calibrated else str(config.get("label", "Mean image intensity (a.u.)"))
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Calculated {len(values)} image measurements.",
            {"image_metric_values": values, "image_metric_label": label, "measurement_roi": roi},
        )


class SelectionSection(PipelineSection):
    id = "selection"
    title = "Image selection"
    requires = ("image_stack",)
    optional_inputs = ("event_image_indices",)
    produces = ("selected_image_indices",)

    def run(self, session, config):
        count = len(session.artifacts.require("image_stack"))
        raw = config.get("indices", ())
        if isinstance(raw, str):
            raw = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
        if raw:
            indices = backends.nci_plots().normalize_selected_image_indices([int(value) for value in raw], count)
        elif config.get("use_detected_events", True) and session.artifacts.has("event_image_indices"):
            indices = backends.nci_plots().normalize_selected_image_indices(
                session.artifacts.require("event_image_indices"), count
            )
        else:
            number = min(int(config.get("representative_count", 4)), count)
            indices = sorted(set(np.linspace(0, count - 1, number).round().astype(int).tolist()))
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Selected image indices: {indices}",
            {"selected_image_indices": indices},
        )

"""Load, combine, select, and measure image stacks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from ..proposal_paths import analysis_root_for_proposal

from pathlib import Path

import numpy as np
import tifffile

from .. import backends
from ..average_collection import copy_average
from ..result_layout import average_filename
from ..products import is_all_runs_sum
from ..section import PipelineSection, SectionOutcome, SectionState
from ..white_spots import filter_settings, remove_white_spots


def preview_white_spots(session, index=0):
    """Read one pre-merge frame for a non-writing before/after preview."""
    chosen = session.artifacts.get("analysis_image_dir") or session.artifacts.get("processed_image_dir")
    if chosen is not None:
        paths = sorted(path for path in Path(chosen).iterdir()
                       if path.is_file() and path.suffix.lower() in {".tif", ".tiff"} and not is_all_runs_sum(path))
    else:
        paths = [product.path for product in session.artifacts.require("source_products")
                 if not product.aggregate and not is_all_runs_sum(product.path)]
    if index < 0 or index >= len(paths):
        raise ValueError(f"Preview frame must be between 0 and {len(paths) - 1}; discover data first.")
    path = Path(paths[index])
    original = np.asarray(tifffile.imread(path), dtype=np.float32)
    settings = filter_settings(session.section_config("images"))
    filtered, mask, stats = remove_white_spots(original, **{k: v for k, v in settings.items() if k != "enabled"})
    return path, original, filtered, mask, stats


def _write_filtered_tiff(path, array, source, timestamp):
    """Keep acquisition/beam/calibration tags and pixel spacing on derived TIFFs."""
    with tifffile.TiffFile(source) as tif:
        page = tif.pages[0]
        tags = []
        for code in (65000, 65001, 65002, 65003, 65004):
            if code in page.tags:
                tag = page.tags[code]
                tags.append((code, tag.dtype, tag.count, tag.value, False))
        if 65002 not in page.tags and isinstance(timestamp, datetime):
            tags.append((65002, "s", 0, timestamp.strftime("%Y-%m-%d %H:%M:%S"), False))
        kwargs = {}
        for tag_name, key in (("DateTime", "datetime"), ("ResolutionUnit", "resolutionunit")):
            if tag_name in page.tags:
                kwargs[key] = page.tags[tag_name].value
        if "XResolution" in page.tags and "YResolution" in page.tags:
            kwargs["resolution"] = (page.tags["XResolution"].value, page.tags["YResolution"].value)
        # Prefix rather than append: a JSON-shaped source description followed
        # by free text makes tifffile attempt to decode invalid shaped metadata.
        description = "WhiteSpotFilter: bright outliers replaced with local median\n" + (page.description or "")
    tifffile.imwrite(path, array, metadata=None, description=description, extratags=tags, **kwargs)


def _filter_loaded_images(session, images, paths, times, settings):
    root = Path(session.artifacts.require("results_dir"))
    output = root / "images" / "white_spot_filtered_frames"
    destinations = [output / f"{i:05d}_{Path(path).stem}_despotted.tiff" for i, path in enumerate(paths)]
    if output.exists() and set(output.glob("*.tiff")) - set(destinations):
        raise ValueError("Filtered-frame inputs changed. Use a new Analysis name to preserve existing results.")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    filtered = []
    for original, source, timestamp, destination in zip(images, paths, times, destinations):
        array, _, stats = remove_white_spots(original, **{k: v for k, v in settings.items() if k != "enabled"})
        _write_filtered_tiff(destination, array, source, timestamp)
        filtered.append(array)
        source_stat = Path(source).stat()
        records.append({"source": str(source), "output": str(destination),
                        "acquisition_time": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
                        "source_size": source_stat.st_size, "source_mtime_ns": source_stat.st_mtime_ns,
                        **stats})
    report = root / "processing" / "white_spot_filter.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    total = sum(row["replaced_pixels"] for row in records)
    report.write_text(json.dumps({
        "algorithm": "local_median_bright_outliers_v1", "inspired_by": "pyerre/proc_functions/filters.py:ro_filter_2D",
        "created_at": datetime.now(timezone.utc).isoformat(), "settings": settings,
        "stage": "after_selected_corrections_before_crop_and_merge",
        "frame_count": len(records), "replaced_pixels": total, "frames": records,
    }, indent=2), encoding="utf-8")
    message = f"White-spot filter replaced {total} pixel(s) in {len(records)} TIFF(s); report: {report}."
    if any(row["replaced_fraction"] > 0.01 for row in records):
        message += " WARNING: over 1% replaced in at least one frame; preview and consider a higher threshold."
    print(message)
    return filtered, [str(path) for path in destinations], report, message


def _source_product_time(dataset, product, tpx_time=2.0, metadata_only=False):
    module = backends.nci_data()
    run = dataset.run(product.run)
    if run is not None:
        windows = module.get_exposure_windows(
            tpx_folder=str(run.tpx3_dir) if run.tpx3_dir and not metadata_only else None,
            metadata_dir=str(run.metadata_dir) if run.metadata_dir else None,
            tpx_time=tpx_time,
        )
        if windows:
            return module.mean_exposure_midpoint(windows)
    if metadata_only:
        from ..image_io import read_image
        return read_image(product.path)[1]["timestamp"]
    return module._source_image_datetime(str(product.path))


class ImageLoadSection(PipelineSection):
    id = "images"
    title = "Image products"
    requires = ("dataset", "source_products")
    optional_inputs = ("processed_image_dir", "analysis_image_dir")
    produces = ("image_times", "image_stack", "image_paths", "source_tiff_count", "white_spot_report", "image_exposure_bounds")

    def run(self, session, config):
        dataset = session.artifacts.require("dataset")
        chosen_dir = session.artifacts.get("analysis_image_dir") or session.artifacts.get("processed_image_dir")
        crop = config.get("crop_roi") or {}
        r0, x0, y0 = crop.get("r0"), crop.get("x0"), crop.get("y0")
        filtering = bool(config.get("white_spot_enabled", False))
        processing = session.config.get("workflow", {}).get("role") == "processing"
        if processing and session.section_config("background").get("enabled", False) and not session.artifacts.has("analysis_image_dir"):
            raise ValueError("Background correction is enabled. Run Background correction before publishing images, or disable it.")
        settings = filter_settings(config) if filtering else None
        if chosen_dir is not None and processing:
            from ..image_io import read_image, natural_key
            records = []
            for path in sorted(Path(chosen_dir).iterdir(), key=natural_key):
                if path.suffix.lower() not in {".tif", ".tiff"} or is_all_runs_sum(path):
                    continue
                array, info = read_image(path, timestamp_timezone="")
                records.append((info["timestamp"], array, str(path)))
            times, images, paths = map(list, zip(*records)) if records else ([], [], [])
            # Correction TIFFs use historical local-naive tags. Prefer the
            # correction log, whose timestamp origin is validated JSON metadata.
            logs = session.artifacts.get("correction_result", {}).get("beam_log", [])
            by_name = {row["corrected_filename"]: row for row in logs}
            for i, path in enumerate(paths):
                row = by_name.get(Path(path).name, {})
                if row.get("exposure_window_count", 0) and row.get("time") is not None:
                    times[i] = row["time"]
        elif chosen_dir is not None:
            times, images, paths = backends.nci_plots().load_corrected_images_with_time(
                str(chosen_dir)
            )
        else:
            records = []
            for product in session.artifacts.require("source_products"):
                if product.aggregate or is_all_runs_sum(product.path):
                    continue
                try:
                    image = np.asarray(tifffile.imread(product.path), dtype=np.float32)
                except Exception as exc:
                    if config.get("skip_unreadable", True):
                        continue
                    raise ValueError(f"Cannot read {product.path}: {exc}") from exc
                if image.ndim != 2:
                    raise ValueError(f"Expected a 2D TIFF, got {image.shape}: {product.path}")
                timestamp = _source_product_time(dataset, product, float(config.get("tpx_time", 2)), processing)
                if processing and isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
                    timestamp = timestamp.astimezone().replace(tzinfo=None)
                records.append((timestamp, image, str(product.path)))
            records.sort(key=lambda row: row[0] if isinstance(row[0], datetime) else datetime.min)
            times = [row[0] for row in records]
            images = [row[1] for row in records]
            paths = [row[2] for row in records]
        if not images:
            raise ValueError("No readable images were loaded.")
        exposure_bounds=[None]*len(images)
        if processing:
            if chosen_dir is not None:
                for i,path in enumerate(paths):
                    row=by_name.get(Path(path).name,{})
                    start,end=row.get('exposure_start_utc'),row.get('exposure_end_utc')
                    if start and end: exposure_bounds[i]=(datetime.fromisoformat(start),datetime.fromisoformat(end))
            else:
                products=session.artifacts.require('source_products')
                for i,path in enumerate(paths):
                    product=next((p for p in products if str(p.path)==path),None)
                    run=dataset.run(product.run) if product else None
                    if run is None or sum(p.run==product.run for p in products)!=1: continue
                    windows=backends.nci_data().get_exposure_windows(metadata_dir=str(run.metadata_dir) if run.metadata_dir else None,
                                                                    tpx_folder=None,tpx_time=float(config.get('tpx_time',2)))
                    if windows:
                        exposure_bounds[i]=(min(w['start'] for w in windows),max(w['end'] for w in windows))
        source_tiff_count = len(images)
        report, filter_message = None, ""
        if filtering:
            images, paths, report, filter_message = _filter_loaded_images(session, images, paths, times, settings)
        if None not in (r0, x0, y0):
            r0_i, x0_i, y0_i = int(r0), int(x0), int(y0)
            origin = (max(0, x0_i-r0_i), max(0, y0_i-r0_i))
            cropped = []
            for image in images:
                image = np.asarray(image[origin[1]:y0_i+r0_i, origin[0]:x0_i+r0_i]).view(backends.nci_plots().ImageArray)
                image.crop_origin_xy = origin
                cropped.append(image)
            images = cropped
        frame_merge = int(config.get("frames_per_merge", 1))
        if frame_merge > 1:
            groups=[exposure_bounds[i:i+frame_merge] for i in range(0,len(exposure_bounds),frame_merge)]
            exposure_bounds=[(min(row[0] for row in group),max(row[1] for row in group))
                             if all(row is not None for row in group) else None for group in groups]
            if processing and any(time is None for time in times):
                method = str(config.get("merge_method", "mean"))
                if method not in {"mean", "sum", "median"}:
                    raise ValueError("Merge method must be mean, sum or median.")
                reducer = {"mean": np.nanmean, "sum": np.nansum, "median": np.nanmedian}[method]
                images = [reducer(np.stack(images[i:i+frame_merge]), axis=0)
                          for i in range(0, len(images), frame_merge)]
                times = [None] * len(images)
            else:
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
            f"Loaded {len(images)} image(s)." + (f" {filter_message}" if filter_message else ""),
            {
                "image_times": list(times),
                "image_stack": list(images),
                "image_paths": list(paths),
                "source_tiff_count": source_tiff_count,
                "white_spot_report": report,
                "image_exposure_bounds": exposure_bounds,
            },
        )


class AveragingSection(PipelineSection):
    id = "average"
    title = "Average image"
    requires = ("image_stack", "results_dir", "experiment")
    produces = ("average_image", "average_image_path", "average_collection_path")

    def run(self, session, config):
        images = session.artifacts.require("image_stack")
        source_tiff_count = int(session.artifacts.get("source_tiff_count", len(images)))
        method = str(config.get("method", "mean")).lower()
        stack = np.stack([np.asarray(image, dtype=np.float32) for image in images], axis=0)
        average = np.nanmedian(stack, axis=0) if method == "median" else np.nanmean(stack, axis=0)
        output_dir = Path(session.artifacts.require("results_dir")) / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / str(
            config.get("filename") or average_filename(
                session.artifacts.require("experiment"),
                session.section_config("project").get("variant", ""),
                session.section_config("project").get("analysis_name", ""),
            )
        )
        tifffile.imwrite(output_path, average.astype(np.float32))
        collection = None
        if config.get("collect_in_proposal", False):
            dataset = session.artifacts.get("dataset")
            proposal = dataset.proposal_root if dataset is not None else None
            analysis_root = (
                analysis_root_for_proposal(proposal) if proposal is not None
                else Path(session.artifacts.get("analysis_root", Path(session.artifacts.require("results_dir")).parent))
            )
            collection = copy_average(
                output_path, analysis_root, session.artifacts.require("experiment"),
                session.section_config("project").get("variant") or "default",
                session.section_config("project").get("analysis_name", ""),
            )
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Wrote {method} of {len(images)} loaded image(s) from "
            f"{source_tiff_count} source TIFF(s) to {output_path}"
            + (f"; comparison copy: {collection}" if collection else ""),
            {
                "average_image": average, "average_image_path": output_path,
                "average_collection_path": collection,
            },
            {
                "loaded_image_count": len(images),
                "source_tiff_count": source_tiff_count,
                "method": method,
            },
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

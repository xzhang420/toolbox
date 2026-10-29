"""Established ex-situ image processing retained by LumaCam compatibility APIs.

This module intentionally keeps the ex-situ workflow small:

1. Copy source TIFFs to ``results/uncorrected``.
2. Optionally apply beam-history intensity correction.
3. Optionally apply open-beam (OB) correction.
4. Optionally correct and subtract an external background (BG) dataset.
5. Write processed TIFFs to ``results/obint_corrected`` or ``results/final``.
6. Average the processed stack into one sample-named TIFF.

The correction math reuses the existing operando helpers where possible, but
the file discovery is more permissive for ex-situ sample folders.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from ..proposal_paths import analysis_root_for_proposal, proposal_root_for_layout

from pathlib import Path

import numpy as np
import tifffile


SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from lumacam_pipeline import backends as _lumacam_backends  # noqa: E402

operando_data = _lumacam_backends.nci_data()


@dataclass(frozen=True)
class ImageRecord:
    """One source TIFF and its optional TPX context."""

    path: Path
    output_name: str
    group_label: str
    group_index: int
    tpx_folder: Path | None = None
    metadata_dir: Path | None = None
    layout_type: str = "legacy"


def _clean_optional_path(path_value):
    value = str(path_value or "").strip()
    return str(Path(value).expanduser().resolve()) if value else None


def _sample_name_from_data_dir(data_dir):
    data_path = Path(data_dir).expanduser().resolve()
    try:
        layout = operando_data.resolve_nci_experiment(str(data_path))
        if layout["layout_type"] == "v2":
            return layout["experiment"]
    except ValueError:
        pass
    parent_name = data_path.parent.name
    return parent_name or data_path.name


def _default_results_path(data_dir):
    data_path = Path(data_dir).expanduser().resolve()
    try:
        layout = operando_data.resolve_nci_experiment(str(data_path))
        if layout["layout_type"] == "v2":
            proposal_root = proposal_root_for_layout(layout["layout_root"])
            return analysis_root_for_proposal(proposal_root) / layout["experiment"]
    except ValueError:
        pass
    return data_path.parent / "analysis" / data_path.name


def _is_tiff(path):
    try:
        return path.is_file() and operando_data.is_tiff_file(str(path)) and not operando_data.is_all_runs_sum(path)
    except Exception:
        return False


def _normalize_file_keywords(file_filter):
    if file_filter is None:
        return []
    if isinstance(file_filter, str):
        value = file_filter.strip()
        if not value:
            return []
        for sep in [",", ";"]:
            value = value.replace(sep, " ")
        return [part.strip() for part in value.split() if part.strip()]
    if isinstance(file_filter, (list, tuple, set)):
        return [str(part).strip() for part in file_filter if str(part).strip()]
    return [str(file_filter).strip()]


def _matches_file_keywords(filename, file_filter):
    keywords = _normalize_file_keywords(file_filter)
    if not keywords:
        return True
    return any(keyword in filename for keyword in keywords)


def _tiff_files(directory, file_filter=None):
    if not directory.is_dir():
        return []
    files = []
    for entry in sorted(directory.iterdir(), key=lambda p: p.name):
        if not _is_tiff(entry):
            continue
        if not _matches_file_keywords(entry.name, file_filter):
            continue
        files.append(entry)
    return files


def _unique_name(name, used):
    path = Path(name)
    stem = path.stem
    suffix = path.suffix or ".tif"
    candidate = f"{stem}{suffix}"
    idx = 2
    while candidate in used:
        candidate = f"{stem}_{idx}{suffix}"
        idx += 1
    used.add(candidate)
    return candidate


def discover_exsitu_images(data_dir, final_subfolder="final", file_filter=None):
    """Find ex-situ TIFF images in flat, legacy-run, or version-2 layouts."""

    data_path = Path(data_dir).expanduser().resolve()
    if not data_path.is_dir():
        raise ValueError(f"data_dir is not a directory: {data_path}")

    records = []
    used_names = set()
    is_v2 = operando_data._find_v2_layout_root(str(data_path)) is not None

    flat_files = _tiff_files(data_path, file_filter=file_filter)
    if flat_files and not is_v2:
        tpx_folder = data_path / "tpx3Files"
        tpx_folder = tpx_folder if tpx_folder.is_dir() else None
        metadata_dir = data_path / "metadata"
        metadata_dir = metadata_dir if metadata_dir.is_dir() else None
        for index, path in enumerate(flat_files):
            name = _unique_name(path.name if path.suffix else f"{path.name}.tif", used_names)
            records.append(
                ImageRecord(
                    path=path,
                    output_name=name,
                    group_label=data_path.name,
                    group_index=index,
                    tpx_folder=tpx_folder,
                    metadata_dir=metadata_dir,
                )
            )
        return records

    try:
        layout = operando_data.resolve_nci_experiment(
            str(data_path),
            final_subfolder=final_subfolder,
        )
    except ValueError:
        if is_v2:
            # A v2 metadata error must not silently fall back to stale TIFFs.
            raise
        layout = None
    if layout is not None:
        for run in layout["runs"]:
            if run.get("image_files") is not None:
                files = [
                    Path(path)
                    for path in run["image_files"]
                    if _is_tiff(Path(path))
                    and _matches_file_keywords(Path(path).name, file_filter)
                ]
            else:
                image_dir = Path(run["image_dir"])
                files = _tiff_files(image_dir, file_filter=file_filter)
            tpx_folder = Path(run["tpx_dir"])
            tpx_folder = tpx_folder if tpx_folder.is_dir() else None
            metadata_dir = Path(run["metadata_dir"]) if run.get("metadata_dir") else None
            metadata_dir = metadata_dir if metadata_dir and metadata_dir.is_dir() else None
            for index, path in enumerate(files):
                base = path.stem
                suffix = path.suffix or ".tif"
                raw_name = f"{run['name']}{suffix}"
                if len(files) > 1:
                    raw_name = f"{run['name']}_{base}{suffix}"
                records.append(
                    ImageRecord(
                        path=path,
                        output_name=_unique_name(raw_name, used_names),
                        group_label=run["name"],
                        group_index=index,
                        tpx_folder=tpx_folder,
                        metadata_dir=metadata_dir,
                        layout_type=run["layout_type"],
                    )
                )
        if records:
            return records
        if is_v2:
            raise ValueError(f"No non-excluded matching TIFF images found in {data_path}")

    for child in sorted(data_path.iterdir(), key=lambda p: p.name):
        if not child.is_dir() or child.name in {"results", "tpx3Files", "__pycache__"}:
            continue

        image_dir = child / final_subfolder
        if not image_dir.is_dir():
            image_dir = child
        files = _tiff_files(image_dir, file_filter=file_filter)
        if not files:
            continue

        tpx_folder = child / "tpx3Files"
        tpx_folder = tpx_folder if tpx_folder.is_dir() else None
        metadata_dir = child / "metadata"
        metadata_dir = metadata_dir if metadata_dir.is_dir() else None
        for index, path in enumerate(files):
            base = path.stem
            suffix = path.suffix or ".tif"
            if len(files) == 1:
                raw_name = f"{child.name}{suffix}"
            else:
                raw_name = f"{child.name}_{base}{suffix}"
            name = _unique_name(raw_name, used_names)
            records.append(
                ImageRecord(
                    path=path,
                    output_name=name,
                    group_label=child.name,
                    group_index=index,
                    tpx_folder=tpx_folder,
                    metadata_dir=metadata_dir,
                    layout_type="legacy",
                )
            )

    if not records:
        keywords = _normalize_file_keywords(file_filter)
        filter_msg = f" matching keyword(s) {keywords}" if keywords else ""
        raise ValueError(
            f"No TIFF images{filter_msg} found in {data_path}. Expected a flat TIFF folder "
            f"or run folders containing TIFFs or '{final_subfolder}' TIFF folders."
        )
    return records


def _copy_uncorrected(records, uncorrected_dir):
    uncorrected_path = Path(uncorrected_dir)
    uncorrected_path.mkdir(parents=True, exist_ok=True)
    copied = []
    for record in records:
        dst = uncorrected_path / record.output_name
        mean_tpx_time = _mean_tpx_time(record.tpx_folder, record.metadata_dir)
        operando_data.copy_tiff_with_metadata(
            str(record.path),
            str(dst),
            mean_tpx_modified_time=mean_tpx_time,
        )
        copied.append(dst)
    return copied


def _mean_tpx_time(tpx_folder, metadata_dir=None, tpx_time=2):
    windows = operando_data.get_exposure_windows(
        tpx_folder=str(tpx_folder) if tpx_folder is not None else None,
        metadata_dir=str(metadata_dir) if metadata_dir is not None else None,
        tpx_time=tpx_time,
    )
    return operando_data.mean_exposure_midpoint(windows)


def _group_tpx_by_image(records, n_tpx_per_image, tpx_time=2):
    groups_by_folder = {}
    for record in records:
        if record.tpx_folder is None and record.metadata_dir is None:
            continue
        groups_by_folder.setdefault((record.tpx_folder, record.metadata_dir), None)

    layout_by_folder = {
        (record.tpx_folder, record.metadata_dir): record.layout_type
        for record in records
        if record.tpx_folder is not None or record.metadata_dir is not None
    }
    image_count_by_folder = {}
    for record in records:
        if record.tpx_folder is not None or record.metadata_dir is not None:
            key = (record.tpx_folder, record.metadata_dir)
            image_count_by_folder[key] = max(
                image_count_by_folder.get(key, 0),
                record.group_index + 1,
            )

    for key in groups_by_folder:
        tpx_folder, metadata_dir = key
        exposure_windows = operando_data.get_exposure_windows(
            tpx_folder=str(tpx_folder) if tpx_folder is not None else None,
            metadata_dir=str(metadata_dir) if metadata_dir is not None else None,
            tpx_time=tpx_time,
        )
        if layout_by_folder.get(key) == "v2":
            groups = [exposure_windows for _ in range(image_count_by_folder.get(key, 1))]
        else:
            groups = []
            for start in range(0, len(exposure_windows), int(n_tpx_per_image)):
                group = exposure_windows[start:start + int(n_tpx_per_image)]
                if group:
                    groups.append(group)
        groups_by_folder[key] = groups

    return groups_by_folder


def _compute_intensity_factors(
    records,
    beamhistory_path,
    tpx_time,
    n_tpx_per_image,
    reference_beam_integral_per_minute=None,
):
    if beamhistory_path is None:
        return {record.path: 1.0 for record in records}, [], None, {}

    beamhistory = operando_data.load_beamhistory(beamhistory_path)
    tpx_groups = _group_tpx_by_image(records, n_tpx_per_image, tpx_time=tpx_time)
    beam_rows = []
    integrals_by_path = {}
    beam_metrics_by_path = {}
    times_by_path = {}
    windows_by_path = {}

    for record in records:
        group = None
        if record.tpx_folder is not None or record.metadata_dir is not None:
            groups = tpx_groups.get((record.tpx_folder, record.metadata_dir)) or []
            if record.group_index < len(groups):
                group = groups[record.group_index]
        windows_by_path[record.path] = group or []

        if not group:
            warnings.warn(
                f"No exposure manifest or TPX3 group found for {record.path}; intensity correction "
                "factor set to 1.0 for this image."
            )
            continue

        operando_data.validate_beamhistory_window_coverage(
            beamhistory,
            group,
            context=f"exsitu:{record.group_label}:{record.path.name}",
        )
        beam_metrics = operando_data.beam_metrics_for_windows(beamhistory, group)
        integral = beam_metrics["beam_integral_per_minute"]
        acq_time = operando_data.mean_exposure_midpoint(group)
        integrals_by_path[record.path] = integral
        beam_metrics_by_path[record.path] = beam_metrics
        times_by_path[record.path] = acq_time

    valid_integrals = [value for value in integrals_by_path.values() if np.isfinite(value)]
    if not valid_integrals:
        warnings.warn(
            "Beam-history path was supplied, but no usable TPX/image matches were "
            "found. Intensity correction was skipped."
        )
        return {record.path: 1.0 for record in records}, [], None, {}

    reference_integral = reference_beam_integral_per_minute
    if reference_integral is None:
        reference_integral = float(np.mean(valid_integrals))

    factors = {}
    for record in records:
        integral = integrals_by_path.get(record.path)
        if integral is None:
            factor = 1.0
            acq_time = _mean_tpx_time(
                record.tpx_folder, record.metadata_dir, tpx_time=tpx_time
            )
        else:
            relative_beam_factor = float(integral) / float(reference_integral)
            acq_time = times_by_path.get(record.path)
        record_windows = windows_by_path.get(record.path, [])
        provenance = operando_data.exposure_group_provenance(record_windows)
        if integral is None:
            relative_beam_factor = 1.0
            factor = 1.0
        else:
            factor = (
                relative_beam_factor * provenance["effective_exposure_fraction"]
            )
        factors[record.path] = factor
        beam_metrics = beam_metrics_by_path.get(record.path, {})
        beam_rows.append(
            {
                "group": record.group_label,
                "image_index": record.group_index,
                "filename": record.path.name,
                "corrected_filename": record.output_name,
                "time": acq_time,
                "beam_integral": integral,
                "beam_integral_per_minute": integral,
                "total_beam_integral": beam_metrics.get("total_beam_integral"),
                "tpx3_beam_integrals_json": json.dumps(
                    beam_metrics.get("tpx3_beam_integrals", []),
                    separators=(",", ":"),
                ),
                "relative_beam_factor": relative_beam_factor,
                "relative_factor": factor,
                "tpx_file_count": provenance["tpx_file_count"],
                "input_tpx_file_count": provenance["input_tpx_file_count"],
                "excluded_tpx_file_count": provenance["excluded_tpx_file_count"],
                "excluded_tpx_files": ";".join(provenance["excluded_tpx_files"]),
                "processing_exclusions_path": provenance["processing_exclusions_path"],
                "timing_metadata_paths": provenance["timing_metadata_paths"],
                "total_effective_exposure_s": provenance[
                    "included_effective_exposure_s"
                ],
                "nominal_effective_exposure_s": provenance[
                    "nominal_effective_exposure_s"
                ],
                "effective_exposure_fraction": provenance[
                    "effective_exposure_fraction"
                ],
                "exposure_timing_source": (
                    "manifest"
                    if record_windows and record_windows[0].get("manifest_path")
                    else ("raw_tpx3_fallback" if record_windows else None)
                ),
            }
        )

    return factors, beam_rows, float(reference_integral), times_by_path


def _load_ob_series(ob_path, final_subfolder="final", file_filter=None, dtype=np.float32):
    if ob_path is None:
        return []

    path = Path(ob_path).expanduser().resolve()
    if path.is_file():
        if not _is_tiff(path):
            raise ValueError(f"OB path is not a TIFF file: {path}")
        return [np.asarray(tifffile.imread(path), dtype=dtype)]

    records = discover_exsitu_images(
        path,
        final_subfolder=final_subfolder,
        file_filter=file_filter,
    )
    series = []
    for record in records:
        img = tifffile.imread(record.path)
        if np.asarray(img).ndim != 2:
            warnings.warn(f"Skipping non-2D OB image: {record.path}")
            continue
        series.append(np.asarray(img, dtype=dtype))
    return series


def _write_processed_images(
    records,
    output_dir,
    correction_factors,
    acquisition_times=None,
    reference_image=None,
    reference_thickness_um=1.0,
    compute_dtype=np.float32,
    output_dtype=np.float32,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written = []
    acquisition_times = acquisition_times or {}
    for record in records:
        dst = output_path / record.output_name
        acquisition_time = acquisition_times.get(record.path)
        if acquisition_time is None:
            acquisition_time = _mean_tpx_time(
                record.tpx_folder, record.metadata_dir
            )
        operando_data.correct_and_save_tiff(
            src_path=str(record.path),
            dst_path=str(dst),
            correction_factor=float(correction_factors.get(record.path, 1.0)),
            acquisition_time=acquisition_time,
            reference_image=reference_image,
            reference_thickness_um=reference_thickness_um,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
        written.append(dst)
    return written


def _average_tiffs(image_paths, output_path, compute_dtype=np.float32, output_dtype=np.float32):
    stack = []
    shape = None
    for path in image_paths:
        arr = np.asarray(tifffile.imread(path), dtype=compute_dtype)
        if arr.ndim != 2:
            warnings.warn(f"Skipping non-2D image during averaging: {path}")
            continue
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(
                f"Cannot average images with different shapes: {path} has {arr.shape}, "
                f"expected {shape}."
            )
        stack.append(arr)

    if not stack:
        raise ValueError("No valid 2D TIFF images were available for averaging.")

    avg = np.mean(np.stack(stack, axis=0), axis=0, dtype=compute_dtype)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(output_path), avg.astype(output_dtype))
    return output_path


def _median_tiffs(image_paths, compute_dtype=np.float32):
    stack = []
    shape = None
    for path in image_paths:
        arr = np.asarray(tifffile.imread(path), dtype=compute_dtype)
        if arr.ndim != 2:
            warnings.warn(f"Skipping non-2D background image: {path}")
            continue
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ValueError(
                f"Cannot combine background images with different shapes: {path} "
                f"has {arr.shape}, expected {shape}."
            )
        if np.all(arr == 0):
            warnings.warn(f"Skipping all-zero background image: {path}")
            continue
        stack.append(arr)
    if not stack:
        raise ValueError("No valid non-zero 2D TIFF images were available for background.")
    return np.median(np.stack(stack, axis=0), axis=0).astype(compute_dtype)


def _safe_filename_part(value):
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))
    return safe.strip("_") or "keyword"


def _keyword_average_path(results_path, sample_name, keyword):
    keyword_part = _safe_filename_part(keyword)
    sample_part = _safe_filename_part(sample_name)
    if sample_part == keyword_part or sample_part.endswith(f"_{keyword_part}"):
        stem = sample_part
    else:
        stem = f"{sample_part}_{keyword_part}"
    return Path(results_path) / f"{stem}.tif"


def _average_tiffs_by_keyword(
    image_paths,
    keywords,
    results_path,
    sample_name,
    compute_dtype=np.float32,
    output_dtype=np.float32,
):
    average_paths = {}
    for keyword in keywords:
        keyword_paths = [path for path in image_paths if keyword in Path(path).name]
        if not keyword_paths:
            warnings.warn(f"No processed TIFF files matched keyword '{keyword}' for averaging.")
            continue
        output_path = _keyword_average_path(results_path, sample_name, keyword)
        average_paths[keyword] = str(
            _average_tiffs(
                keyword_paths,
                output_path,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
            )
        )
    return average_paths


def run_exsitu_pipeline(
    data_dir,
    beamhistory_path=None,
    ob_path=None,
    bg_path=None,
    results_dir=None,
    sample_name=None,
    tpx_time=2,
    n_tpx_per_image=30,
    ob_bin_y=2,
    ob_bin_x=2,
    final_subfolder="final",
    file_filter=None,
    reference_beam_integral_per_minute=None,
    reference_thickness_um=1.0,
    background_correction=False,
    background_n_images=10,
    clip_negatives=True,
    compute_dtype=np.float32,
    output_dtype=np.float32,
    verbose=True,
    organized_output=True,
    output_paths=None,
    write_average=True,
):
    """Run the quick ex-situ image pipeline.

    Parameters
    ----------
    data_dir : str or path-like
        Folder containing sample TIFFs, either flat or in run folders.
    beamhistory_path : str or None, optional
        Beam-history CSV. If omitted, beam intensity correction is skipped.
    ob_path : str or None, optional
        Open-beam TIFF file or folder. If omitted, OB correction is skipped.
    bg_path : str or None, optional
        Background TIFF file or experiment folder. When supplied, its images
        receive the same available intensity/OB corrections, are averaged, and
        are subtracted from the sample. If omitted, background correction is
        skipped unless legacy ``background_correction=True`` is requested.
    results_dir : str or None, optional
        Output folder. The section-based LumaCam pipeline supplies the
        experiment-specific analysis directory explicitly.
    organized_output : bool, optional
        Place images under ``images/`` and correction metadata/intermediates
        under ``processing/``. This is the default; pass ``False`` only for the
        former flat legacy output.
    sample_name : str or None, optional
        Average TIFF stem. Defaults to the parent folder name of ``data_dir``.
    background_correction : bool, optional
        If True, compute a median background image from the first corrected
        images, subtract it from the corrected stack, and average the final
        background-subtracted images.
    background_n_images : int, optional
        Number of corrected images used to compute the median background.
    clip_negatives : bool, optional
        Clip negative values to zero during background subtraction.

    Returns
    -------
    dict
        Paths and counts for the created outputs.
    """

    data_path = Path(data_dir).expanduser().resolve()
    beamhistory_path = _clean_optional_path(beamhistory_path)
    ob_path = _clean_optional_path(ob_path)
    bg_path = _clean_optional_path(bg_path)
    sample_name = sample_name or _sample_name_from_data_dir(data_path)

    if results_dir is None:
        results_path = _default_results_path(data_path)
    else:
        results_path = Path(results_dir).expanduser().resolve()
    if organized_output:
        image_root = results_path / "images"
        processing_root = results_path / "processing"
        uncorrected_dir = image_root / "uncorrected"
        corrected_dir = image_root / "obint_corrected"
        final_dir = image_root / "final"
        background_output_path = processing_root / "background.tif"
        background_input_dir = processing_root / "background_corrected_input"
        average_root = image_root
    else:
        uncorrected_dir = results_path / "uncorrected"
        corrected_dir = results_path / "obint_corrected"
        final_dir = results_path / "final"
        background_output_path = results_path / "background.tif"
        background_input_dir = results_path / "background_corrected_input"
        average_root = results_path
    average_path = average_root / f"{sample_name}.tif"
    if output_paths:
        uncorrected_dir = Path(output_paths.get("uncorrected", uncorrected_dir))
        corrected_dir = Path(output_paths.get("corrected", corrected_dir))
    file_keywords = _normalize_file_keywords(file_filter)

    records = discover_exsitu_images(
        data_path,
        final_subfolder=final_subfolder,
        file_filter=file_filter,
    )
    if verbose:
        print(f"Found {len(records)} sample TIFF image(s).")

    copied_paths = _copy_uncorrected(records, uncorrected_dir)
    if verbose:
        print(f"Copied uncorrected TIFFs to: {uncorrected_dir}")

    factors, beam_log, reference_integral, acquisition_times = _compute_intensity_factors(
        records,
        beamhistory_path=beamhistory_path,
        tpx_time=tpx_time,
        n_tpx_per_image=n_tpx_per_image,
        reference_beam_integral_per_minute=reference_beam_integral_per_minute,
    )
    if verbose:
        if beamhistory_path is None:
            print("No beam-history path supplied; skipped intensity correction.")
        elif reference_integral is None:
            print("Intensity correction skipped because no usable TPX groups were found.")
        else:
            print(f"Applied intensity correction using reference integral {reference_integral:.6g}.")

    processed_ob = None
    ob_image_count = 0
    if ob_path is None:
        if verbose:
            print("No OB path supplied; skipped OB correction.")
    else:
        ob_series = _load_ob_series(
            ob_path,
            final_subfolder=final_subfolder,
            file_filter=file_filter,
            dtype=compute_dtype,
        )
        ob_image_count = len(ob_series)
        if not ob_series:
            raise ValueError(f"No usable OB TIFF images found in: {ob_path}")
        processed_ob = operando_data.process_reference_images(
            ob_series,
            bin_y=ob_bin_y,
            bin_x=ob_bin_x,
            compute_dtype=compute_dtype,
        )
        if verbose:
            print(f"Processed {ob_image_count} OB image(s) from: {ob_path}")

    corrected_paths = _write_processed_images(
        records,
        corrected_dir,
        correction_factors=factors,
        acquisition_times=acquisition_times,
        reference_image=processed_ob,
        reference_thickness_um=reference_thickness_um,
        compute_dtype=compute_dtype,
        output_dtype=output_dtype,
    )
    if verbose:
        print(f"Wrote processed TIFFs to: {corrected_dir}")

    average_input_paths = corrected_paths
    background_summary = None
    background_source_count = 0
    apply_background = bg_path is not None or bool(background_correction)
    if apply_background:
        if bg_path is not None:
            bg_source = Path(bg_path).expanduser().resolve()
            if bg_source.is_file():
                if not _is_tiff(bg_source):
                    raise ValueError(f"BG path is not a TIFF file: {bg_source}")
                bg_records = [
                    ImageRecord(
                        path=bg_source,
                        output_name=bg_source.name if bg_source.suffix else f"{bg_source.name}.tif",
                        group_label=bg_source.stem,
                        group_index=0,
                    )
                ]
            else:
                bg_records = discover_exsitu_images(
                    bg_source,
                    final_subfolder=final_subfolder,
                    file_filter=file_filter,
                )
            background_source_count = len(bg_records)
            bg_factors, _, _, bg_times = _compute_intensity_factors(
                bg_records,
                beamhistory_path=beamhistory_path,
                tpx_time=tpx_time,
                n_tpx_per_image=n_tpx_per_image,
                reference_beam_integral_per_minute=reference_integral,
            )
            bg_processed_paths = _write_processed_images(
                bg_records,
                background_input_dir,
                correction_factors=bg_factors,
                acquisition_times=bg_times,
                reference_image=processed_ob,
                reference_thickness_um=reference_thickness_um,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
            )
            if verbose:
                print(
                    f"Computing background from {len(bg_processed_paths)} corrected "
                    f"image(s) in: {bg_path}"
                )
            background = _median_tiffs(bg_processed_paths, compute_dtype=compute_dtype)
        else:
            if verbose:
                print(
                    "No external BG path supplied; legacy background mode is using "
                    f"the first {int(background_n_images)} corrected sample image(s)."
                )
            background = operando_data.compute_background_image(
                str(corrected_dir),
                n_images=int(background_n_images),
                compute_dtype=compute_dtype,
            )
            background_source_count = min(int(background_n_images), len(corrected_paths))

        background_output_path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(str(background_output_path), background.astype(output_dtype))
        if verbose:
            print(f"Wrote background image: {background_output_path}")

        background_summary = operando_data.background_subtract_directory(
            str(corrected_dir),
            str(final_dir),
            str(background_output_path),
            clip_negatives=bool(clip_negatives),
            skip_existing=False,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
        average_input_paths = [
            final_dir / path.name
            for path in corrected_paths
            if (final_dir / path.name).is_file()
        ]
        if verbose:
            print(f"Wrote background-subtracted TIFFs to: {final_dir}")
    elif verbose:
        print("No BG path supplied; skipped background correction.")

    keyword_average_paths = {}
    if not write_average:
        average_path = None
    elif file_keywords:
        keyword_average_paths = _average_tiffs_by_keyword(
            average_input_paths,
            file_keywords,
            average_root,
            sample_name,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
        if not keyword_average_paths:
            raise ValueError(
                "No keyword-specific averages were written. "
                f"Check file_filter={file_filter!r} and processed filenames."
            )
        if len(keyword_average_paths) == 1:
            average_path = Path(next(iter(keyword_average_paths.values())))
        else:
            average_path = None
        if verbose:
            print("Wrote keyword average image(s):")
            for keyword, path in keyword_average_paths.items():
                print(f"  {keyword}: {path}")
    else:
        average_path = _average_tiffs(
            average_input_paths,
            average_path,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
        )
        if verbose:
            print(f"Wrote average image: {average_path}")

    beam_log_path = None
    if beam_log:
        beam_log_path = (
            results_path / "processing" / "obint_beam_log.csv"
            if organized_output
            else results_path / "obint_beam_log.csv"
        )
        beam_log_path.parent.mkdir(parents=True, exist_ok=True)
        operando_data.save_obint_beam_log(beam_log, str(beam_log_path))

    return {
        "results_dir": str(results_path),
        "uncorrected_dir": str(uncorrected_dir),
        "obint_corrected_dir": str(corrected_dir),
        "final_dir": str(final_dir) if apply_background else None,
        "background_path": str(background_output_path) if apply_background else None,
        "background_source_path": bg_path,
        "background_input_dir": str(background_input_dir) if bg_path is not None else None,
        "average_path": str(average_path) if average_path is not None else None,
        "average_paths": keyword_average_paths,
        "beam_log_path": str(beam_log_path) if beam_log_path else None,
        "image_count": len(records),
        "uncorrected_count": len(copied_paths),
        "obint_corrected_count": len(corrected_paths),
        "final_count": len(average_input_paths) if apply_background else 0,
        "ob_image_count": ob_image_count,
        "intensity_corrected": beamhistory_path is not None and reference_integral is not None,
        "ob_corrected": processed_ob is not None,
        "background_corrected": apply_background,
        "background_source_count": background_source_count,
        "background_summary": background_summary,
    }


def _build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run the quick ex-situ neutron TIFF processing pipeline."
    )
    parser.add_argument("data_dir", help="Sample image folder.")
    parser.add_argument("--beamhistory", default=None, help="Optional beam-history CSV path.")
    parser.add_argument("--ob", default=None, help="Optional OB TIFF file or folder.")
    parser.add_argument("--bg", default=None, help="Optional background TIFF file or folder.")
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Experiment analysis folder. Default: ../analysis/<data_dir.name>",
    )
    parser.add_argument("--sample-name", default=None, help="Average TIFF stem.")
    parser.add_argument("--tpx-time", type=float, default=2, help="TPX exposure time in seconds.")
    parser.add_argument("--n-tpx-per-image", type=int, default=30, help="TPX files per image.")
    parser.add_argument("--ob-bin-y", type=int, default=2, help="OB binning in y.")
    parser.add_argument("--ob-bin-x", type=int, default=2, help="OB binning in x.")
    parser.add_argument("--final-subfolder", default="final", help="Run image subfolder name.")
    parser.add_argument(
        "--file-filter",
        default=None,
        help="Optional filename keyword(s), for example '3to1000' or '3to1000,5to1000'.",
    )
    parser.add_argument(
        "--background-correction",
        action="store_true",
        help="Compute and subtract a median background after OBINT correction.",
    )
    parser.add_argument(
        "--background-n-images",
        type=int,
        default=10,
        help="Number of corrected images used for median background.",
    )
    parser.add_argument(
        "--no-clip-negatives",
        action="store_true",
        help="Do not clip negative values after background subtraction.",
    )
    return parser


def main(argv=None):
    args = _build_arg_parser().parse_args(argv)
    result = run_exsitu_pipeline(
        data_dir=args.data_dir,
        beamhistory_path=args.beamhistory,
        ob_path=args.ob,
        bg_path=args.bg,
        results_dir=args.results_dir,
        sample_name=args.sample_name,
        tpx_time=args.tpx_time,
        n_tpx_per_image=args.n_tpx_per_image,
        ob_bin_y=args.ob_bin_y,
        ob_bin_x=args.ob_bin_x,
        final_subfolder=args.final_subfolder,
        file_filter=args.file_filter,
        background_correction=args.background_correction,
        background_n_images=args.background_n_images,
        clip_negatives=not args.no_clip_negatives,
    )
    print("Summary:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

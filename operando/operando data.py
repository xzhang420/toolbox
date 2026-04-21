import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tifffile
import warnings

# =============================================================================
# Beam history loader
# =============================================================================
"""
Load a beam history file into a pandas DataFrame.

The first column is treated as the start timestamp of each interval, and the
second column is treated as the average beam value for the interval from this
row's time to the next row's time.

Parameters
----------
path : str
    Path to the beam history file.

Returns
-------
pd.DataFrame
    DataFrame with columns:
    - 'timestamp' (datetime): interval start timestamps
    - 'beam_count' (float): average beam value for the interval
    The DataFrame is sorted by timestamp.
"""
def load_beamhistory(path):
    df = pd.read_csv(path, header=None, engine="python", sep=None)

    ts = pd.to_datetime(df.iloc[:, 0], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    beam_col_idx = 1

    if ts.isna().all() and df.shape[1] >= 3:
        ts = pd.to_datetime(
            df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str) + " " + df.iloc[:, 2].astype(str),
            format="%m/%d/%Y %I:%M:%S %p",
            errors="coerce",
        )
        beam_col_idx = 3
    elif ts.isna().all() and df.shape[1] >= 2:
        ts = pd.to_datetime(
            df.iloc[:, 0].astype(str) + " " + df.iloc[:, 1].astype(str),
            errors="coerce",
        )
        beam_col_idx = 2

    if ts.isna().all():
        raise ValueError(
            "Could not parse timestamps from the first columns. "
            "Expected format like '12/1/2025 4:00:10 AM'."
        )

    beam_vals = pd.to_numeric(df.iloc[:, beam_col_idx], errors="coerce")

    out = pd.DataFrame({"timestamp": ts, "beam_count": beam_vals})
    out = out.dropna(subset=["timestamp"]).reset_index(drop=True)
    return out.sort_values("timestamp").reset_index(drop=True)


# =============================================================================
# Get exposure times from TPX folder
# =============================================================================
"""
Extract the modification times of all TPX files in a folder, sorted earliest to latest.

Parameters
----------
tpx_folder : str
    Path to the folder containing TPX files.
extension : str, optional
    File extension to filter by (default is ".tpx3").

Returns
-------
list of datetime.datetime
    Sorted list of acquisition end times corresponding to each TPX file.
"""
def get_exposure_times(tpx_folder, extension=".tpx3"):
    times = []
    for f in sorted(os.listdir(tpx_folder)):
        if f.endswith(extension):
            ts = datetime.fromtimestamp(os.stat(os.path.join(tpx_folder, f)).st_mtime)
            times.append(ts)
    return times


# =============================================================================
# Datetime helpers
# =============================================================================
"""
Compute the mean datetime for a list of datetime objects.

Parameters
----------
times : list of datetime.datetime
    Datetimes to average.

Returns
-------
datetime.datetime or None
    Mean datetime rounded to the nearest second, or None if input is empty.
"""
def mean_datetime(times):
    if not times:
        return None
    avg_ts = sum(t.timestamp() for t in times) / len(times)
    return datetime.fromtimestamp(round(avg_ts))


# =============================================================================
# TIFF helpers
# =============================================================================
"""
Check whether a file is a TIFF image, allowing extensionless TIFFs.
"""
def is_tiff_file(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return True
    if ext == "":
        try:
            with tifffile.TiffFile(path):
                return True
        except Exception:
            return False
    return False


def _normalize_metadata_value(value):
    """
    Convert values into JSON-friendly, comparison-stable representations.
    """
    if isinstance(value, np.dtype):
        return value.name
    if isinstance(value, type) and issubclass(value, np.generic):
        return np.dtype(value).name
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _normalize_metadata_value(v) for k, v in value.items()}
    return value


def write_stage_metadata(metadata_path, stage_name, parameters, extra=None, section_name=None):
    """
    Write or update a JSON metadata manifest for a pipeline stage.

    Metadata is consolidated into one file per results folder. The file contains
    top-level ``sections`` such as ``obint_correction`` and
    ``background_processing``. A section may either store one stage directly or
    contain nested stage entries.
    """
    payload = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "parameters": _normalize_metadata_value(parameters),
    }
    if extra is not None:
        payload["extra"] = _normalize_metadata_value(extra)

    existing = read_stage_metadata(metadata_path, stage_name=None, section_name=None)
    if existing is None:
        existing = {"sections": {}}

    section_key = section_name or stage_name
    sections = existing.setdefault("sections", {})

    if section_name is None:
        sections[section_key] = payload
    else:
        section_payload = sections.setdefault(
            section_key,
            {"updated_at": payload["updated_at"]},
        )
        section_payload["updated_at"] = payload["updated_at"]
        section_payload[stage_name] = payload

    existing["updated_at"] = payload["updated_at"]

    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, sort_keys=True)

    return existing


def read_stage_metadata(metadata_path, stage_name=None, section_name=None):
    """
    Read a JSON metadata manifest, or one stage within it, if it exists.

    Parameters
    ----------
    metadata_path : str
        Path to the JSON metadata file.
    stage_name : str or None, optional
        Stage name to retrieve. If None, return the full metadata payload.
    section_name : str or None, optional
        Optional top-level section containing the stage.
    """
    if not os.path.isfile(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as exc:
        warnings.warn(f"Could not read metadata file {metadata_path}: {exc}")
        return None

    if stage_name is None and section_name is None:
        return payload

    sections = payload.get("sections", {})
    if section_name is None:
        return sections.get(stage_name)

    section_payload = sections.get(section_name, {})
    if stage_name is None:
        return section_payload
    return section_payload.get(stage_name)


def compare_stage_metadata(expected_parameters, stored_metadata):
    """
    Compare expected stage parameters against a stored metadata manifest.
    """
    expected = _normalize_metadata_value(expected_parameters)
    stored_parameters = (stored_metadata or {}).get("parameters", {})

    differences = []
    for key in sorted(expected.keys()):
        expected_value = expected[key]
        if key not in stored_parameters:
            differences.append(
                {"key": key, "expected": expected_value, "stored": "<missing>"}
            )
            continue
        stored_value = stored_parameters[key]
        if expected_value != stored_value:
            differences.append(
                {"key": key, "expected": expected_value, "stored": stored_value}
            )

    return differences


def print_stage_metadata_warnings(stage_name, metadata_path, differences):
    """
    Print readable warnings for parameter mismatches against stored metadata.
    """
    if not differences:
        return

    print(
        f"WARNING: existing {stage_name} outputs were created with different "
        f"parameters than the current notebook settings."
    )
    print(f"Metadata file: {metadata_path}")
    for diff in differences:
        print(
            f"  - {diff['key']}: current={diff['expected']} | stored={diff['stored']}"
        )


# =============================================================================
# Reference image processing (bin -> average -> clamp -> upsample)
# =============================================================================
def block_mean(img, bin_y=2, bin_x=2):
    h, w = img.shape
    h2 = (h // bin_y) * bin_y
    w2 = (w // bin_x) * bin_x
    cropped = img[:h2, :w2]
    return cropped.reshape(h2 // bin_y, bin_y, w2 // bin_x, bin_x).mean(axis=(1, 3))


def upsample_nearest(img, bin_y=2, bin_x=2, target_shape=None):
    up = np.repeat(np.repeat(img, bin_y, axis=0), bin_x, axis=1)
    if target_shape is not None:
        return up[:target_shape[0], :target_shape[1]]
    return up


def process_reference_images(ref_series, bin_y=2, bin_x=2, compute_dtype=np.float32):
    """
    1) Average reference images
    2) Compute global mean of the averaged image
    3) Clamp pixels below the global mean
    4) Bin (smooth) the clamped image
    5) Upsample back to original size for division
    """
    if len(ref_series) == 0:
        raise ValueError("ref_series is empty.")

    ref_stack = np.stack(
        [np.asarray(img, dtype=compute_dtype) for img in ref_series],
        axis=0,
    )
    ref_avg = np.mean(ref_stack, axis=0, dtype=compute_dtype)

    global_mean = float(np.mean(ref_avg))
    ref_clamped = np.maximum(ref_avg, global_mean)

    ref_binned = block_mean(ref_clamped, bin_y=bin_y, bin_x=bin_x)

    ref_processed = upsample_nearest(
        ref_binned,
        bin_y=bin_y,
        bin_x=bin_x,
        target_shape=ref_series[0].shape,
    )
    return ref_processed


def load_reference_image_series(
    reference_dir,
    final_subfolder="final",
    dtype=np.float32,
):
    """
    Load all 2D reference TIFF images from each run's ``final_subfolder``.

    Parameters
    ----------
    reference_dir : str
        Root directory containing reference run subfolders.
    final_subfolder : str, optional
        Per-run subfolder containing the reference TIFF images.
    dtype : numpy dtype, optional
        Data type used when loading the images.

    Returns
    -------
    list of np.ndarray
        Reference image stack in run/file sort order.
    """
    ref_series = []
    for run_dir in sorted(os.listdir(reference_dir)):
        run_path = os.path.join(reference_dir, run_dir)
        if not os.path.isdir(run_path):
            continue
        final_dir = os.path.join(run_path, final_subfolder)
        if not os.path.isdir(final_dir):
            continue

        for fname in sorted(os.listdir(final_dir)):
            path = os.path.join(final_dir, fname)
            if not os.path.isfile(path) or not is_tiff_file(path):
                continue
            try:
                img = tifffile.imread(path)
                if img.ndim != 2:
                    continue
                ref_series.append(np.asarray(img, dtype=dtype))
            except Exception:
                continue
    return ref_series


# =============================================================================
# Run-level beam metrics
# =============================================================================
"""
Compute run-level beam metrics from all TPX files in a run.

Parameters
----------
beamhistory : pd.DataFrame
    DataFrame containing beam history with 'timestamp' and 'beam_count'.
tpx_folder : str
    Folder containing TPX files for a run.
tpx_time : int or float, optional
    Duration of each TPX exposure in seconds (default is 2).
context : str, optional
    Label used in error messages to identify the dataset.

Returns
-------
tuple
    (mean_beam_integral_per_minute, mean_tpx_modified_time)
"""
def compute_run_metrics(beamhistory, tpx_folder, tpx_time=2, context=None):
    exposure_times = get_exposure_times(tpx_folder)
    if len(exposure_times) == 0:
        return np.nan, None

    validate_beamhistory_coverage(
        beamhistory,
        exposure_times,
        tpx_time,
        context=context or os.path.basename(tpx_folder),
    )
    beam_integrals = integrate_beam(beamhistory, exposure_times, tpx_time)
    mean_beam_integral_per_minute = float(np.mean(beam_integrals))
    mean_tpx_time = mean_datetime(exposure_times)
    return mean_beam_integral_per_minute, mean_tpx_time


# =============================================================================
# Validate beam history coverage
# =============================================================================
"""
Validate that beam history covers the full time range of exposure windows.

Parameters
----------
beamhistory : pd.DataFrame
    DataFrame containing beam history with 'timestamp' and 'beam_count'.
exposure_times : list of datetime.datetime
    End times of each exposure.
tpx_time : int or float, optional
    Duration of each TPX exposure in seconds (default is 2).
context : str, optional
    Label used in error messages to identify the dataset.

Raises
------
ValueError
    If beam history does not fully cover the exposure windows.
"""
def validate_beamhistory_coverage(
    beamhistory,
    exposure_times,
    tpx_time=2,
    context="operando",
):
    if len(exposure_times) == 0:
        return

    t_min = min(exposure_times) - timedelta(seconds=tpx_time)
    t_max = max(exposure_times)

    bh_min = beamhistory["timestamp"].min()
    bh_max = beamhistory["timestamp"].max()

    if t_min < bh_min or t_max > bh_max:
        raise ValueError(
            f"Beam history does not cover all {context} exposures. "
            f"Exposure window range: {t_min} to {t_max}. "
            f"Beam history range: {bh_min} to {bh_max}."
        )


# =============================================================================
# Integrate beam counts over exposure windows (time-weighted, per minute)
# =============================================================================
"""
Compute average beam integral per minute for each exposure window.

This uses time-weighting: each beam_count value is treated as the average
over the interval from its timestamp to the next timestamp.

Parameters
----------
beamhistory : pd.DataFrame
    DataFrame containing beam history with 'timestamp' and 'beam_count'.
exposure_times : list of datetime.datetime
    End times of each exposure.
tpx_time : int or float, optional
    Duration of each TPX exposure in seconds (default is 2).

Returns
-------
np.ndarray
    Array of average beam integrals (per minute) for each exposure.
"""
def integrate_beam(beamhistory, exposure_times, tpx_time=2):
    if len(exposure_times) == 0:
        return np.array([])

    ts = beamhistory["timestamp"]
    next_ts = ts.shift(-1)
    beam = pd.to_numeric(beamhistory["beam_count"], errors="coerce")

    valid = next_ts.notna() & beam.notna()
    ts = ts[valid].reset_index(drop=True)
    next_ts = next_ts[valid].reset_index(drop=True)
    beam = beam[valid].astype(float).reset_index(drop=True)

    ts_ns = ts.values.astype("datetime64[ns]").astype("int64")
    next_ns = next_ts.values.astype("datetime64[ns]").astype("int64")

    beam_integrals_per_min = []
    for t_end in exposure_times:
        t_start = t_end - timedelta(seconds=tpx_time)
        t_start_ns = np.int64(pd.Timestamp(t_start).value)
        t_end_ns = np.int64(pd.Timestamp(t_end).value)

        overlap_start = np.maximum(ts_ns, t_start_ns)
        overlap_end = np.minimum(next_ns, t_end_ns)
        dur_ns = overlap_end - overlap_start
        dur_ns[dur_ns < 0] = 0

        dur_sec = dur_ns / 1e9
        window_sum = np.sum(beam.values * dur_sec)
        per_min = window_sum / tpx_time * 60.0
        beam_integrals_per_min.append(per_min)
        if np.all(dur_sec == 0):
            warnings.warn(
                f"No beam history overlap for exposure window {t_start} to {t_end}. "
                "Beam integral set to 0 for this window."
            )

    return np.array(beam_integrals_per_min)


# =============================================================================
# Compute intensity correction factors
# =============================================================================
"""
Compute intensity correction factors from per-minute beam integrals.

Parameters
----------
beam_integrals : array-like
    Average beam integrals per minute for each exposure.
reference_beam_integral_per_minute : float
    Reference beam integral per minute used to normalize images.

Returns
-------
np.ndarray
    Array of correction factors to apply to images.
"""
def compute_intensity_correction(beam_integrals, reference_beam_integral_per_minute):
    return beam_integrals / reference_beam_integral_per_minute


# =============================================================================
# Reference beam integral from reference data (average per minute)
# =============================================================================
"""
Compute the reference beam integral per minute from reference data.

Parameters
----------
reference_dir : str
    Root directory containing reference runs (same structure as operando data).
reference_beamhistory_path : str
    Path to the .dat file containing beam history for the reference data.
tpx_time : int or float, optional
    Duration of each TPX exposure in seconds (default is 2).

Returns
-------
float
    Mean beam integral per minute across all reference exposures.
"""
def compute_reference_beam_integral_per_minute(
    reference_dir,
    reference_beamhistory_path,
    tpx_time=2,
):
    beamhistory = load_beamhistory(reference_beamhistory_path)

    all_beam_integrals = []
    for run_dir in sorted(os.listdir(reference_dir)):
        tpx_folder = os.path.join(reference_dir, run_dir, "tpx3Files")
        if not os.path.isdir(tpx_folder):
            continue

        exposure_times = get_exposure_times(tpx_folder)
        validate_beamhistory_coverage(
            beamhistory,
            exposure_times,
            tpx_time,
            context=f"reference:{run_dir}",
        )
        beam_integrals = integrate_beam(beamhistory, exposure_times, tpx_time)
        all_beam_integrals.extend(beam_integrals.tolist())

    if len(all_beam_integrals) == 0:
        return np.nan

    return float(np.mean(all_beam_integrals))


# =============================================================================
# Apply correction to TIFF and save with metadata
# =============================================================================
"""
Apply intensity correction to a TIFF image and save with metadata.

Parameters
----------
src_path : str
    Path to the source TIFF file.
dst_path : str
    Path where the corrected TIFF will be saved.
correction_factor : float
    Factor by which to divide the image to normalize intensity.
acquisition_time : datetime.datetime, optional
    Acquisition time to be written into TIFF metadata (nearest second).
mean_tpx_modified_time : datetime.datetime, optional
    Mean modified time of corresponding TPX files (nearest second).
reference_image : np.ndarray or None, optional
    Processed averaged reference image used for pixel-wise normalization.
reference_thickness_um : float, optional
    Thickness represented by the reference image. After dividing the
    experiment image by the reference image, the result is multiplied by this
    factor so references other than 1 um can be used directly.
compute_dtype : data-type, optional
    Floating-point dtype used during correction math.
output_dtype : data-type, optional
    Floating-point dtype used when writing corrected TIFFs.

Notes
-----
Writes custom metadata into the TIFF header:
- 'AcquisitionTime': human-readable acquisition timestamp
- 'BeamCorrectionFactor': applied correction factor
- 'MeanTPXModifiedTime': mean modified time of corresponding TPX files
- 'ReferenceThicknessUm': physical thickness represented by the reference image

The corrected image is computed as:
1. beam-normalized experiment image = ``img / correction_factor``
2. direct thickness scaling = ``(beam-normalized image / reference_image) * reference_thickness_um``

This workflow assumes image intensity is directly proportional to thickness after
reference normalization; it does not apply a Beer-Lambert transform.
"""
def correct_and_save_tiff(
    src_path,
    dst_path,
    correction_factor,
    acquisition_time,
    mean_tpx_modified_time=None,
    reference_image=None,
    reference_thickness_um=1.0,
    compute_dtype=np.float32,
    output_dtype=np.float32,
):
    """
    Apply intensity correction to a TIFF image and save with metadata.
    Includes protection against zero/low beam conditions.
    """

    # Load image
    with tifffile.TiffFile(src_path) as tif:
        img = tif.asarray()

    img = np.asarray(img, dtype=compute_dtype)

    # ------------------------------------------------------------------
    # Beam intensity correction with low-beam protection
    # ------------------------------------------------------------------

    LOW_BEAM_THRESHOLD = 1e-6  # relative threshold

    if (not np.isfinite(correction_factor)) or (correction_factor <= LOW_BEAM_THRESHOLD):
        print(f"Low/zero beam detected for {src_path} → writing dark image")
        corrected = np.zeros_like(img, dtype=compute_dtype)
        low_beam_flag = True
    else:
        corrected = img / float(correction_factor)
        low_beam_flag = False

    # ------------------------------------------------------------------
    # Pixel-wise reference normalization (only if valid beam)
    # ------------------------------------------------------------------

    if reference_image is not None and not low_beam_flag:
        if corrected.shape != reference_image.shape:
            raise ValueError(
                f"Reference image shape {reference_image.shape} does not match "
                f"image shape {corrected.shape} for {src_path}."
            )

        safe_ref = np.asarray(reference_image, dtype=compute_dtype)

        corrected = np.divide(
            corrected,
            safe_ref,
            out=np.zeros_like(corrected, dtype=compute_dtype),
            where=safe_ref != 0,
        )
        corrected *= float(reference_thickness_um)

    # ------------------------------------------------------------------
    # Private TIFF tags
    # ------------------------------------------------------------------

    extratags = [
        (65001, "d", 1, float(correction_factor), False),  # BeamCorrectionFactor
        (65004, "d", 1, float(reference_thickness_um), False),  # ReferenceThicknessUm
    ]

    if acquisition_time is not None:
        extratags.append(
            (
                65000,
                "s",
                0,
                acquisition_time.strftime("%Y-%m-%d %H:%M:%S"),
                False,
            )
        )

    if mean_tpx_modified_time is not None:
        extratags.append(
            (
                65002,
                "s",
                0,
                mean_tpx_modified_time.strftime("%Y-%m-%d %H:%M:%S"),
                False,
            )
        )

    # Optional: flag low-beam frames
    if low_beam_flag:
        extratags.append((65003, "s", 0, "LOW_BEAM", False))

    # ------------------------------------------------------------------
    # Human-readable TIFF description text
    # ------------------------------------------------------------------

    desc_lines = []

    if acquisition_time is not None:
        desc_lines.append(
            f"AcquisitionTime: {acquisition_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    desc_lines.append(f"BeamCorrectionFactor: {float(correction_factor)}")
    desc_lines.append(f"ReferenceThicknessUm: {float(reference_thickness_um)}")

    if mean_tpx_modified_time is not None:
        desc_lines.append(
            f"MeanTPXModifiedTime: {mean_tpx_modified_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )

    if low_beam_flag:
        desc_lines.append("NOTE: Low/zero beam - image set to zero")

    description = "\n".join(desc_lines)

    # ------------------------------------------------------------------
    # Write TIFF
    # ------------------------------------------------------------------

    tifffile.imwrite(
        dst_path,
        corrected.astype(output_dtype),
        extratags=extratags,
        description=description,
    )
    
    
    
# =============================================================================
# High-level intensity correction pipeline
# =============================================================================
"""
Apply intensity correction to all images in an experiment using beam history.

Parameters
----------
exp_dir : str
    Root directory containing experiment runs (subfolders with TPX files).
image_dir : str
    Directory containing the original images to correct.
output_dir : str
    Directory where corrected images will be saved.
beamhistory_path : str
    Path to the .dat file containing the beam history.
tpx_time : int or float, optional
    Duration of each TPX exposure in seconds (default 2).
reference_beam_integral_per_minute : float, optional
    Reference beam integral per minute used to normalize images. If not
    provided, it will be computed from reference data.
reference_dir : str
    Root directory containing reference runs (same structure as operando data).
reference_beamhistory_path : str
    Path to the .dat file containing beam history for the reference data.
processed_reference_image : np.ndarray or None, optional
    Processed averaged reference image for pixel-wise division.
compute_dtype : data-type, optional
    Floating-point dtype used for all correction math.
corrected_output_dtype : data-type, optional
    Floating-point dtype used when writing corrected TIFFs.

Notes
-----
For each run, this function:
1. Loads beam history.
2. Extracts acquisition times from TPX files.
3. Computes beam integrals over exposure windows.
4. Computes correction factors.
5. Corrects images and writes them to output_dir with metadata.
"""
def intensity_correct_experiment(
    exp_dir,
    output_dir,
    beamhistory_path,
    tpx_time,
    reference_beam_integral_per_minute,
    reference_image=None,
    reference_thickness_um=1.0,
    compute_dtype=np.float32,
    corrected_output_dtype=np.float32,
    n_tpx_per_image=30,
    verbose=True,
):
    """
    Apply intensity correction run-by-run using grouped TPX files per image.

    Each corrected image is assigned one acquisition time and one beam
    correction factor derived from the corresponding TPX group.
    """

    beamhistory = load_beamhistory(beamhistory_path)
    os.makedirs(output_dir, exist_ok=True)

    beam_log = []

    for run_dir in sorted(os.listdir(exp_dir)):

        run_path = os.path.join(exp_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        tpx_folder = os.path.join(run_path, "tpx3Files")
        image_folder = os.path.join(run_path, "final")

        if not os.path.isdir(tpx_folder) or not os.path.isdir(image_folder):
            continue

        exposure_times = get_exposure_times(tpx_folder)
        image_files = sorted([
            f for f in os.listdir(image_folder)
            if os.path.isfile(os.path.join(image_folder, f)) and is_tiff_file(os.path.join(image_folder, f))
        ])

        if len(exposure_times) == 0 or len(image_files) == 0:
            continue

        # --- GROUP TPX FILES PER IMAGE ---
        beam_integrals_per_image = []
        grouped_times = []

        for i in range(0, len(exposure_times), n_tpx_per_image):
            group = exposure_times[i:i+n_tpx_per_image]

            if len(group) == 0:
                continue

            integrals = integrate_beam(beamhistory, group, tpx_time)

            mean_integral = float(np.mean(integrals))
            beam_integrals_per_image.append(mean_integral)
            grouped_times.append(mean_datetime(group))

        beam_integrals_per_image = np.array(beam_integrals_per_image)

        # --- CORRECTION FACTORS ---
        correction_factors = beam_integrals_per_image / reference_beam_integral_per_minute

        # --- DEBUG PRINT ---
        if verbose:
            print(f"\n[RUN {run_dir}]")
            print(f"  TPX files: {len(exposure_times)}")
            print(f"  Images: {len(image_files)}")
            print(f"  Groups formed: {len(beam_integrals_per_image)}")

        # --- SAFETY CHECK ---
        if len(image_files) != len(correction_factors):
            print(f"WARNING: mismatch in run {run_dir}")
            print(f"  Images: {len(image_files)}, Beam groups: {len(correction_factors)}")

        # --- PROCESS IMAGES ---
        for i, fname in enumerate(image_files):

            if i >= len(correction_factors):
                print(f"Skipping extra image: {fname}")
                break

            src = os.path.join(image_folder, fname)
            base, ext = os.path.splitext(fname)
            if not ext:
                ext = ".tif"
            dst_name = f"{run_dir}_obint_corrected{ext}"
            if len(image_files) > 1:
                dst_name = f"{run_dir}_obint_corrected_{base}{ext}"
            dst = os.path.join(output_dir, dst_name)

            cf = float(correction_factors[i])
            bi = float(beam_integrals_per_image[i])
            acq_time = grouped_times[i]

            # --- DEBUG PRINT PER IMAGE ---
            if verbose:
                print(f"[{i}] {fname}")
                print(f"    Beam integral (avg of TPX): {bi:.3f}")
                print(f"    Correction factor: {cf:.6f}")

            beam_log.append({
                "run": run_dir,
                "image_index": i,
                "filename": fname,
                "corrected_filename": dst_name,
                "time": acq_time,
                "beam_integral": bi,
                "relative_factor": cf,
            })

            correct_and_save_tiff(
                src_path=src,
                dst_path=dst,
                correction_factor=cf,
                acquisition_time=acq_time,
                reference_image=reference_image,
                reference_thickness_um=reference_thickness_um,
                compute_dtype=compute_dtype,
                output_dtype=corrected_output_dtype,
            )

    return beam_log

# =============================================================================
# Collect TIFF files from run folders (fixed)
# =============================================================================
def collect_and_copy_tiffs(
    exp_dir,
    final_subfolder="final",
    tiff_extensions=(".tif", ".tiff"),
    file_filter=None,
):
    """
    Copy TIFF files from each run's "final" folder into a new output folder and
    prefix filenames with the run folder name.

    Parameters
    ----------
    exp_dir : str
        Root directory containing run folders.
    final_subfolder : str, optional
        Subfolder name containing TIFF files (default 'final').
    tiff_extensions : tuple of str, optional
        TIFF extensions to include.
    file_filter : str or None
        Optional substring that must appear in the filename.

    Returns
    -------
    str
        Path to the auto-named output folder inside ``results_<exp_name>``.
    """

    exp_name = os.path.basename(os.path.abspath(exp_dir))
    parent_dir = os.path.dirname(os.path.abspath(exp_dir))
    results_dir = os.path.join(parent_dir, f"results_{exp_name}")

    output_dir = os.path.join(results_dir, f"uncorrected_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)

    for run_dir in sorted(os.listdir(exp_dir)):
        run_path = os.path.join(exp_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        final_dir = os.path.join(run_path, final_subfolder)
        if not os.path.isdir(final_dir):
            continue

        tpx_folder = os.path.join(run_path, "tpx3Files")
        exposure_times = get_exposure_times(tpx_folder) if os.path.isdir(tpx_folder) else []
        mean_tpx_time = mean_datetime(exposure_times) if exposure_times else None

        tiff_files = [
            f for f in sorted(os.listdir(final_dir))
            if (is_tiff_file(os.path.join(final_dir, f)) and
                (file_filter is None or file_filter in f))
        ]

        if len(tiff_files) == 0:
            warnings.warn(
                f"No TIFF images found in {final_dir} matching filter '{file_filter}'; skipping run."
            )
            continue

        if not exposure_times:
            warnings.warn(
                f"No TPX files found for run {run_dir}; uncorrected copies will "
                "not include MeanTPXModifiedTime."
            )

        for fname in tiff_files:
            src = os.path.join(final_dir, fname)
            base, ext = os.path.splitext(fname)
            if not ext:
                ext = ".tif"
            dst_name = f"{run_dir}_uncorrected{ext}"
            if len(tiff_files) > 1:
                dst_name = f"{run_dir}_uncorrected_{base}{ext}"
            dst = os.path.join(output_dir, dst_name)
            copy_tiff_with_metadata(src, dst, mean_tpx_time)

    return output_dir

# =============================================================================
# Copy TIFF with metadata
# =============================================================================
"""
Copy a TIFF image and add metadata for mean TPX modified time.

Parameters
----------
src_path : str
    Source TIFF path.
dst_path : str
    Destination TIFF path.
mean_tpx_modified_time : datetime.datetime or None
    Mean TPX modified time to write to metadata.
"""
def copy_tiff_with_metadata(src_path, dst_path, mean_tpx_modified_time=None):
    with tifffile.TiffFile(src_path) as tif:
        img = tif.asarray()

    extratags = []
    if mean_tpx_modified_time is not None:
        extratags.append(
            (65002, "s", 0, mean_tpx_modified_time.strftime("%Y-%m-%d %H:%M:%S"), False)
        )

    tifffile.imwrite(
        dst_path,
        img,
        extratags=extratags
    )


# =============================================================================
# Capacity events and nearest image indices
# =============================================================================
def find_capacity_turning_points(
    echem_datetimes,
    capacity_series,
    extra_events=None,
):
    """
    Find timestamps of turning points in Capacity/mA.h.
    (No zero detection, no automatic last event.)

    Parameters
    ----------
    echem_datetimes : list of datetime.datetime
        Electrochemical time series.
    capacity_series : array-like
        Capacity/mA.h values aligned with echem_datetimes.

    extra_events : list of dict, optional
        Additional precomputed events to include. Each dict may contain
        ``index``, ``time``, and ``labels``.

    Returns
    -------
    events : list of dict
        Each dict has keys: 'index', 'time', 'labels'.
        labels contains at least {"turning"} for derivative-based events.
    """
    times = np.asarray(echem_datetimes)
    cap = np.asarray(capacity_series, dtype=float)

    if len(cap) < 3:
        return []

    if len(times) != len(cap):
        raise ValueError("echem_datetimes and capacity_series must have the same length.")

    labels_by_idx = {}

    # Turning points (sign change in first derivative)
    d = np.diff(cap)
    sign_d = np.sign(d)
    sign_d[sign_d == 0] = np.nan
    prod_d = np.nan_to_num(sign_d[:-1]) * np.nan_to_num(sign_d[1:])
    turning_idx = set((np.where(prod_d < 0)[0] + 1).tolist())

    for idx in turning_idx:
        labels_by_idx.setdefault(idx, set()).add("turning")

    events = []
    for idx in sorted(labels_by_idx.keys()):
        events.append({
            "index": int(idx),
            "time": times[idx],
            "labels": ["turning"],
        })

    if extra_events:
        for event in extra_events:
            labels = event.get("labels", [])
            if isinstance(labels, str):
                labels = [labels]
            events.append({
                "index": event.get("index"),
                "time": event["time"],
                "labels": list(labels),
            })

        events.sort(key=lambda event: event["time"])

    return events


def find_capacity_turning_points_and_zeros(
    echem_datetimes,
    capacity_series,
    only_initial_zero=False,
    zero_tolerance=1e-12,
):
    """
    Find timestamps of turning points and zero crossings in Capacity/mA.h.

    Parameters
    ----------
    echem_datetimes : list of datetime.datetime
        Electrochemical time series.
    capacity_series : array-like
        Capacity/mA.h values aligned with echem_datetimes.
    only_initial_zero : bool, optional
        If True, keep only the first detected zero event.
    zero_tolerance : float, optional
        Absolute tolerance used when treating capacity values as zero.

    Returns
    -------
    events : list of dict
        Each dict has keys: 'index', 'time', 'labels'.
        labels contains one or both of {"turning", "zero"}.
    """
    times = np.asarray(echem_datetimes)
    cap = np.asarray(capacity_series, dtype=float)

    if len(times) != len(cap):
        raise ValueError("echem_datetimes and capacity_series must have the same length.")

    labels_by_idx = {}

    # Reuse the existing turning-point logic.
    for event in find_capacity_turning_points(times, cap):
        labels_by_idx.setdefault(event["index"], set()).update(event["labels"])

    # Detect exact/near zeros and sign changes through zero.
    zero_indices = set(np.where(np.isclose(cap, 0.0, atol=zero_tolerance))[0].tolist())

    sign_cap = np.sign(cap)
    nonzero_sign_cap = sign_cap.astype(float)
    nonzero_sign_cap[nonzero_sign_cap == 0] = np.nan
    sign_change_idx = np.where(
        np.nan_to_num(nonzero_sign_cap[:-1]) * np.nan_to_num(nonzero_sign_cap[1:]) < 0
    )[0]

    for idx in sign_change_idx:
        left = abs(cap[idx])
        right = abs(cap[idx + 1])
        zero_indices.add(int(idx if left <= right else idx + 1))

    zero_indices = sorted(zero_indices)
    if only_initial_zero and zero_indices:
        zero_indices = zero_indices[:1]

    for idx in zero_indices:
        labels_by_idx.setdefault(int(idx), set()).add("zero")

    events = []
    for idx in sorted(labels_by_idx.keys()):
        events.append({
            "index": int(idx),
            "time": times[idx],
            "labels": sorted(labels_by_idx[idx]),
        })

    return events



def find_nearest_image_indices(event_times, img_datetimes):
    """
    For each event time, find the index of the closest image timestamp.

    The returned indices are unique: if multiple event times map to the same
    nearest image, only the first match is kept. No maximum allowed time
    difference is enforced here.

    Parameters
    ----------
    event_times : list of datetime.datetime
    img_datetimes : list of datetime.datetime

    Returns
    -------
    list of int
        Indices into img_datetimes.
    """
    img_times = np.asarray(img_datetimes)
    if len(img_times) == 0:
        return []
    indices = []
    seen = set()
    for t in event_times:
        deltas = np.abs(img_times - t)
        idx = int(np.argmin(deltas))
        if idx in seen:
            continue
        seen.add(idx)
        indices.append(idx)
    return indices


# =============================================================================
# High-level pipeline function (fixed)
# =============================================================================
def run_operando_pipeline(
    exp_dir,
    beamhistory_path,
    tpx_time=2,
    n_tpx_per_image=30,
    reference_dir=None,
    reference_beamhistory_path=None,
    reference_bin_y=2,
    reference_bin_x=2,
    reference_final_subfolder="final",
    reference_thickness_um=1.0,
    compute_dtype=np.float32,
    corrected_output_dtype=np.float32,
    file_filter=None,
    metadata_path=None,
):
    """
    Run the full obint-correction workflow for one experiment.

    This function:
    1. Computes the reference beam integral per minute
    2. Loads and processes the reference image stack
    3. Collects auto-named uncorrected TIFF copies for provenance
    4. Applies beam and reference normalization to the auto-named corrected images
    5. Writes a JSON metadata manifest describing the correction settings

    Returns
    -------
    dict
        Paths to the corrected and uncorrected outputs, the reference beam
        integral, the per-image beam log, and the metadata-manifest path.
    """

    exp_name = os.path.basename(os.path.abspath(exp_dir))
    parent_dir = os.path.dirname(os.path.abspath(exp_dir))

    results_dir = os.path.join(parent_dir, f"results_{exp_name}")
    os.makedirs(results_dir, exist_ok=True)

    if reference_dir is None or reference_beamhistory_path is None:
        raise ValueError(
            "reference_dir and reference_beamhistory_path are required to compute "
            "reference_beam_integral_per_minute."
        )

    output_dir = os.path.join(results_dir, f"obint_corrected_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)

    print("Computing reference beam integral per minute...")
    reference_beam_integral_per_minute = compute_reference_beam_integral_per_minute(
        reference_dir,
        reference_beamhistory_path,
        tpx_time,
    )

    print("Loading reference images...")
    ref_series = load_reference_image_series(
        reference_dir,
        final_subfolder=reference_final_subfolder,
        dtype=compute_dtype,
    )
    if len(ref_series) == 0:
        raise ValueError(
            "No reference TIFF images found. "
            "Check reference_dir and reference_final_subfolder."
        )

    print("Processing reference images...")
    processed_reference_image = process_reference_images(
        ref_series,
        bin_y=reference_bin_y,
        bin_x=reference_bin_x,
        compute_dtype=compute_dtype,
    )

    print("Collecting uncorrected TIFFs...")
    uncorrected_dir = collect_and_copy_tiffs(
        exp_dir,
        file_filter=file_filter,
    )

    print("Applying intensity correction to operando images...")
    beam_log = intensity_correct_experiment(
        exp_dir=exp_dir,
        output_dir=output_dir,
        beamhistory_path=beamhistory_path,
        tpx_time=tpx_time,
        reference_beam_integral_per_minute=reference_beam_integral_per_minute,
        reference_image=processed_reference_image,
        reference_thickness_um=reference_thickness_um,
        n_tpx_per_image=n_tpx_per_image,
        compute_dtype=compute_dtype,
        corrected_output_dtype=corrected_output_dtype,
    )

    if metadata_path is None:
        metadata_path = os.path.join(results_dir, "processing_metadata.json")

    write_stage_metadata(
        metadata_path,
        stage_name="obint_correction",
        parameters={
            "exp_dir": os.path.abspath(exp_dir),
            "beamhistory_path": os.path.abspath(beamhistory_path),
            "output_dir": os.path.abspath(output_dir),
            "uncorrected_dir": os.path.abspath(uncorrected_dir),
            "tpx_time": tpx_time,
            "n_tpx_per_image": n_tpx_per_image,
            "reference_dir": os.path.abspath(reference_dir),
            "reference_beamhistory_path": os.path.abspath(reference_beamhistory_path),
            "reference_bin_y": reference_bin_y,
            "reference_bin_x": reference_bin_x,
            "reference_final_subfolder": reference_final_subfolder,
            "reference_thickness_um": reference_thickness_um,
            "compute_dtype": compute_dtype,
            "corrected_output_dtype": corrected_output_dtype,
        },
        extra={
            "reference_beam_integral_per_minute": reference_beam_integral_per_minute,
            "beam_log_count": len(beam_log),
        },
        section_name=None,
    )

    return {
        "results_dir": results_dir,
        "corrected_dir": output_dir,
        "uncorrected_dir": uncorrected_dir,
        "reference_beam_integral_per_minute": reference_beam_integral_per_minute,
        "beam_log": beam_log,
        "metadata_path": metadata_path,
    }
    
    

def compute_background_image(
    image_dir,
    n_images=10,
    compute_dtype=np.float32,
    metadata_path=None,
):
    """
    Compute a background image from the first valid corrected images in a directory.

    The background is the pixel-wise median of the first ``n_images`` readable,
    non-zero TIFF images in sort order. If ``metadata_path`` is provided, a JSON
    manifest describing the background settings is written alongside the result.

    Parameters
    ----------
    image_dir : str
        Directory containing TIFF images.
    n_images : int, optional
        Number of images to consider from the start (default = 10).
    compute_dtype : numpy dtype
        Data type used for computation.

    Returns
    -------
    np.ndarray
        Background image in ``compute_dtype``.
    """

    import os
    import numpy as np
    import tifffile

    files = sorted(os.listdir(image_dir))

    valid_images = []
    count_attempted = 0

    for fname in files:
        path = os.path.join(image_dir, fname)

        if not os.path.isfile(path) or not is_tiff_file(path):
            continue

        # Stop after collecting n_images valid, non-zero TIFFs.
        if len(valid_images) >= n_images:
            break

        try:
            img = tifffile.imread(path).astype(compute_dtype)
        except Exception:
            continue

        count_attempted += 1

        # Skip all-zero images
        if np.all(img == 0):
            print(f"Skipping zero image: {fname}")
            continue

        valid_images.append(img)

    if len(valid_images) == 0:
        raise ValueError("No valid (non-zero) images found for background computation.")

    # Stack the selected frames and take their median.
    stack = np.stack(valid_images, axis=0)
    background = np.median(stack, axis=0)

    print(
        f"Background computed from {len(valid_images)} valid images "
        f"(from {count_attempted} TIFFs successfully read)."
    )

    if metadata_path is not None:
        write_stage_metadata(
            metadata_path,
            stage_name="background_image",
            parameters={
                "image_dir": os.path.abspath(image_dir),
                "n_images": n_images,
                "compute_dtype": compute_dtype,
            },
            extra={
                "valid_images_used": len(valid_images),
                "tiffs_successfully_read": count_attempted,
            },
            section_name="background_processing",
        )

    return background.astype(compute_dtype)


# =============================================================================
# Background subtraction (post-processing step)
# =============================================================================
"""
Background subtraction for corrected operando images.

This step is intended to be run AFTER intensity correction, once a suitable
background image has been constructed.

It preserves the image timestamp and beam-correction TIFF tags written during
obint correction.
"""


def read_custom_tags(path):
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            return {
                code: tags[code].value
                for code in (65000, 65001, 65002)
                if code in tags
            }
    except Exception:
        return {}


def background_subtract_and_save(
    src_path,
    dst_path,
    background_image,
    clip_negatives=True,
    compute_dtype=np.float32,
    output_dtype=np.float32,
):
    """
    Subtract background from a single TIFF and preserve metadata.
    """

    # Load image
    img = tifffile.imread(src_path).astype(compute_dtype)

    if img.shape != background_image.shape:
        raise ValueError(
            f"Background shape {background_image.shape} does not match "
            f"image shape {img.shape} for {src_path}"
        )

    # Subtract background
    bgsub = img - background_image

    if clip_negatives:
        bgsub = np.maximum(bgsub, 0)

    # Preserve metadata
    tags = read_custom_tags(src_path)
    extratags = []

    if 65001 in tags:
        extratags.append((65001, "d", 1, float(tags[65001]), False))
    if 65000 in tags:
        extratags.append((65000, "s", 0, str(tags[65000]), False))
    if 65002 in tags:
        extratags.append((65002, "s", 0, str(tags[65002]), False))

    tifffile.imwrite(
        dst_path,
        bgsub.astype(output_dtype),
        extratags=extratags,
    )


def background_subtract_directory(
    input_dir,
    output_dir,
    background_path,
    clip_negatives=True,
    skip_existing=True,
    compute_dtype=np.float32,
    output_dtype=np.float32,
    metadata_path=None,
):
    """
    Apply background subtraction to all TIFF images in a directory.

    Parameters
    ----------
    input_dir : str
        Directory containing corrected TIFF images.
    output_dir : str
        Directory where background-subtracted images will be written.
    background_path : str
        Path to background TIFF image.
    clip_negatives : bool, optional
        Whether to clip negative values to zero.
    skip_existing : bool, optional
        Skip files that already exist in output_dir.
    compute_dtype : numpy dtype, optional
        Data type used while loading and subtracting the images.
    output_dtype : numpy dtype, optional
        Data type used when writing the final TIFF files.
    metadata_path : str or None, optional
        If provided, write a JSON manifest describing the subtraction settings.

    Returns
    -------
    dict
        Processing summary with counts, error details, and optionally the
        metadata-manifest path.
    """

    os.makedirs(output_dir, exist_ok=True)

    # Load background once
    background_image = tifffile.imread(background_path).astype(compute_dtype)

    files = sorted(os.listdir(input_dir))
    tiff_files = [
        fname for fname in files
        if os.path.isfile(os.path.join(input_dir, fname))
        and is_tiff_file(os.path.join(input_dir, fname))
    ]

    processed_count = 0
    skipped_existing_count = 0
    error_count = 0
    errors = []

    for i, fname in enumerate(tiff_files, 1):
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)

        if skip_existing and os.path.exists(dst):
            skipped_existing_count += 1
            continue

        print(f"[{i}/{len(tiff_files)}] {fname}")

        try:
            background_subtract_and_save(
                src,
                dst,
                background_image,
                clip_negatives=clip_negatives,
                compute_dtype=compute_dtype,
                output_dtype=output_dtype,
            )
            processed_count += 1
        except Exception as e:
            error_count += 1
            errors.append((fname, str(e)))
            print(f"Skipping {fname}: {e}")

    print(
        f"Background subtraction complete: processed={processed_count}, "
        f"skipped_existing={skipped_existing_count}, errors={error_count}"
    )

    summary = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "background_path": background_path,
        "total_tiff_files": len(tiff_files),
        "processed_count": processed_count,
        "skipped_existing_count": skipped_existing_count,
        "error_count": error_count,
        "errors": errors,
    }

    if metadata_path is not None:
        write_stage_metadata(
            metadata_path,
            stage_name="background_subtraction",
            parameters={
                "input_dir": os.path.abspath(input_dir),
                "output_dir": os.path.abspath(output_dir),
                "background_path": os.path.abspath(background_path),
                "clip_negatives": clip_negatives,
                "skip_existing": skip_existing,
                "compute_dtype": compute_dtype,
                "output_dtype": output_dtype,
            },
            extra={
                "total_tiff_files": len(tiff_files),
                "processed_count": processed_count,
                "skipped_existing_count": skipped_existing_count,
                "error_count": error_count,
            },
            section_name="background_processing",
        )

        summary["metadata_path"] = metadata_path
    return summary

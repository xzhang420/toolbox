import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tifffile
import shutil
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
exposure_seconds : int or float, optional
    Duration of each exposure in seconds (default is 2).
context : str, optional
    Label used in error messages to identify the dataset.

Returns
-------
tuple
    (mean_beam_integral_per_minute, mean_tpx_modified_time)
"""
def compute_run_metrics(beamhistory, tpx_folder, exposure_seconds=2, context=None):
    exposure_times = get_exposure_times(tpx_folder)
    if len(exposure_times) == 0:
        return np.nan, None

    validate_beamhistory_coverage(
        beamhistory,
        exposure_times,
        exposure_seconds,
        context=context or os.path.basename(tpx_folder),
    )
    beam_integrals = integrate_beam(beamhistory, exposure_times, exposure_seconds)
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
exposure_seconds : int or float, optional
    Duration of each exposure in seconds (default is 2).
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
    exposure_seconds=2,
    context="operando",
):
    if len(exposure_times) == 0:
        return

    t_min = min(exposure_times) - timedelta(seconds=exposure_seconds)
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
exposure_seconds : int or float, optional
    Duration of each exposure in seconds (default is 2).

Returns
-------
np.ndarray
    Array of average beam integrals (per minute) for each exposure.
"""
def integrate_beam(beamhistory, exposure_times, exposure_seconds=2):
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
        t_start = t_end - timedelta(seconds=exposure_seconds)
        t_start_ns = np.int64(pd.Timestamp(t_start).value)
        t_end_ns = np.int64(pd.Timestamp(t_end).value)

        overlap_start = np.maximum(ts_ns, t_start_ns)
        overlap_end = np.minimum(next_ns, t_end_ns)
        dur_ns = overlap_end - overlap_start
        dur_ns[dur_ns < 0] = 0

        dur_sec = dur_ns / 1e9
        window_sum = np.sum(beam.values * dur_sec)
        per_min = window_sum / exposure_seconds * 60.0
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
exposure_seconds : int or float, optional
    Duration of each exposure in seconds (default is 2).

Returns
-------
float
    Mean beam integral per minute across all reference exposures.
"""
def compute_reference_beam_integral_per_minute(
    reference_dir,
    reference_beamhistory_path,
    exposure_seconds=2,
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
            exposure_seconds,
            context=f"reference:{run_dir}",
        )
        beam_integrals = integrate_beam(beamhistory, exposure_times, exposure_seconds)
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

Notes
-----
Writes custom metadata into the TIFF header:
- 'AcquisitionTime': human-readable acquisition timestamp
- 'BeamCorrectionFactor': applied correction factor
- 'MeanTPXModifiedTime': mean modified time of corresponding TPX files
"""
def correct_and_save_tiff(
    src_path,
    dst_path,
    correction_factor,
    acquisition_time,
    mean_tpx_modified_time=None,
):
    with tifffile.TiffFile(src_path) as tif:
        img = tif.asarray()

    corrected = img / correction_factor

    # Add custom metadata
    # Use private TIFF tags to avoid KeyError with custom tag names
    # 65000: AcquisitionTime, 65001: BeamCorrectionFactor, 65002: MeanTPXModifiedTime
    extratags = [
        (65001, "d", 1, float(correction_factor), False),
    ]
    if acquisition_time is not None:
        extratags.append(
            (65000, "s", 0, acquisition_time.strftime("%Y-%m-%d %H:%M:%S"), False)
        )
    if mean_tpx_modified_time is not None:
        extratags.append(
            (65002, "s", 0, mean_tpx_modified_time.strftime("%Y-%m-%d %H:%M:%S"), False)
        )

    tifffile.imwrite(
        dst_path,
        corrected.astype(img.dtype),
        extratags=extratags
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
exposure_seconds : int or float, optional
    Duration of each exposure in seconds (default 2).
reference_beam_integral_per_minute : float, optional
    Reference beam integral per minute used to normalize images. If not
    provided, it will be computed from reference data.
reference_dir : str
    Root directory containing reference runs (same structure as operando data).
reference_beamhistory_path : str
    Path to the .dat file containing beam history for the reference data.

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
    image_dir=None,
    output_dir=None,
    beamhistory_path=None,
    exposure_seconds=2,
    reference_beam_integral_per_minute=None,
    reference_dir=None,
    reference_beamhistory_path=None,
):
    beamhistory = load_beamhistory(beamhistory_path)

    if reference_beam_integral_per_minute is None:
        if reference_dir and reference_beamhistory_path:
            reference_beam_integral_per_minute = compute_reference_beam_integral_per_minute(
                reference_dir,
                reference_beamhistory_path,
                exposure_seconds,
            )
        else:
            raise ValueError(
                "Provide reference_beam_integral_per_minute or "
                "reference_dir and reference_beamhistory_path."
            )
    elif reference_beam_integral_per_minute <= 0:
        raise ValueError(
            "reference_beam_integral_per_minute must be a positive value."
        )

    for run_dir in sorted(os.listdir(exp_dir)):
        tpx_folder = os.path.join(exp_dir, run_dir, "tpx3Files")
        if not os.path.isdir(tpx_folder):
            continue

        if image_dir is None:
            image_dir_run = os.path.join(exp_dir, run_dir, "final")
        else:
            image_dir_run = image_dir

        run_beam_integral_per_minute, mean_tpx_time = compute_run_metrics(
            beamhistory, tpx_folder, exposure_seconds, context=run_dir
        )
        if not np.isfinite(run_beam_integral_per_minute):
            warnings.warn(
                f"No TPX files found for run {run_dir}; skipping correction."
            )
            continue

        correction_factor = compute_intensity_correction(
            run_beam_integral_per_minute, reference_beam_integral_per_minute
        )

        image_files = [
            f for f in sorted(os.listdir(image_dir_run))
            if is_tiff_file(os.path.join(image_dir_run, f))
        ]
        if len(image_files) == 0:
            warnings.warn(f"No TIFF images found in {image_dir_run}; skipping run.")
            continue

        skipped_in_run = 0
        for img_file in image_files:
            src = os.path.join(image_dir_run, img_file)
            dst_name = f"{run_dir}_corrected_{img_file}"
            dst = os.path.join(output_dir, dst_name)
            if os.path.exists(dst):
                skipped_in_run += 1
                continue

            correct_and_save_tiff(
                src, dst, correction_factor, mean_tpx_time, mean_tpx_time
            )
        if skipped_in_run:
            warnings.warn(
                f"Skipped {skipped_in_run} existing corrected files in run {run_dir}."
            )


# =============================================================================
# Collect TIFF files from run folders
# =============================================================================
"""
Copy TIFF files from each run's "final" folder into a new output folder and
prefix filenames with the run folder name.

Parameters
----------
exp_dir : str
    Root directory containing run folders (e.g., exp604_..._00075).
output_folder_name : str, optional
    Name of output folder created alongside exp_dir.
final_subfolder : str, optional
    Subfolder name containing TIFF files (default "final").
tiff_extensions : tuple of str, optional
    TIFF extensions to include (default (".tif", ".tiff")).

Returns
-------
str
    Path to the output folder containing copied TIFFs with run-prefixed filenames.
"""
def collect_and_copy_tiffs(
    exp_dir,
    output_folder_name="uncorrected",
    final_subfolder="final",
    tiff_extensions=(".tif", ".tiff"),
):
    parent_dir = os.path.dirname(os.path.abspath(exp_dir))
    output_dir = os.path.join(parent_dir, output_folder_name)
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
            if is_tiff_file(os.path.join(final_dir, f))
        ]
        if len(tiff_files) == 0:
            warnings.warn(f"No TIFF images found in {final_dir}; skipping run.")
            continue
        if not exposure_times:
            warnings.warn(
                f"No TPX files found for run {run_dir}; uncorrected copies will "
                "not include MeanTPXModifiedTime."
            )

        for fname in tiff_files:
            src = os.path.join(final_dir, fname)
            dst_name = f"{run_dir}_{fname}"
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
# High-level pipeline function
# =============================================================================
"""
Run the full operando correction workflow.

Parameters
----------
exp_dir : str
    Root directory containing run folders (e.g., exp604_..._00075).
beamhistory_path : str
    Path to the .dat file containing beam history for operando data.
output_dir : str, optional
    Directory where corrected images will be saved. If None, creates an
    "obint_corrected_<experiment_folder_name>" folder alongside exp_dir.
exposure_seconds : int or float, optional
    Duration of each exposure in seconds (default 2).
reference_dir : str
    Root directory containing reference runs (same structure as operando data).
reference_beamhistory_path : str
    Path to the .dat file containing beam history for the reference data.
uncorrected_output_folder_name : str, optional
    Output folder name for uncorrected TIFFs. If None, creates an
    "uncorrected_<experiment_folder_name>" folder alongside exp_dir.

Returns
-------
dict
    Summary with keys: "corrected_dir", "uncorrected_dir", "reference_beam_integral_per_minute".
"""
def run_operando_pipeline(
    exp_dir,
    beamhistory_path,
    output_dir=None,
    exposure_seconds=2,
    reference_dir=None,
    reference_beamhistory_path=None,
    uncorrected_output_folder_name=None,
):
    if reference_dir is None or reference_beamhistory_path is None:
        raise ValueError(
            "reference_dir and reference_beamhistory_path are required to compute "
            "reference_beam_integral_per_minute."
        )

    exp_name = os.path.basename(os.path.abspath(exp_dir))
    if output_dir is None:
        parent_dir = os.path.dirname(os.path.abspath(exp_dir))
        output_dir = os.path.join(parent_dir, f"obint_corrected_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)

    reference_beam_integral_per_minute = compute_reference_beam_integral_per_minute(
        reference_dir,
        reference_beamhistory_path,
        exposure_seconds,
    )

    if uncorrected_output_folder_name is None:
        uncorrected_output_folder_name = f"uncorrected_{exp_name}"
    uncorrected_dir = collect_and_copy_tiffs(
        exp_dir,
        output_folder_name=uncorrected_output_folder_name,
    )

    intensity_correct_experiment(
        exp_dir=exp_dir,
        image_dir=None,
        output_dir=output_dir,
        beamhistory_path=beamhistory_path,
        exposure_seconds=exposure_seconds,
        reference_beam_integral_per_minute=reference_beam_integral_per_minute,
        reference_dir=reference_dir,
        reference_beamhistory_path=reference_beamhistory_path,
    )

    return {
        "corrected_dir": output_dir,
        "uncorrected_dir": uncorrected_dir,
        "reference_beam_integral_per_minute": reference_beam_integral_per_minute,
    }

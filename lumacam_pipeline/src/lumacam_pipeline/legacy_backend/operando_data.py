"""Established NCI data-processing calculations used by LumaCam sections."""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tifffile
import warnings
from ..products import is_all_runs_sum
from ..proposal_paths import raw_root_for_layout, proposal_root_for_layout, analysis_root_for_proposal


V2_LAYOUT_MARKER = os.path.join("metadata", "layout.json")
V2_DATA_TREES = {"tpx3Files", "final", "derived", "metadata"}
EXPOSURE_MANIFEST_FILENAME = "tpx3_exposure_manifest.json"
PROCESSING_EXCLUSIONS_FILENAME = "processing_exclusions.json"
ACQUISITION_METADATA_FILENAME = "acquisition.json"
EXPERIMENT_METADATA_FILENAME = "experiment.json"


def clean_optional_path(path_value):
    """Return an absolute path string, or ``None`` for a blank optional path."""
    value = str(path_value or "").strip()
    return os.path.abspath(os.path.expanduser(value)) if value else None


def _find_v2_layout_root(path):
    """Find a canonical, early-v2, or marker-era data-layout root.

    New proposals are recognized from the ``data/experiments`` structure and
    do not need ``metadata/layout.json``.  The marker remains a read-only
    compatibility signal for older proposals.
    """
    from ..layout import find_layout_root
    root = find_layout_root(path)
    return str(root) if root is not None else None


def _v2_experiment_name(layout_root, requested_path):
    requested_path = os.path.abspath(requested_path)
    raw_root = raw_root_for_layout(layout_root)
    try:
        raw_parts = Path(requested_path).relative_to(raw_root).parts
    except ValueError:
        raw_parts = ()
    if raw_parts:
        if len(raw_parts) > 2 or raw_parts[0].startswith("."):
            raise ValueError("Select a published raw experiment or run")
        return raw_parts[0], raw_parts[1] if len(raw_parts) == 2 else None, None
    try:
        relative = os.path.relpath(requested_path, layout_root)
    except ValueError:
        relative = "."
    parts = [] if relative == "." else relative.split(os.sep)
    if len(parts) >= 2 and parts[0] in V2_DATA_TREES:
        experiment = parts[1]
        selected_run = None
        selected_variant = None
        if len(parts) >= 3 and parts[2] != "summed":
            candidate = parts[2]
            if parts[0] == "final" and not (
                os.path.isdir(str(raw_root_for_layout(layout_root) / experiment / candidate))
                or os.path.isdir(os.path.join(layout_root, "metadata", experiment, candidate))
            ):
                selected_variant = candidate
            else:
                selected_run = candidate
        return experiment, selected_run, selected_variant

    experiment_names = set()
    for tree_name in ("final", "tpx3Files"):
        tree = str(raw_root_for_layout(layout_root)) if tree_name == "tpx3Files" else os.path.join(layout_root, tree_name)
        if not os.path.isdir(tree):
            continue
        experiment_names.update(
            entry.name
            for entry in os.scandir(tree)
            if entry.is_dir() and not entry.name.startswith(".")
        )
    if len(experiment_names) == 1:
        return next(iter(experiment_names)), None, None
    if not experiment_names:
        raise ValueError(f"No experiments found in the LumaCam proposal: {layout_root}")
    raise ValueError(
        "The supplied LumaCam proposal contains multiple experiments. "
        "Select tpx3Files/<experiment> or final/<experiment>: "
        f"{layout_root}"
    )


def resolve_nci_experiment(exp_dir, final_subfolder="final"):
    """Resolve legacy and version-2 NCI experiments into run image/raw pairs.

    The returned dictionary contains ``layout_type``, ``experiment``,
    ``experiment_dir``, and a ``runs`` list. Each run entry has ``name``,
    ``image_dir``, ``image_files``, ``tpx_dir`` and ``metadata_dir`` keys.
    Version-2 inputs may point at the
    proposal, ``data/experiments``, ``tpx3Files/<experiment>``,
    ``final/<experiment>``, or one selected run below those trees.
    """
    if isinstance(exp_dir, dict):
        # Explicit, already validated product mapping supplied by the processing
        # adapter. It never changes files or patches the global path resolver.
        required = {"layout_type", "layout_root", "experiment", "experiment_dir", "runs"}
        if not required.issubset(exp_dir):
            raise ValueError("Incomplete normalized experiment mapping")
        return exp_dir
    requested = os.path.abspath(os.path.expanduser(str(exp_dir)))
    if not os.path.isdir(requested):
        raise ValueError(f"Experiment directory does not exist: {requested}")

    layout_root = _find_v2_layout_root(requested)
    if layout_root is not None:
        experiment, selected_run, selected_variant = _v2_experiment_name(layout_root, requested)
        tpx3_root = str(raw_root_for_layout(layout_root) / experiment)
        image_root = os.path.join(layout_root, "final", experiment)
        metadata_root = os.path.join(layout_root, "metadata", experiment)
        run_names = set()
        for tree in (tpx3_root, metadata_root):
            if os.path.isdir(tree):
                run_names.update(
                    entry.name
                    for entry in os.scandir(tree)
                    if entry.is_dir()
                    and entry.name != "summed"
                    and not entry.name.startswith(".")
                )
        if selected_run is not None:
            run_names = {selected_run}
        # Compatibility with the earlier v2 layout where final images lived in
        # final/<experiment>/<run>.  Current acquisitions instead use
        # final/<experiment>/<photon-range>/<run-named TIFF>.
        if os.path.isdir(image_root):
            for entry in os.scandir(image_root):
                if entry.is_dir() and (
                    os.path.isdir(os.path.join(tpx3_root, entry.name))
                    or os.path.isdir(os.path.join(metadata_root, entry.name))
                ) and (selected_run is None or entry.name == selected_run):
                    run_names.add(entry.name)

        variant_dirs = []
        if os.path.isdir(image_root):
            variant_dirs = [
                entry.path
                for entry in sorted(os.scandir(image_root), key=lambda item: item.name)
                if entry.is_dir()
                and not entry.name.startswith(".")
                and entry.name != "summed"
                and entry.name not in run_names
                and (selected_variant is None or entry.name == selected_variant)
            ]

        runs = []
        excluded_runs = []
        for run_name in sorted(run_names):
            metadata_dir = os.path.join(metadata_root, run_name)
            run_exclusion = load_run_processing_exclusion(metadata_dir)
            if run_exclusion is not None:
                excluded_runs.append(run_name)
                continue
            old_image_dir = os.path.join(image_root, run_name)
            image_files = []
            if os.path.isdir(old_image_dir):
                image_files.extend(
                    entry.path
                    for entry in sorted(os.scandir(old_image_dir), key=lambda item: item.name)
                    if entry.is_file() and is_tiff_file(entry.path) and not is_all_runs_sum(entry.path)
                )
                image_dir = old_image_dir
            else:
                for variant_dir in variant_dirs:
                    image_files.extend(
                        entry.path
                        for entry in sorted(os.scandir(variant_dir), key=lambda item: item.name)
                        if entry.is_file()
                        and is_tiff_file(entry.path)
                        and not is_all_runs_sum(entry.path)
                        and (
                            entry.name == f"{run_name}.tiff"
                            or entry.name == f"{run_name}.tif"
                            or entry.name.startswith(f"{run_name}_")
                        )
                    )
                image_dir = variant_dirs[0] if len(variant_dirs) == 1 else image_root
            runs.append(
                {
                    "name": run_name,
                    "image_dir": image_dir,
                    "image_files": image_files,
                    "tpx_dir": os.path.join(tpx3_root, run_name),
                    "metadata_dir": metadata_dir,
                    "layout_type": "v2",
                }
            )
        return {
            "layout_type": "v2",
            "layout_root": layout_root,
            "experiment": experiment,
            "experiment_dir": image_root,
            "runs": runs,
            "excluded_runs": excluded_runs,
        }

    # A legacy run itself may be supplied instead of its experiment parent.
    # Some legacy experiment roots also contain an aggregate ``final`` folder,
    # so prefer recognizable child runs before treating that root as one run.
    child_run_paths = [
        entry.path
        for entry in sorted(os.scandir(requested), key=lambda item: item.name)
        if entry.is_dir()
        and (
            os.path.isdir(os.path.join(entry.path, "tpx3Files"))
            or os.path.isdir(os.path.join(entry.path, final_subfolder))
        )
    ]
    is_single_run = os.path.isdir(os.path.join(requested, "tpx3Files")) or (
        os.path.isdir(os.path.join(requested, final_subfolder)) and not child_run_paths
    )
    if is_single_run:
        run_paths = [requested]
        experiment = os.path.basename(os.path.dirname(requested))
        experiment_dir = os.path.dirname(requested)
    else:
        run_paths = child_run_paths
        experiment = os.path.basename(requested.rstrip(os.sep))
        experiment_dir = requested

    runs = []
    for run_path in run_paths:
        image_dir = os.path.join(run_path, final_subfolder)
        tpx_dir = os.path.join(run_path, "tpx3Files")
        if not os.path.isdir(image_dir) and not os.path.isdir(tpx_dir):
            continue
        runs.append(
            {
                "name": os.path.basename(run_path.rstrip(os.sep)),
                "image_dir": image_dir,
                "tpx_dir": tpx_dir,
                "metadata_dir": os.path.join(run_path, "metadata"),
                "layout_type": "legacy",
            }
        )
    return {
        "layout_type": "legacy",
        "layout_root": None,
        "experiment": experiment,
        "experiment_dir": experiment_dir,
        "runs": runs,
    }


def list_nci_experiment_dirs(root):
    """List selectable experiment directories below a campaign/proposal root."""
    root = os.path.abspath(os.path.expanduser(str(root)))
    layout_root = _find_v2_layout_root(root)
    if layout_root is not None:
        names = set()
        for tree_name in ("final", "tpx3Files"):
            tree = str(raw_root_for_layout(layout_root)) if tree_name == "tpx3Files" else os.path.join(layout_root, tree_name)
            if os.path.isdir(tree):
                names.update(entry.name for entry in os.scandir(tree) if entry.is_dir())
        return [
            os.path.join(layout_root, "final", name)
            if os.path.isdir(os.path.join(layout_root, "final", name))
            else str(raw_root_for_layout(layout_root) / name)
            for name in sorted(names)
            if name != "summed" and not name.startswith(".")
        ]

    paths = []
    for category in ("experiment", "reference"):
        category_root = os.path.join(root, category)
        if not os.path.isdir(category_root):
            continue
        paths.extend(
            entry.path
            for entry in sorted(os.scandir(category_root), key=lambda item: item.name)
            if entry.is_dir()
        )
    return paths


def _matching_tiff_files(directory, file_filter=None):
    if not os.path.isdir(directory):
        return []
    return [
        entry.name
        for entry in sorted(os.scandir(directory), key=lambda item: item.name)
        if entry.is_file()
        and is_tiff_file(entry.path)
        and not is_all_runs_sum(entry.path)
        and (file_filter is None or file_filter in entry.name)
    ]


def _run_image_paths(run, file_filter=None):
    # Normalized processing inputs have already selected an exact variant,
    # which need not appear in a mapped filename. References still use the
    # caller's filter; do not globally clear it for preselected samples.
    if run.get("preselected"):
        file_filter = None
    explicit = run.get("image_files")
    if explicit is not None:
        return [
            path
            for path in explicit
            if os.path.isfile(path)
            and is_tiff_file(path)
            and not is_all_runs_sum(path)
            and (file_filter is None or file_filter in os.path.basename(path))
        ]
    return [
        os.path.join(run["image_dir"], name)
        for name in _matching_tiff_files(run["image_dir"], file_filter=file_filter)
    ]


def count_experiment_images(exp_dir, final_subfolder="final", file_filter=None):
    """Count TIFF inputs selected from a legacy or version-2 experiment."""
    layout = resolve_nci_experiment(exp_dir, final_subfolder=final_subfolder)
    return sum(
        len(_run_image_paths(run, file_filter=file_filter))
        for run in layout["runs"]
    )

# =============================================================================
# Beam history loader
# =============================================================================
"""
Load a beam history file into a pandas DataFrame.

The first column is treated as the start timestamp of each interval, and the
second column is treated as the average beam value for the interval from this
row's time to the next row's time. Timestamp and beam value may instead occupy
separate whitespace-delimited columns; the loader finds the shortest valid
timestamp prefix followed by a numeric beam column.

Parameters
----------
path : str
    Path to one beam-history file or a folder containing beam-history files.

Returns
-------
pd.DataFrame
    DataFrame with columns:
    - 'timestamp' (datetime): interval start timestamps
    - 'beam_count' (float): average beam value for the interval
    The DataFrame is sorted by timestamp.
"""
_BEAMHISTORY_NAIVE_TIMESTAMP_FORMATS = (
    "%m/%d/%Y %I:%M:%S.%f %p",
    "%m/%d/%Y %I:%M:%S %p",
    "%m/%d/%Y %H:%M:%S.%f",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d.%H:%M:%S.%f",
    "%Y-%m-%d.%H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%dT%H:%M:%S",
    "%Y/%m/%d %H:%M:%S.%f",
    "%Y/%m/%d %H:%M:%S",
    "%d.%m.%Y %H:%M:%S.%f",
    "%d.%m.%Y %H:%M:%S",
    "%Y%m%d%H%M%S",
)

_BEAMHISTORY_FILE_SUFFIXES = {"", ".csv", ".dat", ".log", ".tsv", ".txt"}


def _absolute_times_to_local_naive(values):
    """Convert timezone-aware timestamps to pipeline's local naive time basis."""
    converted = []
    for value in values:
        if pd.isna(value):
            converted.append(pd.NaT)
        else:
            converted.append(datetime.fromtimestamp(pd.Timestamp(value).timestamp()))
    return pd.Series(converted, index=values.index, dtype="datetime64[ns]")


def _parse_beamhistory_timestamp_series(values):
    """Parse supported local, ISO-8601, and Unix beam-history timestamps."""
    text = values.astype("string").str.strip().str.strip('"').str.strip("'")
    parsed = pd.Series(pd.NaT, index=text.index, dtype="datetime64[ns]")

    aware_mask = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True, na=False)
    if aware_mask.any():
        aware = pd.to_datetime(text.where(aware_mask), errors="coerce", utc=True)
        parsed.loc[aware.notna()] = _absolute_times_to_local_naive(
            aware.loc[aware.notna()]
        )

    for timestamp_format in _BEAMHISTORY_NAIVE_TIMESTAMP_FORMATS:
        missing = parsed.isna()
        if not missing.any():
            break
        candidate = pd.to_datetime(
            text.where(missing), format=timestamp_format, errors="coerce"
        )
        valid = missing & candidate.notna()
        parsed.loc[valid] = candidate.loc[valid]

    missing = parsed.isna()
    numeric = pd.to_numeric(text.where(missing), errors="coerce")
    plausible_unix = numeric.notna() & numeric.abs().ge(100_000_000)
    if plausible_unix.any():
        magnitude = float(numeric.loc[plausible_unix].abs().median())
        if magnitude < 1e11:
            unit = "s"
        elif magnitude < 1e14:
            unit = "ms"
        elif magnitude < 1e17:
            unit = "us"
        else:
            unit = "ns"
        absolute = pd.to_datetime(
            numeric.where(plausible_unix), unit=unit, errors="coerce", utc=True
        )
        valid = absolute.notna()
        parsed.loc[valid] = _absolute_times_to_local_naive(absolute.loc[valid])

    return parsed


def summarize_beamhistory_cadence(beamhistory, gap_factor=1.5):
    """Summarize sampling interval and internal timestamp gaps."""
    timestamps = pd.to_datetime(beamhistory["timestamp"], errors="coerce").dropna()
    timestamps = timestamps.sort_values().reset_index(drop=True)
    differences = timestamps.diff().dt.total_seconds().dropna()
    positive = differences[differences > 0]
    duplicate_count = int((differences == 0).sum())
    if positive.empty:
        nominal = None
        regular_count = 0
        long_gap_count = 0
        longest_gap = None
        long_gap_threshold = None
    else:
        rounded = positive.round(3)
        nominal = float(rounded.mode().iloc[0])
        tolerance = max(0.05, nominal * 0.1)
        regular_count = int(np.isclose(positive, nominal, atol=tolerance).sum())
        long_gap_threshold = nominal * float(gap_factor)
        long_gap_count = int((positive > long_gap_threshold).sum())
        longest_gap = float(positive.max())
    return {
        "row_count": int(len(timestamps)),
        "interval_count": int(len(differences)),
        "nominal_interval_s": nominal,
        "regular_interval_count": regular_count,
        "duplicate_timestamp_count": duplicate_count,
        "long_gap_threshold_s": long_gap_threshold,
        "long_gap_count": long_gap_count,
        "longest_gap_s": longest_gap,
    }


def _warn_beamhistory_quality(beamhistory, path):
    summary = summarize_beamhistory_cadence(beamhistory)
    invalid_beam_count = int(
        pd.to_numeric(beamhistory["beam_count"], errors="coerce").isna().sum()
    )
    if invalid_beam_count:
        warnings.warn(
            f"Beam history {path} contains {invalid_beam_count} timestamped row(s) "
            "without a numeric beam value; those intervals cannot contribute to correction.",
            stacklevel=2,
        )
    if summary["duplicate_timestamp_count"]:
        warnings.warn(
            f"Beam history {path} contains {summary['duplicate_timestamp_count']} "
            "duplicate timestamp interval(s).",
            stacklevel=2,
        )
    if summary["long_gap_count"]:
        warnings.warn(
            f"Beam-history cadence for {path}: nominal "
            f"{summary['nominal_interval_s']:.6g} s, but {summary['long_gap_count']} "
            f"interval(s) exceed {summary['long_gap_threshold_s']:.6g} s; longest "
            f"gap is {summary['longest_gap_s']:.6g} s. Beam values are held from "
            "each timestamp to the next; check whether long gaps overlap acquisitions.",
            stacklevel=2,
        )
    return summary


def load_beamhistory(path, warn_on_gaps=True):
    """Load one file or combine a folder of flexible beam-history text data."""
    source = Path(path).expanduser()
    if source.is_dir():
        candidates = [
            candidate
            for candidate in sorted(source.rglob("*"))
            if candidate.is_file()
            and candidate.suffix.lower() in _BEAMHISTORY_FILE_SUFFIXES
            and not any(part.startswith(".") for part in candidate.relative_to(source).parts)
        ]
        if not candidates:
            raise ValueError(f"No beam-history data files found in folder: {source}")

        frames = []
        skipped = []
        for candidate in candidates:
            try:
                frames.append(load_beamhistory(candidate, warn_on_gaps=False))
            except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
                skipped.append((candidate, exc))
        if not frames:
            details = "; ".join(
                f"{candidate.name}: {type(exc).__name__}" for candidate, exc in skipped[:5]
            )
            raise ValueError(
                f"No readable beam-history data files found in folder: {source}"
                + (f" ({details})" if details else "")
            )
        if skipped:
            names = ", ".join(candidate.name for candidate, _ in skipped[:5])
            suffix = "..." if len(skipped) > 5 else ""
            warnings.warn(
                f"Skipped {len(skipped)} unreadable beam-history file(s) in {source}: "
                f"{names}{suffix}",
                stacklevel=2,
            )

        out = pd.concat(frames, ignore_index=True)
        out = out.sort_values("timestamp", kind="mergesort")
        overlap_count = int(out.duplicated(subset="timestamp", keep="last").sum())
        if overlap_count:
            warnings.warn(
                f"Combined beam-history folder {source} contains {overlap_count} "
                "overlapping timestamp row(s); values from later filenames were kept.",
                stacklevel=2,
            )
            out = out.drop_duplicates(subset="timestamp", keep="last")
        out = out.reset_index(drop=True)
        cadence = (
            _warn_beamhistory_quality(out, source)
            if warn_on_gaps
            else summarize_beamhistory_cadence(out)
        )
        cadence["source_file_count"] = len(frames)
        cadence["skipped_file_count"] = len(skipped)
        cadence["overlapping_row_count"] = overlap_count
        out.attrs["beamhistory_cadence"] = cadence
        out.attrs["beamhistory_source_files"] = tuple(
            str(candidate.resolve())
            for candidate in candidates
            if candidate not in {item[0] for item in skipped}
        )
        return out
    if not source.is_file():
        raise ValueError(f"Beam-history path is not a file or folder: {source}")

    df = pd.read_csv(
        source,
        header=None,
        engine="python",
        sep=None,
        dtype=str,
        comment="#",
        skip_blank_lines=True,
    )
    if df.shape[1] < 2:
        raise ValueError(
            "Beam history needs a timestamp followed by a numeric beam value."
        )

    best = None
    for beam_col_idx in range(1, min(df.shape[1], 5)):
        timestamp_text = df.iloc[:, 0].astype("string")
        for column in range(1, beam_col_idx):
            timestamp_text = timestamp_text.str.cat(
                df.iloc[:, column].astype("string"), sep=" "
            )
        timestamps = _parse_beamhistory_timestamp_series(timestamp_text)
        beam_values = pd.to_numeric(df.iloc[:, beam_col_idx], errors="coerce")
        score = int((timestamps.notna() & beam_values.notna()).sum())
        if best is None or score > best[0]:
            best = (score, timestamps, beam_values)

    score, timestamps, beam_values = best
    if score == 0:
        raise ValueError(
            "Could not parse beam-history timestamps and values. Supported timestamps "
            "include '12/1/2025 4:00:10 AM', '2026-07-20.12:00:35', ISO 8601, "
            "and Unix seconds/milliseconds/microseconds/nanoseconds."
        )

    out = pd.DataFrame({"timestamp": timestamps, "beam_count": beam_values})
    out = out.dropna(subset=["timestamp"])
    out = out.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    cadence = (
        _warn_beamhistory_quality(out, source)
        if warn_on_gaps
        else summarize_beamhistory_cadence(out)
    )
    out.attrs["beamhistory_cadence"] = cadence
    out.attrs["beamhistory_source_files"] = (str(source.resolve()),)
    return out


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


def load_exposure_manifest(metadata_dir):
    """Load and validate one acquisition exposure manifest.

    Returns ``None`` when the sidecar does not exist.  Invalid sidecars raise a
    ``ValueError`` so processing cannot silently use corrupt timing metadata.
    """
    if metadata_dir is None:
        return None
    path = os.path.join(str(metadata_dir), EXPOSURE_MANIFEST_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read exposure manifest {path}: {error}") from error
    if (
        payload.get("schema") != "lumacam-tpx3-exposure-manifest"
        or payload.get("schema_version") != 1
    ):
        raise ValueError(f"Unsupported exposure manifest schema: {path}")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"Exposure manifest contains no frames: {path}")
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict) or frame.get("frame_index") != index:
            raise ValueError(f"Exposure manifest has invalid frame ordering: {path}")
        start_ns = frame.get("start_unix_ns")
        end_ns = frame.get("end_unix_ns")
        if not isinstance(start_ns, int) or not isinstance(end_ns, int) or start_ns >= end_ns:
            raise ValueError(f"Exposure manifest has an invalid frame window: {path}")
    return payload


def load_processing_exclusions(metadata_dir, exposure_manifest):
    """Load and validate frames deliberately omitted during reconstruction.

    The exclusion sidecar is meaningful only together with the exact exposure
    manifest whose filenames and timing windows it references. Invalid or
    inconsistent metadata is rejected so beam normalization cannot silently
    integrate a raw interval that did not contribute to the reconstructed TIFF.
    """
    if metadata_dir is None:
        return None
    path = os.path.join(str(metadata_dir), PROCESSING_EXCLUSIONS_FILENAME)
    if not os.path.isfile(path):
        return None
    if exposure_manifest is None:
        raise ValueError(
            f"Processing exclusions require {EXPOSURE_MANIFEST_FILENAME}: {path}"
        )
    payload = _read_json_object(path, "processing exclusions")
    if (
        payload.get("schema") != "lumacam-processing-exclusions"
        or payload.get("schema_version") != 1
    ):
        raise ValueError(f"Unsupported processing-exclusion schema: {path}")
    if payload.get("raw_tpx3_retained") is not True:
        raise ValueError(f"Processing exclusions do not confirm retained raw data: {path}")

    manifest_frames = exposure_manifest["frames"]
    manifest_by_name = {frame.get("file_name"): frame for frame in manifest_frames}
    excluded_frames = payload.get("excluded_frames")
    if not isinstance(excluded_frames, list) or not excluded_frames:
        raise ValueError(f"Processing exclusions contain no excluded frames: {path}")
    excluded_names = []
    for excluded in excluded_frames:
        if not isinstance(excluded, dict):
            raise ValueError(f"Processing exclusions contain a non-object frame: {path}")
        name = excluded.get("file_name")
        manifest_frame = manifest_by_name.get(name)
        if manifest_frame is None:
            raise ValueError(
                f"Excluded frame {name!r} is absent from the exposure manifest: {path}"
            )
        if name in excluded_names:
            raise ValueError(f"Processing exclusions contain duplicate frame {name!r}: {path}")
        if (
            excluded.get("start_unix_ns") != manifest_frame.get("start_unix_ns")
            or excluded.get("end_unix_ns") != manifest_frame.get("end_unix_ns")
        ):
            raise ValueError(
                f"Excluded frame timing does not match the exposure manifest for {name}: {path}"
            )
        if not str(excluded.get("reason", "")).strip():
            raise ValueError(f"Excluded frame {name!r} has no reason: {path}")
        excluded_names.append(name)

    input_count = len(manifest_frames)
    excluded_count = len(excluded_names)
    if (
        payload.get("input_frame_count") != input_count
        or payload.get("excluded_frame_count") != excluded_count
        or payload.get("processed_frame_count") != input_count - excluded_count
    ):
        raise ValueError(f"Processing-exclusion frame counts are inconsistent: {path}")
    manifest_duration = sum(
        (frame["end_unix_ns"] - frame["start_unix_ns"]) / 1e9
        for frame in manifest_frames
    )
    excluded_duration = sum(
        (manifest_by_name[name]["end_unix_ns"] - manifest_by_name[name]["start_unix_ns"])
        / 1e9
        for name in excluded_names
    )
    included_duration = manifest_duration - excluded_duration
    duration_fields = {
        "input_effective_exposure_s": manifest_duration,
        "excluded_effective_exposure_s": excluded_duration,
        "included_effective_exposure_s": included_duration,
    }
    for field, expected in duration_fields.items():
        try:
            actual = float(payload[field])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Processing exclusions have invalid {field}: {path}") from error
        if not np.isclose(actual, expected, rtol=0, atol=1e-6):
            raise ValueError(f"Processing exclusions have inconsistent {field}: {path}")
    payload = dict(payload)
    payload["metadata_path"] = path
    payload["excluded_file_names"] = excluded_names
    return payload


def load_run_processing_exclusion(metadata_dir):
    """Return validated metadata when reduction deliberately omitted a whole run."""
    if metadata_dir is None or not os.path.isfile(
        os.path.join(str(metadata_dir), PROCESSING_EXCLUSIONS_FILENAME)
    ):
        return None
    manifest = load_exposure_manifest(metadata_dir)
    exclusions = load_processing_exclusions(metadata_dir, manifest)
    if exclusions is None or exclusions.get("run_excluded") is not True:
        return None
    manifest_names = [frame.get("file_name") for frame in manifest["frames"]]
    if set(exclusions["excluded_file_names"]) != set(manifest_names):
        raise ValueError(
            "Whole-run processing exclusion does not list every TPX3 frame: "
            f"{exclusions['metadata_path']}"
        )
    if not np.isclose(
        float(exclusions["included_effective_exposure_s"]), 0.0, rtol=0, atol=1e-9
    ):
        raise ValueError(
            "Whole-run processing exclusion has non-zero included exposure: "
            f"{exclusions['metadata_path']}"
        )
    return exclusions


def _positive_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) and result > 0 else None


def _nominal_effective_exposure_s(exposure_manifest, metadata_dir):
    """Resolve the intended detector-open time represented by one run TIFF."""
    acquisition = None
    if metadata_dir is not None:
        acquisition_path = os.path.join(
            str(metadata_dir), ACQUISITION_METADATA_FILENAME
        )
        if os.path.isfile(acquisition_path):
            acquisition = _read_json_object(acquisition_path, "acquisition metadata")
            planned = _positive_float(acquisition.get("planned_effective_exposure_s"))
            if planned is not None:
                return planned

    planned = _positive_float(exposure_manifest.get("planned_effective_exposure_s"))
    if planned is not None:
        return planned

    requested = _positive_float(exposure_manifest.get("requested_measurement_time_s"))
    detector = exposure_manifest.get("detector_configuration")
    if requested is not None and isinstance(detector, dict):
        exposure = _positive_float(detector.get("exposure_time_s"))
        period = _positive_float(detector.get("trigger_period_s"))
        if exposure is not None and period is not None and exposure <= period:
            remaining_after_first = max(0.0, requested - exposure)
            frame_count = max(
                1, 1 + math.ceil(remaining_after_first / period - 1e-12)
            )
            return frame_count * exposure

    summary = exposure_manifest.get("summary")
    summary_duration = _positive_float(
        summary.get("total_effective_exposure_s")
        if isinstance(summary, dict)
        else None
    )
    if summary_duration is not None:
        return summary_duration
    return float(
        sum(
            (frame["end_unix_ns"] - frame["start_unix_ns"]) / 1e9
            for frame in exposure_manifest["frames"]
        )
    )


def _read_json_object(path, label):
    """Read a JSON object and report its path in validation errors."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label.capitalize()} must contain a JSON object: {path}")
    return payload


def load_run_acquisition_metadata(metadata_dir):
    """Load run-level acquisition timing from durable version-2 JSON metadata.

    The per-run ``acquisition.json`` is preferred.  For early version-2 data,
    the matching entry in the experiment-level ``experiment.json`` is used as
    a fallback.  ``None`` is returned when neither file contains a complete
    start/end pair.
    """
    if metadata_dir is None:
        return None
    metadata_dir = os.path.abspath(os.path.expanduser(str(metadata_dir)))
    run_name = os.path.basename(metadata_dir.rstrip(os.sep))
    candidates = []

    acquisition_path = os.path.join(metadata_dir, ACQUISITION_METADATA_FILENAME)
    if os.path.isfile(acquisition_path):
        payload = _read_json_object(acquisition_path, "acquisition metadata")
        candidates.append((payload, acquisition_path, "acquisition_json"))

    experiment_path = os.path.join(
        os.path.dirname(metadata_dir), EXPERIMENT_METADATA_FILENAME
    )
    if os.path.isfile(experiment_path):
        payload = _read_json_object(experiment_path, "experiment metadata")
        run_payload = payload.get("runs", {}).get(run_name)
        if isinstance(run_payload, dict):
            candidates.append((run_payload, experiment_path, "experiment_json"))

    for payload, path, source in candidates:
        if payload.get("started_at") and payload.get("ended_at"):
            return {
                "started_at": payload["started_at"],
                "ended_at": payload["ended_at"],
                "run": payload.get("run", run_name),
                "status": payload.get("status"),
                "metadata_path": path,
                "metadata_source": source,
            }
    return None


def _unix_ns_from_metadata_datetime(value, path, field_name):
    """Parse an ISO metadata timestamp as UTC and return Unix nanoseconds."""
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(
            f"Invalid {field_name!r} timestamp in acquisition metadata {path}: {value!r}"
        )
    return int(parsed.value)


def get_run_metadata_window(metadata_dir):
    """Return one run envelope from acquisition/experiment JSON metadata."""
    metadata = load_run_acquisition_metadata(metadata_dir)
    if metadata is None:
        return None
    path = metadata["metadata_path"]
    start_ns = _unix_ns_from_metadata_datetime(metadata["started_at"], path, "started_at")
    end_ns = _unix_ns_from_metadata_datetime(metadata["ended_at"], path, "ended_at")
    if start_ns >= end_ns:
        raise ValueError(
            f"Acquisition metadata end must follow start in {path}: "
            f"{metadata['started_at']!r} -> {metadata['ended_at']!r}"
        )
    midpoint_ns = start_ns + (end_ns - start_ns) // 2
    return {
        "start": _datetime_from_unix_ns(start_ns),
        "end": _datetime_from_unix_ns(end_ns),
        "midpoint": _datetime_from_unix_ns(midpoint_ns),
        "start_unix_ns": start_ns,
        "end_unix_ns": end_ns,
        "midpoint_unix_ns": midpoint_ns,
        "beam_start_ns": _local_naive_ns_from_unix_ns(start_ns),
        "beam_end_ns": _local_naive_ns_from_unix_ns(end_ns),
        "duration_s": (end_ns - start_ns) / 1e9,
        "file_name": None,
        "timing_source": metadata["metadata_source"],
        "timing_quality": "run_envelope",
        "manifest_path": None,
        "metadata_path": path,
        "timing_metadata_kind": metadata["metadata_source"],
        "tpx_file_count": 0,
    }


def _datetime_from_unix_ns(timestamp_ns):
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    return datetime.fromtimestamp(seconds).replace(microsecond=nanoseconds // 1000)


def _local_naive_ns_from_unix_ns(timestamp_ns):
    """Map Unix ns to the local naive time basis used by beam-history CSVs."""
    seconds, nanoseconds = divmod(int(timestamp_ns), 1_000_000_000)
    local_second = datetime.fromtimestamp(seconds)
    return int(pd.Timestamp(local_second).value) + nanoseconds


def get_exposure_windows(
    tpx_folder=None,
    metadata_dir=None,
    tpx_time=2,
    extension=".tpx3",
):
    """Return exposure windows, preferring durable JSON metadata.

    Each result is a dictionary with ``start``, ``end``, ``midpoint``,
    ``duration_s``, ``file_name``, ``timing_source`` and ``timing_quality``.
    The detailed TPX3 exposure manifest is preferred.  If it is unavailable,
    a run envelope from ``acquisition.json`` (or ``experiment.json``) is used.
    Legacy datasets without timing JSON retain the historical
    ``mtime - tpx_time`` fallback.
    """
    manifest = None
    manifest_error = None
    try:
        manifest = load_exposure_manifest(metadata_dir)
    except ValueError as error:
        manifest_error = error
    if manifest is not None:
        exclusions = load_processing_exclusions(metadata_dir, manifest)
        excluded_names = set(
            exclusions["excluded_file_names"] if exclusions is not None else []
        )
        manifest_path = os.path.join(str(metadata_dir), EXPOSURE_MANIFEST_FILENAME)
        exclusions_path = (
            exclusions["metadata_path"] if exclusions is not None else None
        )
        timing_metadata_paths = [manifest_path]
        if exclusions_path is not None:
            timing_metadata_paths.append(exclusions_path)
        nominal_effective_exposure_s = _nominal_effective_exposure_s(
            manifest, metadata_dir
        )
        windows = []
        for frame in manifest["frames"]:
            if frame.get("file_name") in excluded_names:
                continue
            start = _datetime_from_unix_ns(frame["start_unix_ns"])
            end = _datetime_from_unix_ns(frame["end_unix_ns"])
            midpoint = _datetime_from_unix_ns(
                frame.get(
                    "midpoint_unix_ns",
                    frame["start_unix_ns"]
                    + (frame["end_unix_ns"] - frame["start_unix_ns"]) // 2,
                )
            )
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "midpoint": midpoint,
                    "start_unix_ns": frame["start_unix_ns"],
                    "end_unix_ns": frame["end_unix_ns"],
                    "midpoint_unix_ns": frame.get(
                        "midpoint_unix_ns",
                        frame["start_unix_ns"]
                        + (frame["end_unix_ns"] - frame["start_unix_ns"]) // 2,
                    ),
                    "beam_start_ns": _local_naive_ns_from_unix_ns(
                        frame["start_unix_ns"]
                    ),
                    "beam_end_ns": _local_naive_ns_from_unix_ns(
                        frame["end_unix_ns"]
                    ),
                    "duration_s": (frame["end_unix_ns"] - frame["start_unix_ns"]) / 1e9,
                    "file_name": frame.get("file_name"),
                    "timing_source": frame.get("timing_source", "exposure_manifest"),
                    "timing_quality": frame.get("timing_quality", "unknown"),
                    "manifest_path": manifest_path,
                    "metadata_path": manifest_path,
                    "timing_metadata_paths": timing_metadata_paths,
                    "processing_exclusions_path": exclusions_path,
                    "input_tpx_file_count": len(manifest["frames"]),
                    "nominal_effective_exposure_s": nominal_effective_exposure_s,
                    "excluded_tpx_file_count": len(excluded_names),
                    "excluded_tpx_files": list(exclusions["excluded_file_names"])
                    if exclusions is not None
                    else [],
                    "timing_metadata_kind": "exposure_manifest",
                    "tpx_file_count": 1,
                }
            )
        if not windows:
            raise ValueError(
                f"Processing exclusions omit every exposure in {manifest_path}"
            )
        return windows

    exclusions_path = (
        os.path.join(str(metadata_dir), PROCESSING_EXCLUSIONS_FILENAME)
        if metadata_dir is not None
        else None
    )
    if exclusions_path is not None and os.path.isfile(exclusions_path):
        if manifest_error is not None:
            raise ValueError(
                f"{manifest_error}; cannot safely apply {exclusions_path}"
            ) from manifest_error
        raise ValueError(
            f"Processing exclusions require {EXPOSURE_MANIFEST_FILENAME}: {exclusions_path}"
        )

    metadata_error = None
    try:
        metadata_window = get_run_metadata_window(metadata_dir)
    except ValueError as error:
        metadata_window = None
        metadata_error = error
    if metadata_window is not None:
        if manifest_error is not None:
            warnings.warn(
                f"{manifest_error}; using run-level acquisition metadata instead."
            )
        return [metadata_window]

    if tpx_folder is None or not os.path.isdir(tpx_folder):
        if manifest_error is not None:
            warnings.warn(f"{manifest_error}; no usable TPX3 fallback was found.")
        if metadata_error is not None:
            warnings.warn(f"{metadata_error}; image metadata/file time will be used.")
        return []
    if manifest_error is not None:
        warnings.warn(f"{manifest_error}; using raw TPX3 modification times.")
    if metadata_error is not None:
        warnings.warn(f"{metadata_error}; using raw TPX3 modification times.")
    duration = float(tpx_time)
    if duration <= 0:
        raise ValueError("tpx_time must be positive")
    windows = []
    for filename in sorted(os.listdir(tpx_folder)):
        if not filename.endswith(extension):
            continue
        stat_result = os.stat(os.path.join(tpx_folder, filename))
        end_ns = stat_result.st_mtime_ns
        start_ns = end_ns - int(round(duration * 1e9))
        midpoint_ns = start_ns + (end_ns - start_ns) // 2
        end = _datetime_from_unix_ns(end_ns)
        start = _datetime_from_unix_ns(start_ns)
        windows.append(
            {
                "start": start,
                "end": end,
                "midpoint": _datetime_from_unix_ns(midpoint_ns),
                "start_unix_ns": start_ns,
                "end_unix_ns": end_ns,
                "midpoint_unix_ns": midpoint_ns,
                "beam_start_ns": int(pd.Timestamp(start).value),
                "beam_end_ns": int(pd.Timestamp(end).value),
                "duration_s": duration,
                "file_name": filename,
                "timing_source": "raw_tpx3_mtime_minus_tpx_time",
                "timing_quality": "estimated",
                "manifest_path": None,
                "metadata_path": None,
                "timing_metadata_kind": "raw_tpx3",
                "tpx_file_count": 1,
            }
        )
    return windows


def get_run_exposure_windows(run, tpx_time=2, extension=".tpx3"):
    """Resolve one run's manifest or legacy raw-file timing fallback."""
    return get_exposure_windows(
        tpx_folder=run.get("tpx_dir"),
        metadata_dir=run.get("metadata_dir"),
        tpx_time=tpx_time,
        extension=extension,
    )


def exposure_group_provenance(exposure_windows):
    """Summarize included and explicitly excluded TPX3 inputs for one TIFF."""
    if not exposure_windows:
        return {
            "tpx_file_count": 0,
            "input_tpx_file_count": 0,
            "excluded_tpx_file_count": 0,
            "excluded_tpx_files": [],
            "processing_exclusions_path": None,
            "timing_metadata_paths": [],
            "included_effective_exposure_s": 0.0,
            "nominal_effective_exposure_s": 0.0,
            "effective_exposure_fraction": 1.0,
        }
    included_count = sum(
        int(window.get("tpx_file_count", 1)) for window in exposure_windows
    )
    metadata_paths = set()
    for window in exposure_windows:
        metadata_paths.update(
            str(path) for path in window.get("timing_metadata_paths", []) if path
        )
        if window.get("metadata_path"):
            metadata_paths.add(str(window["metadata_path"]))
    first = exposure_windows[0] if exposure_windows else {}
    excluded_files = list(first.get("excluded_tpx_files", []))
    included_exposure_s = float(
        sum(window["duration_s"] for window in exposure_windows)
    )
    nominal_exposure_s = float(
        first.get("nominal_effective_exposure_s", included_exposure_s)
    )
    if nominal_exposure_s <= 0 or not np.isfinite(nominal_exposure_s):
        raise ValueError("Nominal effective exposure must be positive")
    return {
        "tpx_file_count": included_count,
        "input_tpx_file_count": int(
            first.get("input_tpx_file_count", included_count)
        ),
        "excluded_tpx_file_count": int(
            first.get("excluded_tpx_file_count", len(excluded_files))
        ),
        "excluded_tpx_files": excluded_files,
        "processing_exclusions_path": first.get("processing_exclusions_path"),
        "timing_metadata_paths": sorted(metadata_paths),
        "included_effective_exposure_s": included_exposure_s,
        "nominal_effective_exposure_s": nominal_exposure_s,
        "effective_exposure_fraction": included_exposure_s / nominal_exposure_s,
    }


def mean_exposure_midpoint(exposure_windows):
    """Return the duration-weighted midpoint of a collection of exposures."""
    if not exposure_windows:
        return None
    weights = np.asarray([window["duration_s"] for window in exposure_windows], dtype=float)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise ValueError("Exposure windows contain invalid durations")
    midpoint_seconds = np.asarray([
        window.get("midpoint_unix_ns", int(window["midpoint"].timestamp() * 1e9)) / 1e9
        for window in exposure_windows
    ], dtype=float)
    return datetime.fromtimestamp(float(np.average(midpoint_seconds, weights=weights)))


def summarize_experiment_tpx_time_range(exp_dir, extension=".tpx3"):
    """
    Summarize earliest and latest TPX file modification times for one experiment.

    Parameters
    ----------
    exp_dir : str
        Experiment directory containing run subfolders with ``tpx3Files``.
    extension : str, optional
        TPX file extension to scan (default ``.tpx3``).

    Returns
    -------
    dict
        Summary including experiment name, run count, TPX file count, earliest time,
        latest time, and duration in hours. If no TPX files are found, the datetime
        fields are ``None`` and ``tpx_file_count`` is 0.
    """
    layout = resolve_nci_experiment(exp_dir)
    all_windows = []
    run_dirs = []
    for run in layout["runs"]:
        run_windows = get_run_exposure_windows(run, extension=extension)
        if run_windows:
            all_windows.extend(run_windows)
            run_dirs.append(run["name"])

    if not all_windows:
        return {
            "exp_name": layout["experiment"],
            "exp_dir": layout["experiment_dir"],
            "layout_type": layout["layout_type"],
            "run_count": len(run_dirs),
            "tpx_file_count": 0,
            "timing_window_count": 0,
            "timing_sources": [],
            "earliest_datetime": None,
            "latest_datetime": None,
            "duration_hours": None,
        }

    earliest = min(window["start"] for window in all_windows)
    latest = max(window["end"] for window in all_windows)
    tpx_file_count = sum(
        int(window.get("tpx_file_count", 1)) for window in all_windows
    )
    timing_sources = sorted({
        str(window.get("timing_metadata_kind", window.get("timing_source", "unknown")))
        for window in all_windows
    })
    return {
        "exp_name": layout["experiment"],
        "exp_dir": layout["experiment_dir"],
        "layout_type": layout["layout_type"],
        "run_count": len(run_dirs),
        "tpx_file_count": tpx_file_count,
        "timing_window_count": len(all_windows),
        "timing_sources": timing_sources,
        "earliest_datetime": earliest,
        "latest_datetime": latest,
        "duration_hours": (latest - earliest).total_seconds() / 3600.0,
    }


def summarize_experiment_root_tpx_ranges(experiments_root, extension=".tpx3"):
    """
    Summarize TPX time ranges for all experiment folders in a root directory.

    Parameters
    ----------
    experiments_root : str
        Directory containing one subfolder per experiment.
    extension : str, optional
        TPX file extension to scan (default ``.tpx3``).

    Returns
    -------
    list of dict
        One summary per experiment folder, sorted by experiment name.
    """
    experiments_root = os.path.abspath(experiments_root)
    summaries = []
    experiment_dirs = list_nci_experiment_dirs(experiments_root)
    if not experiment_dirs and os.path.isdir(experiments_root):
        experiment_dirs = [
            entry.path
            for entry in sorted(os.scandir(experiments_root), key=lambda e: e.name)
            if entry.is_dir()
        ]
    for experiment_dir in experiment_dirs:
        try:
            summaries.append(
                summarize_experiment_tpx_time_range(experiment_dir, extension=extension)
            )
        except ValueError:
            continue
    return summaries


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

    header = (
        f"WARNING: existing {stage_name} outputs were created with different "
        f"parameters than the current notebook settings."
    )
    detail = f"Metadata file: {metadata_path}"
    border_width = max(len(header), len(detail), 72)
    border = "!" * border_width

    print("")
    print(border)
    print(header)
    print(detail)
    print("The existing files may not match this run. Recompute the stage if these changes matter.")
    print(border)
    for diff in differences:
        print(
            f"  - {diff['key']}: current={diff['expected']} | stored={diff['stored']}"
        )
    print(border)
    print("")


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
    file_filter=None,
    beamhistory_path=None,
    reference_beam_integral_per_minute=None,
    tpx_time=2,
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
    file_filter : str or None
        Optional substring that must appear in the filename.

    Returns
    -------
    list of np.ndarray
        Reference image stack in run/file sort order.
    """
    layout = resolve_nci_experiment(reference_dir, final_subfolder=final_subfolder)
    correction_factors = {}
    if (
        clean_optional_path(beamhistory_path) is not None
        and _positive_float(reference_beam_integral_per_minute) is not None
    ):
        _, correction_plan, intensity_enabled = _correction_plan(
            reference_dir,
            beamhistory_path=beamhistory_path,
            tpx_time=tpx_time,
            reference_beam_integral_per_minute=reference_beam_integral_per_minute,
            file_filter=file_filter,
            final_subfolder=final_subfolder,
        )
        if intensity_enabled:
            correction_factors = {
                os.path.abspath(row["source_path"]): float(row["relative_factor"])
                for row in correction_plan
            }
    ref_series = []
    for run in layout["runs"]:
        for path in _run_image_paths(run, file_filter=file_filter):
            try:
                img = tifffile.imread(path)
            except Exception:
                continue
            if img.ndim != 2:
                continue
            image = np.asarray(img, dtype=dtype)
            factor = correction_factors.get(os.path.abspath(path), 1.0)
            if not np.isfinite(factor) or factor <= 1e-6:
                raise ValueError(
                    f"Reference beam-dose correction factor is invalid for {path}: {factor}"
                )
            ref_series.append(image / factor)
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
def compute_run_metrics(
    beamhistory,
    tpx_folder,
    tpx_time=2,
    context=None,
    metadata_dir=None,
):
    exposure_windows = get_exposure_windows(
        tpx_folder=tpx_folder,
        metadata_dir=metadata_dir,
        tpx_time=tpx_time,
    )
    if len(exposure_windows) == 0:
        return np.nan, None

    validate_beamhistory_window_coverage(
        beamhistory,
        exposure_windows,
        context=context or os.path.basename(str(tpx_folder or metadata_dir or "run")),
    )
    mean_beam_integral_per_minute = mean_beam_integral_for_windows(
        beamhistory, exposure_windows
    )
    mean_tpx_time = mean_exposure_midpoint(exposure_windows)
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


def validate_beamhistory_window_coverage(
    beamhistory,
    exposure_windows,
    context="operando",
):
    """Validate beam-history coverage against explicit start/end windows."""
    if not exposure_windows:
        return
    t_min = min(window["start"] for window in exposure_windows)
    t_max = max(window["end"] for window in exposure_windows)
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


def integrate_beam_windows(beamhistory, exposure_windows):
    """Compute the per-minute beam level for explicit exposure windows."""
    if not exposure_windows:
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

    values = []
    for window in exposure_windows:
        duration_s = float(window["duration_s"])
        if duration_s <= 0:
            raise ValueError("Exposure window duration must be positive")
        start_ns = np.int64(
            window.get("beam_start_ns", pd.Timestamp(window["start"]).value)
        )
        end_ns = np.int64(
            window.get("beam_end_ns", pd.Timestamp(window["end"]).value)
        )
        overlap_start = np.maximum(ts_ns, start_ns)
        overlap_end = np.minimum(next_ns, end_ns)
        duration_ns = overlap_end - overlap_start
        duration_ns[duration_ns < 0] = 0
        overlap_seconds = duration_ns / 1e9
        window_sum = np.sum(beam.values * overlap_seconds)
        values.append(window_sum / duration_s * 60.0)
        if np.all(overlap_seconds == 0):
            warnings.warn(
                f"No beam history overlap for exposure window "
                f"{window['start']} to {window['end']}. Beam integral set to 0."
            )
    return np.asarray(values, dtype=float)


def mean_beam_integral_for_windows(beamhistory, exposure_windows):
    """Return one duration-weighted beam level for an image acquisition."""
    values = integrate_beam_windows(beamhistory, exposure_windows)
    if len(values) == 0:
        return np.nan
    durations = np.asarray(
        [window["duration_s"] for window in exposure_windows], dtype=float
    )
    return float(np.average(values, weights=durations))


def beam_metrics_for_windows(beamhistory, exposure_windows):
    """Return average beam level, total dose, and per-TPX3 beam integrals."""
    values = integrate_beam_windows(beamhistory, exposure_windows)
    if len(values) == 0:
        return {
            "beam_integral_per_minute": np.nan,
            "total_beam_integral": np.nan,
            "tpx3_beam_integrals": [],
        }
    durations = np.asarray(
        [window["duration_s"] for window in exposure_windows], dtype=float
    )
    total_integrals = values * durations / 60.0
    details = []
    for window, per_minute, total in zip(exposure_windows, values, total_integrals):
        details.append(
            {
                "file_name": window.get("file_name"),
                "start": window["start"].isoformat(),
                "end": window["end"].isoformat(),
                "duration_s": float(window["duration_s"]),
                "beam_integral_per_minute": float(per_minute),
                "total_beam_integral": float(total),
            }
        )
    return {
        "beam_integral_per_minute": float(np.average(values, weights=durations)),
        "total_beam_integral": float(np.sum(total_integrals)),
        "tpx3_beam_integrals": details,
    }


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
    reference_beamhistory_path = clean_optional_path(reference_beamhistory_path)
    if reference_beamhistory_path is None:
        return np.nan
    beamhistory = load_beamhistory(reference_beamhistory_path)
    layout = resolve_nci_experiment(reference_dir)

    all_beam_integrals = []
    all_exposure_durations = []
    for run in layout["runs"]:
        exposure_windows = get_run_exposure_windows(run, tpx_time=tpx_time)
        if not exposure_windows:
            continue

        validate_beamhistory_window_coverage(
            beamhistory,
            exposure_windows,
            context=f"reference:{run['name']}",
        )
        beam_integrals = integrate_beam_windows(beamhistory, exposure_windows)
        all_beam_integrals.extend(beam_integrals.tolist())
        all_exposure_durations.extend(
            window["duration_s"] for window in exposure_windows
        )

    if len(all_beam_integrals) == 0:
        return np.nan

    return float(np.average(all_beam_integrals, weights=all_exposure_durations))


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
def _source_image_datetime(path):
    """Read a useful source timestamp, falling back to file modification time."""
    values = []
    try:
        with tifffile.TiffFile(path) as tif:
            tags = tif.pages[0].tags
            for tag_id in (65002, 65000, "DateTime", "DateTimeOriginal"):
                if tag_id in tags:
                    values.append(tags[tag_id].value)
            if 270 in tags:
                for line in str(tags[270].value).splitlines():
                    if line.startswith(("MeanTPXModifiedTime:", "AcquisitionTime:")):
                        values.append(line.split(":", 1)[1].strip())
    except Exception:
        pass
    for value in values:
        parsed = pd.to_datetime(value, errors="coerce")
        if not pd.isna(parsed):
            return parsed.to_pydatetime()
    return datetime.fromtimestamp(os.path.getmtime(path))


def _exposure_groups_for_images(run, image_count, n_tpx_per_image, tpx_time=2):
    exposure_windows = get_run_exposure_windows(run, tpx_time=tpx_time)
    if run["layout_type"] == "v2":
        # Every filtered TIFF variant in a v2 run is reconstructed from the same
        # complete raw acquisition, so all variants share one TPX interval.
        return exposure_windows, [exposure_windows for _ in range(image_count)]
    group_size = max(1, int(n_tpx_per_image))
    groups = [
        exposure_windows[start:start + group_size]
        for start in range(0, len(exposure_windows), group_size)
    ]
    return exposure_windows, groups


def _correction_plan(
    exp_dir,
    beamhistory_path=None,
    tpx_time=2,
    reference_beam_integral_per_minute=None,
    n_tpx_per_image=30,
    file_filter=None,
    final_subfolder="final",
):
    layout = resolve_nci_experiment(exp_dir, final_subfolder=final_subfolder)
    beamhistory_path = clean_optional_path(beamhistory_path)
    reference_value = reference_beam_integral_per_minute
    intensity_enabled = (
        beamhistory_path is not None
        and reference_value is not None
        and np.isfinite(reference_value)
        and float(reference_value) > 0
    )
    beamhistory = load_beamhistory(beamhistory_path) if intensity_enabled else None
    rows = []

    for run in layout["runs"]:
        image_paths = _run_image_paths(run, file_filter=file_filter)
        if not image_paths:
            continue
        exposure_windows, groups = _exposure_groups_for_images(
            run,
            len(image_paths),
            n_tpx_per_image,
            tpx_time=tpx_time,
        )
        if intensity_enabled and not exposure_windows:
            warnings.warn(
                f"No timing JSON or TPX3 files found for run {run['name']}; "
                "intensity correction was skipped for its images."
            )

        for index, src in enumerate(image_paths):
            fname = os.path.basename(src)
            group = groups[index] if index < len(groups) else []
            acquisition_time = mean_exposure_midpoint(group) if group else _source_image_datetime(src)
            timing_kind = group[0].get("timing_metadata_kind") if group else None
            timing_source = {
                "exposure_manifest": "manifest",
                "raw_tpx3": "raw_tpx3_fallback",
            }.get(timing_kind, timing_kind)
            if timing_source is None:
                timing_source = "source_tiff_metadata_or_mtime"
            provenance = exposure_group_provenance(group)
            beam_integral = None
            total_beam_integral = None
            tpx3_beam_integrals = []
            relative_beam_factor = 1.0
            factor = 1.0
            applied = False
            if intensity_enabled and group:
                validate_beamhistory_window_coverage(
                    beamhistory,
                    group,
                    context=f"experiment:{layout['experiment']}:{run['name']}",
                )
                beam_metrics = beam_metrics_for_windows(beamhistory, group)
                beam_integral = beam_metrics["beam_integral_per_minute"]
                total_beam_integral = beam_metrics["total_beam_integral"]
                tpx3_beam_integrals = beam_metrics["tpx3_beam_integrals"]
                relative_beam_factor = beam_integral / float(reference_value)
                factor = (
                    relative_beam_factor
                    * provenance["effective_exposure_fraction"]
                )
                applied = True

            base, ext = os.path.splitext(fname)
            ext = ext or ".tif"
            dst_name = f"{run['name']}_obint_corrected{ext}"
            if len(image_paths) > 1:
                dst_name = f"{run['name']}_obint_corrected_{base}{ext}"
            rows.append(
                {
                    "run": run["name"],
                    "layout_type": run["layout_type"],
                    "image_index": index,
                    "filename": fname,
                    "source_path": src,
                    "corrected_filename": dst_name,
                    "time": acquisition_time,
                    "beam_integral": beam_integral,
                    "beam_integral_per_minute": beam_integral,
                    "total_beam_integral": total_beam_integral,
                    "tpx3_beam_integrals_json": json.dumps(
                        tpx3_beam_integrals, separators=(",", ":")
                    ),
                    "relative_beam_factor": float(relative_beam_factor),
                    "relative_factor": float(factor),
                    "intensity_correction_applied": applied,
                    "tpx_file_count": provenance["tpx_file_count"],
                    "input_tpx_file_count": provenance["input_tpx_file_count"],
                    "excluded_tpx_file_count": provenance["excluded_tpx_file_count"],
                    "excluded_tpx_files": ";".join(provenance["excluded_tpx_files"]),
                    "processing_exclusions_path": provenance["processing_exclusions_path"],
                    "exposure_window_count": len(group),
                    "exposure_start_utc": min(w['start'] for w in group).astimezone(timezone.utc).isoformat()
                        if group and timing_source not in {'raw_tpx3_fallback','source_tiff_metadata_or_mtime'} else None,
                    "exposure_end_utc": max(w['end'] for w in group).astimezone(timezone.utc).isoformat()
                        if group and timing_source not in {'raw_tpx3_fallback','source_tiff_metadata_or_mtime'} else None,
                    "total_effective_exposure_s": provenance[
                        "included_effective_exposure_s"
                    ],
                    "nominal_effective_exposure_s": provenance[
                        "nominal_effective_exposure_s"
                    ],
                    "effective_exposure_fraction": provenance[
                        "effective_exposure_fraction"
                    ],
                    "exposure_timing_source": timing_source,
                    "timing_metadata_paths": provenance["timing_metadata_paths"],
                }
            )
    return layout, rows, intensity_enabled


def intensity_correct_experiment(
    exp_dir,
    output_dir,
    beamhistory_path=None,
    tpx_time=2,
    reference_beam_integral_per_minute=None,
    reference_image=None,
    reference_thickness_um=1.0,
    compute_dtype=np.float32,
    corrected_output_dtype=np.float32,
    n_tpx_per_image=30,
    file_filter=None,
    verbose=True,
    final_subfolder="final",
):
    """Apply available beam/OB corrections to legacy or version-2 NCI images.

    A blank beam-history path produces correction factor 1.0 and continues with
    reference-image normalization when a reference image is supplied.
    """
    os.makedirs(output_dir, exist_ok=True)
    layout, plan, intensity_enabled = _correction_plan(
        exp_dir,
        beamhistory_path=beamhistory_path,
        tpx_time=tpx_time,
        reference_beam_integral_per_minute=reference_beam_integral_per_minute,
        n_tpx_per_image=n_tpx_per_image,
        file_filter=file_filter,
        final_subfolder=final_subfolder,
    )
    if verbose:
        layout_label = (
            "LumaCam proposal" if layout["layout_type"] == "v2" else "legacy LumaCam folder"
        )
        print(f"Resolved {layout_label} for {layout['experiment']}.")
        if not intensity_enabled:
            print("Beam history/reference integral unavailable; skipped intensity correction.")

    beam_log = []
    for index, row in enumerate(plan, 1):
        dst = os.path.join(output_dir, row["corrected_filename"])
        if verbose:
            correction_label = (
                f"factor={row['relative_factor']:.6g}"
                if row["intensity_correction_applied"]
                else "intensity skipped"
            )
            print(f"[{index}/{len(plan)}] {row['run']}/{row['filename']} ({correction_label})")
        correct_and_save_tiff(
            src_path=row["source_path"],
            dst_path=dst,
            correction_factor=row["relative_factor"],
            acquisition_time=row["time"],
            # Tag 65002 is the established downstream image-time tag.  It now
            # represents the selected timing metadata as well as legacy TPX3
            # modification times.
            mean_tpx_modified_time=row["time"] if row["exposure_window_count"] else None,
            reference_image=reference_image,
            reference_thickness_um=reference_thickness_um,
            compute_dtype=compute_dtype,
            output_dtype=corrected_output_dtype,
        )
        beam_log.append({key: value for key, value in row.items() if key != "source_path"})

    if not plan:
        raise ValueError(
            f"No TIFF images found for experiment {layout['experiment']!r} "
            f"matching filter {file_filter!r}."
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
    output_dir=None,
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
        Path to the auto-named output folder.
    """

    layout = resolve_nci_experiment(exp_dir, final_subfolder=final_subfolder)
    if output_dir is None:
        exp_name = layout["experiment"]
        if layout["layout_type"] == "v2":
            proposal_root = proposal_root_for_layout(layout["layout_root"])
            results_dir = str(analysis_root_for_proposal(proposal_root) / exp_name)
        else:
            parent_dir = os.path.dirname(layout["experiment_dir"])
            results_dir = os.path.join(parent_dir, "analysis", exp_name)
        output_dir = os.path.join(results_dir, "images", f"uncorrected_{exp_name}")

    os.makedirs(output_dir, exist_ok=True)

    for run in layout["runs"]:
        run_dir = run["name"]
        # Prefer durable sidecar timing; raw TPX3 mtimes remain a legacy fallback.
        exposure_windows = get_run_exposure_windows(run)
        mean_tpx_time = mean_exposure_midpoint(exposure_windows)

        tiff_paths = _run_image_paths(run, file_filter=file_filter)

        if len(tiff_paths) == 0:
            warnings.warn(
                f"No TIFF images found for run {run_dir} matching filter '{file_filter}'; skipping run."
            )
            continue

        if not exposure_windows:
            warnings.warn(
                f"No timing JSON or TPX3 files found for run {run_dir}; uncorrected copies will "
                "use their existing image/file time."
            )

        for src in tiff_paths:
            fname = os.path.basename(src)
            base, ext = os.path.splitext(fname)
            if not ext:
                ext = ".tif"
            dst_name = f"{run_dir}_uncorrected{ext}"
            if len(tiff_paths) > 1:
                dst_name = f"{run_dir}_uncorrected_{base}{ext}"
            dst = os.path.join(output_dir, dst_name)
            copy_tiff_with_metadata(
                src,
                dst,
                mean_tpx_time or _source_image_datetime(src),
            )

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
def save_obint_beam_log(beam_log, path):
    """Persist OBINT correction beam-log rows for later inspection."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df = pd.DataFrame(beam_log)
    if not df.empty and "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    df.to_csv(path, index=False)
    return path


def load_obint_beam_log(path):
    """Load a persisted OBINT correction beam log."""
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    if df.empty:
        return []
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    return df.to_dict("records")


def compute_obint_beam_log(
    exp_dir,
    beamhistory_path=None,
    reference_beam_integral_per_minute=None,
    tpx_time=2,
    n_tpx_per_image=30,
    file_filter=None,
):
    """Compute correction-log rows without rewriting corrected images."""
    _, rows, _ = _correction_plan(
        exp_dir,
        beamhistory_path=beamhistory_path,
        tpx_time=tpx_time,
        reference_beam_integral_per_minute=reference_beam_integral_per_minute,
        n_tpx_per_image=n_tpx_per_image,
        file_filter=file_filter,
    )
    return [{key: value for key, value in row.items() if key != "source_path"} for row in rows]


def circular_roi_values(image, r0, x0, y0):
    """
    Return finite pixel values inside a circular ROI.
    """
    arr = np.asarray(image)
    yy, xx = np.ogrid[:arr.shape[0], :arr.shape[1]]
    mask = (xx - float(x0)) ** 2 + (yy - float(y0)) ** 2 <= float(r0) ** 2
    values = arr[mask]
    return values[np.isfinite(values)]


def clip_image_to_circular_roi_percentile(image, roi, percentile=5.0, compute_dtype=np.float32):
    """
    Floor the entire image to a percentile computed inside a circular ROI.

    The threshold is computed from pixels inside ``roi`` only. Every pixel in
    the full image with a value below that threshold is then set to the
    threshold value.
    """
    required = {"r0", "x0", "y0"}
    missing = required.difference(roi or {})
    if missing:
        raise ValueError(f"ROI is missing required keys: {sorted(missing)}")

    arr = np.asarray(image, dtype=compute_dtype)
    values = circular_roi_values(arr, roi["r0"], roi["x0"], roi["y0"])
    if values.size == 0:
        raise ValueError("Small ROI contains no finite pixels for OB clipping.")

    threshold = float(np.percentile(values, percentile))
    if not np.isfinite(threshold):
        raise ValueError("Computed OB clipping percentile is not finite.")

    return np.maximum(arr, threshold).astype(compute_dtype, copy=False), threshold


def run_operando_pipeline(
    exp_dir,
    beamhistory_path=None,
    tpx_time=2,
    n_tpx_per_image=30,
    reference_dir=None,
    reference_beamhistory_path=None,
    reference_bin_y=2,
    reference_bin_x=2,
    reference_final_subfolder="final",
    reference_thickness_um=1.0,
    clip_ob_to_small_roi_p5=False,
    ob_clip_roi=None,
    ob_clip_percentile=5.0,
    compute_dtype=np.float32,
    corrected_output_dtype=np.float32,
    file_filter=None,
    metadata_path=None,
    results_dir=None,
    corrected_dir=None,
    uncorrected_dir=None,
    reference_beam_integral_per_minute=None,
    processed_reference_path=None,
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

    layout = resolve_nci_experiment(exp_dir)
    exp_name = layout["experiment"]
    parent_dir = os.path.dirname(layout["experiment_dir"])
    beamhistory_path = clean_optional_path(beamhistory_path)
    reference_dir = clean_optional_path(reference_dir)
    reference_beamhistory_path = clean_optional_path(reference_beamhistory_path)
    processed_reference_path = clean_optional_path(processed_reference_path)

    if results_dir is None:
        if layout["layout_type"] == "v2":
            proposal_root = proposal_root_for_layout(layout["layout_root"])
            results_dir = str(analysis_root_for_proposal(proposal_root) / exp_name)
        else:
            results_dir = os.path.join(parent_dir, "analysis", exp_name)
    os.makedirs(results_dir, exist_ok=True)

    output_dir = corrected_dir
    if output_dir is None:
        output_dir = os.path.join(results_dir, "images", f"obint_corrected_{exp_name}")
    os.makedirs(output_dir, exist_ok=True)
    if uncorrected_dir is None:
        uncorrected_dir = os.path.join(
            results_dir, "images", f"uncorrected_{exp_name}"
        )

    intensity_correction_requested = beamhistory_path is not None
    if reference_beam_integral_per_minute is None and (
        beamhistory_path is not None
        and reference_dir is not None
        and reference_beamhistory_path is not None
    ):
        print("Computing reference beam integral per minute...")
        reference_beam_integral_per_minute = compute_reference_beam_integral_per_minute(
            reference_dir,
            reference_beamhistory_path,
            tpx_time,
        )
    intensity_correction_applied = (
        beamhistory_path is not None
        and reference_beam_integral_per_minute is not None
        and np.isfinite(reference_beam_integral_per_minute)
        and float(reference_beam_integral_per_minute) > 0
    )
    if intensity_correction_requested and not intensity_correction_applied:
        warnings.warn(
            "Experiment beam history was supplied without a usable reference beam "
            "integral/reference beam history; intensity correction was skipped."
        )
    elif not intensity_correction_requested:
        print("No experiment beam-history path supplied; skipped intensity correction.")

    processed_reference_image = None
    ref_series = []
    if processed_reference_path is not None:
        if not os.path.isfile(processed_reference_path):
            raise ValueError(
                f"Prepared reference image does not exist: {processed_reference_path}"
            )
        print(f"Loading prepared reference image: {processed_reference_path}")
        processed_reference_image = np.asarray(
            tifffile.imread(processed_reference_path), dtype=compute_dtype
        )
        if processed_reference_image.ndim != 2:
            raise ValueError(
                "Prepared reference image must be a 2D TIFF; "
                f"got shape {processed_reference_image.shape}."
            )
        ref_series = [processed_reference_path]
    elif reference_dir is not None:
        print("Loading reference images...")
        ref_series = load_reference_image_series(
            reference_dir,
            final_subfolder=reference_final_subfolder,
            dtype=compute_dtype,
            file_filter=file_filter,
            beamhistory_path=reference_beamhistory_path,
            reference_beam_integral_per_minute=reference_beam_integral_per_minute,
            tpx_time=tpx_time,
        )
        if len(ref_series) == 0:
            raise ValueError(
                "No reference TIFF images found. "
                "Check reference_dir, layout, reference_final_subfolder, and file_filter."
            )
        print("Processing reference images...")
        processed_reference_image = process_reference_images(
            ref_series,
            bin_y=reference_bin_y,
            bin_x=reference_bin_x,
            compute_dtype=compute_dtype,
        )
    else:
        print("No reference image path supplied; skipped OB/reference correction.")
    ob_clip_threshold = None
    if clip_ob_to_small_roi_p5 and processed_reference_image is not None:
        processed_reference_image, ob_clip_threshold = clip_image_to_circular_roi_percentile(
            processed_reference_image,
            ob_clip_roi,
            percentile=ob_clip_percentile,
            compute_dtype=compute_dtype,
        )
        print(
            "Applied OB image floor from small ROI: "
            f"percentile={float(ob_clip_percentile):g}, threshold={ob_clip_threshold:.6g}"
        )
    elif clip_ob_to_small_roi_p5:
        warnings.warn("OB clipping was requested without a reference image; skipped OB clipping.")

    print("Collecting uncorrected TIFFs...")
    uncorrected_dir = collect_and_copy_tiffs(
        exp_dir,
        file_filter=file_filter,
        output_dir=uncorrected_dir,
    )

    print("Writing corrected/normalized operando images...")
    beam_log = intensity_correct_experiment(
        exp_dir=exp_dir,
        output_dir=output_dir,
        beamhistory_path=beamhistory_path,
        tpx_time=tpx_time,
        reference_beam_integral_per_minute=(
            reference_beam_integral_per_minute if intensity_correction_applied else None
        ),
        reference_image=processed_reference_image,
        reference_thickness_um=reference_thickness_um,
        n_tpx_per_image=n_tpx_per_image,
        compute_dtype=compute_dtype,
        corrected_output_dtype=corrected_output_dtype,
        file_filter=file_filter,
    )

    if metadata_path is None:
        metadata_path = os.path.join(results_dir, "processing", "processing_metadata.json")
    processing_dir = os.path.dirname(os.path.abspath(metadata_path))
    beam_log_path = os.path.join(processing_dir, "obint_beam_log.csv")
    save_obint_beam_log(beam_log, beam_log_path)
    timing_sources = sorted({
        str(row.get("exposure_timing_source"))
        for row in beam_log
        if row.get("exposure_timing_source")
    })
    timing_metadata_paths = sorted({
        str(path)
        for row in beam_log
        for path in row.get("timing_metadata_paths", [])
    })

    write_stage_metadata(
        metadata_path,
        stage_name="obint_correction",
        parameters={
            "exp_dir": os.path.abspath(layout["experiment_dir"]),
            "beamhistory_path": beamhistory_path,
            "output_dir": os.path.abspath(output_dir),
            "uncorrected_dir": os.path.abspath(uncorrected_dir),
            "tpx_time": tpx_time,
            "n_tpx_per_image": n_tpx_per_image,
            "reference_dir": reference_dir,
            "processed_reference_path": processed_reference_path,
            "reference_beamhistory_path": reference_beamhistory_path,
            "reference_bin_y": reference_bin_y,
            "reference_bin_x": reference_bin_x,
            "reference_final_subfolder": reference_final_subfolder,
            "reference_thickness_um": reference_thickness_um,
            "clip_ob_to_small_roi_p5": clip_ob_to_small_roi_p5,
            "ob_clip_roi": ob_clip_roi if clip_ob_to_small_roi_p5 else None,
            "ob_clip_percentile": float(ob_clip_percentile) if clip_ob_to_small_roi_p5 else None,
            "compute_dtype": compute_dtype,
            "corrected_output_dtype": corrected_output_dtype,
            "file_filter": file_filter,
            "source_layout_type": layout["layout_type"],
        },
        extra={
            "reference_beam_integral_per_minute": reference_beam_integral_per_minute,
            "ob_clip_threshold": ob_clip_threshold,
            "beam_log_count": len(beam_log),
            "beam_log_path": os.path.abspath(beam_log_path),
            "intensity_correction_applied": intensity_correction_applied,
            "reference_correction_applied": processed_reference_image is not None,
            "reference_image_count": len(ref_series),
            "timing_sources": timing_sources,
            "timing_metadata_paths": timing_metadata_paths,
        },
        section_name=None,
    )

    return {
        "results_dir": results_dir,
        "corrected_dir": output_dir,
        "uncorrected_dir": uncorrected_dir,
        "reference_beam_integral_per_minute": reference_beam_integral_per_minute,
        "beam_log": beam_log,
        "beam_log_path": beam_log_path,
        "metadata_path": metadata_path,
        "source_layout_type": layout["layout_type"],
        "timing_sources": timing_sources,
        "timing_metadata_paths": timing_metadata_paths,
        "intensity_corrected": intensity_correction_applied,
        "reference_corrected": processed_reference_image is not None,
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

        if not os.path.isfile(path) or not is_tiff_file(path) or is_all_runs_sum(path):
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
        and not is_all_runs_sum(os.path.join(input_dir, fname))
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

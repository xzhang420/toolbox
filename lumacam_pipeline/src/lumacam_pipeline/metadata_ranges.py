"""Metadata-only acquisition time ranges for LumaCam proposals."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Iterable

from .layout import find_layout_root, resolve_experiment_name


UTC = timezone.utc
_ADMIN_METADATA_DIRS = {"migrations", "summed"}


@dataclass(frozen=True)
class ExperimentMetadataRange:
    experiment: str
    start_utc: datetime | None
    end_utc: datetime | None
    run_count: int
    timed_run_count: int
    exposure_frame_count: int
    sources: tuple[str, ...]
    metadata_files: tuple[Path, ...]
    missing_runs: tuple[str, ...]

    @property
    def duration_seconds(self):
        if self.start_utc is None or self.end_utc is None:
            return None
        return (self.end_utc - self.start_utc).total_seconds()


@dataclass(frozen=True)
class ProposalMetadataRange:
    proposal_root: Path
    layout_root: Path
    experiments: tuple[ExperimentMetadataRange, ...]

    @property
    def start_utc(self):
        values = [row.start_utc for row in self.experiments if row.start_utc is not None]
        return min(values) if values else None

    @property
    def end_utc(self):
        values = [row.end_utc for row in self.experiments if row.end_utc is not None]
        return max(values) if values else None

    @property
    def duration_seconds(self):
        if self.start_utc is None or self.end_utc is None:
            return None
        return (self.end_utc - self.start_utc).total_seconds()

    @property
    def timed_experiment_count(self):
        return sum(row.start_utc is not None and row.end_utc is not None for row in self.experiments)

    def padded_utc_range(self, minutes=5):
        if self.start_utc is None or self.end_utc is None:
            return None, None
        margin = timedelta(minutes=float(minutes))
        return self.start_utc - margin, self.end_utc + margin


def _read_json_object(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _datetime_from_unix_ns(value):
    try:
        seconds, nanoseconds = divmod(int(value), 1_000_000_000)
    except (TypeError, ValueError):
        return None
    return datetime.fromtimestamp(seconds, tz=UTC).replace(
        microsecond=nanoseconds // 1000
    )


def _datetime_from_iso(value):
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    # Python 3.10 accepts microseconds; trim longer nanosecond fractions while
    # preserving an explicit timezone suffix.
    match = re.match(r"^(.*?\.)(\d{7,})([+-]\d\d:\d\d)$", text)
    if match:
        text = match.group(1) + match.group(2)[:6] + match.group(3)
    try:
        value = datetime.fromisoformat(text)
    except ValueError:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _manifest_range(path):
    payload = _read_json_object(path)
    frames = payload.get("frames", ())
    if not isinstance(frames, list):
        return None
    starts = []
    ends = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        start = _datetime_from_unix_ns(frame.get("start_unix_ns"))
        end = _datetime_from_unix_ns(frame.get("end_unix_ns"))
        if start is None:
            start = _datetime_from_iso(frame.get("start_time_utc"))
        if end is None:
            end = _datetime_from_iso(frame.get("end_time_utc"))
        if start is not None and end is not None and end >= start:
            starts.append(start)
            ends.append(end)
    if not starts:
        return None
    return min(starts), max(ends), len(starts)


def _json_interval(payload):
    start = _datetime_from_iso(payload.get("started_at"))
    end = _datetime_from_iso(payload.get("ended_at"))
    if start is None or end is None or end < start:
        return None
    return start, end


def _metadata_experiment_names(metadata_root):
    if not metadata_root.is_dir():
        return []
    return sorted(
        child.name
        for child in metadata_root.iterdir()
        if child.is_dir()
        and child.name not in _ADMIN_METADATA_DIRS
        and not child.name.startswith(".")
    )


def _inspect_experiment(metadata_root, experiment):
    experiment_root = metadata_root / experiment
    experiment_payload = _read_json_object(experiment_root / "experiment.json")
    experiment_runs = experiment_payload.get("runs", {})
    if not isinstance(experiment_runs, dict):
        experiment_runs = {}
    run_names = {
        child.name
        for child in experiment_root.iterdir()
        if child.is_dir()
        and child.name not in _ADMIN_METADATA_DIRS
        and not child.name.startswith(".")
    } if experiment_root.is_dir() else set()
    run_names.update(str(name) for name in experiment_runs)

    starts = []
    ends = []
    sources = set()
    used_files = set()
    missing_runs = []
    exposure_frame_count = 0
    for run_name in sorted(run_names):
        run_root = experiment_root / run_name
        manifest_path = run_root / "tpx3_exposure_manifest.json"
        manifest_interval = _manifest_range(manifest_path) if manifest_path.is_file() else None
        if manifest_interval is not None:
            start, end, frame_count = manifest_interval
            source = "exposure_manifest"
            source_path = manifest_path
            exposure_frame_count += frame_count
        else:
            acquisition_path = run_root / "acquisition.json"
            acquisition_interval = (
                _json_interval(_read_json_object(acquisition_path))
                if acquisition_path.is_file()
                else None
            )
            if acquisition_interval is not None:
                start, end = acquisition_interval
                source = "acquisition_json"
                source_path = acquisition_path
            else:
                run_interval = _json_interval(experiment_runs.get(run_name, {}))
                if run_interval is None:
                    missing_runs.append(run_name)
                    continue
                start, end = run_interval
                source = "experiment_json"
                source_path = experiment_root / "experiment.json"
        starts.append(start)
        ends.append(end)
        sources.add(source)
        used_files.add(source_path)

    return ExperimentMetadataRange(
        experiment=experiment,
        start_utc=min(starts) if starts else None,
        end_utc=max(ends) if ends else None,
        run_count=len(run_names),
        timed_run_count=len(starts),
        exposure_frame_count=exposure_frame_count,
        sources=tuple(sorted(sources)),
        metadata_files=tuple(sorted(used_files)),
        missing_runs=tuple(missing_runs),
    )


def inspect_metadata_time_ranges(data_path, experiments: Iterable[str] | None = None):
    """Inspect acquisition ranges using JSON metadata only.

    No TPX3 file is opened or statted. Exposure manifests are preferred;
    ``acquisition.json`` and then ``experiment.json`` are metadata fallbacks.
    """

    layout_root = find_layout_root(data_path)
    if layout_root is None:
        raise ValueError(f"No supported LumaCam proposal layout found around: {data_path}")
    metadata_root = layout_root / "metadata"
    available = _metadata_experiment_names(metadata_root)
    if not available:
        raise ValueError(f"No experiment metadata directories found in: {metadata_root}")
    if experiments is None:
        selected = available
    else:
        selected = []
        for requested in experiments:
            resolved = resolve_experiment_name(requested, available)
            if resolved not in selected:
                selected.append(resolved)
        if not selected:
            raise ValueError("Select at least one experiment for metadata time inspection.")
    proposal_root = (
        layout_root.parent.parent if layout_root.parent.name == "data" else layout_root.parent
    )
    return ProposalMetadataRange(
        proposal_root=proposal_root.resolve(),
        layout_root=layout_root.resolve(),
        experiments=tuple(_inspect_experiment(metadata_root, name) for name in selected),
    )


def _local_time(value):
    return value.astimezone() if value is not None else None


def _iso(value):
    return value.isoformat(timespec="seconds") if value is not None else ""


def save_metadata_time_ranges_csv(report, output_path=None, margin_minutes=5):
    """Save proposal and per-experiment metadata ranges as a portable CSV."""

    target = Path(output_path) if output_path else (
        report.proposal_root
        / "analysis"
        / "proposal_summary"
        / "metadata_datetime_ranges.csv"
    )
    target = target.expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "scope",
        "experiment",
        "start_utc",
        "end_utc",
        "start_local",
        "end_local",
        "suggested_start_utc",
        "suggested_end_utc",
        "duration_seconds",
        "run_count",
        "timed_run_count",
        "missing_run_count",
        "exposure_frame_count",
        "metadata_sources",
        "metadata_file_count",
    )
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        proposal_padded_start, proposal_padded_end = report.padded_utc_range(margin_minutes)
        writer.writerow(
            {
                "scope": "proposal",
                "experiment": "",
                "start_utc": _iso(report.start_utc),
                "end_utc": _iso(report.end_utc),
                "start_local": _iso(_local_time(report.start_utc)),
                "end_local": _iso(_local_time(report.end_utc)),
                "suggested_start_utc": _iso(proposal_padded_start),
                "suggested_end_utc": _iso(proposal_padded_end),
                "duration_seconds": report.duration_seconds,
                "run_count": sum(row.run_count for row in report.experiments),
                "timed_run_count": sum(row.timed_run_count for row in report.experiments),
                "missing_run_count": sum(len(row.missing_runs) for row in report.experiments),
                "exposure_frame_count": sum(row.exposure_frame_count for row in report.experiments),
                "metadata_sources": ";".join(
                    sorted({source for row in report.experiments for source in row.sources})
                ),
                "metadata_file_count": sum(len(row.metadata_files) for row in report.experiments),
            }
        )
        for row in report.experiments:
            margin = timedelta(minutes=float(margin_minutes))
            writer.writerow(
                {
                    "scope": "experiment",
                    "experiment": row.experiment,
                    "start_utc": _iso(row.start_utc),
                    "end_utc": _iso(row.end_utc),
                    "start_local": _iso(_local_time(row.start_utc)),
                    "end_local": _iso(_local_time(row.end_utc)),
                    "suggested_start_utc": _iso(
                        row.start_utc - margin if row.start_utc is not None else None
                    ),
                    "suggested_end_utc": _iso(
                        row.end_utc + margin if row.end_utc is not None else None
                    ),
                    "duration_seconds": row.duration_seconds,
                    "run_count": row.run_count,
                    "timed_run_count": row.timed_run_count,
                    "missing_run_count": len(row.missing_runs),
                    "exposure_frame_count": row.exposure_frame_count,
                    "metadata_sources": ";".join(row.sources),
                    "metadata_file_count": len(row.metadata_files),
                }
            )
    return target.resolve(strict=False)


__all__ = [
    "ExperimentMetadataRange",
    "ProposalMetadataRange",
    "inspect_metadata_time_ranges",
    "save_metadata_time_ranges_csv",
]

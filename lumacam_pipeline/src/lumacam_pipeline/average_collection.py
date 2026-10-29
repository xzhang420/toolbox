"""Collect experiment averages for convenient proposal-wide comparison."""

from pathlib import Path
import filecmp
import os
import re
import shutil
import tempfile

from .config import load_config
from .result_layout import average_filename, snapshot_path


def collection_path(analysis_root, experiment, variant, analysis_name=""):
    """Include the variant so averages from different TIFF products stay separate."""
    def safe(value):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-") or "default"

    return (
        Path(analysis_root) / "proposal_summary" / "averaged_images"
        / average_filename(safe(experiment), safe(variant), analysis_name)
    )


def copy_average(source, analysis_root, experiment, variant, analysis_name=""):
    """Publish an exact copy; reruns refresh the same experiment/variant entry."""
    source = Path(source)
    target = collection_path(analysis_root, experiment, variant, analysis_name)
    if target.is_file() and filecmp.cmp(source, target, shallow=False):
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".average_", dir=target.parent)
    os.close(descriptor)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target


def collect_completed_batch(status_path, *, apply=False):
    """Audit or copy existing ex-situ averages without rerunning corrections."""
    import csv
    import hashlib

    status_path = Path(status_path).resolve()
    # analysis/proposal_summary/batches/<batch>/batch_status.csv
    batches = next((parent for parent in status_path.parents if parent.name == "batches"), None)
    if batches is None:
        raise ValueError("Select a pipeline batch_status.csv under analysis/batches.")
    analysis_root = batches.parent
    if analysis_root.name == "proposal_summary":
        analysis_root = analysis_root.parent
    plans = []
    with status_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["status"] not in {"completed", "skipped"}:
            continue
        experiment = row["experiment"]
        result_dir = Path(row.get("results_dir") or analysis_root / experiment).resolve()
        if not result_dir.is_relative_to(analysis_root / experiment):
            raise ValueError(f"Invalid experiment folder: {experiment}")
        config_path = snapshot_path(result_dir)
        if not config_path.is_file():
            config_path = result_dir / "lumacam_pipeline.ini"
        preset, config = load_config(config_path)
        if preset != "exsitu":
            continue
        variant = config.get("project", {}).get("variant") or "default"
        analysis_name = config.get("project", {}).get("analysis_name", "")
        filename = config.get("average", {}).get("filename") or average_filename(experiment, variant, analysis_name)
        if not (result_dir / "images" / filename).exists() and config_path.name == "lumacam_pipeline.ini":
            filename = f"average_{experiment}.tiff"
        source = (result_dir / "images" / filename).resolve()
        if source.parent != result_dir / "images":
            raise ValueError(f"Average is outside the experiment images folder: {source}")
        if not source.is_file():
            raise FileNotFoundError(source)
        target = collection_path(analysis_root, experiment, variant, analysis_name)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise FileExistsError(f"A different comparison copy already exists: {target}")
        plans.append((source, target, experiment, variant, analysis_name, digest))
    for source, target, experiment, variant, analysis_name, digest in plans:
        if apply:
            copy_average(source, analysis_root, experiment, variant, analysis_name)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"Copy verification failed: {target}")
        print(f"{'Verified' if apply else 'Would copy'}: {source} -> {target}")
    print(f"{len(plans)} experiment average(s); {'copied and verified' if apply else 'audit only'}.")
    return [target for _, target, _, _, _, _ in plans]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-status", required=True)
    parser.add_argument("--apply", action="store_true", help="Copy after auditing all targets")
    args = parser.parse_args()
    collect_completed_batch(args.batch_status, apply=args.apply)

"""Shared rules distinguishing preview sums from individual TIFF inputs."""

from pathlib import Path
import re


def is_all_runs_sum(path):
    """Also recognize copied/processed aggregate filenames and summed folders."""
    path = Path(path)
    return "_sumallruns" in path.stem.lower() or path.parent.name == "summed"


def final_sum_variant(path):
    """Read the original event-image variant independently of storage folder."""
    match = re.search(r"_sumImage_([0-9]+to[0-9]+)_sumAllRuns$", Path(path).stem, re.I)
    return match.group(1).lower() if match else None

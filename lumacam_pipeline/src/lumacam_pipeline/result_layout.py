"""Canonical analysis paths, shared by individual runs, batches and migration."""

from .proposal_paths import analysis_root_for_proposal

from pathlib import Path
import re


def component(value, default="default"):
    name = str(value or "").strip() or default
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name.endswith("."):
        raise ValueError(f"Use letters, numbers, underscores or hyphens for folder names: {name!r}")
    return name


def named_directory(base, name=""):
    return Path(base) / "alternatives" / component(name) if str(name or "").strip() else Path(base)


def result_directory(root, experiment, variant="", analysis_name=""):
    root = Path(root)
    experiment_root = root if root.name == experiment else root / component(experiment)
    return named_directory(experiment_root / component(variant), analysis_name)


def shared_directory(proposal, config):
    variant = config.get("project", {}).get("variant") or config.get("corrections", {}).get("file_filter", "")
    name = config.get("shared_setup", {}).get("reference_set", "")
    return named_directory(analysis_root_for_proposal(proposal) / "shared_corrections" / component(variant), name)


def snapshot_path(results_dir):
    return Path(results_dir) / "processing" / "config_used.ini"


def average_filename(experiment, variant="", analysis_name=""):
    suffix = f"_{component(analysis_name)}" if str(analysis_name or "").strip() else ""
    return f"average_{component(experiment)}_{component(variant)}{suffix}.tiff"

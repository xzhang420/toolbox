"""Prepared OB files are correction inputs, not source stacks to process again."""

import json
from pathlib import Path

import numpy as np
import tifffile

from .products import is_all_runs_sum


def uses_prepared_ob(config):
    return config.get("reference_input_mode") == "prepared" or (
        bool(config.get("prepared_reference_path")) and not config.get("reference_dir")
    )


def validate_prepared_ob(value):
    path = Path(value).expanduser()
    if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Prepared OB must be an existing TIFF file: {path}")
    if is_all_runs_sum(path):
        raise ValueError("Select a processed, averaged OB TIFF, not an all-runs preview sum.")
    with tifffile.TiffFile(path) as tif:
        if len(tif.series[0].shape) != 2:
            raise ValueError("Prepared OB must be a single 2D TIFF image.")
    return path


def prepared_ob_integral(value):
    """Recover normalization from our preparation sidecar, when available."""
    path = Path(value).expanduser()
    metadata = path.with_name("ob_preparation.json")
    if path.name != "ob.tiff" or not metadata.is_file():
        return None
    settings = json.loads(metadata.read_text(encoding="utf-8"))
    integral = settings.get("reference_beam_integral_per_minute")
    if integral is None:
        return None
    integral = float(integral)
    return integral if np.isfinite(integral) and integral > 0 else None

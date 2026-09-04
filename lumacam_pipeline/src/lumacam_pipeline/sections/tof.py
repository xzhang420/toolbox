"""Analysis of an already reconstructed time-of-flight image cube.

This section deliberately does not interpret raw TPX3 packets. It accepts a
3-D TIFF/NumPy cube produced by an upstream reconstruction and keeps ToF
analysis independent from NCI and electrochemistry sections.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import tifffile

from ..section import PipelineSection, SectionOutcome, SectionState


def _load_cube(path, npz_key="cube"):
    suffix = path.suffix.lower()
    if suffix == ".npy":
        cube = np.load(path)
    elif suffix == ".npz":
        with np.load(path) as archive:
            if npz_key not in archive:
                raise ValueError(f"NPZ cube key {npz_key!r} is unavailable; found {list(archive)}")
            cube = archive[npz_key]
    elif suffix in {".tif", ".tiff"}:
        cube = tifffile.imread(path)
    else:
        raise ValueError("ToF cubes must be .tif/.tiff, .npy, or .npz")
    cube = np.asarray(cube, dtype=np.float32)
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3-D ToF cube, got shape {cube.shape}: {path}")
    return cube


class TimeOfFlightCubeSection(PipelineSection):
    id = "tof_cube"
    title = "Time-of-flight cube"
    requires = ("results_dir", "experiment")
    produces = ("tof_axis", "tof_cube", "tof_feature_paths", "tof_spectrum_path")

    def is_configured(self, config):
        return bool(config.get("enabled", False) and str(config.get("path", "")).strip())

    def run(self, session, config):
        path = Path(config["path"]).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"ToF cube does not exist: {path}")
        cube = _load_cube(path, str(config.get("npz_key", "cube")))
        axis_order = str(config.get("axis_order", "CYX")).upper()
        if axis_order == "YXC":
            cube = np.moveaxis(cube, -1, 0)
        elif axis_order != "CYX":
            raise ValueError("axis_order must be CYX or YXC")
        raw_axis = config.get("axis_values")
        if isinstance(raw_axis, str):
            raw_axis = [part.strip() for part in raw_axis.replace(";", ",").split(",") if part.strip()]
        if raw_axis:
            axis = np.asarray(raw_axis, dtype=float)
        else:
            start = float(config.get("axis_start", 0.0))
            step = float(config.get("axis_step", 1.0))
            axis = start + np.arange(cube.shape[0], dtype=float) * step
        if len(axis) != cube.shape[0]:
            raise ValueError(f"ToF axis has {len(axis)} values but cube has {cube.shape[0]} channels")

        total = cube.sum(axis=0, dtype=np.float32)
        weighted = (cube * axis[:, None, None]).sum(axis=0, dtype=np.float64)
        mean = np.divide(weighted, total, out=np.zeros_like(total, dtype=float), where=total > 0)
        dominant = axis[cube.argmax(axis=0)]
        dominant = np.where(total > 0, dominant, np.nan)
        global_counts = cube.reshape(cube.shape[0], -1).sum(axis=1, dtype=np.float64)

        results_dir = Path(session.artifacts.require("results_dir"))
        output_root = results_dir / "images" / "tof"
        table_root = results_dir / "tables" / "tof"
        output_root.mkdir(parents=True, exist_ok=True)
        table_root.mkdir(parents=True, exist_ok=True)
        feature_arrays = {
            "tof_integrated_counts.tiff": total,
            "tof_mean_axis_value.tiff": mean.astype(np.float32),
            "tof_dominant_axis_value.tiff": np.nan_to_num(dominant, nan=0).astype(np.float32),
        }
        for band in config.get("bands", ()) or ():
            low, high = map(float, band)
            mask = (axis >= low) & (axis <= high)
            feature_arrays[f"tof_band_{low:g}_to_{high:g}.tiff"] = (
                cube[mask].sum(axis=0, dtype=np.float32) if mask.any() else np.zeros(cube.shape[1:], np.float32)
            )
        feature_paths = {}
        for name, image in feature_arrays.items():
            output = output_root / name
            tifffile.imwrite(output, image)
            feature_paths[name] = output
        spectrum_path = table_root / "tof_global_spectrum.csv"
        with spectrum_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([str(config.get("axis_label", "tof")), "count"])
            writer.writerows(zip(axis.tolist(), global_counts.tolist()))
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Analyzed reconstructed ToF cube {cube.shape} from {path.name}.",
            {
                "tof_axis": axis,
                "tof_cube": cube,
                "tof_feature_paths": feature_paths,
                "tof_spectrum_path": spectrum_path,
            },
        )

"""Photon-count hyperspectral cube extraction from exact ``NtoN`` TIFFs."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile

from ..section import PipelineSection, SectionOutcome, SectionState


EXACT_PHOTON_RE = re.compile(r"(?:^|_)sumImage_(\d+)to(\d+)(?:_|\.)", re.IGNORECASE)


def exact_photon_number(path):
    match = EXACT_PHOTON_RE.search(Path(path).name)
    if not match or match.group(1) != match.group(2):
        return None
    return int(match.group(1))


def robust_scale(image, low=1.0, high=99.7):
    array = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(array)
    if not finite.any():
        return np.zeros(array.shape, dtype=np.float32)
    minimum, maximum = np.percentile(array[finite], [low, high])
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.float32)
    return np.clip((array - minimum) / (maximum - minimum), 0, 1).astype(np.float32)


def _band_sum(cube, photon_numbers, low, high):
    mask = (photon_numbers >= low) & (photon_numbers <= high)
    return cube[mask].sum(axis=0, dtype=np.float32) if mask.any() else np.zeros(cube.shape[1:], np.float32)


class PhotonHyperspectralSection(PipelineSection):
    id = "photon_hyperspectral"
    title = "Photon-count hyperspectral analysis"
    requires = ("dataset", "results_dir", "experiment")
    produces = (
        "photon_numbers",
        "photon_histogram_cube",
        "photon_feature_paths",
        "photon_histogram_path",
    )

    def is_configured(self, config):
        return bool(config.get("enabled", False))

    def run(self, session, config):
        from matplotlib import colors

        dataset = session.artifacts.require("dataset")
        minimum = int(config.get("photon_min", 2))
        maximum = int(config.get("photon_max", 100))
        files_by_photon = defaultdict(list)
        for product in dataset.products:
            if product.kind != "final" or product.aggregate:
                continue
            number = exact_photon_number(product.path)
            if number is not None and minimum <= number <= maximum:
                files_by_photon[number].append(product.path)
        if not files_by_photon:
            raise FileNotFoundError(
                f"No per-run *_sumImage_NtoN TIFFs were found for photon numbers {minimum}–{maximum}."
            )

        results_dir = Path(session.artifacts.require("results_dir"))
        output_root = results_dir / "images" / "photon_hyperspectral"
        table_root = results_dir / "tables" / "photon_hyperspectral"
        averaged_root = output_root / "averaged_tiffs"
        averaged_root.mkdir(parents=True, exist_ok=True)
        table_root.mkdir(parents=True, exist_ok=True)
        photon_numbers = []
        images = []
        expected_shape = None
        for number, paths in sorted(files_by_photon.items()):
            channel_images = []
            for path in paths:
                image = np.asarray(tifffile.imread(path), dtype=np.float64)
                expected_shape = image.shape if expected_shape is None else expected_shape
                if image.shape != expected_shape:
                    raise ValueError(f"Shape mismatch for {path}: {image.shape} != {expected_shape}")
                channel_images.append(image)
            average = np.mean(channel_images, axis=0, dtype=np.float64).astype(np.float32)
            tifffile.imwrite(
                averaged_root / f"average_sumImage_{number}to{number}.tiff",
                average,
                metadata={"photon_number": number, "averaged_input_count": len(paths)},
            )
            photon_numbers.append(number)
            images.append(average)
        photon_numbers_array = np.asarray(photon_numbers, dtype=np.float32)
        cube = np.stack(images, axis=0).astype(np.float32)
        cube_path = output_root / "photon_histogram_cube_CYX.tiff"
        tifffile.imwrite(
            cube_path,
            cube,
            metadata={"axes": "CYX", "photon_numbers": photon_numbers},
        )

        eps = np.float32(1e-12)
        photon_axis = photon_numbers_array[:, None, None]
        total = cube.sum(axis=0, dtype=np.float32)
        weighted = (cube * photon_axis).sum(axis=0, dtype=np.float32)
        mean = np.divide(weighted, total, out=np.zeros_like(total), where=total > 0)
        dominant = photon_numbers_array[cube.argmax(axis=0)]
        dominant = np.where(total > 0, dominant, np.nan)
        maximum_fraction = np.divide(cube.max(axis=0), total, out=np.zeros_like(total), where=total > 0)
        probabilities = np.divide(cube, total[None] + eps, out=np.zeros_like(cube), where=total[None] > 0)
        entropy = -(probabilities * np.log2(probabilities + eps)).sum(axis=0, dtype=np.float32)
        entropy /= np.log2(max(len(photon_numbers), 2))

        thirds = config.get("rgb_bands") or (
            (minimum, min(5, maximum)),
            (max(minimum, 6), min(12, maximum)),
            (max(minimum, 13), maximum),
        )
        low = _band_sum(cube, photon_numbers_array, *thirds[0])
        middle = _band_sum(cube, photon_numbers_array, *thirds[1])
        high = _band_sum(cube, photon_numbers_array, *thirds[2])
        rgb = np.dstack([
            np.sqrt(robust_scale(high)),
            np.sqrt(robust_scale(middle)),
            np.sqrt(robust_scale(low)),
        ]).astype(np.float32)
        span = max(float(photon_numbers_array.max() - photon_numbers_array.min()), 1.0)
        hue = (1 - np.clip((mean - photon_numbers_array.min()) / span, 0, 1)) * 0.66
        hsv = np.dstack([
            hue,
            np.clip(np.sqrt(maximum_fraction), 0, 1),
            robust_scale(np.log1p(total), low=0.5, high=99.8),
        ])
        hsv_rgb = colors.hsv_to_rgb(hsv).astype(np.float32)

        feature_arrays = {
            "total_events.tiff": total,
            "mean_photon_number.tiff": mean,
            "dominant_photon_number.tiff": np.nan_to_num(dominant, nan=0),
            "photon_histogram_entropy.tiff": entropy,
            "hyperspectral_rgb_low_mid_high.tiff": (rgb * 65535).astype(np.uint16),
            "hyperspectral_hsv_summary_rgb.tiff": (hsv_rgb * 65535).astype(np.uint16),
        }
        feature_paths = {}
        for name, array in feature_arrays.items():
            path = output_root / name
            tifffile.imwrite(path, array)
            feature_paths[name] = path

        global_counts = cube.reshape(cube.shape[0], -1).sum(axis=1, dtype=np.float64)
        total_count = max(global_counts.sum(), 1.0)
        histogram_path = table_root / "global_photon_histogram.csv"
        with histogram_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["photon_number", "event_count", "fraction"])
            for number, count in zip(photon_numbers, global_counts):
                writer.writerow([number, float(count), float(count / total_count)])
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Built a {cube.shape} photon-count cube from {sum(map(len, files_by_photon.values()))} TIFFs.",
            {
                "photon_numbers": photon_numbers_array,
                "photon_histogram_cube": cube,
                "photon_feature_paths": feature_paths,
                "photon_histogram_path": histogram_path,
            },
        )

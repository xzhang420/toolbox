"""Portable, versioned processed-image datasets independent of acquisition paths."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import tifffile

from .image_io import IMAGE_SUFFIXES, natural_key, parse_timestamp, read_image
from .products import is_all_runs_sum

SCHEMA_VERSION = 1
MANIFEST = "processing/processed_dataset.json"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, allow_nan=False, default=str)
            handle.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contained_path(root, relative):
    path = (Path(root) / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Dataset path escapes its root: {relative}")
    return path


def publish_processed_dataset(session):
    """Write the exact loaded stack, after all selected crop/merge/filter steps.

    Existing exports are immutable unless their TIFF bytes are identical. Use a
    named alternative for changed inputs/settings. No source files are modified.
    """
    from . import __version__

    root = Path(session.artifacts.require("results_dir"))
    images = session.artifacts.require("image_stack")
    times = session.artifacts.require("image_times")
    paths = session.artifacts.require("image_paths")
    if not images or len(times) != len(images):
        raise ValueError("Cannot publish an empty or inconsistently timed image stack.")
    image_settings = dict(session.section_config("images"))
    pixel_size = image_settings.get("pixel_size_um")
    if pixel_size not in (None, ""):
        pixel_size = float(pixel_size) * int(image_settings.get("spatial_pixel_merge", 1))
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError("images.pixel_size_um must be positive when supplied.")
    else:
        pixel_size = None
    output = root / "images/processed_frames"
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=output.parent) as temporary:
        stage = Path(temporary) / "frames"
        stage.mkdir()
        for index, (array, stamp) in enumerate(zip(images, times)):
            filename = f"frame_{index:05d}.tiff"
            target = stage / filename
            # Existing processing timestamps are local-naive; convert on the
            # processing device, then store explicit UTC for other devices.
            timestamp = stamp.astimezone(timezone.utc).isoformat() if isinstance(stamp, datetime) else None
            tifffile.imwrite(target, np.asarray(array, dtype=np.float32), metadata=None,
                             description=f"AcquisitionTime:{timestamp}" if timestamp else "Time unavailable")
            frames.append({"path": f"images/processed_frames/{filename}",
                           "timestamp": timestamp, "shape": list(np.shape(array)),
                           "sha256": file_digest(target),
                           "source": Path(paths[index]).name if index < len(paths) else ""})
            bounds=session.artifacts.get('image_exposure_bounds')
            if bounds and bounds[index] is not None:
                start,end=bounds[index]
                if start>end: raise ValueError('Exposure starts after its end.')
                frames[-1].update(exposure_start_utc=start.astimezone(timezone.utc).isoformat(),
                                  exposure_end_utc=end.astimezone(timezone.utc).isoformat())
        if output.exists():
            existing = {path.name for path in output.iterdir() if path.is_file()}
            if existing != {Path(row["path"]).name for row in frames} or any(
                file_digest(output / Path(row["path"]).name) != row["sha256"] for row in frames
            ):
                raise ValueError("Published frames differ. Choose a new Analysis name; existing frames were preserved.")
        else:
            stage.rename(output)
    calibrated = bool(session.artifacts.get("nci_calibrated", False))
    payload = {
        "schema_version": SCHEMA_VERSION, "software_version": __version__,
        "experiment": session.artifacts.require("experiment"),
        "variant": session.section_config("project").get("variant", ""),
        "frames": frames, "frame_count": len(frames),
        "source_image_count": session.artifacts.get("source_tiff_count", len(frames)),
        "timestamp_convention": "UTC; null means unavailable, never filesystem mtime",
        "pixel_size_um": pixel_size, "unit": "um" if calibrated else image_settings.get("unit", ""),
        "nci_calibrated": calibrated,
        "processing_settings": {key: session.config.get(key, {}) for key in
                                ("layout", "corrections", "background", "images")},
        "config_snapshot": "processing/config_used.ini",
    }
    identity = {key: payload[key] for key in ("frames", "pixel_size_um", "unit", "nci_calibrated")}
    payload["dataset_id"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    previous = root / MANIFEST
    if previous.is_file() and json.loads(previous.read_text()).get("dataset_id") != payload["dataset_id"]:
        raise ValueError("Published dataset metadata differs. Use a new Analysis name; previous manifest was preserved.")
    average = session.artifacts.get("average_image_path")
    if average and Path(average).is_file() and Path(average).is_relative_to(root):
        payload["average"] = Path(average).relative_to(root).as_posix()
    atomic_json(root / MANIFEST, payload)
    session.artifacts.put("processed_dataset_path", root / MANIFEST, "images")
    return root / MANIFEST


def load_processed_images(source, *, pattern="", fits_hdu=None, timestamp_timezone="",
                          pixel_size_um=None, unit="", start_time="", interval_seconds=None):
    """Open a portable export or an explicitly selected TIFF/FITS collection.

    A plain directory is non-recursive by default, preventing accidental mixing
    of variants, averages, figures and intermediate frames. Use a deliberate
    pattern for other arrangements. Timing is optional for image-only work.
    """
    source = Path(source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    manifest = source if source.is_file() and source.name == "processed_dataset.json" else None
    if source.is_dir() and (source / MANIFEST).is_file():
        manifest = source / MANIFEST
    payload = {}
    if manifest:
        root = manifest.parent.parent
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported processed dataset schema; use a compatible pipeline version.")
        frames = payload.get("frames", [])
        paths = [contained_path(root, frame["path"]) for frame in frames]
        if len(set(paths)) != len(paths) or len(paths) != payload.get("frame_count"):
            raise ValueError("Processed manifest contains duplicate frames or an incorrect frame count.")
    else:
        root = source.parent if source.is_file() else source
        if source.is_file():
            paths = [source]
        else:
            if Path(pattern or ".").is_absolute() or ".." in Path(pattern or ".").parts:
                raise ValueError("Image pattern must stay inside the selected folder.")
            paths = sorted((path for path in root.glob(pattern or "*") if path.is_file()
                            and path.suffix.lower() in IMAGE_SUFFIXES and not is_all_runs_sum(path)), key=natural_key)
        frames = [{} for _ in paths]
    if not paths:
        raise ValueError("No images found. Select the experiment's frame folder or its processed_dataset.json.")
    images, times, units, records = [], [], set(), []
    for path, frame in zip(paths, frames):
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Image path escapes the selected folder: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Processed image missing: {path}")
        if is_all_runs_sum(path):
            raise ValueError(f"Preview sums are not frame inputs: {path}")
        digest = file_digest(path)
        if frame.get("sha256") and digest != frame["sha256"]:
            raise ValueError(f"Processed image changed since publication: {path}")
        array, info = read_image(path, fits_hdu=fits_hdu, timestamp_timezone=timestamp_timezone)
        if frame.get("shape") and list(array.shape) != frame["shape"]:
            raise ValueError(f"Image shape disagrees with manifest: {path}")
        images.append(array)
        times.append(parse_timestamp(frame.get("timestamp"), "UTC") if manifest else info["timestamp"])
        if info["unit"]:
            units.add(info["unit"])
        records.append({"path": path.relative_to(root).as_posix(), "sha256": digest})
    if len({array.shape for array in images}) != 1:
        raise ValueError("Selected images have different shapes. Select one compatible image stack.")
    if len(units) > 1:
        raise ValueError(f"Selected FITS images have inconsistent units: {sorted(units)}")
    if start_time:
        start = parse_timestamp(start_time, timestamp_timezone)
        if start is None or interval_seconds in (None, "") or float(interval_seconds) <= 0:
            raise ValueError("Explicit timing requires a timezone-qualified start and positive interval_seconds.")
        from datetime import timedelta
        times = [start + timedelta(seconds=i * float(interval_seconds)) for i in range(len(images))]
    size = pixel_size_um if pixel_size_um not in (None, "") else payload.get("pixel_size_um")
    if size is not None and (not np.isfinite(float(size)) or float(size) <= 0):
        raise ValueError("Pixel size must be positive, or blank when unknown.")
    known_times = all(time is not None for time in times)
    if known_times and times != sorted(times):
        if manifest:
            raise ValueError("Published frame timestamps are not chronological. Check the source manifest.")
        # Resumed legacy runs can have names that do not sort by acquisition
        # time. Keep every array, timestamp and provenance record aligned.
        order = sorted(range(len(times)), key=times.__getitem__)
        paths, images, times, records = ([values[i] for i in order]
                                        for values in (paths, images, times, records))
    identity = payload.get("dataset_id") or hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    exposures = []
    for frame in frames:
        start = parse_timestamp(frame.get('exposure_start_utc'), 'UTC')
        end = parse_timestamp(frame.get('exposure_end_utc'), 'UTC')
        if (start is None) != (end is None) or (start is not None and start>end):
            raise ValueError('Exposure metadata requires ordered start/end UTC bounds.')
        exposures.append((start,end) if start is not None else None)
    return {"root": root, "paths": paths, "images": images, "times": times, "exposure_bounds": exposures,
            "pixel_size_um": float(size) if size is not None else None,
            "unit": str(unit or payload.get("unit") or next(iter(units), "")),
            "nci_calibrated": bool(payload.get("nci_calibrated", False)),
            "experiment": payload.get("experiment") or root.name,
            "dataset_id": identity, "manifest": manifest,
            "timing_available": known_times, "records": records}

"""Validation and resumable sequential execution for LumaCam experiment batches."""

from __future__ import annotations

import configparser
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from .proposal_paths import analysis_root_for_proposal, proposal_root_for_layout

from pathlib import Path
import re
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Iterable, Mapping

import numpy as np
import tifffile

from . import backends
from .background import (
    background_frame_count,
    background_source as resolve_background_source,
    effective_background_settings,
    shared_background_path,
)
from .config import save_config
from .layout import discover_dataset, find_layout_root, list_experiments, resolve_experiment_name
from .presets import create_session
from .prepared_inputs import uses_prepared_ob, validate_prepared_ob, prepared_ob_integral
from .result_layout import average_filename, named_directory, result_directory, shared_directory, snapshot_path
from .section import SectionState
from .modes import LEGACY_MODE_ALIASES
from .white_spots import effective_image_settings, filter_settings


BATCH_SECTION_LABELS = {
    "project": "Discover project data",
    "corrections": "Corrections using shared reference/OB",
    "background": "Background correction",
    "images": "Load processed images",
    "average": "Write average TIFF",
    "image_measurements": "Measure image ROI",
    "nci_plating_metrics": "Calculate NCI plating metrics",
    "photon_hyperspectral": "Build photon-count cube",
    "electrochemistry": "Load electrochemistry",
    "synchronization": "Synchronize images and electrochemistry",
    "event_detection": "Detect electrochemistry events",
    "selection": "Select representative images",
    "image_plot": "Save representative image panel",
    "individual_image_plots": "Save every image as an individual figure",
    "combined_plot": "Save combined electrochemistry/image plot",
    "nci_event_outputs": "Save established NCI event outputs",
    "image_video": "Export image video",
    "combined_video": "Export synchronized video",
}

BATCH_SECTION_ORDER = tuple(BATCH_SECTION_LABELS)
DEFAULT_BATCH_SECTIONS = (
    "corrections",
    "images",
    "average",
    "individual_image_plots",
)

_PREREQUISITES = {
    "corrections": ("project",),
    "background": ("corrections",),
    "images": ("project",),
    "average": ("images",),
    "image_measurements": ("images",),
    "nci_plating_metrics": ("images",),
    "photon_hyperspectral": ("project",),
    "electrochemistry": ("project",),
    "synchronization": ("images", "image_measurements", "electrochemistry"),
    "event_detection": ("images", "electrochemistry"),
    "selection": ("images",),
    "image_plot": ("selection",),
    "individual_image_plots": ("images",),
    "combined_plot": ("images", "image_measurements", "electrochemistry"),
    "nci_event_outputs": ("images", "image_measurements", "selection"),
    "image_video": ("images",),
    "combined_video": ("images", "electrochemistry"),
}


@dataclass(frozen=True)
class BatchValidationRow:
    experiment: str
    valid: bool
    message: str
    run_count: int
    image_count: int
    result_dir: Path
    input_signature: str
    setup_like: bool = False


@dataclass(frozen=True)
class BatchValidationReport:
    proposal_root: Path
    layout_root: Path
    sections: tuple[str, ...]
    rows: tuple[BatchValidationRow, ...]
    common_errors: tuple[str, ...] = ()

    @property
    def valid(self):
        return not self.common_errors and bool(self.rows) and all(row.valid for row in self.rows)


@dataclass(frozen=True)
class BatchJobResult:
    experiment: str
    status: str
    message: str
    failed_section: str
    started_at: str
    finished_at: str
    config_path: Path
    results_dir: Path
    recipe_hash: str


@dataclass(frozen=True)
class BatchRunSummary:
    batch_name: str
    batch_dir: Path
    recipe_path: Path
    status_path: Path
    log_path: Path
    results: tuple[BatchJobResult, ...]
    report_error: str = ""

    @property
    def failed_count(self):
        return sum(result.status == "failed" for result in self.results)


@dataclass(frozen=True)
class SharedPreparationResult:
    proposal_root: Path
    ob_experiment: str
    background_experiment: str
    reference_source_path: Path
    prepared_reference_path: Path
    reference_tiff_count: int
    reference_beam_integral_per_minute: float | None
    background_source_path: Path | None
    corrected_background_dir: Path | None
    prepared_background_path: Path | None
    background_source_tiff_count: int
    background_tiff_count: int
    metadata_path: Path
    reused_background_cache: bool


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(payload):
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_batch_name(name, variant=""):
    value = str(name or "").strip() or f"batch_{variant or 'images'}"
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    if not value:
        raise ValueError("Batch name is blank after removing unsafe characters.")
    return value


def is_setup_experiment(name):
    """Flag likely setup data for GUI filtering; explicit selection remains allowed."""

    value = str(name).casefold()
    return (
        value in {"migration", "migrations", "summed"}
        or value.startswith(("test", "text", "focus"))
        or "_ob" in value
        or "_bg" in value
        or "openbeam" in value
        or "open_beam" in value
    )


def normalize_batch_sections(sections: Iterable[str]):
    requested = {str(section).strip() for section in sections if str(section).strip()}
    unknown = requested.difference(BATCH_SECTION_ORDER)
    if unknown:
        raise ValueError(f"Unknown batch section(s): {sorted(unknown)}")
    if not requested:
        raise ValueError("Select at least one batch processing section.")
    requested.add("project")
    pending = list(requested)
    while pending:
        section = pending.pop()
        for prerequisite in _PREREQUISITES.get(section, ()):
            if prerequisite not in requested:
                requested.add(prerequisite)
                pending.append(prerequisite)
    return tuple(section for section in BATCH_SECTION_ORDER if section in requested)


def _proposal_paths(data_path, config=None):
    if (config or {}).get("layout", {}).get("images_pattern"):
        root = Path(data_path).expanduser().resolve()
        return root, root
    layout_root = find_layout_root(data_path)
    if layout_root is None:
        raise ValueError(f"No supported LumaCam proposal layout found around: {data_path}")
    proposal_root = (
        proposal_root_for_layout(layout_root)
    )
    return proposal_root.resolve(), layout_root.resolve()


def discover_batch_experiments(data_path, *, hide_setup=True, filter_text="", config=None):
    """Return experiment names for an explicit GUI selection list."""

    _, layout_root = _proposal_paths(data_path, config)
    query = str(filter_text or "").strip().casefold()
    names = []
    from .inspection import mapped_experiments
    available = mapped_experiments(config) if (config or {}).get("layout", {}).get("images_pattern") else list_experiments(layout_root)
    for name in available:
        if hide_setup and is_setup_experiment(name):
            continue
        if query and query not in name.casefold():
            continue
        names.append(name)
    return names


def _parse_kinds(value):
    if isinstance(value, str):
        return tuple(part.strip() for part in value.replace(";", ",").split(",") if part.strip())
    return tuple(value or ("final",))


def _path_signature(paths):
    records = []
    for path in sorted({Path(path).resolve(strict=False) for path in paths}):
        try:
            stat = path.stat()
            records.append((str(path), stat.st_size, stat.st_mtime_ns))
        except OSError:
            records.append((str(path), None, None))
    return _digest(records)


def _experiment_override(config, experiment, overrides):
    merged = deepcopy(config)
    for section, values in (overrides or {}).get(experiment, {}).items():
        merged.setdefault(section, {}).update(values)
    return merged


def _common_validation_errors(config, sections, variant, *, check_echem=True):
    errors = []
    if "images" in sections and config.get("images", {}).get("white_spot_enabled", False):
        try:
            filter_settings(config["images"])
        except ValueError as exc:
            errors.append(str(exc))
    corrections = config.get("corrections", {})
    if "corrections" in sections:
        prepared_mode = uses_prepared_ob(corrections)
        paths = [("Beam history", "beamhistory_path")]
        if prepared_mode:
            prepared = str(corrections.get("prepared_reference_path", "") or "").strip()
            if prepared:
                try:
                    validate_prepared_ob(prepared)
                except (OSError, ValueError, tifffile.TiffFileError) as exc:
                    errors.append(str(exc))
        else:
            paths.extend((("Reference/OB", "reference_dir"),
                          ("Reference beam history", "reference_beamhistory_path")))
        for label, key in paths:
            value = str(corrections.get(key, "") or "").strip()
            if value and not Path(value).expanduser().exists():
                errors.append(f"{label} path does not exist: {value}")
        reference = "" if prepared_mode else str(corrections.get("reference_dir", "") or "").strip()
        if reference:
            try:
                candidates = _reference_source_files(reference,
                    str(corrections.get("file_filter") or variant),
                    str(corrections.get("reference_final_subfolder", "final")))
            except (OSError, ValueError) as exc:
                candidates = []
                errors.append(f"Reference/OB selection failed: {exc}")
            if not candidates:
                errors.append(
                    f"Reference/OB contains no individual TIFF products for variant {variant!r}: {reference}"
                )
    if check_echem and "electrochemistry" in sections:
        value = str(config.get("electrochemistry", {}).get("path", "") or "").strip()
        if not value:
            errors.append("Electrochemistry is selected but its path is blank.")
        elif not Path(value).expanduser().exists():
            errors.append(f"Electrochemistry path does not exist: {value}")
    if "background" in sections:
        try:
            background_config = config.get("background", {})
            background = shared_background_path(background_config)
            if background and not Path(background).expanduser().exists():
                errors.append(f"Background path does not exist: {background}")
            if not background or not Path(background).expanduser().is_file():
                background_frame_count(background_config)
        except ValueError as exc:
            errors.append(str(exc))
    return tuple(errors)


def validate_batch(
    data_path,
    experiments,
    *,
    preset="nci_images_only",
    config=None,
    sections=DEFAULT_BATCH_SECTIONS,
    experiment_overrides=None,
):
    """Validate selected experiments and shared inputs without creating outputs."""

    proposal_root, layout_root = _proposal_paths(data_path, config)
    normalized_sections = normalize_batch_sections(sections)
    base_config = deepcopy(config or {})
    project = base_config.setdefault("project", {})
    project["data_path"] = str(data_path)
    variant = str(project.get("variant", "") or "").strip()
    from .inspection import mapped_experiments, discover_for_config
    available = mapped_experiments(base_config) if base_config.get("layout", {}).get("images_pattern") else list_experiments(layout_root)
    requested = []
    for experiment in experiments:
        resolved = resolve_experiment_name(experiment, available)
        if resolved not in requested:
            requested.append(resolved)
    if not requested:
        raise ValueError("Select at least one experiment for batch processing.")

    common_errors = _common_validation_errors(
        base_config,
        normalized_sections,
        variant,
        check_echem=not bool(experiment_overrides),
    )
    reference_shape = None
    reference_settings = base_config.get("corrections", {})
    reference_value = str(reference_settings.get(
        "prepared_reference_path" if uses_prepared_ob(reference_settings) else "reference_dir", ""
    ) or "").strip()
    if "corrections" in normalized_sections and reference_value and not common_errors:
        reference_settings = base_config.get("corrections", {})
        reference_files = _reference_source_files(reference_value,
            str(reference_settings.get("file_filter") or variant),
            str(reference_settings.get("reference_final_subfolder", "final")))
        if reference_files:
            try:
                with tifffile.TiffFile(reference_files[0]) as tif:
                    reference_shape = tuple(tif.pages[0].shape)
                if len(reference_shape) != 2:
                    common_errors = common_errors + (
                        f"Reference/OB TIFF must be 2D; got shape {reference_shape}.",
                    )
            except Exception as exc:
                common_errors = common_errors + (
                    f"Cannot read reference/OB TIFF {reference_files[0]}: {exc}",
                )
    rows = []
    for experiment in requested:
        row_config = _experiment_override(base_config, experiment, experiment_overrides)
        row_project = row_config.get("project", {})
        row_variant = str(row_project.get("variant", variant) or "").strip()
        try:
            if "electrochemistry" in normalized_sections:
                echem_value = str(
                    row_config.get("electrochemistry", {}).get("path", "") or ""
                ).strip()
                if not echem_value:
                    raise ValueError("Electrochemistry is selected but its path is blank.")
                if not Path(echem_value).expanduser().exists():
                    raise ValueError(f"Electrochemistry path does not exist: {echem_value}")
            row_config.setdefault("project", {})["data_path"] = str(data_path)
            dataset = discover_for_config(row_config, experiment=experiment, variant=row_variant or None)
            if base_config.get("workflow", {}).get("role") == "processing":
                from .processing import PROCESSING_SECTIONS
                invalid = set(normalized_sections) - PROCESSING_SECTIONS
                if invalid:
                    raise ValueError(f"Analysis/visualization steps do not belong in a processing batch: {sorted(invalid)}")
            products = dataset.image_products(
                variant=row_variant or None,
                kinds=_parse_kinds(row_project.get("product_kinds", "final")),
                include_aggregates=False,
            )
            if not products:
                raise ValueError(f"No matching per-run TIFF products for variant {row_variant!r}.")
            if row_config.get("workflow", {}).get("role") == "processing":
                if len({product.variant for product in products}) > 1:
                    raise ValueError("Select one image variant before processing.")
                from .processing import available_beamhistory
                if "corrections" in normalized_sections and available_beamhistory(row_config.get("corrections", {}).get("beamhistory_path")):
                    from .inspection import processing_layout
                    runs = processing_layout(dataset, products)["runs"]
                    missing = [run["name"] for run in runs
                               if not backends.nci_data().get_run_exposure_windows(run)]
                    if missing:
                        raise ValueError(f"Beam correction requires timing metadata; missing: {missing}")
            first = products[0].path
            with tifffile.TiffFile(first) as tif:
                shape = tif.pages[0].shape
            if len(shape) != 2:
                raise ValueError(f"Expected 2D TIFF products; first image has shape {shape}.")
            if reference_shape is not None and tuple(shape) != tuple(reference_shape):
                raise ValueError(
                    f"Image shape {tuple(shape)} does not match shared reference/OB "
                    f"shape {tuple(reference_shape)}."
                )
            metadata_files = []
            for run in dataset.runs:
                if run.metadata_dir and run.metadata_dir.is_dir():
                    metadata_files.extend(run.metadata_dir.glob("*.json"))
            signature = _path_signature([product.path for product in products] + metadata_files)
            rows.append(
                BatchValidationRow(
                    experiment=dataset.experiment,
                    valid=True,
                    message=f"{len(dataset.runs)} run record(s), {len(products)} TIFF image(s)",
                    run_count=len(dataset.runs),
                    image_count=len(products),
                    result_dir=result_directory(analysis_root_for_proposal(proposal_root), dataset.experiment,
                                                row_variant, row_project.get("analysis_name", "")),
                    input_signature=signature,
                    setup_like=is_setup_experiment(dataset.experiment),
                )
            )
        except Exception as exc:
            rows.append(
                BatchValidationRow(
                    experiment=experiment,
                    valid=False,
                    message=f"{type(exc).__name__}: {exc}",
                    run_count=0,
                    image_count=0,
                    result_dir=result_directory(analysis_root_for_proposal(proposal_root), experiment,
                                                row_variant, row_project.get("analysis_name", "")),
                    input_signature="",
                    setup_like=is_setup_experiment(experiment),
                )
            )
    return BatchValidationReport(
        proposal_root=proposal_root,
        layout_root=layout_root,
        sections=normalized_sections,
        rows=tuple(rows),
        common_errors=common_errors,
    )


def _batch_dir(proposal_root, batch_name, analysis_name=""):
    return named_directory(analysis_root_for_proposal(proposal_root) / "batches" / batch_name, analysis_name)


def _write_recipe(path, preset, config, report, batch_name, recipe_hash):
    parser = configparser.ConfigParser()
    parser["batch"] = {
        "name": batch_name,
        "experiments": json.dumps([row.experiment for row in report.rows]),
        "sections": json.dumps(list(report.sections)),
        "recipe_hash": recipe_hash,
        "updated_at": _now(),
    }
    parser["pipeline"] = {"preset": preset}
    for section, values in config.items():
        encoded = {}
        for key, value in values.items():
            if isinstance(value, (dict, list, tuple)):
                encoded[key] = json.dumps(_jsonable(value))
            elif value is None:
                encoded[key] = ""
            else:
                encoded[key] = str(value)
        parser[section] = encoded
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        parser.write(handle)


_STATUS_FIELDS = (
    "experiment",
    "status",
    "message",
    "failed_section",
    "started_at",
    "finished_at",
    "config_path",
    "results_dir",
    "recipe_hash",
)


def _write_status(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_STATUS_FIELDS)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["config_path"] = str(row["config_path"])
            row["results_dir"] = str(row["results_dir"])
            writer.writerow(row)


def save_batch_plan(report, *, preset, config, batch_name=""):
    """Persist a validation-only recipe and initial status table."""

    saved_config = deepcopy(config or {})
    saved_project = saved_config.setdefault("project", {})
    if not str(saved_project.get("data_path", "") or "").strip():
        saved_project["data_path"] = str(report.proposal_root)
    saved_project["result_root"] = ""
    variant = str(saved_project.get("variant", "") or "")
    safe_name = _safe_batch_name(batch_name, variant)
    target = _batch_dir(report.proposal_root, safe_name, saved_project.get("analysis_name", ""))
    # Dashboard visibility is recorded but cannot change scientific results.
    recipe_hash = _digest(
        {"preset": preset,
         "config": {key: value for key, value in saved_config.items() if key != "gui"},
         "sections": report.sections}
    )
    recipe = target / "batch_recipe.ini"
    _write_recipe(recipe, preset, saved_config, report, safe_name, recipe_hash)
    (target / "experiment_list.txt").write_text(
        "\n".join(row.experiment for row in report.rows) + "\n", encoding="utf-8"
    )
    initial = tuple(
        BatchJobResult(
            experiment=row.experiment,
            status="pending" if row.valid and not report.common_errors else "invalid",
            message=row.message if row.valid else row.message,
            failed_section="",
            started_at="",
            finished_at="",
            config_path=snapshot_path(row.result_dir),
            results_dir=row.result_dir,
            recipe_hash=recipe_hash,
        )
        for row in report.rows
    )
    _write_status(target / "batch_status.csv", initial)
    return target


def _reference_source_files(reference_path, variant, final_subfolder="final"):
    """Use the same individual inputs as reference loading, including exclusions."""
    from .products import is_all_runs_sum
    reference = Path(reference_path).expanduser().resolve()
    if reference.is_file():
        if is_all_runs_sum(reference):
            raise ValueError(f"All-runs sum is a preview, not an individual OB input: {reference}")
        return [reference]
    module = backends.nci_data()
    layout = module.resolve_nci_experiment(str(reference), final_subfolder=final_subfolder)
    return sorted({Path(path) for run in layout["runs"]
        for path in module._run_image_paths(run, file_filter=variant or None)})


def _input_files(path_value):
    """Return deterministic non-hidden files represented by one path."""

    if path_value is None or not str(path_value).strip():
        return []
    path = Path(path_value).expanduser()
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return [
        candidate
        for candidate in sorted(path.rglob("*"))
        if candidate.is_file()
        and not any(part.startswith(".") for part in candidate.relative_to(path).parts)
    ]


def _reuse_shared(image_path, metadata_path, settings):
    if not image_path.exists() and not metadata_path.exists():
        return False
    if image_path.is_file() and metadata_path.is_file():
        saved = json.loads(metadata_path.read_text(encoding="utf-8"))
        if all(saved.get(key) == value for key, value in settings.items()):
            return True
    raise ValueError(
        f"Shared corrections at {image_path.parent} use different inputs/settings or are incomplete. "
        "Choose a new Preparation name (Reference set) in Prepare OB/BG files; existing files are preserved."
    )


def _prepare_shared_reference(proposal_root, config, log_path):
    from .processing import prepare_reference_timing
    prepare_reference_timing(config)
    corrections = config.setdefault("corrections", {})
    if uses_prepared_ob(corrections):
        value = str(corrections.get("prepared_reference_path", "") or "").strip()
        return validate_prepared_ob(value) if value else None
    reference_value = str(corrections.get("reference_dir", "") or "").strip()
    if not reference_value:
        corrections.pop("prepared_reference_path", None)
        return None
    reference = Path(reference_value).expanduser().resolve()
    variant = str(corrections.get("file_filter") or config.get("project", {}).get("variant", ""))
    source_files = _reference_source_files(reference, variant,
        str(corrections.get("reference_final_subfolder", "final")))
    if not source_files:
        raise ValueError(f"No individual reference TIFF inputs for variant {variant!r}: {reference}")
    # Exclusions/timing can change without any TIFF changing. Include the
    # sidecars in the cache key so excluded reference counts cannot be reused.
    reference_metadata = []
    if reference.is_dir():
        layout = backends.nci_data().resolve_nci_experiment(
            str(reference),
            final_subfolder=str(corrections.get("reference_final_subfolder", "final")),
        )
        for run in layout["runs"]:
            if run.get("metadata_dir"):
                reference_metadata.extend(Path(run["metadata_dir"]).glob("*.json"))
        if layout.get("layout_type") == "v2":
            for run_name in layout.get("excluded_runs", []):
                reference_metadata.extend(
                    (Path(layout["layout_root"]) / "metadata" / layout["experiment"] / run_name).glob("*.json")
                )
    signature = _path_signature(source_files)
    settings = {
        "input_policy": "individual-runs-only-v1",
        "reference": str(reference),
        "variant": variant,
        "reference_bin_y": int(corrections.get("reference_bin_y", 2)),
        "reference_bin_x": int(corrections.get("reference_bin_x", 2)),
        "reference_final_subfolder": str(corrections.get("reference_final_subfolder", "final")),
        "reference_beamhistory_path": str(
            corrections.get("reference_beamhistory_path", "") or ""
        ),
        "reference_beam_integral_per_minute": corrections.get(
            "reference_beam_integral_per_minute"
        ),
        "tpx_time": float(corrections.get("tpx_time", 2)),
        "source_signature": signature,
        "metadata_signature": _path_signature(reference_metadata),
    }
    settings["reference_beamhistory_signature"] = _path_signature(
        _input_files(settings["reference_beamhistory_path"])
    )
    cache_dir = shared_directory(proposal_root, config)
    cache_path = cache_dir / "ob.tiff"
    metadata_path = cache_dir / "ob_preparation.json"
    if _reuse_shared(cache_path, metadata_path, settings):
        corrections["prepared_reference_path"] = str(cache_path)
        return cache_path

    module = backends.nci_data()
    if reference.is_file():
        series = [np.asarray(tifffile.imread(reference), dtype=np.float32)]
    else:
        series = module.load_reference_image_series(
            str(reference),
            final_subfolder=settings["reference_final_subfolder"],
            dtype=np.float32,
            file_filter=variant or None,
            beamhistory_path=settings["reference_beamhistory_path"] or None,
            reference_beam_integral_per_minute=settings[
                "reference_beam_integral_per_minute"
            ],
            tpx_time=settings["tpx_time"],
        )
    if not series:
        raise ValueError(f"No reference TIFF images found for variant {variant!r}: {reference}")
    processed = module.process_reference_images(
        series,
        bin_y=settings["reference_bin_y"],
        bin_x=settings["reference_bin_x"],
        compute_dtype=np.float32,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(cache_path, np.asarray(processed, dtype=np.float32))
    metadata_path.write_text(
        json.dumps(
            {**settings, "source_count": len(series), "created_at": _now()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} prepared shared reference: {cache_path}\n")
    corrections["prepared_reference_path"] = str(cache_path)
    return cache_path


def _prepare_reference_integral(config, log_path):
    from .processing import prepare_reference_timing
    prepare_reference_timing(config)
    corrections = config.setdefault("corrections", {})
    if corrections.get("reference_beam_integral_per_minute") not in (None, ""):
        return corrections["reference_beam_integral_per_minute"]
    if uses_prepared_ob(corrections):
        value = str(corrections.get("prepared_reference_path", "") or "").strip()
        integral = prepared_ob_integral(value) if value else None
        corrections["reference_beam_integral_per_minute"] = integral
        return integral
    reference = str(corrections.get("reference_dir", "") or "").strip()
    reference_beam = str(corrections.get("reference_beamhistory_path", "") or "").strip()
    experiment_beam = str(corrections.get("beamhistory_path", "") or "").strip()
    if not (reference and reference_beam and experiment_beam):
        return None
    value = backends.nci_data().compute_reference_beam_integral_per_minute(
        reference,
        reference_beam,
        float(corrections.get("tpx_time", 2)),
    )
    corrections["reference_beam_integral_per_minute"] = float(value)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} shared reference beam integral per minute: {value}\n")
    return value


def prepare_shared_ob_background(
    data_path,
    ob_experiment,
    background_experiment="",
    *,
    preset="exsitu",
    config=None,
    prepare_background=None,
):
    """Prepare shared OB, optionally also a separate shared BG.

    Explicit first_frames mode needs only OB. Older callers without a mode
    retain the original OB-and-BG preparation behavior.
    """

    values = config if config is not None else {}
    if prepare_background is None:
        background_config = values.get("background", {})
        raw_mode = str(background_config.get("source_mode", "auto") or "auto").strip().lower()
        prepare_background = raw_mode == "auto" or resolve_background_source(background_config) == "shared"
    background_n_images = background_frame_count(values.get("background", {})) if prepare_background else None
    proposal_root, layout_root = _proposal_paths(data_path)
    available = list_experiments(layout_root)
    ob_name = resolve_experiment_name(str(ob_experiment).strip(), available)
    background_name = (resolve_experiment_name(str(background_experiment).strip(), available)
                       if prepare_background else "")
    if prepare_background and ob_name == background_name:
        raise ValueError("OB and background experiments must be different.")

    project = values.setdefault("project", {})
    variant = str(project.get("variant", "") or "").strip()
    if not variant:
        raise ValueError("Select a TIFF variant before preparing shared OB and background.")
    project["data_path"] = str(Path(data_path).expanduser())

    ob_dataset = discover_dataset(data_path, experiment=ob_name, variant=variant)
    background_dataset = discover_dataset(
        data_path, experiment=background_name, variant=variant
    ) if prepare_background else None
    ob_products = ob_dataset.image_products(
        variant=variant, kinds=("final",), include_aggregates=False
    )
    background_products = background_dataset.image_products(
        variant=variant, kinds=("final",), include_aggregates=False
    ) if prepare_background else []
    if not ob_products:
        raise ValueError(f"No per-run OB TIFFs found for {ob_name!r}, variant {variant!r}.")
    if prepare_background and not background_products:
        raise ValueError(
            f"No per-run background TIFFs found for {background_name!r}, variant {variant!r}."
        )

    reference_source = layout_root / "final" / ob_name
    background_source = layout_root / "final" / background_name
    if not reference_source.is_dir() or (prepare_background and not background_source.is_dir()):
        raise ValueError(
            "Shared OB/background preparation requires a LumaCam proposal with "
            "final/<experiment> folders."
        )

    shared_root = shared_directory(proposal_root, values)
    shared_root.mkdir(parents=True, exist_ok=True)
    log_path = shared_root / "preparation.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{_now()} preparing OB={ob_name}, background={background_name}, "
            f"variant={variant}\n"
        )

    corrections = values.setdefault("corrections", {})
    corrections["enabled"] = True
    corrections["reference_input_mode"] = "source"
    corrections["reference_dir"] = str(reference_source)
    corrections["file_filter"] = variant
    # The shared routine always uses the same history for OB, BG, and samples.
    # Keep the backend key for compatibility with older single-experiment configs.
    corrections["reference_beamhistory_path"] = corrections.get("beamhistory_path", "")
    corrections.pop("prepared_reference_path", None)
    corrections.pop("reference_beam_integral_per_minute", None)
    reference_integral = _prepare_reference_integral(values, log_path)
    prepared_reference = _prepare_shared_reference(proposal_root, values, log_path)
    if prepared_reference is None:
        raise ValueError("Shared OB preparation did not produce a reference image.")
    reference_metadata_path = prepared_reference.parent / "ob_preparation.json"
    try:
        reference_metadata = json.loads(reference_metadata_path.read_text(encoding="utf-8"))
        reference_count = int(reference_metadata.get("source_count", len(ob_products)))
    except (OSError, ValueError, json.JSONDecodeError):
        reference_count = len(ob_products)

    if not prepare_background:
        values.setdefault("shared_setup", {}).update({
            "ob_experiment": ob_name,
            "reference_source_path": str(reference_source),
            "prepared_reference_path": str(prepared_reference),
            "reference_tiff_count": reference_count,
        })
        return SharedPreparationResult(
            proposal_root=proposal_root, ob_experiment=ob_name, background_experiment="",
            reference_source_path=reference_source, prepared_reference_path=prepared_reference,
            reference_tiff_count=reference_count,
            reference_beam_integral_per_minute=(float(reference_integral) if reference_integral is not None else None),
            background_source_path=None, corrected_background_dir=None, prepared_background_path=None,
            background_source_tiff_count=0, background_tiff_count=0,
            metadata_path=reference_metadata_path, reused_background_cache=False,
        )
    background_metadata_files = []
    for run in background_dataset.runs:
        if run.metadata_dir and run.metadata_dir.is_dir():
            background_metadata_files.extend(run.metadata_dir.glob("*.json"))
    background_settings = {
        "background_experiment": background_name,
        "background_source_signature": _path_signature(
            [product.path for product in background_products]
        ),
        "background_metadata_signature": _path_signature(background_metadata_files),
        "beam_history_signature": _path_signature(
            _input_files(corrections.get("beamhistory_path", ""))
        ),
        "prepared_reference_path": str(prepared_reference),
        "reference_beam_integral_per_minute": reference_integral,
        "background_n_images": background_n_images,
        "variant": variant,
    }
    cache_dir = shared_root
    prepared_background = cache_dir / "background.tiff"
    metadata_path = cache_dir / "preparation.json"
    reused_background_cache = _reuse_shared(prepared_background, metadata_path, background_settings)

    corrected_background_dir = result_directory(cache_dir / "processing" / "work", background_name, variant) / "processing" / "intermediate" / "beam_ob_corrected"
    background_count = 0
    if reused_background_cache:
        try:
            background_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            corrected_background_dir = Path(background_metadata["corrected_background_dir"])
            background_count = int(background_metadata["background_tiff_count"])
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            reused_background_cache = False

    if not reused_background_cache:
        background_config = deepcopy(values)
        background_config.setdefault("background", {})["enabled"] = True
        background_config.setdefault("project", {}).update(
            {
                "data_path": str(Path(data_path).expanduser()),
                "experiment": background_name,
                "variant": variant,
                "result_root": str(cache_dir / "processing" / "work"),
                "analysis_name": "",
                "include_aggregates": False,
            }
        )
        background_config.setdefault("corrections", {}).update(
            {
                "enabled": True,
                "reference_dir": str(reference_source),
                "prepared_reference_path": str(prepared_reference),
                "reference_beam_integral_per_minute": reference_integral,
                "file_filter": variant,
            }
        )
        session = create_session(preset, background_config)
        project_outcome = session.run("project")
        if project_outcome.state != SectionState.COMPLETED:
            raise ValueError(f"Background discovery failed: {project_outcome.message}")
        correction_outcome = session.run("corrections")
        if correction_outcome.state != SectionState.COMPLETED:
            raise ValueError(f"Background correction failed: {correction_outcome.message}")
        corrected_background_dir = Path(
            session.artifacts.require("processed_image_dir")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        calculation_metadata_path = cache_dir / "background_calculation.json"
        background_image = backends.nci_data().compute_background_image(
            str(corrected_background_dir),
            n_images=background_n_images,
            compute_dtype=np.float32,
            metadata_path=str(calculation_metadata_path),
        )
        tifffile.imwrite(prepared_background, np.asarray(background_image, dtype=np.float32))
        try:
            calculation_metadata = json.loads(
                calculation_metadata_path.read_text(encoding="utf-8")
            )
            background_count = int(
                calculation_metadata["sections"]["background_processing"]
                ["background_image"]["extra"]["valid_images_used"]
            )
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            background_count = min(background_n_images, len(background_products))
        metadata_path.write_text(
            json.dumps(
                {
                    **background_settings,
                    "ob_experiment": ob_name,
                    "reference_source_path": str(reference_source),
                    "reference_tiff_count": reference_count,
                    "background_source_path": str(background_source),
                    "background_source_tiff_count": len(background_products),
                    "background_tiff_count": background_count,
                    "corrected_background_dir": str(corrected_background_dir),
                    "prepared_background_path": str(prepared_background),
                    "created_at": _now(),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"{_now()} prepared shared background from {background_count} TIFF(s): "
                f"{prepared_background}\n"
            )

    background = values.setdefault("background", {})
    background.update(
        {
            "enabled": True,
            "source_mode": "shared",
            "background_path": str(prepared_background),
            "n_images": background_n_images,
        }
    )
    shared_setup = values.setdefault("shared_setup", {})
    shared_setup.update(
        {
            "ob_experiment": ob_name,
            "background_experiment": background_name,
            "reference_source_path": str(reference_source),
            "prepared_reference_path": str(prepared_reference),
            "reference_tiff_count": reference_count,
            "prepared_background_path": str(prepared_background),
            "background_source_tiff_count": len(background_products),
            "background_tiff_count": background_count,
            "metadata_path": str(metadata_path),
        }
    )
    return SharedPreparationResult(
        proposal_root=proposal_root,
        ob_experiment=ob_name,
        background_experiment=background_name,
        reference_source_path=reference_source,
        prepared_reference_path=prepared_reference,
        reference_tiff_count=reference_count,
        reference_beam_integral_per_minute=(
            float(reference_integral) if reference_integral is not None else None
        ),
        background_source_path=background_source,
        corrected_background_dir=corrected_background_dir,
        prepared_background_path=prepared_background,
        background_source_tiff_count=len(background_products),
        background_tiff_count=background_count,
        metadata_path=metadata_path,
        reused_background_cache=reused_background_cache,
    )


def _sentinel_matches(path, job_hash):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("job_hash") == job_hash and payload.get("status") == "completed"


def _relocated_sentinel_matches(path, config_path, input_signature, sections, preset, config):
    """A verified move or retired GUI mode can change hashes without changing science."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (payload.get("status") != "completed"
                or payload.get("input_signature") != input_signature
                or tuple(payload.get("sections", ())) != tuple(sections)):
            return False
        from .config import load_config
        old_preset, saved = load_config(config_path)
        renamed_mode = LEGACY_MODE_ALIASES.get(old_preset) == preset
        if (payload.get("relocated_config_snapshot") != str(config_path) and not renamed_mode):
            return False
        if old_preset != preset and not renamed_mode:
            return False
        previous = create_session(old_preset, saved).config
        current = create_session(preset, config).config
        if renamed_mode:
            previous["average"].setdefault("collect_in_proposal", False)
            current["average"].setdefault("collect_in_proposal", False)
        previous["background"] = effective_background_settings(previous["background"])
        current["background"] = effective_background_settings(current["background"])
        previous["images"] = effective_image_settings(previous["images"])
        current["images"] = effective_image_settings(current["images"])
        for key in ("variant", "product_kinds", "include_aggregates"):
            if previous["project"].get(key) != current["project"].get(key):
                return False
        for section in sections:
            if section == "project":
                continue
            ignored = {"enabled", "verbose", "recompute"}
            old = {key: value for key, value in previous.get(section, {}).items() if key not in ignored}
            new = {key: value for key, value in current.get(section, {}).items() if key not in ignored}
            if old != new:
                return False
        if "average" in sections:
            project = current["project"]
            filename = current["average"].get("filename") or average_filename(
                project["experiment"], project.get("variant", ""), project.get("analysis_name", ""))
            if not (config_path.parent.parent / "images" / filename).is_file():
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def run_batch(
    data_path,
    experiments,
    *,
    preset="nci_images_only",
    config=None,
    sections=DEFAULT_BATCH_SECTIONS,
    batch_name="",
    overwrite=False,
    experiment_overrides=None,
    progress: Callable[[str, Mapping], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
):
    """Run validated experiment jobs sequentially and continue after failures."""

    base_config = deepcopy(config or {})
    base_config.setdefault("project", {})["data_path"] = str(Path(data_path).expanduser())
    base_config["project"]["result_root"] = ""
    if not uses_prepared_ob(base_config.setdefault("corrections", {})):
        base_config["corrections"].pop("prepared_reference_path", None)
    report = validate_batch(
        data_path,
        experiments,
        preset=preset,
        config=base_config,
        sections=sections,
        experiment_overrides=experiment_overrides,
    )
    if not report.valid:
        details = list(report.common_errors) + [
            f"{row.experiment}: {row.message}" for row in report.rows if not row.valid
        ]
        raise ValueError("Batch validation failed: " + " | ".join(details))

    variant = str(base_config["project"].get("variant", "") or "")
    safe_name = _safe_batch_name(batch_name, variant)
    batch_dir = _batch_dir(report.proposal_root, safe_name, base_config["project"].get("analysis_name", ""))
    batch_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = batch_dir / "batch_recipe.ini"
    status_path = batch_dir / "batch_status.csv"
    log_path = batch_dir / "batch.log"
    (batch_dir / "experiment_list.txt").write_text(
        "\n".join(row.experiment for row in report.rows) + "\n", encoding="utf-8"
    )
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} starting/resuming batch {safe_name}\n")

    if "corrections" in report.sections:
        _prepare_reference_integral(base_config, log_path)
        _prepare_shared_reference(report.proposal_root, base_config, log_path)
    # Toggling the advanced view must not invalidate completed batch jobs.
    hash_config = {key: value for key, value in base_config.items() if key != "gui"}
    if "background" in hash_config:
        hash_config["background"] = effective_background_settings(hash_config["background"])
    if "images" in hash_config:
        hash_config["images"] = effective_image_settings(hash_config["images"])
    recipe_hash = _digest(
        {"preset": preset,
         "config": hash_config,
         "sections": report.sections}
    )
    _write_recipe(recipe_path, preset, base_config, report, safe_name, recipe_hash)

    result_map = {
        row.experiment: BatchJobResult(
            row.experiment,
            "pending",
            "Waiting to run.",
            "",
            "",
            "",
            snapshot_path(row.result_dir),
            row.result_dir,
            recipe_hash,
        )
        for row in report.rows
    }

    def ordered_results():
        return tuple(result_map[row.experiment] for row in report.rows)

    _write_status(status_path, ordered_results())
    for row in report.rows:
        if stop_requested is not None and stop_requested():
            break
        experiment_config = _experiment_override(base_config, row.experiment, experiment_overrides)
        project = experiment_config.setdefault("project", {})
        project.update(
            {
                "data_path": str(Path(data_path).expanduser()),
                "experiment": row.experiment,
                "result_root": "",
            }
        )
        for section in report.sections:
            if section != "project":
                experiment_config.setdefault(section, {})["enabled"] = True
        override_hash = _digest((experiment_overrides or {}).get(row.experiment, {}))
        job_hash = _digest(
            {
                "recipe_hash": recipe_hash,
                "input_signature": row.input_signature,
                "override_hash": override_hash,
            }
        )
        config_path = snapshot_path(row.result_dir)
        sentinel = row.result_dir / "processing" / f"batch_{safe_name}.json"
        if not overwrite and (
            _sentinel_matches(sentinel, job_hash)
            or _relocated_sentinel_matches(sentinel, config_path, row.input_signature,
                                           report.sections, preset, experiment_config)
        ):
            result = BatchJobResult(
                row.experiment,
                "skipped",
                "Already completed with the same recipe and inputs.",
                "",
                "",
                _now(),
                config_path,
                row.result_dir,
                recipe_hash,
            )
            result_map[row.experiment] = result
            _write_status(status_path, ordered_results())
            if progress:
                progress("experiment_skipped", asdict(result))
            continue

        started = _now()
        result_map[row.experiment] = BatchJobResult(
            row.experiment,
            "running",
            "Processing selected sections.",
            "",
            started,
            "",
            config_path,
            row.result_dir,
            recipe_hash,
        )
        _write_status(status_path, ordered_results())
        if progress:
            progress("experiment_started", {"experiment": row.experiment})
        failed_section = ""
        message = "Completed selected sections."
        status = "failed"
        try:
            session = create_session(preset, experiment_config)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{started} [{row.experiment}] started\n")
                with redirect_stdout(handle), redirect_stderr(handle):
                    for section in report.sections:
                        outcome = session.run(section)
                        print(
                            f"[{row.experiment}] {section}: "
                            f"{outcome.state.value}: {outcome.message}"
                        )
                        if outcome.state != SectionState.COMPLETED:
                            failed_section = section
                            message = outcome.message
                            break
            finished = _now()
            if not failed_section:
                save_config(config_path, preset, session.config)
                status = "completed"
                sentinel.parent.mkdir(parents=True, exist_ok=True)
                sentinel.write_text(
                    json.dumps(
                        {
                            "batch_name": safe_name,
                            "experiment": row.experiment,
                            "status": status,
                            "job_hash": job_hash,
                            "recipe_hash": recipe_hash,
                            "input_signature": row.input_signature,
                            "sections": list(report.sections),
                            "completed_at": finished,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
        except Exception as exc:
            finished = _now()
            status = "failed"
            failed_section = failed_section or "batch_runner"
            message = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"{finished} [{row.experiment}] failed: {message}\n")
        result = BatchJobResult(
            row.experiment,
            status,
            message,
            failed_section,
            started,
            finished,
            config_path,
            row.result_dir,
            recipe_hash,
        )
        result_map[row.experiment] = result
        _write_status(status_path, ordered_results())
        if progress:
            progress("experiment_finished", asdict(result))

    with log_path.open("a", encoding="utf-8") as handle:
        completed_count = sum(
            result.status in {"completed", "failed", "skipped"}
            for result in ordered_results()
        )
        handle.write(
            f"{_now()} batch {safe_name} stopped with {completed_count} finished result(s)\n"
        )
    report_error = ""
    if (base_config.get("workflow", {}).get("role") == "processing"
            and base_config.get("reports", {}).get("enabled", True)):
        from .handover import MANIFEST
        from .reports import create_report
        published = [result.results_dir for result in ordered_results()
                     if (result.results_dir / MANIFEST).is_file()]
        if published:
            try:
                create_report(published, output_dir=batch_dir, title=f"LumaCam batch: {safe_name}",
                    notes=["Batch job status (a report is not evidence that every requested section completed):"] +
                          [f"{result.experiment}: {result.status}; {result.message}" for result in ordered_results()])
            except Exception as exc:
                report_error = f"Batch overview failed; processing results retained: {type(exc).__name__}: {exc}"
                with log_path.open("a", encoding="utf-8") as handle:
                    handle.write(report_error + "\n")
                if progress:
                    progress("report_failed", {"message": report_error})
    return BatchRunSummary(
        batch_name=safe_name,
        batch_dir=batch_dir,
        recipe_path=recipe_path,
        status_path=status_path,
        log_path=log_path,
        results=ordered_results(),
        report_error=report_error,
    )


__all__ = [
    "BATCH_SECTION_LABELS",
    "BATCH_SECTION_ORDER",
    "DEFAULT_BATCH_SECTIONS",
    "BatchJobResult",
    "BatchRunSummary",
    "BatchValidationReport",
    "BatchValidationRow",
    "SharedPreparationResult",
    "discover_batch_experiments",
    "is_setup_experiment",
    "normalize_batch_sections",
    "prepare_shared_ob_background",
    "run_batch",
    "save_batch_plan",
    "validate_batch",
]

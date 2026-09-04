"""Validation and resumable sequential execution for LumaCam experiment batches."""

from __future__ import annotations

import configparser
import csv
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from contextlib import redirect_stderr, redirect_stdout
from typing import Callable, Iterable, Mapping

import numpy as np
import tifffile

from . import backends
from .config import save_config
from .layout import discover_dataset, find_layout_root, list_experiments, resolve_experiment_name
from .presets import create_session
from .section import SectionState


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

    @property
    def failed_count(self):
        return sum(result.status == "failed" for result in self.results)


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


def _proposal_paths(data_path):
    layout_root = find_layout_root(data_path)
    if layout_root is None:
        raise ValueError(f"No supported LumaCam proposal layout found around: {data_path}")
    proposal_root = (
        layout_root.parent.parent if layout_root.parent.name == "data" else layout_root.parent
    )
    return proposal_root.resolve(), layout_root.resolve()


def discover_batch_experiments(data_path, *, hide_setup=True, filter_text=""):
    """Return experiment names for an explicit GUI selection list."""

    _, layout_root = _proposal_paths(data_path)
    query = str(filter_text or "").strip().casefold()
    names = []
    for name in list_experiments(layout_root):
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
    corrections = config.get("corrections", {})
    if "corrections" in sections:
        for label, key in (
            ("Reference/OB", "reference_dir"),
            ("Beam history", "beamhistory_path"),
            ("Reference beam history", "reference_beamhistory_path"),
        ):
            value = str(corrections.get(key, "") or "").strip()
            if value and not Path(value).expanduser().exists():
                errors.append(f"{label} path does not exist: {value}")
        reference = str(corrections.get("reference_dir", "") or "").strip()
        if reference:
            reference_path = Path(reference).expanduser()
            if reference_path.is_file():
                candidates = [reference_path]
            else:
                candidates = [
                    path
                    for path in reference_path.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".tif", ".tiff"}
                    and (not variant or variant in str(path))
                ]
            if not candidates:
                errors.append(
                    f"Reference/OB contains no TIFF products for variant {variant!r}: {reference}"
                )
    if check_echem and "electrochemistry" in sections:
        value = str(config.get("electrochemistry", {}).get("path", "") or "").strip()
        if not value:
            errors.append("Electrochemistry is selected but its path is blank.")
        elif not Path(value).expanduser().exists():
            errors.append(f"Electrochemistry path does not exist: {value}")
    background = str(config.get("background", {}).get("background_path", "") or "").strip()
    if "background" in sections and background and not Path(background).expanduser().exists():
        errors.append(f"Background path does not exist: {background}")
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

    proposal_root, layout_root = _proposal_paths(data_path)
    normalized_sections = normalize_batch_sections(sections)
    base_config = deepcopy(config or {})
    project = base_config.setdefault("project", {})
    variant = str(project.get("variant", "") or "").strip()
    available = list_experiments(layout_root)
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
    reference_value = str(
        base_config.get("corrections", {}).get("reference_dir", "") or ""
    ).strip()
    if "corrections" in normalized_sections and reference_value and not common_errors:
        reference_files = _reference_source_files(reference_value, variant)
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
            dataset = discover_dataset(data_path, experiment=experiment, variant=row_variant or None)
            products = dataset.image_products(
                variant=row_variant or None,
                kinds=_parse_kinds(row_project.get("product_kinds", "final")),
                include_aggregates=bool(row_project.get("include_aggregates", False)),
            )
            if not products:
                raise ValueError(f"No matching per-run TIFF products for variant {row_variant!r}.")
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
                    result_dir=proposal_root / "analysis" / dataset.experiment,
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
                    result_dir=proposal_root / "analysis" / experiment,
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


def _batch_dir(proposal_root, batch_name):
    return proposal_root / "analysis" / "proposal_summary" / "batches" / batch_name


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
    target = _batch_dir(report.proposal_root, safe_name)
    recipe_hash = _digest(
        {"preset": preset, "config": saved_config, "sections": report.sections}
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
            config_path=row.result_dir / "lumacam_pipeline.ini",
            results_dir=row.result_dir,
            recipe_hash=recipe_hash,
        )
        for row in report.rows
    )
    _write_status(target / "batch_status.csv", initial)
    return target


def _reference_source_files(reference_path, variant):
    reference = Path(reference_path).expanduser().resolve()
    if reference.is_file():
        return [reference]
    return [
        path
        for path in reference.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".tif", ".tiff"}
        and (not variant or variant in str(path))
    ]


def _prepare_shared_reference(proposal_root, config, log_path):
    corrections = config.setdefault("corrections", {})
    reference_value = str(corrections.get("reference_dir", "") or "").strip()
    if not reference_value:
        corrections.pop("prepared_reference_path", None)
        return None
    reference = Path(reference_value).expanduser().resolve()
    variant = str(corrections.get("file_filter") or config.get("project", {}).get("variant", ""))
    source_files = _reference_source_files(reference, variant)
    signature = _path_signature(source_files)
    settings = {
        "reference": str(reference),
        "variant": variant,
        "reference_bin_y": int(corrections.get("reference_bin_y", 2)),
        "reference_bin_x": int(corrections.get("reference_bin_x", 2)),
        "reference_final_subfolder": str(corrections.get("reference_final_subfolder", "final")),
        "source_signature": signature,
    }
    cache_key = _digest(settings)[:16]
    cache_dir = proposal_root / "analysis" / "proposal_summary" / "reference_cache" / cache_key
    safe_variant = re.sub(r"[^A-Za-z0-9._-]+", "_", variant).strip("._-") or "default"
    cache_path = cache_dir / f"processed_reference_{safe_variant}.tiff"
    metadata_path = cache_dir / "reference_cache.json"
    if cache_path.is_file() and metadata_path.is_file():
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
    corrections = config.setdefault("corrections", {})
    if corrections.get("reference_beam_integral_per_minute") not in (None, ""):
        return corrections["reference_beam_integral_per_minute"]
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


def _sentinel_matches(path, job_hash):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("job_hash") == job_hash and payload.get("status") == "completed"


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
    base_config.setdefault("corrections", {}).pop("prepared_reference_path", None)
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
    batch_dir = _batch_dir(report.proposal_root, safe_name)
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
        _prepare_shared_reference(report.proposal_root, base_config, log_path)
        _prepare_reference_integral(base_config, log_path)
    recipe_hash = _digest(
        {"preset": preset, "config": base_config, "sections": report.sections}
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
            row.result_dir / "lumacam_pipeline.ini",
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
        config_path = row.result_dir / "lumacam_pipeline.ini"
        sentinel = row.result_dir / "processing" / f"batch_{safe_name}.json"
        if not overwrite and _sentinel_matches(sentinel, job_hash):
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
            save_config(config_path, preset, experiment_config)
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
    return BatchRunSummary(
        batch_name=safe_name,
        batch_dir=batch_dir,
        recipe_path=recipe_path,
        status_path=status_path,
        log_path=log_path,
        results=ordered_results(),
    )


__all__ = [
    "BATCH_SECTION_LABELS",
    "BATCH_SECTION_ORDER",
    "DEFAULT_BATCH_SECTIONS",
    "BatchJobResult",
    "BatchRunSummary",
    "BatchValidationReport",
    "BatchValidationRow",
    "discover_batch_experiments",
    "is_setup_experiment",
    "normalize_batch_sections",
    "run_batch",
    "save_batch_plan",
    "validate_batch",
]

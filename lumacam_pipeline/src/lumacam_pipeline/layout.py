"""Discover LumaCam datasets without requiring ``metadata/layout.json``.

The canonical layout is recognized structurally as ``data/experiments`` with
the acquisition trees produced by ``lumacam_measurementcontrol``.  Marker-era
and early ``data/experiment`` proposals remain readable, but this module never
creates or requires a layout marker.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .models import Dataset, ImageProduct, RunRecord
from .products import final_sum_variant, is_all_runs_sum
from .proposal_paths import proposal_root_for_layout, raw_root_for_layout, split_proposal_root


TIFF_SUFFIXES = {".tif", ".tiff"}
DATA_TREES = {"tpx3Files", "final", "derived", "metadata", "logs", ".work"}
LAYOUT_CONTAINER_NAMES = {"experiments", "experiment"}
LAYOUT_MARKER = Path("metadata") / "layout.json"


class LayoutError(ValueError):
    """Raised when a dataset path is missing or structurally ambiguous."""


def _resolved(path: str | Path) -> Path:
    return Path(path).expanduser().resolve(strict=False)


def _is_structural_layout_root(path: Path) -> bool:
    if path.name not in LAYOUT_CONTAINER_NAMES or path.parent.name != "data":
        return False
    present = {name for name in DATA_TREES if (path / name).is_dir()}
    return "metadata" in present and bool(present.intersection({"tpx3Files", "final", "derived"}))


def is_layout_root(path: str | Path) -> bool:
    """Return whether *path* is a canonical, early-v2, or marker-era root."""

    root = _resolved(path)
    return _is_structural_layout_root(root) or (root / LAYOUT_MARKER).is_file()


def _candidate_layout_roots(path: Path) -> list[Path]:
    candidates: list[Path] = []
    current = path.parent if path.is_file() else path
    for ancestor in (current, *current.parents):
        if ancestor not in candidates:
            candidates.append(ancestor)
    for base in list(candidates):
        for relative in (
            Path("reduced/data/experiment"),
            Path("reduced/data/experiments"),
            Path("data/experiments"),
            Path("data/experiment"),
            Path("experiments"),
            Path("experiment"),
        ):
            candidate = base / relative
            if candidate not in candidates:
                candidates.append(candidate)
    return candidates


def find_layout_root(path: str | Path) -> Path | None:
    """Find the nearest supported proposal data root around *path*."""

    resolved = _resolved(path)
    structural = [candidate for candidate in _candidate_layout_roots(resolved) if _is_structural_layout_root(candidate)]
    if structural:
        root = structural[0]
        outer = proposal_root_for_layout(root)
        named = [outer / relative for relative in (
            "data/experiment", "data/experiments", "reduced/data/experiment", "reduced/data/experiments")]
        if sum(path.exists() or path.is_symlink() for path in named) > 1:
            raise LayoutError(f"Multiple layout roots found below: {outer}")
        if split_proposal_root(root) is not None and any((root / name).exists() for name in ("tpx3Files", "raw")):
            raise LayoutError(f"Mixed raw/reduced and nested raw layout: {outer}")
        if split_proposal_root(root) is None and (outer / "raw").exists():
            raise LayoutError(f"Multiple raw layout roots found below: {outer}")
        return structural[0]
    marked = [candidate for candidate in _candidate_layout_roots(resolved) if (candidate / LAYOUT_MARKER).is_file()]
    return marked[0] if marked else None


def list_experiments(layout_root: str | Path) -> list[str]:
    root = _resolved(layout_root)
    names: set[str] = set()
    for tree_name in ("metadata", "tpx3Files", "final", "derived"):
        tree = raw_root_for_layout(root) if tree_name == "tpx3Files" else root / tree_name
        if not tree.is_dir():
            continue
        names.update(
            child.name
            for child in tree.iterdir()
            if child.is_dir()
            and child.name not in {"summed", "migrations"}
            and not child.name.startswith(".")
        )
    return sorted(names)


def _experiment_number(value: str, *, short_form: bool) -> int | None:
    pattern = r"exp[\s_-]*0*(\d+)" if short_form else r"exp[\s_-]*0*(\d+)(?:$|[_-])"
    match = re.fullmatch(pattern, value, flags=re.IGNORECASE) if short_form else re.match(
        pattern, value, flags=re.IGNORECASE
    )
    return int(match.group(1)) if match else None


def resolve_experiment_name(requested: str, available: Iterable[str]) -> str:
    """Resolve an exact name, short ``expNNN`` id, or unique name prefix."""

    query = str(requested).strip()
    names = sorted({str(name) for name in available})
    if not query:
        raise LayoutError("Experiment selection is blank.")

    if query in names:
        return query

    case_matches = [name for name in names if name.casefold() == query.casefold()]
    if len(case_matches) == 1:
        return case_matches[0]

    requested_number = _experiment_number(query, short_form=True)
    if requested_number is not None:
        matches = [
            name
            for name in names
            if _experiment_number(name, short_form=False) == requested_number
        ]
        if not matches:
            matches = [name for name in names if name.casefold().startswith(query.casefold())]
    else:
        matches = [name for name in names if name.casefold().startswith(query.casefold())]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise LayoutError(f"Experiment {query!r} is ambiguous; matches: {matches}")

    short_ids = []
    for name in names:
        number = _experiment_number(name, short_form=False)
        label = f"exp{number:03d}" if number is not None else name
        if label not in short_ids:
            short_ids.append(label)
    raise LayoutError(
        f"Experiment {query!r} was not found. Available experiment IDs: {short_ids}"
    )


def _selection_from_path(layout_root: Path, requested: Path) -> tuple[str | None, str | None, str | None]:
    if split_proposal_root(layout_root) is not None:
        try:
            raw_parts = requested.relative_to(raw_root_for_layout(layout_root)).parts
        except ValueError:
            raw_parts = ()
        if raw_parts:
            if raw_parts[0].startswith(".") or len(raw_parts) > 2:
                raise LayoutError("Select a published raw experiment or run")
            return raw_parts[0], raw_parts[1] if len(raw_parts) == 2 else None, None
    try:
        relative = requested.relative_to(layout_root)
    except ValueError:
        return None, None, None
    parts = relative.parts
    if len(parts) < 2 or parts[0] not in DATA_TREES:
        return None, None, None
    experiment = parts[1]
    selected_run = None
    selected_variant = None
    if len(parts) >= 3 and parts[2] != "summed":
        if parts[0] == "final":
            selected_variant = parts[2]
        elif parts[0] in {"tpx3Files", "metadata", "derived", "logs", ".work"}:
            selected_run = parts[2]
    return experiment, selected_run, selected_variant


def _read_json_object(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _aggregation_sources(metadata_root: Path) -> tuple[set[Path], dict[Path, tuple[Path, ...]]]:
    outputs: set[Path] = set()
    sources: dict[Path, tuple[Path, ...]] = {}
    summed = metadata_root / "summed"
    if not summed.is_dir():
        return outputs, sources
    for manifest_path in sorted(summed.glob("aggregation_manifest*.json")):
        manifest = _read_json_object(manifest_path)
        for raw_output in manifest.get("outputs", []):
            try:
                output = _resolved(raw_output)
            except TypeError:
                continue
            outputs.add(output)
        for raw_output, raw_inputs in manifest.get("inputs", {}).items():
            try:
                output = _resolved(raw_output)
                inputs = tuple(_resolved(value) for value in raw_inputs)
            except (TypeError, ValueError):
                continue
            outputs.add(output)
            sources[output] = inputs
    return outputs, sources


def _tiffs(directory: Path, recursive: bool = False) -> list[Path]:
    if not directory.is_dir():
        return []
    iterator: Iterable[Path] = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES)


def _match_run(filename: str, run_names: Iterable[str]) -> str | None:
    for run_name in sorted(run_names, key=len, reverse=True):
        if filename in {f"{run_name}.tif", f"{run_name}.tiff"} or filename.startswith(f"{run_name}_"):
            return run_name
    return None


def _run_names(layout_root: Path, experiment: str) -> list[str]:
    names: set[str] = set()
    manifest = _read_json_object(layout_root / "metadata" / experiment / "experiment.json")
    names.update(str(name) for name in manifest.get("expected_runs", []) if str(name).strip())
    for tree_name in ("tpx3Files", "metadata", "derived"):
        root = (raw_root_for_layout(layout_root) if tree_name == "tpx3Files" else layout_root / tree_name) / experiment
        if root.is_dir():
            names.update(
                child.name
                for child in root.iterdir()
                if child.is_dir() and child.name != "summed" and not child.name.startswith(".")
            )
    return sorted(names)


def _canonical_products(layout_root: Path, experiment: str, run_names: list[str]) -> list[ImageProduct]:
    products: list[ImageProduct] = []
    metadata_root = layout_root / "metadata" / experiment
    aggregate_outputs, aggregate_sources = _aggregation_sources(metadata_root)
    final_root = layout_root / "final" / experiment

    if final_root.is_dir():
        for child in sorted(final_root.iterdir()):
            if child.is_file() and child.suffix.lower() in TIFF_SUFFIXES:
                files = [child]
                variant = "default"
            elif child.is_dir() and not child.name.startswith("."):
                files = _tiffs(child)
                variant = "default" if child.name in run_names else child.name
            else:
                continue
            for path in files:
                aggregate = path.resolve(strict=False) in aggregate_outputs or is_all_runs_sum(path)
                run = None if aggregate else _match_run(path.name, run_names)
                if not aggregate and child.is_dir() and child.name in run_names:
                    run = child.name
                products.append(
                    ImageProduct(
                        path=path,
                        experiment=experiment,
                        run=run,
                        kind="final",
                        variant=(final_sum_variant(path) or variant) if aggregate else variant,
                        aggregate=aggregate,
                        metadata_dir=(metadata_root / run) if run else (metadata_root / "summed"),
                        source_paths=aggregate_sources.get(path.resolve(strict=False), ()),
                    )
                )

    derived_root = layout_root / "derived" / experiment
    if derived_root.is_dir():
        for path in _tiffs(derived_root, recursive=True):
            relative = path.relative_to(derived_root)
            top = relative.parts[0] if len(relative.parts) > 1 else None
            aggregate = top == "summed" or is_all_runs_sum(path) or path.resolve(strict=False) in aggregate_outputs
            run = None if aggregate else (top if top in run_names else _match_run(path.name, run_names))
            lower = path.name.lower()
            kind = "derived_photon" if "photon" in lower else "derived_pixel" if "pixel" in lower else "derived"
            summed_variant = final_sum_variant(path)
            if aggregate and summed_variant:
                kind = "final"
            products.append(
                ImageProduct(
                    path=path,
                    experiment=experiment,
                    run=run,
                    kind=kind,
                    variant=summed_variant or "default",
                    aggregate=aggregate,
                    metadata_dir=(metadata_root / run) if run else (metadata_root / "summed"),
                    source_paths=aggregate_sources.get(path.resolve(strict=False), ()),
                )
            )
    return products


def _discover_canonical(
    requested: Path,
    layout_root: Path,
    experiment: str | None,
    variant: str | None,
) -> Dataset:
    selected_experiment, selected_run, selected_variant = _selection_from_path(layout_root, requested)
    experiment = experiment or selected_experiment
    available = list_experiments(layout_root)
    if experiment is None:
        if len(available) != 1:
            raise LayoutError(
                "Select an experiment because the proposal contains "
                f"{len(available)} experiments: {available}"
            )
        experiment = available[0]
    else:
        experiment = resolve_experiment_name(experiment, available)
    run_names = _run_names(layout_root, experiment)
    runs = tuple(
        RunRecord(
            name=name,
            tpx3_dir=raw_root_for_layout(layout_root) / experiment / name,
            metadata_dir=layout_root / "metadata" / experiment / name,
            derived_dir=layout_root / "derived" / experiment / name,
        )
        for name in run_names
    )
    experiment_manifest = _read_json_object(layout_root / "metadata" / experiment / "experiment.json")
    excluded_runs = set()
    for run in runs:
        if (run.metadata_dir / "processing_exclusions.json").is_file():
            from . import backends

            if backends.nci_data().load_run_processing_exclusion(run.metadata_dir) is not None:
                excluded_runs.add(run.name)
    products = _canonical_products(layout_root, experiment, run_names)
    return Dataset(
        input_path=requested,
        layout_type="v2",
        experiment=experiment,
        proposal_root=proposal_root_for_layout(layout_root),
        layout_root=layout_root,
        runs=runs,
        products=tuple(product for product in products if product.run not in excluded_runs),
        selected_variant=variant or selected_variant,
        selected_run=selected_run,
        metadata={"experiment_manifest": experiment_manifest, "excluded_runs": sorted(excluded_runs)},
    )


def _discover_legacy_or_flat(requested: Path, variant: str | None) -> Dataset:
    if not requested.is_dir():
        raise LayoutError(f"Dataset path is not a directory: {requested}")
    flat_files = _tiffs(requested)
    if flat_files:
        products = tuple(
            ImageProduct(
                path=path,
                experiment=requested.name,
                run=None,
                kind="final",
                variant=final_sum_variant(path) or variant or "default",
                aggregate=is_all_runs_sum(path),
                metadata_dir=(requested / "metadata") if (requested / "metadata").is_dir() else None,
            )
            for path in flat_files
        )
        return Dataset(requested, "flat", requested.name, None, None, products=products, selected_variant=variant)

    child_runs = [
        child
        for child in sorted(requested.iterdir())
        if child.is_dir() and ((child / "final").is_dir() or (child / "tpx3Files").is_dir())
    ]
    if (requested / "tpx3Files").is_dir() or ((requested / "final").is_dir() and not child_runs):
        child_runs = [requested]
        experiment = requested.parent.name
        experiment_root = requested.parent
    else:
        experiment = requested.name
        experiment_root = requested
    runs: list[RunRecord] = []
    products: list[ImageProduct] = []
    for run_path in child_runs:
        metadata_dir = run_path / "metadata"
        runs.append(
            RunRecord(
                run_path.name,
                run_path / "tpx3Files",
                metadata_dir if metadata_dir.is_dir() else None,
                None,
            )
        )
        for path in _tiffs(run_path / "final"):
            products.append(
                ImageProduct(
                    path,
                    experiment,
                    None if is_all_runs_sum(path) else run_path.name,
                    "final",
                    final_sum_variant(path) or variant or "default",
                    is_all_runs_sum(path),
                    metadata_dir if metadata_dir.is_dir() else None,
                )
            )
    # Experiment-level legacy sums can live in the historical final folder
    # or in the new derived/summed folder. They are preview products only.
    known_paths = {product.path for product in products}
    for directory in (experiment_root / "final", experiment_root / "derived" / "summed"):
        for path in _tiffs(directory):
            if path not in known_paths and is_all_runs_sum(path):
                products.append(ImageProduct(path, experiment, None, "final",
                    final_sum_variant(path) or variant or "default", True,
                    experiment_root / "metadata" / "summed"))
    if not products and not runs:
        raise LayoutError(f"No supported LumaCam, legacy-run, or flat TIFF data found in {requested}")
    return Dataset(
        requested,
        "legacy",
        experiment,
        None,
        None,
        tuple(runs),
        tuple(products),
        selected_variant=variant,
        metadata={"experiment_root": str(experiment_root)},
    )


def discover_dataset(
    path: str | Path,
    *,
    experiment: str | None = None,
    variant: str | None = None,
) -> Dataset:
    """Normalize any supported input path into one :class:`Dataset`."""

    requested = _resolved(path)
    if not requested.exists():
        raise LayoutError(f"Dataset path does not exist: {requested}")
    layout_root = find_layout_root(requested)
    if layout_root is not None:
        return _discover_canonical(requested, layout_root, experiment, variant)
    return _discover_legacy_or_flat(requested, variant)

"""Read-only layout inspection and explicit mappings for unfamiliar proposals."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import re

from .layout import discover_dataset, find_layout_root, list_experiments, resolve_experiment_name, LayoutError
from .models import Dataset, ImageProduct, RunRecord
from .products import is_all_runs_sum
from .proposal_paths import analysis_root_for_proposal, proposal_root_for_layout, reduced_root_for_proposal


def _mapped_records(config):
    project, mapping = config.get("project", {}), config.get("layout", {})
    root = Path(project.get("data_path", "")).expanduser().resolve()
    pattern = str(mapping.get("images_pattern", ""))
    if not pattern or Path(pattern).is_absolute() or ".." in Path(pattern).parts:
        raise ValueError("layout.images_pattern must be a relative pattern inside project.data_path.")
    if "{experiment}" not in pattern:
        raise ValueError("layout.images_pattern must include {experiment}; use a flat input folder without a mapping otherwise.")
    regex = re.escape(pattern)
    glob = pattern
    for name in ("experiment", "variant", "run"):
        if pattern.count("{" + name + "}") > 1:
            raise ValueError(f"Repeated mapping field: {name}")
        regex = regex.replace(re.escape("{" + name + "}"), f"(?P<{name}>[^/]+)")
        glob = glob.replace("{" + name + "}", "*")
    if "{" in glob or "}" in glob:
        raise ValueError("Supported mapping fields: {experiment}, {variant}, {run}.")
    regex = regex.replace(r"\*\*/", "(?:.*/)?").replace(r"\*", "[^/]*").replace(r"\?", "[^/]")
    matcher = re.compile("^" + regex + "$")
    records = []
    for path in sorted(root.glob(glob)):
        if not path.is_file() or path.suffix.lower() not in {".tif", ".tiff"} or is_all_runs_sum(path):
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Mapped image escapes proposal: {path}")
        match = matcher.match(path.relative_to(root).as_posix())
        if match:
            fields = match.groupdict()
            fields["variant"] = fields.get("variant") or str(project.get("variant") or "default")
            if not fields.get("run") and mapping.get("run_regex"):
                run_match = re.search(str(mapping["run_regex"]), path.stem)
                if run_match is None or "run" not in run_match.groupdict():
                    raise ValueError(f"layout.run_regex must match a named (?P<run>...) group: {path.name}")
                fields["run"] = run_match.group("run")
            records.append((path, fields))
    if not records:
        raise ValueError(f"No per-frame TIFFs match layout.images_pattern under {root}")
    return root, records


def mapped_experiments(config):
    return sorted({row[1]["experiment"] for row in _mapped_records(config)[1]})


def discover_for_config(config, *, experiment=None, variant=None):
    project, mapping = config.get("project", {}), config.get("layout", {})
    experiment = experiment if experiment is not None else project.get("experiment")
    variant = variant if variant is not None else project.get("variant")
    if not mapping.get("images_pattern"):
        return discover_dataset(project["data_path"], experiment=experiment or None, variant=variant or None)
    from . import backends
    root, records = _mapped_records(config)
    names = sorted({fields["experiment"] for _, fields in records})
    if not experiment and len(names) != 1:
        raise LayoutError(f"Select an experiment: {names}")
    name = resolve_experiment_name(experiment, names) if experiment else names[0]
    chosen = [(path, fields) for path, fields in records if fields["experiment"] == name]
    variants = sorted({fields["variant"] for _, fields in chosen})
    if not variant:
        if len(variants) != 1:
            raise LayoutError(f"Select one image variant: {variants}")
        variant = variants[0]
    products, runs, excluded = [], {}, []
    metadata_pattern = str(mapping.get("metadata_pattern", ""))
    if Path(metadata_pattern).is_absolute() or ".." in Path(metadata_pattern).parts:
        raise ValueError("layout.metadata_pattern must be relative to project.data_path.")
    for path, fields in chosen:
        if fields["variant"] != variant:
            continue
        run_name = fields.get("run") or path.stem
        metadata = None
        if metadata_pattern:
            metadata = (root / metadata_pattern.format(experiment=name, variant=variant, run=run_name)).resolve()
            if not metadata.is_relative_to(root):
                raise ValueError("Mapped metadata escapes proposal root.")
            if not metadata.is_dir():
                metadata = None
        if metadata and backends.nci_data().load_run_processing_exclusion(str(metadata)) is not None:
            excluded.append(run_name)
            continue
        runs[run_name] = RunRecord(run_name, metadata_dir=metadata)
        products.append(ImageProduct(path, name, run_name, variant=variant, metadata_dir=metadata))
    if not products:
        raise ValueError(f"No usable mapped images for {name}, variant {variant}")
    return Dataset(root, "mapped", name, root, None, tuple(runs.values()), tuple(products),
                   selected_variant=variant, metadata={"mapping": mapping, "excluded_runs": sorted(set(excluded))})


def inspect_layout(data_path, *, config=None, experiment=None, variant=None):
    """Return machine-readable evidence and suggestions; never write config/data."""
    values = deepcopy(config or {})
    values.setdefault("project", {})["data_path"] = str(Path(data_path).expanduser().resolve())
    root = find_layout_root(data_path) if not values.get("layout", {}).get("images_pattern") else None
    mapped = bool(values.get("layout", {}).get("images_pattern"))
    if mapped:
        proposal = Path(values["project"]["data_path"])
        names = mapped_experiments(values)
        kind = "explicit_mapping"
    elif root:
        proposal = proposal_root_for_layout(root)
        names = list_experiments(root)
        kind = "raw_reduced" if root.is_relative_to(proposal / "reduced") else "legacy_proposal"
    else:
        dataset = discover_for_config(values, experiment=experiment, variant=variant)
        proposal = dataset.proposal_root or dataset.input_path
        names = [dataset.experiment]
        kind = dataset.layout_type
    reduced = reduced_root_for_proposal(proposal)
    report = {"layout": kind, "proposal_root": str(proposal),
              "layout_root": str(root) if root else None,
              "analysis_root": str(analysis_root_for_proposal(proposal)),
              "experiments": names, "warnings": [],
              "suggested_paths": {"data_path": str(proposal),
                  "beamhistory_path": str(reduced / "data/beamhistory"),
                  "electrochemistry_path": str(reduced / "data/echem")}}
    selected = experiment or values["project"].get("experiment")
    rows = []
    for name in ([resolve_experiment_name(selected, names)] if selected else names):
        try:
            dataset = discover_for_config(values, experiment=name, variant=variant or values["project"].get("variant"))
            frames = dataset.image_products(include_aggregates=False)
            row = {"experiment": dataset.experiment, "variants": sorted({p.variant for p in dataset.products}),
                   "image_count": len(frames), "run_count": len(dataset.runs),
                   "metadata_run_count": sum(bool(r.metadata_dir and r.metadata_dir.is_dir()) for r in dataset.runs),
                   "missing_timing_runs": []}
            from . import backends
            for run in dataset.runs:
                if not backends.nci_data().get_exposure_windows(metadata_dir=run.metadata_dir, tpx_folder=None):
                    row["missing_timing_runs"].append(run.name)
            rows.append(row)
        except (ValueError, OSError) as exc:
            rows.append({"experiment": name, "error": str(exc)})
    report["datasets"] = rows
    report["valid"] = bool(rows) and all("error" not in row for row in rows)
    if any(row.get("missing_timing_runs") for row in rows):
        report["warnings"].append("Some frames lack timing metadata. Do not enable beam correction for those runs.")
    return report


def processing_layout(dataset, products):
    """Adapt normalized products to established correction math, metadata only."""
    groups = defaultdict(list)
    for product in products:
        groups[product.run or product.path.stem].append(str(product.path))
    runs = []
    for name, files in groups.items():
        record = dataset.run(name)
        runs.append({"name": name, "image_dir": str(Path(files[0]).parent), "image_files": files,
                     "preselected": True,
                     "metadata_dir": str(record.metadata_dir) if record and record.metadata_dir else None,
                     "tpx_dir": None, "layout_type": "v2" if dataset.layout_type in {"v2", "mapped"} else "legacy"})
    return {"layout_type": dataset.layout_type, "layout_root": str(dataset.layout_root or dataset.input_path),
            "experiment": dataset.experiment, "experiment_dir": str(dataset.input_path), "runs": runs}

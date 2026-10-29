"""Project and product discovery section."""

from __future__ import annotations

from collections import Counter
from ..proposal_paths import analysis_root_for_proposal, reduced_root_for_proposal

from pathlib import Path

from ..layout import discover_dataset
from ..result_layout import result_directory
from ..section import PipelineSection, SectionOutcome, SectionState


class ProjectSection(PipelineSection):
    id = "project"
    title = "Project and data"
    produces = ("dataset", "experiment", "results_dir", "analysis_root", "source_products", "aggregate_products")

    def is_configured(self, config):
        return bool(str(config.get("data_path", "")).strip())

    def run(self, session, config):
        def counted(count, singular, plural=None):
            return f"{count} {singular if count == 1 else (plural or singular + 's')}"

        requested_experiment = str(config.get("experiment", "")).strip()
        from ..inspection import discover_for_config
        dataset = discover_for_config(session.config)
        include_aggregates = bool(config.get("include_aggregates", False))
        kinds = config.get("product_kinds", ("final",))
        if isinstance(kinds, str):
            kinds = tuple(part.strip() for part in kinds.replace(";", ",").split(",") if part.strip())
        matching_products = dataset.image_products(
            variant=str(config.get("variant", "")).strip() or None,
            kinds=kinds,
            include_aggregates=True,
        )
        per_run_products = [product for product in matching_products if not product.aggregate]
        aggregate_products = [product for product in matching_products if product.aggregate]
        products = per_run_products
        if session.config.get("workflow", {}).get("role") == "processing":
            variants = sorted({product.variant for product in products})
            if len(variants) > 1:
                raise ValueError(f"Select one image variant before processing: {variants}")
        if not products and not (include_aggregates and aggregate_products):
            available = sorted({(product.kind, product.variant) for product in dataset.products})
            raise ValueError(
                "No image products match the selection. "
                f"Available kind/variant pairs: {available}"
            )
        raw_result_root = str(config.get("result_root", "")).strip()
        if raw_result_root:
            result_root = Path(raw_result_root).expanduser()
            if not result_root.is_absolute():
                base = reduced_root_for_proposal(dataset.proposal_root) if dataset.proposal_root else dataset.input_path
                result_root = base / result_root
        elif dataset.proposal_root is not None:
            result_root = analysis_root_for_proposal(dataset.proposal_root)
        else:
            result_root = dataset.input_path.parent / "analysis"
        result_root = result_root.resolve(strict=False)
        results_dir = result_directory(
            result_root, dataset.experiment, config.get("variant", ""),
            config.get("analysis_name", ""),
        )
        resolved = ""
        if requested_experiment and requested_experiment != dataset.experiment:
            resolved = f"Resolved experiment {requested_experiment!r} to {dataset.experiment!r}. "
        layout_labels = {
            "v2": "LumaCam proposal",
            "legacy": "legacy LumaCam folder",
            "flat": "flat TIFF folder",
        }
        run_summary = counted(len(dataset.runs), "run record")
        experiment_manifest = dataset.metadata.get("experiment_manifest", {})
        manifest_runs = experiment_manifest.get("runs", {})
        if isinstance(manifest_runs, dict) and manifest_runs:
            statuses = Counter(
                str(details.get("status", "unknown"))
                for details in manifest_runs.values()
                if isinstance(details, dict)
            )
            status_parts = []
            if statuses.get("completed"):
                status_parts.append(f"{statuses['completed']} completed")
            if statuses.get("planned"):
                status_parts.append(f"{statuses['planned']} planned/not acquired")
            status_parts.extend(
                f"{count} {status.replace('_', ' ')}"
                for status, count in sorted(statuses.items())
                if status not in {"completed", "planned"}
            )
            if status_parts:
                run_summary += f" ({', '.join(status_parts)})"
            interrupted = sum(
                bool(details.get("interrupted_by_ctrl_c"))
                for details in manifest_runs.values()
                if isinstance(details, dict)
            )
        else:
            interrupted = 0
        variant = str(config.get("variant", "")).strip()
        variant_summary = f" for variant {variant!r}" if variant else ""
        facts = [
            run_summary,
            counted(len(per_run_products), "matching per-run TIFF"),
            counted(len(aggregate_products), "matching aggregate TIFF"),
        ]
        if interrupted:
            facts.append(counted(interrupted, "interrupted run"))
        message = resolved + (
            f"Found {', '.join(facts)}{variant_summary} in the "
            f"{layout_labels.get(dataset.layout_type, 'supported data')}."
        )
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            message,
            {
                "dataset": dataset,
                "experiment": dataset.experiment,
                "results_dir": results_dir,
                "analysis_root": result_root.parent if result_root.name == dataset.experiment else result_root,
                "source_products": products,
                "aggregate_products": aggregate_products if include_aggregates else [],
            },
            {"layout_type": dataset.layout_type},
        )

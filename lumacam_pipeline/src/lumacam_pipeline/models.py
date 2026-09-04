"""Data models shared by all LumaCam processing sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class RunRecord:
    """One acquisition run and the locations that describe it."""

    name: str
    tpx3_dir: Path | None = None
    metadata_dir: Path | None = None
    derived_dir: Path | None = None


@dataclass(frozen=True)
class ImageProduct:
    """One TIFF product discovered in a LumaCam or legacy dataset."""

    path: Path
    experiment: str
    run: str | None
    kind: str = "final"
    variant: str = "default"
    aggregate: bool = False
    metadata_dir: Path | None = None
    source_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class Dataset:
    """A normalized view of canonical, marker-era, legacy, or flat data."""

    input_path: Path
    layout_type: str
    experiment: str
    proposal_root: Path | None
    layout_root: Path | None
    runs: tuple[RunRecord, ...] = ()
    products: tuple[ImageProduct, ...] = ()
    selected_variant: str | None = None
    selected_run: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def image_products(
        self,
        *,
        variant: str | None = None,
        kinds: Iterable[str] = ("final",),
        include_aggregates: bool = False,
    ) -> list[ImageProduct]:
        """Return deterministically ordered products matching the selection."""

        accepted_kinds = set(kinds)
        selected_variant = variant or self.selected_variant
        products = [
            product
            for product in self.products
            if product.kind in accepted_kinds
            and (selected_variant is None or product.variant == selected_variant)
            and (self.selected_run is None or product.run == self.selected_run)
            and (include_aggregates or not product.aggregate)
        ]
        return sorted(
            products,
            key=lambda item: (
                item.aggregate,
                item.run or "",
                item.variant,
                item.path.name,
            ),
        )

    def run(self, name: str | None) -> RunRecord | None:
        if name is None:
            return None
        return next((run for run in self.runs if run.name == name), None)


@dataclass
class Artifact:
    """A value produced by a section, with lightweight provenance."""

    name: str
    value: Any
    producer: str
    metadata: dict[str, Any] = field(default_factory=dict)

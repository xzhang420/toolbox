"""Section-based processing for LumaCam image datasets."""

from .layout import (
    LayoutError,
    discover_dataset,
    find_layout_root,
    is_layout_root,
    list_experiments,
    resolve_experiment_name,
)
from .models import Dataset, ImageProduct, RunRecord
from .section import PipelineSection, PipelineSession, SectionOutcome, SectionState
from .presets import PRESETS, create_session, preset_config
from .batch import (
    DEFAULT_BATCH_SECTIONS,
    discover_batch_experiments,
    run_batch,
    validate_batch,
)
from .metadata_ranges import (
    ExperimentMetadataRange,
    ProposalMetadataRange,
    inspect_metadata_time_ranges,
    save_metadata_time_ranges_csv,
)

__all__ = [
    "Dataset",
    "DEFAULT_BATCH_SECTIONS",
    "ExperimentMetadataRange",
    "ImageProduct",
    "LayoutError",
    "PipelineSection",
    "PipelineSession",
    "PRESETS",
    "ProposalMetadataRange",
    "RunRecord",
    "SectionOutcome",
    "SectionState",
    "discover_dataset",
    "discover_batch_experiments",
    "create_session",
    "find_layout_root",
    "inspect_metadata_time_ranges",
    "is_layout_root",
    "list_experiments",
    "preset_config",
    "resolve_experiment_name",
    "run_batch",
    "save_metadata_time_ranges_csv",
    "validate_batch",
]

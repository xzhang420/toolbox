"""Section-based processing for LumaCam image datasets."""

__version__ = "0.4.0"

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
    prepare_shared_ob_background,
    run_batch,
    validate_batch,
)
from .metadata_ranges import (
    ExperimentMetadataRange,
    ProposalMetadataRange,
    inspect_metadata_time_ranges,
    save_metadata_time_ranges_csv,
)
from .processing import create_processing_session
from .inspection import inspect_layout
from .visualization import open_visualization, load_visualization_settings
from .reports import create_report
from .processed_variant import derive_background_variant

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
    "create_processing_session",
    "create_report",
    "derive_background_variant",
    "open_visualization",
    "load_visualization_settings",
    "inspect_layout",
    "find_layout_root",
    "inspect_metadata_time_ranges",
    "is_layout_root",
    "list_experiments",
    "preset_config",
    "prepare_shared_ob_background",
    "resolve_experiment_name",
    "run_batch",
    "save_metadata_time_ranges_csv",
    "validate_batch",
]

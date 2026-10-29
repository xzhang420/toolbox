"""Dashboard visibility only; all modes use the same processing sections."""

MODE_OPTIONS = (
    ("Operando NCI", "operando_nci"),
    ("Ex situ NCI", "exsitu"),
    ("Time of flight", "tof"),
)
LEGACY_MODE_ALIASES = {"image_only": "exsitu", "nci_images_only": "operando_nci"}


def display_mode(mode):
    """Map older config/notebook mode names to the simplified GUI choices."""
    return LEGACY_MODE_ALIASES.get(mode, mode)


IMAGE_SECTIONS = frozenset({
    "project", "images", "average", "image_measurements", "quality", "selection", "time_window", "csv_export",
    "image_plot", "individual_image_plots", "image_video",
})
CORRECTION_SECTIONS = frozenset({"corrections", "background"})
EC_SECTIONS = frozenset({
    "electrochemistry", "synchronization", "event_detection",
    "combined_plot", "nci_event_outputs", "combined_video", "cycle_summary",
})

MODE_SECTIONS = {
    "exsitu": IMAGE_SECTIONS | CORRECTION_SECTIONS | {"inhomogeneity"},
    "operando_nci": IMAGE_SECTIONS | CORRECTION_SECTIONS | EC_SECTIONS | {"nci_plating_metrics", "inhomogeneity"},
    "tof": IMAGE_SECTIONS | {"tof_cube"},
}

MODE_DESCRIPTIONS = {
    "exsitu": "NCI sample corrections, averages, ROI measurements, image exports and batches. Electrochemistry is hidden.",
    "operando_nci": "NCI time-series images with optional electrochemistry, synchronization, event analysis and combined outputs.",
    "tof": "Analyze a reconstructed time-of-flight cube, with standard image tools available for 2D TIFFs.",
}


def visible_sections(mode, all_sections, *, show_all=False):
    """Return visible ids without disabling or discarding hidden results."""
    mode = display_mode(mode)
    if mode not in MODE_SECTIONS:
        raise ValueError(f"Unknown processing mode: {mode!r}")
    return frozenset(all_sections) if show_all else MODE_SECTIONS[mode]

"""Small configuration presets built from the same selectable sections."""

from __future__ import annotations

from copy import deepcopy

from .section import PipelineSession
from .sections import default_sections
from .white_spots import DEFAULTS as WHITE_SPOT_DEFAULTS


COMMON = {
    "shared_setup": {
        "reference_set": "",
        "ob_experiment": "",
        "background_experiment": "",
        "reference_source_path": "",
        "prepared_reference_path": "",
        "reference_tiff_count": 0,
        "prepared_background_path": "",
        "background_source_tiff_count": 0,
        "background_tiff_count": 0,
        "metadata_path": "",
    },
    "project": {
        "data_path": "",
        "experiment": "",
        "variant": "3to1000",
        "result_root": "",
        "analysis_name": "",
        "product_kinds": "final",
        "include_aggregates": False,
    },
    "corrections": {
        "enabled": True,
        "beamhistory_path": "",
        "reference_dir": "",
        "reference_beamhistory_path": "",
        "reference_bin_y": 2,
        "reference_bin_x": 2,
        "reference_thickness_um": 1.0,
        "file_filter": "3to1000",
        "tpx_time": 2,
        "n_tpx_per_image": 30,
    },
    "background": {"enabled": False, "source_mode": "auto", "background_path": "", "n_images": 7, "clip_negatives": True},
    "images": {
        **WHITE_SPOT_DEFAULTS,
        "enabled": True,
        "frames_per_merge": 1,
        "spatial_pixel_merge": 1,
        "merge_method": "mean",
        "tpx_time": 2,
    },
    "average": {"enabled": True, "method": "mean"},
    "image_measurements": {"enabled": True, "roi": {}, "calibrated": False},
    "nci_plating_metrics": {"enabled": False, "roi": {}, "pixel_size_um": 55, "threshold_fraction": 0.05},
    "photon_hyperspectral": {"enabled": False, "photon_min": 2, "photon_max": 100},
    "tof_cube": {"enabled": False, "path": "", "axis_order": "CYX", "axis_start": 0, "axis_step": 1},
    "electrochemistry": {
        "enabled": True,
        "path": "",
        "capacity_to_thickness": 3.66,
        "time_column": "time/s",
        "thickness_capacity_mode": "current_sign_half_cycle",
    },
    "synchronization": {"enabled": True, "direction": "nearest", "tolerance_seconds": ""},
    "event_detection": {"enabled": True, "initial_ocv_image_count": None},
    "selection": {"enabled": True, "indices": "", "representative_count": 4, "use_detected_events": True},
    "image_plot": {"enabled": True, "pixel_size_um": 55, "scalebar_length_um": 5000, "scalebar_label": "5 mm"},
    "individual_image_plots": {
        "enabled": False,
        "indices": "",
        "subfolder": "individual_images",
        "format": "png",
        "color_scale": "global",
        "pixel_size_um": 55,
        "scalebar_length_um": 5000,
        "scalebar_label": "5 mm",
        "dpi": 300,
    },
    "combined_plot": {"enabled": True, "thickness_traces": "both", "thickness_layout": "separate"},
    "nci_event_outputs": {"enabled": True, "pixel_size_um": 55, "scalebar_length_um": 5000},
    "image_video": {"enabled": False, "fps": 10, "dt_seconds": 60, "dpi": 150},
    "combined_video": {"enabled": False, "fps": 10, "dt_seconds": 60, "dpi": 150},
}


def _preset(**updates):
    config = deepcopy(COMMON)
    for section, values in updates.items():
        config.setdefault(section, {}).update(values)
    return config


PRESETS = {
    # Keep the legacy presets for old configs and Python/CLI callers. The GUI
    # exposes only the three choices in modes.MODE_OPTIONS.
    "operando_nci": _preset(),
    "nci_images_only": _preset(
        electrochemistry={"enabled": False, "path": ""},
        combined_plot={"enabled": False},
        event_detection={"enabled": False},
        combined_video={"enabled": False},
    ),
    "exsitu": _preset(
        average={"collect_in_proposal": True},
        electrochemistry={"enabled": False, "path": ""},
        combined_plot={"enabled": False},
        event_detection={"enabled": False},
        nci_event_outputs={"enabled": False},
        combined_video={"enabled": False},
        nci_plating_metrics={"enabled": False},
    ),
    "image_only": _preset(
        corrections={"enabled": False},
        background={"enabled": False},
        electrochemistry={"enabled": False, "path": ""},
        nci_plating_metrics={"enabled": False},
        combined_plot={"enabled": False},
        event_detection={"enabled": False},
        nci_event_outputs={"enabled": False},
        combined_video={"enabled": False},
    ),
    "tof": _preset(
        project={"variant": "", "product_kinds": "derived,derived_photon,derived_pixel"},
        corrections={"enabled": False},
        background={"enabled": False},
        electrochemistry={"enabled": False, "path": ""},
        event_detection={"enabled": False},
        nci_plating_metrics={"enabled": False},
        nci_event_outputs={"enabled": False},
        combined_plot={"enabled": False},
        combined_video={"enabled": False},
        tof_cube={"enabled": True},
    ),
}


def preset_config(name="operando_nci"):
    if name not in PRESETS:
        raise ValueError(f"Unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return deepcopy(PRESETS[name])


def create_session(name="operando_nci", config=None):
    values = preset_config(name)
    for section, section_values in (config or {}).items():
        values.setdefault(section, {}).update(section_values)
    return PipelineSession(default_sections(), values, preset=name)

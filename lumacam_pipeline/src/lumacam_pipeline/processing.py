"""Processing-only public entry points; visualization is a separate session."""

from copy import deepcopy
from pathlib import Path

from .presets import create_session

PROCESSING_SECTIONS = frozenset({"project", "corrections", "background", "images", "average",
                                "photon_hyperspectral", "tof_cube"})


def create_processing_session(preset="operando_nci", config=None):
    values = deepcopy(config or {})
    values.setdefault("workflow", {})["role"] = "processing"
    return create_session(preset, values)


def resolve_processing_paths(config, config_path):
    """Resolve relative root against the INI, other inputs against proposal root."""
    values = deepcopy(config)
    project = values.setdefault("project", {})
    root = Path(project.get("data_path") or ".").expanduser()
    if not root.is_absolute():
        root = Path(config_path).resolve().parent / root
    project["data_path"] = str(root.resolve())
    for section, keys in {
        "corrections": ("beamhistory_path", "reference_beamhistory_path", "reference_dir", "prepared_reference_path"),
        "background": ("background_path",), "electrochemistry": ("path",), "tof_cube": ("path",),
    }.items():
        for key in keys:
            value = values.get(section, {}).get(key)
            if value and not Path(value).expanduser().is_absolute():
                values[section][key] = str((root / value).resolve())
    return values


def available_beamhistory(value):
    """An empty default folder is optional; a bad non-empty input remains an error."""
    if not str(value or "").strip():
        return None
    path = Path(value).expanduser()
    if path.is_dir() and not any(p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
                                 for p in path.rglob("*")):
        return None
    return str(path.resolve())


def prepare_reference_timing(config):
    """Normalize optional histories and reject raw timing in shared preparation."""
    if config.get("workflow", {}).get("role") != "processing":
        return
    from . import backends
    from .prepared_inputs import uses_prepared_ob
    corrections = config.setdefault("corrections", {})
    beam = available_beamhistory(corrections.get("beamhistory_path"))
    reference_beam = available_beamhistory(corrections.get("reference_beamhistory_path"))
    corrections["beamhistory_path"] = beam or ""
    if uses_prepared_ob(corrections):
        return
    reference = corrections.get("reference_dir")
    corrections["reference_beamhistory_path"] = reference_beam or beam or ""
    if reference and corrections["reference_beamhistory_path"]:
        module = backends.nci_data()
        layout = module.resolve_nci_experiment(str(reference),
            final_subfolder=str(corrections.get("reference_final_subfolder", "final")))
        missing = [run["name"] for run in layout["runs"]
                   if not module.get_exposure_windows(metadata_dir=run.get("metadata_dir"), tpx_folder=None,
                       tpx_time=float(corrections.get("tpx_time", 2)))]
        if missing:
            raise ValueError(f"Shared reference beam correction requires JSON timing metadata; missing: {missing}")

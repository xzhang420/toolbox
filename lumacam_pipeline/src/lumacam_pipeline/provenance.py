"""Processing records and protection against overwriting a different recipe."""

from pathlib import Path
import json

from .config import save_config
from .prepared_inputs import uses_prepared_ob
from .background import effective_background_settings
from .result_layout import snapshot_path
from .white_spots import effective_image_settings

GUARDED_SECTIONS = {"corrections", "background", "images", "average"}
READ_ONLY_SECTIONS = {"project", "selection", "electrochemistry", "synchronization", "event_detection"}


def _settings(config):
    ignored = {"enabled", "verbose", "recompute", "prepared_reference_path", "collect_in_proposal", "filename"}
    if uses_prepared_ob(config):
        ignored.discard("prepared_reference_path")
    return {key: value for key, value in config.items() if key not in ignored}


def recipe_for(session, section):
    project = session.section_config("project")
    settings = session.section_config(section)
    if section == "background":
        settings = effective_background_settings(settings)
    elif section == "images":
        settings = effective_image_settings(settings)
    recipe = {"variant": project.get("variant", ""), "section": section,
              "settings": _settings(settings)}
    if section == "background":
        recipe["corrections"] = _settings(session.section_config("corrections"))
    if section in {"images", "average"}:
        if section == "average":
            recipe["images"] = _settings(effective_image_settings(session.section_config("images")))
        stage = ("background" if session.artifacts.has("analysis_image_dir") else
                 "corrections" if session.artifacts.has("processed_image_dir") else "source")
        recipe["source_stage"] = stage
        if stage != "source":
            recipe["corrections"] = _settings(session.section_config("corrections"))
        if stage == "background":
            recipe["background"] = _settings(effective_background_settings(session.section_config("background")))
    return json.loads(json.dumps(recipe, default=str))


def record_path(session):
    root = session.artifacts.get("results_dir")
    return Path(root) / "processing" / "recipes.json" if root is not None else None


def check_recipe(session, section):
    if session.config.get("workflow", {}).get("role") == "visualization":
        return
    if (section == "images" and session.config.get("workflow", {}).get("role") != "processing"
            and not session.section_config("images").get("white_spot_enabled", False)):
        return
    path = record_path(session)
    if section not in GUARDED_SECTIONS or path is None or not path.exists():
        return
    records = json.loads(path.read_text(encoding="utf-8"))
    saved = records.get(section)
    average_stage = records.get("average", {}).get("source_stage")
    filtered_chain_changed = False
    if section == "images" and "average" in records:
        current = recipe_for(session, "images")
        average = records["average"]
        filtered_chain_changed = (
            average.get("images") != current["settings"]
            or any(average.get(key) != current.get(key) for key in ("source_stage", "corrections", "background"))
        )
    changes_finished_chain = (
        section == "corrections" and (
            average_stage == "source"
            or ("background" in records and not session.section_config("background").get("enabled", False))
        )
        or section == "background" and average_stage in {"source", "corrections"}
    )
    if filtered_chain_changed or changes_finished_chain or (saved is not None and saved != recipe_for(session, section)):
        raise ValueError(
            f"Existing {section} results use different settings. Set a new Analysis name "
            "in Project and data, then Discover data and process there. Existing results were not overwritten."
        )


def record_success(session, section):
    role = session.config.get("workflow", {}).get("role")
    if role == "visualization":
        from .visualization import record_visualization
        record_visualization(session, section)
        return
    if role == "processing" and section in {"images", "average"}:
        from .handover import publish_processed_dataset
        publish_processed_dataset(session)
        save_config(snapshot_path(session.artifacts.require("results_dir")), session.preset, session.config)
    if section == "images" and role != "processing" and not session.section_config("images").get("white_spot_enabled", False):
        return
    path = record_path(session)
    if path is None or section in READ_ONLY_SECTIONS:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if section in GUARDED_SECTIONS:
        saved = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        saved[section] = recipe_for(session, section)
        path.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
    save_config(snapshot_path(session.artifacts.require("results_dir")), session.preset, session.config)

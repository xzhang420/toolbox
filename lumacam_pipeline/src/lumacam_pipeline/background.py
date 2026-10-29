"""Background source selection shared by the GUI, batches, and processing."""


def background_source(config):
    """Infer older configs from the path; explicit choices always take priority."""
    mode = str(config.get("source_mode", "auto") or "auto").strip().lower()
    if mode == "auto":
        return "shared" if str(config.get("background_path", "") or "").strip() else "first_frames"
    if mode not in {"shared", "first_frames"}:
        raise ValueError("BG source must be 'shared' or 'first_frames'.")
    return mode


def background_frame_count(config):
    value = config.get("n_images", 7)
    try:
        count = int(str(value).strip())
    except (TypeError, ValueError):
        raise ValueError("BG first images must be a positive whole number (for example, 7).") from None
    if count < 1:
        raise ValueError("BG first images must be at least 1.")
    return count


def effective_background_settings(config):
    """Canonical scientific settings, compatible with pre-selector recipes.

    A retained shared path is inactive in first-frames mode. Omitting the UI
    selector here also keeps an old shared recipe equivalent to an explicit
    shared selection; full config snapshots still retain the user's choice.
    """
    values = dict(config)
    if background_source(values) == "first_frames":
        values["background_path"] = ""
    values.pop("source_mode", None)
    return values


def shared_background_path(config):
    if background_source(config) == "first_frames":
        return ""
    path = str(config.get("background_path", "") or "").strip()
    if not path:
        raise ValueError("Shared BG needs a Background file/folder. Prepare shared OB and BG or select an existing BG.")
    return path

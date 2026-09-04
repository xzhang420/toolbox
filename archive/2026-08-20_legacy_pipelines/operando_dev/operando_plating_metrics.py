"""Compatibility import for the shared LumaCam plating-metrics backend."""

from lumacam_pipeline.legacy_backend import operando_plating_metrics as _backend

globals().update(
    {name: value for name, value in vars(_backend).items() if not name.startswith("__")}
)

"""Compatibility import for the shared LumaCam interactive backend."""

from lumacam_pipeline.legacy_backend import operando_interactive as _backend

globals().update(
    {name: value for name, value in vars(_backend).items() if not name.startswith("__")}
)

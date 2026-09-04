"""Compatibility import for the shared LumaCam image plotting engine."""

from lumacam_pipeline import image_plots as _backend

globals().update(
    {name: value for name, value in vars(_backend).items() if not name.startswith("__")}
)

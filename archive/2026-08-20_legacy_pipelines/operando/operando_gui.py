"""Compatibility import for the retained operando notebook interface.

New workflows should open ``toolbox/lumacam_pipeline.ipynb``.  This module
keeps existing notebooks working while sharing the single maintained legacy
controller from ``operando_dev``.
"""

from operando_dev import operando_gui as _backend

globals().update(
    {name: value for name, value in vars(_backend).items() if not name.startswith("__")}
)

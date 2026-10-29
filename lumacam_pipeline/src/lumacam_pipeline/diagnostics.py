"""Portable dependency diagnostics; no installation or data mutation."""

import importlib.metadata
import platform
import shutil
import sys


def doctor():
    packages = {}
    for name in ("numpy", "scipy", "pandas", "tifffile", "matplotlib", "astropy", "Pillow", "matplotlib-scalebar", "ipywidgets", "jupyterlab"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    from . import __version__
    return {"pipeline_version": __version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "packages": packages,
            "ffmpeg": shutil.which("ffmpeg"),
            "core_ready": all(packages[name] for name in ("numpy", "scipy", "pandas", "tifffile", "matplotlib", "astropy", "Pillow", "matplotlib-scalebar")),
            "note": "FFmpeg is required for MP4; notebook controls require ipywidgets. No software was installed."}

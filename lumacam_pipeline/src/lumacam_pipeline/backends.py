"""Lazy adapters to established toolbox calculations.

The imports remain lazy so image discovery and configuration work even on a
machine that does not have every plotting/notebook dependency installed.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=None)
def nci_data():
    from .legacy_backend import operando_data

    return operando_data


@lru_cache(maxsize=None)
def nci_plots():
    from .legacy_backend import operando_plots

    return operando_plots


@lru_cache(maxsize=None)
def plating_metrics():
    from .legacy_backend import operando_plating_metrics

    return operando_plating_metrics


@lru_cache(maxsize=None)
def exsitu_pipeline():
    from .legacy_backend import exsitu_pipeline as module

    return module


@lru_cache(maxsize=None)
def image_plots():
    from . import image_plots as module

    return module

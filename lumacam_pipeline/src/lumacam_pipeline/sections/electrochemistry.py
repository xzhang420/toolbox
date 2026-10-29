"""Independent electrochemistry and image/electrochemistry synchronization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import backends
from ..section import PipelineSection, SectionOutcome, SectionState


class ElectrochemistrySection(PipelineSection):
    id = "electrochemistry"
    title = "Electrochemistry"
    produces = ("echem_times", "echem_potential", "echem_thickness", "echem_cycles", "echem_current", "echem_capacity", "echem_provenance")

    def is_configured(self, config):
        return bool(config.get("enabled", True) and str(config.get("path", "")).strip())

    def run(self, session, config):
        path = Path(config["path"]).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Electrochemistry path does not exist: {path}")
        values = backends.nci_plots().load_echem_data(
            str(path),
            conv_thickness=float(config.get("capacity_to_thickness", 3.66)),
            time_column=str(config.get("time_column", "time/s")),
            report_summary=bool(config.get("report_summary", True)),
            include_cycle=True,
            include_current=True,
            include_capacity=True,
            include_provenance=True,
            zero_current_ma=float(config.get("zero_current_ma", .01)),
            cycle_column=str(config.get("cycle_column", "cycle number")),
            window_start=config.get("window_start"),
            window_end=config.get("window_end"),
            thickness_capacity_mode=str(config.get("thickness_capacity_mode", "current_sign_half_cycle")),
        )
        # Older/custom backends may still return the original six-item tuple.
        times, potential, thickness, cycles, current, capacity = values[:6]
        provenance = values[6] if len(values)>6 else {}
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Loaded {len(times)} electrochemistry points.",
            {
                "echem_times": list(times),
                "echem_potential": np.asarray(potential),
                "echem_thickness": np.asarray(thickness),
                "echem_cycles": np.asarray(cycles),
                "echem_current": None if current is None else np.asarray(current),
                "echem_capacity": np.asarray(capacity),
                "echem_provenance": provenance,
            },
        )


class SynchronizationSection(PipelineSection):
    id = "synchronization"
    title = "Synchronize and correlate"
    requires = (
        "image_times",
        "image_metric_values",
        "echem_times",
        "echem_potential",
    )
    produces = ("synchronized_table",)

    def run(self, session, config):
        image_table = pd.DataFrame(
            {
                "image_time": pd.to_datetime(session.artifacts.require("image_times")),
                "image_metric": np.asarray(session.artifacts.require("image_metric_values"), dtype=float),
            }
        ).sort_values("image_time")
        echem_table = pd.DataFrame(
            {
                "echem_time": pd.to_datetime(session.artifacts.require("echem_times")),
                "potential": np.asarray(session.artifacts.require("echem_potential"), dtype=float),
            }
        )
        thickness = session.artifacts.get("echem_thickness")
        if thickness is not None and len(thickness) == len(echem_table):
            echem_table["echem_thickness"] = np.asarray(thickness, dtype=float)
        cycles = session.artifacts.get("echem_cycles")
        if cycles is not None and len(cycles) == len(echem_table):
            echem_table["cycle"] = cycles
        echem_table = echem_table.sort_values("echem_time")
        if image_table.empty or echem_table.empty:
            raise ValueError("Synchronization needs nonempty image and electrochemistry time series.")
        tolerance = config.get("tolerance_seconds")
        table = pd.merge_asof(
            image_table,
            echem_table,
            left_on="image_time",
            right_on="echem_time",
            direction=str(config.get("direction", "nearest")),
            tolerance=pd.Timedelta(seconds=float(tolerance)) if tolerance not in (None, "") else None,
        )
        if not config.get("allow_extrapolation", True):
            outside = ((table["image_time"] < echem_table["echem_time"].min()) |
                       (table["image_time"] > echem_table["echem_time"].max()))
            table.loc[outside, "echem_time"] = pd.NaT
            for column in ("potential", "echem_thickness", "cycle"):
                if column in table:
                    table.loc[outside, column] = np.nan
        table["time_offset_s"] = (table["image_time"] - table["echem_time"]).dt.total_seconds()
        matched = int(table["echem_time"].notna().sum())
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Matched {matched}/{len(table)} image measurements with electrochemistry; {len(table)-matched} unmatched.",
            {"synchronized_table": table},
        )

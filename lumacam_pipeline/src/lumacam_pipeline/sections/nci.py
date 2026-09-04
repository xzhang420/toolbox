"""Optional NCI-specific analysis sections."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import backends
from ..section import PipelineSection, SectionOutcome, SectionState


class PlatingMetricsSection(PipelineSection):
    id = "nci_plating_metrics"
    title = "NCI plating metrics"
    requires = ("image_stack", "image_times", "results_dir", "experiment")
    produces = ("plating_metrics_table", "plating_metrics_path")

    def is_configured(self, config):
        return bool(config.get("enabled", False))

    def run(self, session, config):
        images = session.artifacts.require("image_stack")
        times = session.artifacts.require("image_times")
        if not images:
            raise ValueError("No images are available for plating metrics.")
        shape = np.asarray(images[0]).shape
        roi = config.get("roi") or {}
        x0 = float(roi.get("x0", shape[1] / 2))
        y0 = float(roi.get("y0", shape[0] / 2))
        radius = float(roi.get("r0", min(shape) / 2))
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        mask = (xx - x0) ** 2 + (yy - y0) ** 2 <= radius ** 2
        rows = []
        for index, (time, image) in enumerate(zip(times, images)):
            metrics = backends.nci_plots().calculate_plating_metrics(
                image,
                mask,
                pixel_size=float(config.get("pixel_size_um", 55)),
                center=(x0, y0),
                threshold=config.get("threshold"),
                threshold_fraction=float(config.get("threshold_fraction", 0.05)),
                anode_radius=radius,
            )
            rows.append({"image_index": index, "image_time": time, **metrics})
        table = pd.DataFrame(rows)
        output_dir = Path(session.artifacts.require("results_dir")) / "tables"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"plating_metrics_{session.artifacts.require('experiment')}.csv"
        table.to_csv(output_path, index=False)
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Wrote plating metrics to {output_path}",
            {"plating_metrics_table": table, "plating_metrics_path": output_path},
        )

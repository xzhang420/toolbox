"""Native-cadence and explicitly matched tables for humans and AI."""
import numpy as np
import pandas as pd
from .section import PipelineSection, SectionOutcome, SectionState
from .time_window import bounds, time_mask


def utc_times(times):
    return pd.to_datetime(times, utc=True)


def native_tables(session):
    images = session.artifacts.require("image_metrics_table").copy()
    images = images.rename(columns={"time": "image_time_utc"})
    images["image_time_utc"] = utc_times(images.image_time_utc)
    valid_times = images.image_time_utc.dropna()
    origin = valid_times.min() if len(valid_times) else pd.NaT
    images["elapsed_h"] = (images.image_time_utc-origin).dt.total_seconds()/3600
    images = images.drop(columns=["value", "label"])
    paths = session.artifacts.get("image_paths")
    if paths is not None:
        from pathlib import Path
        images["source_image"] = [Path(path).name for path in paths]
    ec_times = session.artifacts.get("echem_times")
    if not ec_times:
        return images, None
    ec = pd.DataFrame({"echem_time_utc": utc_times(ec_times),
                       "potential_V": session.artifacts.require("echem_potential"),
                       "ec_thickness_um": session.artifacts.require("echem_thickness")})
    ec["elapsed_h"] = (ec.echem_time_utc-origin).dt.total_seconds()/3600
    for artifact, column in (("echem_current", "current_mA"), ("echem_capacity", "capacity_mAh"), ("echem_cycles", "cycle")):
        value = session.artifacts.get(artifact)
        if value is not None:
            ec[column] = value
    provenance = session.artifacts.get("echem_provenance", {})
    for key, column in (("source_files", "source_file"), ("net_capacity_mAh", "net_capacity_mAh")):
        if key in provenance:
            ec[column] = provenance[key]
    ec["thickness_basis"] = "net transferred charge; unknown initial Li inventory"
    return images, ec


def comparison_table(images, ec, tolerance_seconds=30):
    tolerance = float(tolerance_seconds)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Matching tolerance must be finite and nonnegative.")
    if images.image_time_utc.isna().any():
        raise ValueError("Comparison export requires real timestamps for every image.")
    if ec.empty or ec.echem_time_utc.isna().any():
        raise ValueError("Comparison export requires real electrochemistry timestamps.")
    right = ec.drop(columns=["elapsed_h"])
    table = pd.merge_asof(images.sort_values("image_time_utc"), right.sort_values("echem_time_utc"),
        left_on="image_time_utc", right_on="echem_time_utc", direction="nearest",
        tolerance=pd.Timedelta(seconds=tolerance))
    outside = (table.image_time_utc < ec.echem_time_utc.min()) | (table.image_time_utc > ec.echem_time_utc.max())
    table.loc[outside, "echem_time_utc"] = pd.NaT
    for column in set(right.columns)-{"echem_time_utc"}:
        table.loc[outside, column] = np.nan
    table["matched"] = table.echem_time_utc.notna()
    table["match_status"] = np.where(table.matched, "matched", "no_sample_within_tolerance")
    table.loc[table.image_time_utc < ec.echem_time_utc.min(), "match_status"] = "before_ec_record"
    table.loc[table.image_time_utc > ec.echem_time_utc.max(), "match_status"] = "after_ec_record"
    table["time_offset_s"] = (table.image_time_utc-table.echem_time_utc).dt.total_seconds()
    return table.sort_values("frame").reset_index(drop=True)


class CsvExportSection(PipelineSection):
    id = "csv_export"
    title = "Export thickness / electrochemistry CSV"
    requires = ("image_metrics_table", "image_times", "results_dir")
    optional_inputs = ("echem_times", "echem_potential", "echem_thickness", "echem_current", "echem_capacity", "echem_cycles", "view_window")
    produces = ("csv_export_paths",)

    def run(self, session, config):
        from .sections.outputs import _category_dir
        scope = config.get("scope", "full")
        if scope not in {"full", "window"}:
            raise ValueError("CSV scope must be full or window.")
        window = bounds(session, force=True) if scope == "window" else None
        images, ec = native_tables(session)
        matched = None
        if config.get("include_comparison", True) and ec is not None:
            # Matching uses the complete record, before selecting rows to export.
            matched = comparison_table(images, ec, config.get("tolerance_seconds", 30))
        keep = time_mask(images.image_time_utc, window)
        if not keep.any():
            raise ValueError("No image samples lie inside the requested CSV window.")
        tables = {"thickness": images.loc[keep]}
        if ec is not None:
            tables["electrochemistry"] = ec.loc[time_mask(ec.echem_time_utc, window)]
        if matched is not None:
            tables["thickness_comparison"] = matched.loc[time_mask(matched.image_time_utc, window)]
        folder = _category_dir(session, "tables")
        suffix = "_window" if scope == "window" else ""
        paths = {}
        for name, table in tables.items():
            path = folder/f"{name}{suffix}.csv"
            table.to_csv(path, index=False, float_format="%.12g")
            paths[name] = path
        message = f"Exported {len(paths)} CSV(s), {scope} experiment; elapsed hours remain relative to the first image."
        if ec is None:
            message += " No electrochemistry loaded; image-only CSV written."
        return SectionOutcome(self.id, SectionState.COMPLETED, message, {"csv_export_paths": paths})

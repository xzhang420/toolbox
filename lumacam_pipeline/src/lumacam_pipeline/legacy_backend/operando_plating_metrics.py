"""Utilities for loading and comparing NCI plating metrics.

The GUI writes one plating-metrics CSV per processed dataset.  This module
keeps the downstream comparison workflow small: point it at result folders or
CSV files, choose metrics, and save publication-ready comparison figures.
"""

from pathlib import Path
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_METRICS = ("f_plate", "CV_active", "CV_global", "rho_Li", "d_COM_norm")

METRIC_LABELS = {
    "f_plate": "Plated footprint fraction",
    "CV_active": "CV of active plated region",
    "CV_global": "CV over nominal anode",
    "rho_Li": "Li-weighted radial position",
    "d_COM_norm": "COM offset / anode radius",
    "active_plating_area_mm2": "Active plated area (mm2)",
    "mean_active": "Mean active signal",
    "mean_nominal": "Mean nominal-anode signal",
    "sum_nominal": "Integrated nominal-anode signal",
}

METRIC_EXPLANATIONS = {
    "f_plate": {
        "formula": "f_plate = N_active / N_ROI",
        "meaning": "Plated footprint fraction. Higher means plating covers more anode area; it is not a thickness-uniformity metric.",
    },
    "CV_active": {
        "formula": "CV_active = std(h_active) / mean(h_active)",
        "meaning": "Homogeneity within active plated pixels. Higher means the active plated region is less uniform.",
    },
    "CV_global": {
        "formula": "CV_global = std(h_ROI) / mean(h_ROI)",
        "meaning": "Homogeneity over the full nominal anode ROI. Higher means the whole anode Li map is less uniform.",
    },
    "rho_Li": {
        "formula": "rho_Li = sum(r_i h_i) / (R sum(h_i))",
        "meaning": "Li-weighted radial position. Higher means deposition is weighted closer to the anode edge.",
    },
    "d_COM_norm": {
        "formula": "d_COM_norm = distance(COM, center) / R",
        "meaning": "Li-weighted center-of-mass offset. Higher means more off-center or one-sided plating.",
    },
    "active_plating_area_mm2": {
        "formula": "area_active = N_active pixel_area",
        "meaning": "Absolute active plated footprint area.",
    },
    "mean_active": {
        "formula": "mean_active = mean(h_active)",
        "meaning": "Mean Li signal only where pixels are above the active-plating threshold.",
    },
    "mean_nominal": {
        "formula": "mean_nominal = mean(h_ROI)",
        "meaning": "Mean Li signal across the nominal anode ROI.",
    },
    "sum_nominal": {
        "formula": "sum_nominal = sum(h_ROI)",
        "meaning": "Integrated Li signal across the nominal anode ROI.",
    },
}

STYLE_PRESETS = {
    "nature": {
        "fig_width": 7.0,
        "panel_height": 1.45,
        "linewidth": 1.45,
        "markersize": 2.6,
        "alpha": 0.95,
        "dpi": 300,
        "formats": ("png", "pdf"),
        "colors": (
            "#1f77b4",
            "#d62728",
            "#2ca02c",
            "#9467bd",
            "#8c564b",
            "#17becf",
            "#7f7f7f",
        ),
    },
    "screen": {
        "fig_width": 8.5,
        "panel_height": 1.8,
        "linewidth": 1.8,
        "markersize": 3.4,
        "alpha": 0.95,
        "dpi": 180,
        "formats": ("png",),
        "colors": (
            "#0072B2",
            "#D55E00",
            "#009E73",
            "#CC79A7",
            "#F0E442",
            "#56B4E9",
            "#000000",
        ),
    },
    "print": {
        "fig_width": 7.0,
        "panel_height": 1.45,
        "linewidth": 1.3,
        "markersize": 2.3,
        "alpha": 0.9,
        "dpi": 600,
        "formats": ("pdf", "svg", "png"),
        "colors": (
            "#000000",
            "#666666",
            "#999999",
            "#333333",
            "#BBBBBB",
            "#555555",
            "#888888",
        ),
    },
}


def recommended_styles_and_formats():
    """Return practical plotting presets for metric-comparison figures."""
    return pd.DataFrame(
        [
            {
                "style": "nature",
                "use": "default manuscript figure",
                "formats": "PNG for quick inspection; PDF for vector editing",
                "notes": "colorblind-safe lines, compact stacked panels",
            },
            {
                "style": "screen",
                "use": "notebook review or group meeting",
                "formats": "PNG",
                "notes": "larger text and markers",
            },
            {
                "style": "print",
                "use": "black-and-white draft or supplement",
                "formats": "PDF, SVG, PNG",
                "notes": "grayscale lines for journal-safe printing",
            },
        ]
    )


def resolve_plating_metrics_csv(dataset):
    """Resolve a CSV path from a CSV file, result directory, or output folder."""
    path = Path(dataset).expanduser()
    if path.is_file():
        return path

    candidates = []
    search_root_groups = [
        [path, path / "tables", path / "processing"],
        [path / "output"],
    ]
    if path.is_dir():
        search_root_groups.extend(
            [
                [candidate.parent for candidate in sorted(path.glob("results_*/processing/plating_metrics*.csv"))],
                [candidate.parent for candidate in sorted(path.glob("results_*/output/plating_metrics*.csv"))],
                [candidate.parent for candidate in sorted(path.glob("*/tables/plating_metrics*.csv"))],
            ]
        )

    for search_roots in search_root_groups:
        group_candidates = []
        for root in search_roots:
            if root.is_dir():
                group_candidates.extend(sorted(root.glob("plating_metrics*.csv")))
        if group_candidates:
            candidates = group_candidates
            break

    unique = []
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            unique.append(candidate)
            seen.add(resolved)

    if not unique:
        raise FileNotFoundError(f"No plating_metrics*.csv found for {path}")
    if len(unique) > 1:
        listed = "\n  ".join(str(candidate) for candidate in unique)
        raise ValueError(
            "More than one plating metrics CSV matched. Pass the exact CSV path:\n"
            f"  {listed}"
        )
    return unique[0]


def _default_dataset_label(csv_path):
    csv_path = Path(csv_path)
    stem = csv_path.stem
    if stem.startswith("plating_metrics_"):
        return stem.replace("plating_metrics_", "", 1)
    if csv_path.parent.name in {"output", "tables"}:
        return csv_path.parent.parent.name.replace("results_", "", 1)
    return stem


def add_elapsed_time_columns(df, datetime_col="datetime", frame_col="frame_index", inplace=False):
    """Add parsed datetimes plus elapsed seconds and hours to a metrics table."""
    out = df if inplace else df.copy()
    if datetime_col in out.columns:
        parsed = pd.to_datetime(out[datetime_col], errors="coerce")
        out[datetime_col] = parsed
        valid = parsed.dropna()
        if not valid.empty:
            origin = valid.iloc[0]
            elapsed_s = (parsed - origin).dt.total_seconds()
            out["elapsed_time_s"] = elapsed_s
            out["elapsed_time_h"] = elapsed_s / 3600.0
            return out

    if frame_col in out.columns:
        frames = pd.to_numeric(out[frame_col], errors="coerce")
        out["elapsed_time_s"] = np.nan
        out["elapsed_time_h"] = frames
    else:
        out["elapsed_time_s"] = np.nan
        out["elapsed_time_h"] = np.arange(len(out), dtype=float)
    return out


def load_plating_metrics(dataset, label=None):
    """Load one plating-metrics CSV and attach a dataset label and time columns."""
    csv_path = resolve_plating_metrics_csv(dataset)
    df = pd.read_csv(csv_path)
    df = add_elapsed_time_columns(df)
    df["dataset"] = label or _default_dataset_label(csv_path)
    df["metrics_csv_path"] = str(csv_path)
    return df


def _iter_dataset_specs(datasets):
    if isinstance(datasets, dict):
        for label, dataset in datasets.items():
            yield str(label), dataset
        return
    for item in datasets:
        if isinstance(item, dict):
            dataset = item.get("path") or item.get("csv_path") or item.get("results_dir")
            label = item.get("label")
            if dataset is None:
                raise ValueError("Dataset dictionaries need a path, csv_path, or results_dir entry.")
            yield label, dataset
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            yield item[0], item[1]
        else:
            yield None, item


def load_many_plating_metrics(datasets, align_time="dataset"):
    """
    Load multiple metrics CSVs.

    Parameters
    ----------
    datasets : dict or sequence
        ``{"label": path}``, ``[(label, path)]``, plain paths, or dictionaries
        with ``label`` and ``path``/``csv_path``/``results_dir``.
    align_time : {"dataset", "global"}
        ``dataset`` starts each line at t=0. ``global`` uses the earliest
        timestamp across all datasets as t=0 when datetimes are available.
    """
    tables = [load_plating_metrics(dataset, label=label) for label, dataset in _iter_dataset_specs(datasets)]
    if not tables:
        raise ValueError("No datasets were provided.")
    combined = pd.concat(tables, ignore_index=True)

    if align_time == "global" and "datetime" in combined.columns:
        datetimes = pd.to_datetime(combined["datetime"], errors="coerce")
        valid = datetimes.dropna()
        if not valid.empty:
            origin = valid.min()
            elapsed_s = (datetimes - origin).dt.total_seconds()
            combined["elapsed_time_s"] = elapsed_s
            combined["elapsed_time_h"] = elapsed_s / 3600.0
    elif align_time != "dataset":
        raise ValueError("align_time must be 'dataset' or 'global'.")

    return combined


def export_combined_plating_metrics(datasets, output_csv, align_time="dataset"):
    """Load multiple datasets and write one combined long-format CSV."""
    combined = load_many_plating_metrics(datasets, align_time=align_time)
    output_csv = Path(output_csv).expanduser()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)
    return combined, output_csv


def _style(style):
    if isinstance(style, dict):
        merged = STYLE_PRESETS["nature"].copy()
        merged.update(style)
        return merged
    if style not in STYLE_PRESETS:
        raise ValueError(f"Unknown style {style!r}. Choose one of {sorted(STYLE_PRESETS)}.")
    return STYLE_PRESETS[style].copy()


def _save_figure(fig, save_path, formats=None, dpi=300):
    if save_path is None:
        return []

    path = Path(save_path).expanduser()
    saved = []
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return [path]

    path.mkdir(parents=True, exist_ok=True)
    for fmt in formats or ("png",):
        out_path = path / f"plating_metrics_comparison.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved.append(out_path)
    return saved


def _metric_explanation_text(metric, wrap_width=44):
    explanation = METRIC_EXPLANATIONS.get(metric)
    if explanation is None:
        return metric
    formula = textwrap.fill(explanation["formula"], width=wrap_width)
    meaning = textwrap.fill(explanation["meaning"], width=wrap_width)
    return f"{metric}\nFormula: {formula}\nMeaning: {meaning}"


def plot_plating_metric_timeseries(
    metrics_df,
    metrics=DEFAULT_METRICS,
    x="elapsed_time_h",
    dataset_col="dataset",
    style="nature",
    title="Lithium plating distribution metrics",
    show_explanations=True,
    save_path=None,
    formats=None,
    dpi=None,
):
    """Plot one or more plating-metric time series from a combined dataframe."""
    df = add_elapsed_time_columns(metrics_df)
    selected = [metric for metric in metrics if metric in df.columns]
    if not selected:
        raise ValueError(f"None of the requested metrics exist in the dataframe: {metrics}")
    if x not in df.columns:
        raise ValueError(f"x column {x!r} is not present in the dataframe.")

    preset = _style(style)
    dpi = int(dpi or preset["dpi"])
    formats = tuple(formats or preset["formats"])
    fig_height = max(2.2, preset["panel_height"] * len(selected) + 0.65)
    if show_explanations:
        fig, axes_grid = plt.subplots(
            len(selected),
            2,
            figsize=(preset["fig_width"] + 3.9, fig_height),
            sharex="col",
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.0, 0.56]},
        )
        if len(selected) == 1:
            axes_grid = np.asarray([axes_grid])
        axes = list(axes_grid[:, 0])
        explanation_axes = list(axes_grid[:, 1])
    else:
        fig, axes = plt.subplots(
            len(selected),
            1,
            figsize=(preset["fig_width"], fig_height),
            sharex=True,
            constrained_layout=True,
        )
        if len(selected) == 1:
            axes = [axes]
        explanation_axes = []

    if dataset_col and dataset_col in df.columns:
        groups = list(df.groupby(dataset_col, sort=False))
    else:
        groups = [("", df)]

    colors = preset["colors"]
    for row_index, (metric, ax) in enumerate(zip(selected, axes)):
        for idx, (label, group) in enumerate(groups):
            group = group.sort_values(x)
            x_values = pd.to_numeric(group[x], errors="coerce").to_numpy(dtype=float)
            y_values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
            ax.plot(
                x_values,
                y_values,
                marker="o",
                markersize=preset["markersize"],
                linewidth=preset["linewidth"],
                alpha=preset["alpha"],
                color=colors[idx % len(colors)],
                label=str(label) if label != "" else None,
            )
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.grid(True, alpha=0.24, linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if show_explanations:
            explanation_ax = explanation_axes[row_index]
            explanation_ax.axis("off")
            explanation_ax.text(
                0.0,
                0.5,
                _metric_explanation_text(metric),
                ha="left",
                va="center",
                fontsize=7,
                linespacing=1.22,
            )

    if title:
        fig.suptitle(title, y=1.01)
    x_labels = {
        "elapsed_time_h": "Elapsed time (h)",
        "elapsed_time_s": "Elapsed time (s)",
        "frame_index": "Processed frame index",
    }
    axes[-1].set_xlabel(x_labels.get(x, x))
    if len(groups) > 1:
        axes[0].legend(frameon=False, loc="best")

    saved_paths = _save_figure(fig, save_path, formats=formats, dpi=dpi)
    return fig, axes, saved_paths


def plot_plating_metric_comparison(
    datasets,
    metrics=DEFAULT_METRICS,
    x="elapsed_time_h",
    align_time="dataset",
    style="nature",
    title="Lithium plating distribution metrics",
    show_explanations=True,
    save_path=None,
    formats=None,
    dpi=None,
):
    """Load multiple datasets and plot metric time series in stacked panels."""
    combined = load_many_plating_metrics(datasets, align_time=align_time)
    fig, axes, saved_paths = plot_plating_metric_timeseries(
        combined,
        metrics=metrics,
        x=x,
        dataset_col="dataset",
        style=style,
        title=title,
        show_explanations=show_explanations,
        save_path=save_path,
        formats=formats,
        dpi=dpi,
    )
    return fig, axes, combined, saved_paths

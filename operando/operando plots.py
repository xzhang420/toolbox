import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.patches as patches
from bisect import bisect_right
from matplotlib_scalebar.scalebar import ScaleBar


def find_paused_intervals(datetime_series, min_gap_minutes=5):
    """
    Detect long gaps in a datetime series that likely correspond to paused acquisition.

    Returns
    -------
    list of tuple(datetime, datetime)
        Start and end datetimes of each detected pause.
    """
    if datetime_series is None or len(datetime_series) < 2:
        return []

    dt_series = pd.to_datetime(list(datetime_series))
    time_diffs = dt_series.to_series().diff().iloc[1:]
    if time_diffs.empty:
        return []

    gap_threshold = pd.Timedelta(minutes=min_gap_minutes)

    pauses = []
    for i, delta in enumerate(time_diffs, start=1):
        if delta > gap_threshold:
            pauses.append((dt_series[i - 1].to_pydatetime(), dt_series[i].to_pydatetime()))

    return pauses


def build_series_with_pauses(x_values, y_values, pauses, time_origin):
    """
    Insert NaN separators into a time series so plotted lines break over pauses.
    """
    if len(x_values) == 0:
        return np.asarray(x_values), np.asarray(y_values)

    pause_starts = {
        (pause_start - time_origin).total_seconds() / 3600.0
        for pause_start, _ in pauses
    }

    x_plot = []
    y_plot = []
    for i, (x_val, y_val) in enumerate(zip(x_values, y_values)):
        x_plot.append(x_val)
        y_plot.append(y_val)
        if i < len(x_values) - 1 and x_val in pause_starts:
            x_plot.append(np.nan)
            y_plot.append(np.nan)

    return np.asarray(x_plot), np.asarray(y_plot)


def build_datetime_series_with_pauses(datetime_values, y_values, pauses, time_origin):
    """
    Insert NaN separators into a datetime-aligned series when segments cross pauses.
    """
    if len(datetime_values) == 0:
        return np.array([]), np.asarray(y_values)

    dt_values = list(pd.to_datetime(list(datetime_values)).to_pydatetime())
    x_values = np.array([(t - time_origin).total_seconds() for t in dt_values]) / 3600.0
    y_values = np.asarray(y_values)

    x_plot = []
    y_plot = []
    for i, (x_val, y_val) in enumerate(zip(x_values, y_values)):
        x_plot.append(x_val)
        y_plot.append(y_val)
        if i >= len(x_values) - 1:
            continue

        seg_start = dt_values[i]
        seg_end = dt_values[i + 1]
        crosses_pause = any(
            seg_start < pause_end and seg_end > pause_start
            for pause_start, pause_end in pauses
        )
        if crosses_pause:
            x_plot.append(np.nan)
            y_plot.append(np.nan)

    return np.asarray(x_plot), np.asarray(y_plot)



def load_images(
    root_folder,
    datetime_formats=("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"),
    r0=None,
    x0=None,
    y0=None
):
    """
    Recursively load all TIFF and FITS images in a folder tree,
    extract datetimes from metadata, optionally crop to a square region,
    sort them, and return images + datetimes.

    Parameters
    ----------
    root_folder : str
        Root directory containing images (including subfolders)
    datetime_formats : tuple of str
        Accepted datetime string formats
    r0 : int, optional
        Half side length of the square crop region
    x0 : int, optional
        X index of the crop center
    y0 : int, optional
        Y index of the crop center

    Returns
    -------
    datetime_series : list of datetime.datetime
        Corresponding datetimes
    img_series : list of np.ndarray
        2D image arrays (cropped if r0, x0, y0 are provided)
    paths : list of str
        File paths (useful for debugging / provenance)
    """

    records = []

    for root, _, files in os.walk(root_folder):
        for fname in files:
            path = os.path.join(root, fname)
            ext = fname.lower().split(".")[-1]

            img = None
            dt = None

            # ---- TIFF files ----
            if ext in ("tif", "tiff"):
                try:
                    img = tifffile.imread(path)
                except Exception:
                    img = np.array(Image.open(path))

                # Extract datetime from TIFF metadata
                try:
                    with tifffile.TiffFile(path) as tif:
                        tags = tif.pages[0].tags
                        for key in ("DateTime", "DateTimeOriginal"):
                            if key in tags:
                                dt_str = tags[key].value
                                for fmt in datetime_formats:
                                    try:
                                        dt = datetime.strptime(dt_str, fmt)
                                        break
                                    except ValueError:
                                        pass
                            if dt is not None:
                                break
                except Exception:
                    pass

                # Fallback: PIL EXIF
                if dt is None:
                    try:
                        with Image.open(path) as im:
                            exif = im.getexif()
                            if exif:
                                dt_str = exif.get(306)  # DateTime
                                if dt_str:
                                    for fmt in datetime_formats:
                                        try:
                                            dt = datetime.strptime(dt_str, fmt)
                                            break
                                        except ValueError:
                                            pass
                    except Exception:
                        pass

            # ---- FITS files ----
            elif ext == "fits":
                try:
                    with fits.open(path) as hdul:
                        img = hdul[0].data
                        header = hdul[0].header
                        # Try common datetime header keywords
                        for key in ("DATE-OBS", "DATE"):
                            if key in header:
                                dt_str = header[key]
                                for fmt in datetime_formats:
                                    try:
                                        dt = datetime.strptime(dt_str, fmt)
                                        break
                                    except ValueError:
                                        pass
                            if dt is not None:
                                break
                except Exception:
                    raise ValueError(f"Failed to read FITS file: {path}")

            else:
                # Skip unsupported file types
                continue

            if img is None or dt is None:
                raise ValueError(f"No valid image or datetime found in: {path}")

            # ---- Crop image if requested ----
            if r0 is not None and x0 is not None and y0 is not None:
                y_start = max(0, y0 - r0)
                y_end = min(img.shape[0], y0 + r0)
                x_start = max(0, x0 - r0)
                x_end = min(img.shape[1], x0 + r0)
                img = img[y_start:y_end, x_start:x_end]

            records.append((dt, img, path))

    # ---- Sort by datetime ----
    records.sort(key=lambda x: x[0])

    datetime_series = [r[0] for r in records]
    img_series = [r[1] for r in records]
    paths = [r[2] for r in records]

    return datetime_series, img_series, paths


def load_echem_data(root_folder, conv_thickness=1.0, time_column="time/s", report_summary=True):
    """
    Recursively load electrochemical text files from a folder tree.

    The returned electrochemical thickness series is computed directly from the
    raw ``Capacity/mA.h`` column as:

    ``echem_thickness_series = Capacity/mA.h * conv_thickness``

    Supports time either as seconds ('time/s') or as datetime strings
    with optional sub-second precision.

    Parameters
    ----------
    root_folder : str
        Root directory containing the electrochemical txt files
    conv_thickness : float
        Factor used to convert ``Capacity/mA.h`` into electrochemical thickness.
    time_column : str
        Column name containing the time (seconds or datetime string)
    report_summary : bool, optional
        If True, print a lightweight validation summary including file ranges,
        duplicate timestamps, and overlaps/gaps between files.

    Returns
    -------
    datetime_series : list of datetime.datetime
        Time series as datetime objects
    potential_series : np.ndarray
        Ewe/V series (potential)
    echem_thickness_series : np.ndarray
        Electrochemical thickness series calculated from capacity.
    """

    all_data = []
    file_summaries = []

    for root, _, files in os.walk(root_folder):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            path = os.path.join(root, fname)

            try:
                # Read tab-separated file
                df = pd.read_csv(path, sep="\t", engine="python")

                required_cols = [time_column, "Ewe/V", "Capacity/mA.h"]
                if not all(col in df.columns for col in required_cols):
                    raise ValueError(f"Missing required columns in {path}")

                subset = df[required_cols].copy()
                all_data.append(subset)

                # Build per-file validation summary.
                sample_val = subset[time_column].iloc[0]
                if isinstance(sample_val, str):
                    file_datetimes = []
                    for val in subset[time_column]:
                        dt_str = str(val).split(".")[0]
                        dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
                        file_datetimes.append(dt)
                else:
                    start_time = datetime(2000, 1, 1)
                    file_datetimes = [
                        start_time + timedelta(seconds=float(t))
                        for t in subset[time_column]
                    ]

                capacity_values = subset["Capacity/mA.h"].to_numpy(dtype=float)
                file_summaries.append(
                    {
                        "file": path,
                        "rows": len(subset),
                        "start": min(file_datetimes),
                        "end": max(file_datetimes),
                        "capacity_min": float(np.nanmin(capacity_values)),
                        "capacity_max": float(np.nanmax(capacity_values)),
                    }
                )
            except Exception as e:
                print(f"Skipping {path} due to error: {e}")

    if not all_data:
        raise ValueError("No valid electrochemical data found in the folder tree.")

    # Concatenate all data
    combined = pd.concat(all_data, ignore_index=True)

    # ---- Parse time column ----
    sample_val = combined[time_column].iloc[0]

    datetime_series = []
    if isinstance(sample_val, str):
        # Datetime string with optional sub-second precision (DD/MM/YYYY)
        for val in combined[time_column]:
            dt_str = str(val).split(".")[0]  # drop fractional seconds
            dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
            datetime_series.append(dt)
    else:
        # Numeric seconds
        start_time = datetime(2000, 1, 1)  # arbitrary reference
        datetime_series = [
            start_time + timedelta(seconds=float(t))
            for t in combined[time_column]
        ]

    # ---- Sort by datetime ----
    sorted_idx = np.argsort(datetime_series)

    datetime_series = [datetime_series[i] for i in sorted_idx]
    potential_series = combined["Ewe/V"].to_numpy(dtype=float)[sorted_idx]
    echem_thickness_series = (
        combined["Capacity/mA.h"].to_numpy(dtype=float)[sorted_idx]
        * conv_thickness
    )

    if report_summary:
        print("Electrochemical data summary:")
        print(f"  Loaded files: {len(file_summaries)}")
        for summary in sorted(file_summaries, key=lambda x: x["start"]):
            print(
                "  - "
                f"{os.path.basename(summary['file'])}: "
                f"{summary['start']} to {summary['end']} | "
                f"rows={summary['rows']} | "
                f"Capacity range {summary['capacity_min']:.6f} to {summary['capacity_max']:.6f} mA.h"
            )

        duplicate_timestamp_count = int(
            pd.Series(pd.to_datetime(datetime_series)).duplicated().sum()
        )
        print(
            f"  Combined range: {min(datetime_series)} to {max(datetime_series)} | "
            f"points={len(datetime_series)}"
        )
        print(f"  Duplicate timestamps: {duplicate_timestamp_count}")

        sorted_summaries = sorted(file_summaries, key=lambda x: x["start"])
        overlaps = []
        gaps = []
        for prev, curr in zip(sorted_summaries[:-1], sorted_summaries[1:]):
            if curr["start"] <= prev["end"]:
                overlaps.append((prev, curr))
            else:
                gap = curr["start"] - prev["end"]
                gaps.append((prev, curr, gap))

        if overlaps:
            print("  Overlapping file ranges detected:")
            for prev, curr in overlaps:
                print(
                    "    "
                    f"{os.path.basename(prev['file'])} overlaps "
                    f"{os.path.basename(curr['file'])}"
                )
        else:
            print("  Overlapping file ranges: none")

        if gaps:
            print("  Gaps between files:")
            for prev, curr, gap in gaps:
                print(
                    "    "
                    f"{os.path.basename(prev['file'])} -> "
                    f"{os.path.basename(curr['file'])}: {gap}"
                )
        else:
            print("  Gaps between files: none")

    return datetime_series, potential_series, echem_thickness_series



def summarize_echem_files_in_window(root_folder, window_start, window_end, time_column="time/s"):
    """
    List electrochemical files whose timestamp ranges overlap a given time window.

    Parameters
    ----------
    root_folder : str
        Root directory containing electrochemical txt files.
    window_start : datetime.datetime
        Start of the time window of interest.
    window_end : datetime.datetime
        End of the time window of interest.
    time_column : str
        Column name containing the time values.

    Returns
    -------
    list of dict
        One summary dict per overlapping file.
    """
    used_files = []

    for root, _, files in os.walk(root_folder):
        for fname in files:
            if not fname.lower().endswith(".txt"):
                continue

            path = os.path.join(root, fname)

            try:
                df = pd.read_csv(path, sep="\t", engine="python")
                required_cols = [time_column, "Ewe/V", "Capacity/mA.h"]
                if not all(col in df.columns for col in required_cols):
                    continue

                subset = df[required_cols]
                sample_val = subset[time_column].iloc[0]
                if isinstance(sample_val, str):
                    file_datetimes = []
                    for val in subset[time_column]:
                        dt_str = str(val).split(".")[0]
                        dt = datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S")
                        file_datetimes.append(dt)
                else:
                    start_time = datetime(2000, 1, 1)
                    file_datetimes = [
                        start_time + timedelta(seconds=float(t))
                        for t in subset[time_column]
                    ]

                file_start = min(file_datetimes)
                file_end = max(file_datetimes)
                overlaps_window = file_start <= window_end and file_end >= window_start
                if not overlaps_window:
                    continue

                used_files.append(
                    {
                        "file": path,
                        "rows": len(subset),
                        "start": file_start,
                        "end": file_end,
                        "points_in_window": sum(window_start <= dt <= window_end for dt in file_datetimes),
                    }
                )
            except Exception:
                continue

    used_files.sort(key=lambda x: x["start"])
    return used_files



def make_potential_image_video_uniform_time(
    echem_datetimes,
    potential_series,
    img_datetimes,
    img_series,
    output_path,
    echem_thickness_series=None,
    avg_thickness_series=None,
    plot_thickness=False,
    fps=10,
    dt_seconds=None,
    dpi=150,
    cmap="inferno",
    vmin=None,
    vmax=None,
    potential_ylim=None,
    pixel_size=55,  # Pixel size in micrometers
    scalebar_length=5000,  # Length of scale bar in micrometers
    scale_label="5 mm",  # Label for scale bar
    colorbar_label=None,  # Backward-compatible alias for cbar_label
    cbar_label="Li thickness",  # Label for colorbar
    cbar_n_ticks=5,
    # Additional scale bar customization options
    scalebar_color='white',
    scale_fontsize=12,
    scalebar_width_frac=0.015,  # Height of scale bar as fraction of image height
    scalebar_position='lower right',  # 'lower right', 'lower left', 'upper right', 'upper left'
    scalebar_margin=0.1,  # Margin from edge as fraction of image dimensions
    pause_gap_minutes=5,
    potential_color="tab:blue",
    ec_thickness_color="tab:green",
    nci_thickness_color="tab:orange",
    thickness_unit="μm",
):
    """
    Create a synchronized potential-plus-image video on a uniform time grid.

    The electrochemical trace is sampled by zero-order hold between measured
    points. Each video frame displays the nearest image in time. Optional EC
    thickness and NCI thickness traces are drawn on the same left-hand panel.
    """

    if colorbar_label is not None:
        cbar_label = colorbar_label

    if len(img_datetimes) == 0 or len(img_series) == 0:
        raise ValueError(
            "No image timestamps/images were loaded. Check the input directory and TIFF metadata."
        )

    # ------------------------------------------------------------------
    # Limit potential series to the time range covered by images
    # ------------------------------------------------------------------
    img_start = min(img_datetimes)
    img_end = max(img_datetimes)
    echem_mask = [(t >= img_start) and (t <= img_end) for t in echem_datetimes]
    echem_datetimes = [t for t, m in zip(echem_datetimes, echem_mask) if m]
    echem_mask_arr = np.array(echem_mask, dtype=bool)
    potential_series = np.asarray(potential_series)[echem_mask_arr]
    if plot_thickness:
        if echem_thickness_series is None or avg_thickness_series is None:
            raise ValueError(
                "Thickness plotting in the video requires echem_thickness_series and avg_thickness_series."
            )
        echem_thickness_series = np.asarray(echem_thickness_series)[echem_mask_arr]
        avg_thickness_series = np.asarray(avg_thickness_series)

    if len(echem_datetimes) == 0:
        raise ValueError("No electrochemical data within image time range.")

    # ------------------------------------------------------------------
    # Convert datetimes to seconds (for speed & simplicity)
    # ------------------------------------------------------------------
    t0 = min(min(echem_datetimes), min(img_datetimes))

    echem_t = np.array([(t - t0).total_seconds() for t in echem_datetimes])
    img_t = np.array([(t - t0).total_seconds() for t in img_datetimes])
    
    # Define echem_t_h HERE before it's used
    echem_t_h = echem_t / 3600.0

    paused_intervals = find_paused_intervals(
        echem_datetimes,
        min_gap_minutes=pause_gap_minutes,
    )

    # ------------------------------------------------------------------
    # Define uniform animation time grid
    # ------------------------------------------------------------------
    t_start = min(echem_t.min(), img_t.min())
    t_end = max(echem_t.max(), img_t.max())

    if dt_seconds is None:
        # sensible default: median image spacing
        dt_seconds = np.median(np.diff(img_t))

    anim_t = np.arange(t_start, t_end, dt_seconds)

    # ------------------------------------------------------------------
    # Helper lookups
    # ------------------------------------------------------------------
    def is_paused_time(t):
        current_dt = t0 + timedelta(seconds=float(t))
        return any(start < current_dt < end for start, end in paused_intervals)

    def potential_at(t):
        """Zero-order hold potential outside paused intervals."""
        if is_paused_time(t):
            return np.nan
        i = bisect_right(echem_t, t) - 1
        return potential_series[i] if i >= 0 else potential_series[0]

    def nearest_image_index(t):
        i = bisect_right(img_t, t)
        if i == 0:
            return 0
        if i == len(img_t):
            return len(img_t) - 1
        before = img_t[i - 1]
        after = img_t[i]
        return i - 1 if abs(t - before) <= abs(after - t) else i

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    potential_panel_aspect = 8.0 / 4.5
    image_panel_aspect = 1.0
    fig_height = 4.5
    fig_width = fig_height * (potential_panel_aspect + image_panel_aspect)

    fig, (ax_v, ax_img) = plt.subplots(
        1,
        2,
        figsize=(fig_width, fig_height),
        gridspec_kw={"width_ratios": [potential_panel_aspect, image_panel_aspect]},
    )

    # Potential curve (static), broken across pauses.
    echem_t_plot, potential_plot = build_series_with_pauses(
        echem_t_h,
        potential_series,
        paused_intervals,
        t0,
    )
    ax_v.plot(echem_t_plot, potential_plot, lw=1.5, color=potential_color, label="Potential (Ewe/V)")
    marker, = ax_v.plot([], [], "ro", ms=6)
    paused_text = ax_img.text(
        0.5,
        0.5,
        "paused",
        transform=ax_img.transAxes,
        color="white",
        fontsize=max(scale_fontsize, 14),
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(facecolor="black", alpha=0.45, edgecolor="none", pad=6),
        visible=False,
        zorder=11,
    )

    ax_v.set_xlabel("Time (h)")
    ax_v.set_ylabel("Potential (V)")
    ax_v.set_title("Potential evolution")
    if potential_ylim is not None:
        ax_v.set_ylim(potential_ylim)

    if plot_thickness:
        ax_v2 = ax_v.twinx()
        t_echem_thickness_plot, echem_thickness_plot = build_series_with_pauses(
            echem_t_h,
            echem_thickness_series,
            paused_intervals,
            t0,
        )
        t_img_plot, nci_thickness_plot = build_datetime_series_with_pauses(
            img_datetimes,
            avg_thickness_series,
            paused_intervals,
            t0,
        )
        ax_v2.plot(
            t_echem_thickness_plot,
            echem_thickness_plot,
            lw=1.2,
            color=ec_thickness_color,
            label="EC thickness",
        )
        ax_v2.plot(
            t_img_plot,
            nci_thickness_plot,
            lw=1.2,
            color=nci_thickness_color,
            label="NCI thickness",
        )
        ax_v2.set_ylabel(f"Thickness ({thickness_unit})")
        lines_1, labels_1 = ax_v.get_legend_handles_labels()
        lines_2, labels_2 = ax_v2.get_legend_handles_labels()
        ax_v.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax_v.legend(loc="best")

    for pause_start, pause_end in paused_intervals:
        pause_start_h = (pause_start - t0).total_seconds() / 3600.0
        pause_end_h = (pause_end - t0).total_seconds() / 3600.0
        ax_v.axvspan(pause_start_h, pause_end_h, color="tab:red", alpha=0.18)
        ylim = ax_v.get_ylim()
        y_text = ylim[1] - 0.08 * (ylim[1] - ylim[0])
        ax_v.text(
            0.5 * (pause_start_h + pause_end_h),
            y_text,
            "paused",
            color="tab:red",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="top",
        )

    # Get image dimensions for physical extent
    first_img = img_series[0]
    ny, nx = first_img.shape
    
    # Calculate physical extent in micrometers
    x_min = 0
    x_max = nx * pixel_size
    y_min = 0
    y_max = ny * pixel_size

        # Image with physical extent
    im = ax_img.imshow(
        img_series[0],
        cmap=cmap,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Remove ticks and tick labels from the image axis
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.set_xticklabels([])
    ax_img.set_yticklabels([])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    
    # Set ticks - exactly like in plot_images()!
    if vmin is not None and vmax is not None:
        ticks = np.linspace(vmin, vmax, cbar_n_ticks)
        cbar.set_ticks(ticks)
        
        # Simple tick labels without units (units are in the colorbar label)
        labels = []
        for t in ticks:
            if np.isclose(t, 0):
                labels.append("0")
            else:
                labels.append(f"{t:g}")
        cbar.set_ticklabels(labels)
    
    # ------------------------------------------------------------------
    # Create scale bar parameters (but don't add it yet)
    # ------------------------------------------------------------------
    
    # Determine scale bar position based on user preference
    if scalebar_position == 'lower right':
        scale_x = x_max - scalebar_length - scalebar_margin * (x_max - x_min)
        scale_y = y_min + scalebar_margin * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    elif scalebar_position == 'lower left':
        scale_x = x_min + scalebar_margin * (x_max - x_min)
        scale_y = y_min + scalebar_margin * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    elif scalebar_position == 'upper right':
        scale_x = x_max - scalebar_length - scalebar_margin * (x_max - x_min)
        scale_y = y_max - scalebar_margin * (y_max - y_min) - scalebar_width_frac * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    elif scalebar_position == 'upper left':
        scale_x = x_min + scalebar_margin * (x_max - x_min)
        scale_y = y_max - scalebar_margin * (y_max - y_min) - scalebar_width_frac * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    else:
        # Default to lower right
        scale_x = x_max - scalebar_length - scalebar_margin * (x_max - x_min)
        scale_y = y_min + scalebar_margin * (y_max - y_min)
        text_ha = 'center'
        text_x_offset = scalebar_length / 2
    
    # Calculate scale bar height in physical units (micrometers)
    scalebar_height = scalebar_width_frac * (y_max - y_min)
    
    # Create scale bar rectangle (we'll add it in the animation loop)
    scalebar_rect = patches.Rectangle(
        (scale_x, scale_y),
        scalebar_length,
        scalebar_height,
        facecolor=scalebar_color,
        edgecolor='none',
        alpha=0.9,
        zorder=10  # High zorder to appear on top of image
    )

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=dpi):
        for t in anim_t:

            # Potential (ZOH)
            v = potential_at(t)
            if np.isfinite(v):
                marker.set_data([t / 3600.0], [v])
            else:
                marker.set_data([], [])

            # Image (nearest)
            j = nearest_image_index(t)
            im.set_data(img_series[j])
            paused_text.set_visible(is_paused_time(t))

            # Remove previous scale bar rectangles and text
            for artist in ax_img.patches[:]:
                if isinstance(artist, patches.Rectangle) and hasattr(artist, 'get_width') and artist.get_width() == scalebar_length:
                    artist.remove()
            for artist in ax_img.texts[:]:
                if scale_label in artist.get_text():
                    artist.remove()
            
            # Add fresh scale bar rectangle
            ax_img.add_patch(scalebar_rect)
            
            # Add fresh scale bar label
            ax_img.text(
                scale_x + text_x_offset,
                scale_y + scalebar_height * 1.5,  # Position above the scale bar
                scale_label,
                color=scalebar_color,
                ha=text_ha,
                va='bottom',
                fontsize=scale_fontsize,
                fontweight='bold',
                zorder=10  # High zorder to appear on top of image
            )

            ax_img.set_title(
                f"Image @ {img_datetimes[j].strftime('%Y-%m-%d %H:%M:%S')}"
            )

            writer.grab_frame()

    plt.close(fig)


    
def merge_images_by_time(
    img_datetimes,
    img_series,
    merge_minutes=1,
    method="mean",
):
    """
    Merge images into longer effective acquisition windows.

    Parameters
    ----------
    img_datetimes : list of datetime.datetime
    img_series : list of 2D np.ndarray
    merge_minutes : float
        Duration of each merged frame in minutes
    method : str
        'mean' or 'sum'

    Returns
    -------
    merged_datetimes : list of datetime.datetime
        Timestamp at center of each merged window
    merged_images : list of 2D np.ndarray
    """

    if merge_minutes <= 1:
        return img_datetimes, img_series

    dt = timedelta(minutes=merge_minutes)

    merged_images = []
    merged_datetimes = []

    window_start = img_datetimes[0]
    buffer = []

    for t, img in zip(img_datetimes, img_series):

        if t < window_start + dt:
            buffer.append(img)
        else:
            stack = np.stack(buffer, axis=0)
            merged = stack.mean(axis=0) if method == "mean" else stack.sum(axis=0)

            merged_images.append(merged)
            merged_datetimes.append(window_start + dt / 2)

            # reset window
            window_start = t
            buffer = [img]

    # flush remainder
    if buffer:
        stack = np.stack(buffer, axis=0)
        merged = stack.mean(axis=0) if method == "mean" else stack.sum(axis=0)
        merged_images.append(merged)
        merged_datetimes.append(window_start + dt / 2)

    return merged_datetimes, merged_images




def circular_average_thickness_series(img_series, x0=None, y0=None, r0=None):
    """
    Compute the average pixel value within a circular region
    for each image in an image series. Silently rejects infinite and NaN values.

    Parameters
    ----------
    img_series : list or np.ndarray
        List or array of 2D images
    x0 : int, optional
        X index of circle center
    y0 : int, optional
        Y index of circle center
    r0 : int, optional
        Radius of the circular region (in pixels)

    Returns
    -------
    avg_thickness_series : np.ndarray
        Average value inside the circular region for each image
        (returns NaN for images with no valid pixels in ROI)
    """

    avg_thickness_series = []

    for img in img_series:
        ny, nx = img.shape

        # ---- Default ROI: center of image, maximal inscribed circle ----
        if x0 is None or y0 is None or r0 is None:
            cx = nx // 2
            cy = ny // 2
            radius = min(nx, ny) // 2
        else:
            cx = int(x0)
            cy = int(y0)
            radius = int(r0)

        # ---- Create circular mask ----
        y, x = np.ogrid[:ny, :nx]
        mask = (x - cx)**2 + (y - cy)**2 <= radius**2
        
        # ---- Extract values in ROI and handle invalid values ----
        roi_values = img[mask].astype(float)
        
        # Replace inf with nan, then compute the mean only if any valid pixels remain.
        roi_values[~np.isfinite(roi_values)] = np.nan

        if np.any(np.isfinite(roi_values)):
            avg = np.nanmean(roi_values)
        else:
            avg = np.nan
        avg_thickness_series.append(avg)

    return np.asarray(avg_thickness_series)



def load_corrected_images_with_time(corrected_dir, tag_id=65002, r0=None, x0=None, y0=None):
    """
    Load corrected or final TIFF images and recover timestamps from TIFF metadata.

    Parameters
    ----------
    corrected_dir : str
        Directory containing corrected TIFF images.
    tag_id : int, optional
        Preferred TIFF tag ID containing the timestamp string. If it is missing,
        the loader falls back to tag 65000 and then to the TIFF description text.
    r0 : int, optional
        Half side length of the square crop region.
    x0 : int, optional
        X index of the crop center.
    y0 : int, optional
        Y index of the crop center.

    Returns
    -------
    img_datetimes : list of datetime.datetime
        Image timestamps sorted in ascending order.
    img_series : list of np.ndarray
        Loaded image arrays, cropped if ROI arguments are provided.
    paths : list of str
        Source file paths corresponding to each loaded image.
    """
    records = []
    for fname in sorted(os.listdir(corrected_dir)):
        path = os.path.join(corrected_dir, fname)
        if not os.path.isfile(path):
            continue
        # Try to open as TIFF (works even if no extension)
        try:
            with tifffile.TiffFile(path) as tif:
                img = tif.asarray()
                if r0 is not None and x0 is not None and y0 is not None:
                    y_start = max(0, y0 - r0)
                    y_end = min(img.shape[0], y0 + r0)
                    x_start = max(0, x0 - r0)
                    x_end = min(img.shape[1], x0 + r0)
                    img = img[y_start:y_end, x_start:x_end]
                tags = tif.pages[0].tags
                dt_str = None

                for candidate_tag in (tag_id, 65000):
                    if candidate_tag in tags:
                        dt_str = str(tags[candidate_tag].value)
                        break

                if dt_str is None and 270 in tags:
                    description = str(tags[270].value)
                    for line in description.splitlines():
                        if line.startswith("MeanTPXModifiedTime:"):
                            dt_str = line.split(":", 1)[1].strip()
                            break
                        if line.startswith("AcquisitionTime:"):
                            dt_str = line.split(":", 1)[1].strip()
                            break

                if dt_str is None:
                    continue
        except Exception:
            continue

        # Parse datetime
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        except Exception:
            continue

        records.append((dt, img, path))

    records.sort(key=lambda x: x[0])
    img_datetimes = [r[0] for r in records]
    img_series = [r[1] for r in records]
    paths = [r[2] for r in records]
    return img_datetimes, img_series, paths


def build_turning_point_event_context(
    echem_datetimes,
    echem_thickness_series,
    img_datetimes,
    event_finder,
    image_index_finder,
    conv_thickness=1.0,
    n_tpx_per_image=30,
    tpx_time=2,
    initial_ocv_image_count=None,
):
    """
    Build a matched event context between electrochemical turning points and images.

    Parameters
    ----------
    echem_datetimes : sequence of datetime.datetime
    echem_thickness_series : sequence of float
    img_datetimes : sequence of datetime.datetime
    event_finder : callable
        Function like ``find_capacity_turning_points(times, capacity_series=values)``.
    image_index_finder : callable
        Function like ``find_nearest_image_indices(event_times, img_datetimes)``.
    conv_thickness : float, optional
        Divisor applied to the electrochemical thickness series before event detection.
    n_tpx_per_image : int, optional
        Number of TPX files grouped into one image. Used for far-match warnings.
    tpx_time : int or float, optional
        Duration of each TPX exposure in seconds. Used for far-match warnings.
    initial_ocv_image_count : int, optional
        If provided and positive, prepend an ``ocv_end`` event corresponding to the
        last image used to build the background image, i.e. ``initial_ocv_image_count - 1``.

    Returns
    -------
    dict
        Event context with keys ``events``, ``event_times``, ``img_indices``,
        ``event_times_for_plot``, ``matched_count``, ``total_events_found``,
        ``img_start``, ``img_end``, ``match_warning_threshold_seconds``, and
        ``far_match_warnings``.
    """
    if len(img_datetimes) == 0:
        raise ValueError("No image timestamps found. Check corrected image metadata.")

    img_start = min(img_datetimes)
    img_end = max(img_datetimes)

    cap_times = []
    cap_values = []
    for t, c in zip(echem_datetimes, echem_thickness_series):
        if img_start <= t <= img_end:
            cap_times.append(t)
            cap_values.append(c / conv_thickness)

    event_context = {
        "events": [],
        "event_times": [],
        "img_indices": [],
        "event_times_for_plot": [],
        "matched_count": 0,
        "total_events_found": 0,
        "match_warning_threshold_seconds": float(2 * n_tpx_per_image * tpx_time),
        "far_match_warnings": [],
        "img_start": img_start,
        "img_end": img_end,
    }

    if len(cap_times) == 0:
        return event_context

    extra_events = []
    if initial_ocv_image_count is not None and initial_ocv_image_count > 0:
        ocv_img_idx = min(int(initial_ocv_image_count) - 1, len(img_datetimes) - 1)
        ocv_img_time = img_datetimes[ocv_img_idx]
        if len(cap_times) > 0:
            echem_time_diffs = np.array([abs((t - ocv_img_time).total_seconds()) for t in cap_times])
            nearest_echem_idx = int(np.argmin(echem_time_diffs))
            extra_events.append({
                "index": nearest_echem_idx,
                "time": cap_times[nearest_echem_idx],
                "labels": ["ocv_end"],
            })
        else:
            extra_events.append({
                "index": None,
                "time": ocv_img_time,
                "labels": ["ocv_end"],
            })

    try:
        all_events = event_finder(cap_times, capacity_series=cap_values, extra_events=extra_events)
    except TypeError:
        all_events = event_finder(cap_times, capacity_series=cap_values)

    all_event_times = [e["time"] for e in all_events]
    matched_img_indices = image_index_finder(all_event_times, img_datetimes)

    matched_count = min(len(all_events), len(matched_img_indices))
    matched_events = all_events[:matched_count]
    matched_img_indices = matched_img_indices[:matched_count]
    match_warning_threshold_seconds = float(2 * n_tpx_per_image * tpx_time)

    far_match_warnings = []
    for i, (event, idx) in enumerate(zip(matched_events, matched_img_indices)):
        image_time = img_datetimes[idx]
        time_diff_seconds = abs((image_time - event["time"]).total_seconds())
        if time_diff_seconds > match_warning_threshold_seconds:
            label = chr(97 + i) if i < 26 else str(i)
            far_match_warnings.append(
                {
                    "event_index": i,
                    "event_label": label,
                    "event_time": event["time"],
                    "matched_image_index": idx,
                    "matched_image_time": image_time,
                    "time_diff_seconds": time_diff_seconds,
                }
            )

    if far_match_warnings:
        print(
            "WARNING: Some turning-point/image matches are farther apart than the "
            f"warning threshold of {match_warning_threshold_seconds:.1f} s "
            f"(2 * n_tpx_per_image * tpx_time = 2 * {n_tpx_per_image} * {tpx_time})."
        )
        for warning in far_match_warnings:
            print(
                "  "
                f"{warning['event_label']} (event {warning['event_index']}): "
                f"turning point {warning['event_time'].strftime('%Y-%m-%d %H:%M:%S')} -> "
                f"image idx {warning['matched_image_index']} at "
                f"{warning['matched_image_time'].strftime('%Y-%m-%d %H:%M:%S')} "
                f"({warning['time_diff_seconds']:.1f} s)"
            )

    deduped_events = []
    deduped_img_indices = []
    deduped_event_times = []
    deduped_event_times_for_plot = []
    img_position_by_idx = {}

    for event, idx in zip(matched_events, matched_img_indices):
        if idx in img_position_by_idx:
            pos = img_position_by_idx[idx]
            existing_labels = set(deduped_events[pos].get("labels", []))
            new_labels = set(event.get("labels", []))
            merged_labels = existing_labels | new_labels
            if "ocv_end" in merged_labels and "turning" in merged_labels:
                ordered_labels = ["ocv_end", "turning"] + sorted(
                    label for label in merged_labels if label not in {"ocv_end", "turning"}
                )
            elif "ocv_end" in merged_labels:
                ordered_labels = ["ocv_end"] + sorted(label for label in merged_labels if label != "ocv_end")
            else:
                ordered_labels = sorted(merged_labels)
            deduped_events[pos]["labels"] = ordered_labels
        else:
            img_position_by_idx[idx] = len(deduped_events)
            deduped_events.append(event)
            deduped_img_indices.append(idx)
            deduped_event_times.append(event["time"])
            deduped_event_times_for_plot.append(img_datetimes[idx])

    event_context["events"] = deduped_events
    event_context["event_times"] = deduped_event_times
    event_context["img_indices"] = deduped_img_indices
    event_context["event_times_for_plot"] = deduped_event_times_for_plot
    event_context["matched_count"] = matched_count
    event_context["total_events_found"] = len(all_events)
    event_context["match_warning_threshold_seconds"] = match_warning_threshold_seconds
    event_context["far_match_warnings"] = far_match_warnings

    return event_context


def print_event_context_table(event_context, img_datetimes):
    """
    Print a compact table of detected turning points and matched image times.
    """
    if not event_context["events"]:
        print("No capacity events found.")
        return

    if event_context["matched_count"] < event_context["total_events_found"]:
        print(
            f"Matched {event_context['matched_count']} image-linked events out of "
            f"{event_context['total_events_found']} turning points. "
            "Only matched events will be used below."
        )

    print("All detected turning points and matched images:")
    print("idx | turning point time       | image idx | matched image time")
    print("-" * 72)
    for i, (event, idx) in enumerate(zip(event_context["events"], event_context["img_indices"])):
        image_time = img_datetimes[idx].strftime("%Y-%m-%d %H:%M:%S")
        event_time = event["time"].strftime("%Y-%m-%d %H:%M:%S")
        print(f"{i:>3} | {event_time} | {idx:>9} | {image_time}")

    print(
        f"Detected {len(event_context['img_indices'])} event-linked images "
        "from all detected turning points."
    )


def format_event_context_preview_text(event_context, img_datetimes, label_start=0):
    """
    Format detected turning-point/image matches for display below the EC preview.
    """
    if not event_context["events"]:
        return "No detected turning points available for preview."

    lines = []
    if event_context["matched_count"] < event_context["total_events_found"]:
        lines.append(
            f"Matched {event_context['matched_count']} image-linked events out of "
            f"{event_context['total_events_found']} turning points."
        )
        lines.append("Only matched events are shown below.")
        lines.append("")

    lines.append(f"Detected image indices: {event_context['img_indices']}")
    lines.append("")
    lines.append("All detected turning points and matched images:")
    lines.append("lbl | turning point time   | image idx | matched image time")
    lines.append("-" * 62)

    for i, (event, idx) in enumerate(zip(event_context["events"], event_context["img_indices"])):
        image_time = img_datetimes[idx].strftime("%Y-%m-%d %H:%M:%S")
        event_time = event["time"].strftime("%Y-%m-%d %H:%M:%S")
        label = chr(97 + int(label_start) + i)
        lines.append(f" {label}  | {event_time} | {idx:>9} | {image_time}")

    lines.append("")
    lines.append(
        f"Detected {len(event_context['img_indices'])} event-linked images from all detected turning points."
    )
    return "\n".join(lines)


def print_detected_image_indices_for_copy(event_context):
    """
    Print detected image indices in a copy/paste-friendly format.
    """
    detected_image_indices = event_context["img_indices"]
    if not detected_image_indices:
        print("No detected turning points available for preview.")
        return

    print(f"Detected image indices for copy/paste: {detected_image_indices}")


def normalize_selected_image_indices(selected_image_indices, valid_image_count):
    """
    Validate and deduplicate selected image indices while preserving order.
    """
    selected_image_indices = list(selected_image_indices)
    invalid_indices = [i for i in selected_image_indices if i < 0 or i >= valid_image_count]
    if invalid_indices:
        raise ValueError(
            f"selected_image_indices contains out-of-range values {invalid_indices}. "
            f"Valid image indices are 0 to {valid_image_count - 1}."
        )

    deduped_selected_image_indices = []
    for i in selected_image_indices:
        if i not in deduped_selected_image_indices:
            deduped_selected_image_indices.append(i)

    return deduped_selected_image_indices


def auto_event_panel_figsize(selected_count, cbar_mode=None):
    """
    Choose a sensible image-panel figure size from the number of selected images.

    This is only used when the caller does not provide an explicit ``figsize``.
    """
    if selected_count <= 0:
        return (8.0, 6.0)

    if selected_count <= 2:
        ncols = selected_count
    elif selected_count <= 4:
        ncols = 2
    elif selected_count <= 9:
        ncols = 3
    elif selected_count <= 16:
        ncols = 4
    else:
        ncols = 5

    nrows = int(np.ceil(selected_count / ncols))

    panel_width = 3.25
    panel_height = 3.2
    width = 0.8 + ncols * panel_width
    height = 0.8 + nrows * panel_height

    if cbar_mode and "shared" in str(cbar_mode):
        width += 1.2

    return (round(width, 2), round(height, 2))


def save_selected_event_outputs(
    event_context,
    selected_image_indices,
    img_datetimes,
    img_series,
    avg_thickness_series,
    echem_datetimes,
    potential_series,
    echem_thickness_series,
    output_plot_dir,
    exp_name,
    roi_small,
    event_plot_kwargs,
    plot_images_func,
    potential_plot_kwargs=None,
    label_start=0,
    event_marker_size=6,
    event_label_fontsize=18,
):
    """
    Save final selected-image outputs: summary text, EC plot, and image panel.

    Returns
    -------
    dict
        Includes normalized image indices, selected plot times, and saved paths.
    """
    normalized_image_indices = normalize_selected_image_indices(
        selected_image_indices,
        len(img_series),
    )

    if len(normalized_image_indices) != len(list(selected_image_indices)):
        print(
            f"Removed duplicate selected image indices. Using {normalized_image_indices} "
            f"instead of {list(selected_image_indices)}."
        )

    if not normalized_image_indices:
        print("selected_image_indices is empty. No final plots or summary files were saved.")
        return {
            "selected_image_indices": [],
            "selected_plot_times": [],
            "saved_paths": {},
        }

    selected_plot_times = [img_datetimes[idx] for idx in normalized_image_indices]
    detected_event_pos_by_image_idx = {
        image_idx: pos for pos, image_idx in enumerate(event_context.get("img_indices", []))
    }

    print(f"Saving final outputs for selected image indices: {normalized_image_indices}")

    out_text_path = os.path.join(output_plot_dir, f"selected_image_summary_{exp_name}.txt")

    def print_and_save(message, file_handle):
        print(message)
        file_handle.write(message + "\n")

    with open(out_text_path, "w") as f:
        print_and_save("\n" + "=" * 60, f)
        print_and_save("SELECTED IMAGE SUMMARY", f)
        print_and_save("=" * 60, f)

        thickness_values = []
        for j, (image_idx, plot_time) in enumerate(
            zip(normalized_image_indices, selected_plot_times)
        ):
            thickness_value = avg_thickness_series[image_idx]
            thickness_values.append(thickness_value)
            event_label = chr(97 + int(label_start) + j)
            detected_event_pos = detected_event_pos_by_image_idx.get(image_idx)
            image_time = img_datetimes[image_idx]
            closest_time_str = image_time.strftime("%Y-%m-%d %H:%M:%S")
            plot_time_str = plot_time.strftime("%Y-%m-%d %H:%M:%S")

            print_and_save(f"Image {event_label} (image index {image_idx}):", f)
            if detected_event_pos is not None:
                turning_time = event_context["event_times"][detected_event_pos]
                turning_time_str = turning_time.strftime("%Y-%m-%d %H:%M:%S")
                turning_to_image_seconds = abs((image_time - turning_time).total_seconds())
                print_and_save(
                    f"  Detected event:  yes (detected image index list position {detected_event_pos})",
                    f,
                )
                print_and_save(f"  Turning point:   {turning_time_str}", f)
                print_and_save(
                    f"  Turning point to image: {turning_to_image_seconds:.1f} seconds",
                    f,
                )
            else:
                print_and_save("  Detected event:  no (manually selected image)", f)
                print_and_save("  Turning point:   n/a", f)
            print_and_save(f"  Plot time:       {plot_time_str}", f)
            print_and_save(f"  Image time:      {closest_time_str}", f)
            print_and_save(f"  Avg thickness:   {thickness_value:.4f}", f)
            print_and_save(f"  Image index:     {image_idx}", f)
            print_and_save("", f)

        print_and_save("-" * 60, f)
        print_and_save("SUMMARY:", f)
        print_and_save(
            f"  ROI: center=({roi_small['x0']}, {roi_small['y0']}), radius={roi_small['r0']} pixels",
            f,
        )
        print_and_save(f"  Total detected turning-point images: {len(event_context['img_indices'])}", f)
        print_and_save(f"  Selected image indices: {normalized_image_indices}", f)
        print_and_save(f"  Number of selected images: {len(normalized_image_indices)}", f)
        if thickness_values:
            print_and_save(
                f"  Thickness range:  {np.nanmin(thickness_values):.4f} to {np.nanmax(thickness_values):.4f}",
                f,
            )
            print_and_save(f"  Mean thickness:   {np.nanmean(thickness_values):.4f}", f)
        print_and_save("=" * 60 + "\n", f)

    print(f"Text output saved to: {out_text_path}")

    potential_plot_kwargs = dict(potential_plot_kwargs or {})

    fig = plot_potential_and_thickness(
        echem_datetime_series=echem_datetimes,
        potential_series=potential_series,
        echem_thickness_series=echem_thickness_series,
        img_datetime_series=img_datetimes,
        avg_thickness_series=avg_thickness_series,
        event_times=selected_plot_times,
        label_start=label_start,
        event_marker_size=event_marker_size,
        event_label_fontsize=event_label_fontsize,
        **potential_plot_kwargs,
    )

    potential_plot_path = os.path.join(output_plot_dir, f"potential_{exp_name}.png")
    fig.savefig(potential_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Image saved to: {potential_plot_path}")

    event_plot_kwargs = dict(event_plot_kwargs or {})
    if event_plot_kwargs.get("figsize") is None:
        event_plot_kwargs["figsize"] = auto_event_panel_figsize(
            len(normalized_image_indices),
            cbar_mode=event_plot_kwargs.get("cbar_mode"),
        )

    fig, axes = plot_images_func(
        img_series=img_series,
        img_indices=normalized_image_indices,
        label_start=label_start,
        **event_plot_kwargs,
    )

    selected_event_plot_path = os.path.join(output_plot_dir, f"nci_{exp_name}.png")
    fig.savefig(selected_event_plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved: {selected_event_plot_path}")

    return {
        "selected_image_indices": normalized_image_indices,
        "selected_plot_times": selected_plot_times,
        "saved_paths": {
            "summary": out_text_path,
            "potential": potential_plot_path,
            "images": selected_event_plot_path,
        },
    }



def plot_potential_and_thickness(
    echem_datetime_series,
    potential_series,
    echem_thickness_series=None,
    img_datetime_series=None,
    avg_thickness_series=None,
    plot_thickness=False,
    potential_ylim=None,
    event_times=None,
    label_start=0,
    event_marker_size=3,
    event_label_fontsize=10,
    show=True,
    pause_gap_minutes=5,
    potential_color="tab:blue",
    ec_thickness_color="tab:green",
    nci_thickness_color="tab:orange",
    thickness_unit="μm",
    side_text=None,
    side_text_fontsize=8,
):
    """
    Plot potential and, optionally, EC and NCI thickness on a common time axis.

    The plot is restricted to the image time window when image timestamps are
    supplied. Paused electrochemical intervals are detected from absolute gaps
    in the EC timestamps and shown as shaded regions. Optional event labels are
    added to the potential trace using the provided ``event_times``.
    """

    # ---- Convert to numpy arrays ----
    echem_datetime_series = np.asarray(echem_datetime_series)
    img_datetime_series = np.asarray(img_datetime_series)

    potential_series = np.asarray(potential_series)
    if plot_thickness:
        if echem_thickness_series is None or img_datetime_series is None or avg_thickness_series is None:
            raise ValueError("Thickness plotting requires echem_thickness_series, img_datetime_series, and avg_thickness_series.")
        echem_thickness_series = np.asarray(echem_thickness_series)
        avg_thickness_series = np.asarray(avg_thickness_series)

    # ---- Determine overlapping time window ----
    if img_datetime_series is not None and len(img_datetime_series) > 0:
        t_start = img_datetime_series.min()
        t_end = img_datetime_series.max()
        # ---- Mask electrochemical data to image time window ----
        echem_mask = (echem_datetime_series >= t_start) & (echem_datetime_series <= t_end)
        echem_datetime_series = echem_datetime_series[echem_mask]
        potential_series = potential_series[echem_mask]
        if plot_thickness:
            echem_thickness_series = echem_thickness_series[echem_mask]
    else:
        t_start = echem_datetime_series.min()
        t_end = echem_datetime_series.max()

    # ---- Define common time origin ----
    t0 = t_start

    # ---- Convert datetimes to elapsed time (hours) ----
    t_echem = np.array([(t - t0).total_seconds() for t in echem_datetime_series]) / 3600.0
    t_img = np.array([(t - t0).total_seconds() for t in img_datetime_series]) / 3600.0
    paused_intervals = find_paused_intervals(
        echem_datetime_series,
        min_gap_minutes=pause_gap_minutes,
    )

    # ---- Plot ----
    if plot_thickness:
        if side_text:
            fig = plt.figure(figsize=(16, 8.5))
            gs = fig.add_gridspec(
                2,
                2,
                width_ratios=[2.2, 1.8],
                height_ratios=[1, 1],
                hspace=0.10,
                wspace=0.20,
            )
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
            ax_text = fig.add_subplot(gs[:, 1])
            ax_text.axis("off")
            ax_text.text(
                0.0,
                1.0,
                side_text,
                transform=ax_text.transAxes,
                va="top",
                ha="left",
                fontsize=side_text_fontsize,
                family="monospace",
            )
        else:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(8, 6.2),
                sharex=True,
                gridspec_kw={"height_ratios": [1.0, 1.0], "hspace": 0.08},
            )
    else:
        if side_text:
            fig, (ax1, ax_text) = plt.subplots(
                1,
                2,
                figsize=(16, 8.5),
                gridspec_kw={"width_ratios": [2.2, 1.8]},
            )
            ax_text.axis("off")
            ax_text.text(
                0.0,
                1.0,
                side_text,
                transform=ax_text.transAxes,
                va="top",
                ha="left",
                fontsize=side_text_fontsize,
                family="monospace",
            )
        else:
            fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Potential (left axis), broken across paused intervals.
    t_echem_plot, potential_plot = build_series_with_pauses(
        t_echem,
        potential_series,
        paused_intervals,
        t0,
    )
    ax1.plot(t_echem_plot, potential_plot, label="Potential (Ewe/V)", color=potential_color)
    ax1.set_ylabel("Potential (V)")
    ax1.tick_params(axis="y")
    ax1.set_title("Potential and Thickness Evolution" if plot_thickness else "Potential evolution")
    if potential_ylim is not None:
        ax1.set_ylim(potential_ylim)

    for pause_start, pause_end in paused_intervals:
        pause_start_h = (pause_start - t0).total_seconds() / 3600.0
        pause_end_h = (pause_end - t0).total_seconds() / 3600.0
        ax1.axvspan(pause_start_h, pause_end_h, color="tab:red", alpha=0.18)
        ylim = ax1.get_ylim()
        y_text = ylim[1] - 0.08 * (ylim[1] - ylim[0])
        ax1.text(
            0.5 * (pause_start_h + pause_end_h),
            y_text,
            "paused",
            color="tab:red",
            fontsize=10,
            fontweight="bold",
            ha="center",
            va="top",
        )
        if plot_thickness:
            ax2.axvspan(pause_start_h, pause_end_h, color="tab:red", alpha=0.18)

    # Optional event markers/labels on the potential trace.
    if event_times is not None and len(event_times) > 0:
        for j, raw_time in enumerate(event_times):
            event_dt = raw_time
            if isinstance(raw_time, str):
                event_dt = pd.to_datetime(raw_time).to_pydatetime()

            if event_dt < echem_datetime_series.min() or event_dt > echem_datetime_series.max():
                continue

            dt_seconds = np.array([abs((t - event_dt).total_seconds()) for t in echem_datetime_series])
            idx = int(np.argmin(dt_seconds))

            label = chr(97 + int(label_start) + j)
            ax1.plot(t_echem[idx], potential_series[idx], "ko", ms=event_marker_size)
            ax1.annotate(
                label,
                (t_echem[idx], potential_series[idx]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=event_label_fontsize,
                fontweight="bold",
            )


    # Thickness (bottom panel when enabled)
    if plot_thickness:
        t_echem_thickness_plot, echem_thickness_plot = build_series_with_pauses(
            t_echem,
            echem_thickness_series,
            paused_intervals,
            t0,
        )
        t_img_plot, nci_thickness_plot = build_datetime_series_with_pauses(
            img_datetime_series,
            avg_thickness_series,
            paused_intervals,
            t0,
        )
        ax2.plot(
            t_echem_thickness_plot,
            echem_thickness_plot,
            label="EC thickness",
            color=ec_thickness_color,
            lw=1.8,
        )
        ax2.plot(
            t_img_plot,
            nci_thickness_plot,
            label="NCI thickness",
            color=nci_thickness_color,
            lw=1.8,
        )
        ax2.set_ylabel(f"Thickness ({thickness_unit})")
        ax2.set_xlabel("Time elapsed (h)")
        ax2.legend(loc="best")
        ax1.legend(loc="best")
        plt.setp(ax1.get_xticklabels(), visible=False)

    else:
        ax1.set_xlabel("Time elapsed (h)")
        ax1.legend(loc="best")
    
    plt.tight_layout()
    if show:
        plt.show()
    return fig

    
    
    
    
    

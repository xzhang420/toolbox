import os
from datetime import datetime
import numpy as np
import tifffile
from PIL import Image
from astropy.io import fits

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




import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def load_echem_data(root_folder, conv_thickness=1.0, time_column="time/s"):
    """
    Recursively load all electrochemical txt files in a folder tree,
    parse important columns, calculate thickness, and return series.

    Supports time either as seconds ('time/s') or as datetime strings
    with optional sub-second precision.

    Parameters
    ----------
    root_folder : str
        Root directory containing the electrochemical txt files
    conv_thickness : float
        Factor to multiply Capacity (mA.h) to convert to thickness
    time_column : str
        Column name containing the time (seconds or datetime string)

    Returns
    -------
    datetime_series : list of datetime.datetime
        Time series as datetime objects
    voltage_series : np.ndarray
        Ewe/V series
    echem_thickness_series : np.ndarray
        Thickness series calculated from capacity
    """

    all_data = []

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

                all_data.append(df[required_cols])
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
    voltage_series = combined["Ewe/V"].to_numpy(dtype=float)[sorted_idx]
    echem_thickness_series = (
        combined["Capacity/mA.h"].to_numpy(dtype=float)[sorted_idx]
        * conv_thickness
    )

    return datetime_series, voltage_series, echem_thickness_series








import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from bisect import bisect_right
from datetime import timedelta


def make_voltage_image_video_uniform_time(
    echem_datetimes,
    voltage_series,
    img_datetimes,
    img_series,
    output_path,
    fps=10,
    dt_seconds=None,
    dpi=150,
    cmap="inferno",
    vmin=None,
    vmax=None,
):
    """
    Create a synchronized voltage–image video using a uniform global time axis.

    Voltage: zero-order hold (previous value)
    Image: nearest in time

    Parameters
    ----------
    echem_datetimes : list of datetime.datetime
    voltage_series : np.ndarray
    img_datetimes : list of datetime.datetime
    img_series : list of 2D np.ndarray
    output_path : str
        Output video filename (mp4)
    fps : int
        Frames per second
    dt_seconds : float or None
        Time step of animation grid.
        If None, uses median image timestep.
    dpi : int
        Video DPI
    cmap : str
        Image colormap
    vmin, vmax : float or None
        Fixed image color scale
    """

    # ------------------------------------------------------------------
    # Convert datetimes to seconds (for speed & simplicity)
    # ------------------------------------------------------------------
    t0 = min(min(echem_datetimes), min(img_datetimes))

    echem_t = np.array([(t - t0).total_seconds() for t in echem_datetimes])
    img_t = np.array([(t - t0).total_seconds() for t in img_datetimes])

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
    def voltage_at(t):
        """Zero-order hold voltage."""
        i = bisect_right(echem_t, t) - 1
        return voltage_series[i] if i >= 0 else voltage_series[0]

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
    fig, (ax_v, ax_img) = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # Voltage curve (static)
    ax_v.plot(echem_t, voltage_series, lw=1.5)
    marker, = ax_v.plot([], [], "ro", ms=6)

    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_title("Voltage evolution")

    # Image
    im = ax_img.imshow(
        img_series[0],
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax_img.set_xticks([])
    ax_img.set_yticks([])

    plt.tight_layout()

    # ------------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------------
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=dpi):
        for t in anim_t:

            # Voltage (ZOH)
            v = voltage_at(t)
            marker.set_data(t, v)

            # Image (nearest)
            j = nearest_image_index(t)
            im.set_data(img_series[j])

            ax_img.set_title(
                f"Image @ {img_datetimes[j].strftime('%Y-%m-%d %H:%M:%S')}"
            )

            writer.grab_frame()

    plt.close(fig)





import numpy as np
from datetime import timedelta


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








import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from bisect import bisect_right


def make_echem_neutron_video(
    echem_folder,
    image_folder,
    output_path,
    *,
    conv_thickness=1.0,
    merge_minutes=1,
    merge_method="mean",
    fps=10,
    dt_seconds=None,
    cmap="inferno",
    vmin=None,
    vmax=None,
    dpi=150,
    image_kwargs=None,
):
    """
    Load electrochemical + neutron image data and create a synchronized video.

    Parameters
    ----------
    echem_folder : str
        Folder containing electrochemical txt files
    image_folder : str
        Folder containing neutron images
    output_path : str
        Output video filename (.mp4)
    conv_thickness : float
        Capacity → thickness conversion (passed to loader)
    merge_minutes : float
        Merge images into longer acquisition windows (minutes)
    merge_method : str
        'mean' or 'sum'
    fps : int
        Video frames per second
    dt_seconds : float or None
        Uniform animation timestep (seconds)
    cmap : str
        Image colormap
    vmin, vmax : float or None
        Fixed image color scale
    dpi : int
        Video DPI
    image_kwargs : dict or None
        Passed to load_images (ROI cropping, etc.)
    """

    image_kwargs = image_kwargs or {}

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    img_dt, img_series, _ = load_images(image_folder, **image_kwargs)
    echem_dt, voltage, _ = load_echem_data(
        echem_folder,
        conv_thickness=conv_thickness,
    )

    # ------------------------------------------------------------
    # Merge images in time
    # ------------------------------------------------------------
    img_dt, img_series = merge_images_by_time(
        img_dt,
        img_series,
        merge_minutes=merge_minutes,
        method=merge_method,
    )

    # ------------------------------------------------------------
    # Convert times to seconds
    # ------------------------------------------------------------
    t0 = min(min(echem_dt), min(img_dt))

    echem_t = np.array([(t - t0).total_seconds() for t in echem_dt])
    img_t = np.array([(t - t0).total_seconds() for t in img_dt])

    if dt_seconds is None:
        dt_seconds = np.median(np.diff(img_t))

    t_anim = np.arange(
        min(echem_t.min(), img_t.min()),
        max(echem_t.max(), img_t.max()),
        dt_seconds,
    )

    # ------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------
    def voltage_at(t):
        i = bisect_right(echem_t, t) - 1
        return voltage[i] if i >= 0 else voltage[0]

    def nearest_image_index(t):
        i = bisect_right(img_t, t)
        if i == 0:
            return 0
        if i == len(img_t):
            return len(img_t) - 1
        return i - 1 if abs(t - img_t[i - 1]) <= abs(img_t[i] - t) else i

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------
    fig, (ax_v, ax_i) = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    ax_v.plot(echem_t, voltage, lw=1.5)
    marker, = ax_v.plot([], [], "ro", ms=6)
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Voltage (V)")
    ax_v.set_title("Voltage evolution")

    im = ax_i.imshow(
        img_series[0],
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    ax_i.set_xticks([])
    ax_i.set_yticks([])

    plt.tight_layout()

    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=dpi):
        for t in t_anim:

            marker.set_data(t, voltage_at(t))

            j = nearest_image_index(t)
            im.set_data(img_series[j])
            ax_i.set_title(
                f"Image @ {img_dt[j].strftime('%Y-%m-%d %H:%M:%S')}"
            )

            writer.grab_frame()

    plt.close(fig)




















import numpy as np

def circular_average_thickness_series(img_series, x0=None, y0=None, r0=None):
    """
    Compute the average pixel value within a circular region
    for each image in an image series.

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

        # ---- Compute average ----
        avg_thickness_series.append(np.nanmean(img[mask]))

    return np.asarray(avg_thickness_series)




import numpy as np
import matplotlib.pyplot as plt

def plot_voltage_and_thickness(
    echem_datetime_series,
    voltage_series,
    echem_thickness_series,
    img_datetime_series,
    avg_thickness_series,
):
    """
    Plot voltage, electrochemical thickness, and image-derived average thickness
    on a common time axis (elapsed time from earliest overlapping time).

    Time ranges where only electrochemical data exists (before or after image data)
    are truncated.
    """

    # ---- Convert to numpy arrays ----
    echem_datetime_series = np.asarray(echem_datetime_series)
    img_datetime_series = np.asarray(img_datetime_series)

    voltage_series = np.asarray(voltage_series)
    echem_thickness_series = np.asarray(echem_thickness_series)
    avg_thickness_series = np.asarray(avg_thickness_series)

    # ---- Determine overlapping time window ----
    t_start = img_datetime_series.min()
    t_end = img_datetime_series.max()

    # ---- Mask electrochemical data to image time window ----
    echem_mask = (echem_datetime_series >= t_start) & (echem_datetime_series <= t_end)

    echem_datetime_series = echem_datetime_series[echem_mask]
    voltage_series = voltage_series[echem_mask]
    echem_thickness_series = echem_thickness_series[echem_mask]

    # ---- Define common time origin ----
    t0 = t_start

    # ---- Convert datetimes to elapsed time (seconds) ----
    t_echem = np.array([(t - t0).total_seconds() for t in echem_datetime_series])
    t_img = np.array([(t - t0).total_seconds() for t in img_datetime_series])

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Voltage (left axis)
    ax1.plot(t_echem, voltage_series, label="Voltage (Ewe/V)")
    ax1.set_xlabel("Time elapsed (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.tick_params(axis="y")

    # Thickness (right axis)
    ax2 = ax1.twinx()
    ax2.plot(t_echem, echem_thickness_series, label="EC thickness")
    ax2.plot(t_img, avg_thickness_series, label="Image avg thickness")
    ax2.set_ylabel("Thickness (a.u.)")

    # ---- Legend handling ----
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")

    plt.tight_layout()
    plt.show()

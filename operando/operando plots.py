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
    potential_series : np.ndarray
        Ewe/V series (potential)
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
    potential_series = combined["Ewe/V"].to_numpy(dtype=float)[sorted_idx]
    echem_thickness_series = (
        combined["Capacity/mA.h"].to_numpy(dtype=float)[sorted_idx]
        * conv_thickness
    )

    return datetime_series, potential_series, echem_thickness_series




def make_potential_image_video_uniform_time(
    echem_datetimes,
    potential_series,
    img_datetimes,
    img_series,
    output_path,
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
    colorbar_label="Li thickness",  # Label for colorbar
    cbar_n_ticks=5,
    # Additional scale bar customization options
    scalebar_color='white',
    scale_fontsize=12,
    scalebar_width_frac=0.015,  # Height of scale bar as fraction of image height
    scalebar_position='lower right',  # 'lower right', 'lower left', 'upper right', 'upper left'
    scalebar_margin=0.1,  # Margin from edge as fraction of image dimensions
):
    """
    Create a synchronized potential–image video using a uniform global time axis.
    
    Potential: zero-order hold (previous value)
    Image: nearest in time
    """

    # ------------------------------------------------------------------
    # Limit potential series to the time range covered by images
    # ------------------------------------------------------------------
    img_start = min(img_datetimes)
    img_end = max(img_datetimes)
    echem_mask = [(t >= img_start) and (t <= img_end) for t in echem_datetimes]
    echem_datetimes = [t for t, m in zip(echem_datetimes, echem_mask) if m]
    potential_series = np.asarray(potential_series)[np.array(echem_mask, dtype=bool)]

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
    def potential_at(t):
        """Zero-order hold potential."""
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
    fig, (ax_v, ax_img) = plt.subplots(
        1, 2, figsize=(10, 4),
        gridspec_kw={"width_ratios": [1.2, 1]}
    )

    # Potential curve (static)
    ax_v.plot(echem_t_h, potential_series, lw=1.5)
    marker, = ax_v.plot([], [], "ro", ms=6)

    ax_v.set_xlabel("Time (h)")
    ax_v.set_ylabel("Potential (V)")
    ax_v.set_title("Potential evolution")
    if potential_ylim is not None:
        ax_v.set_ylim(potential_ylim)

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
    cbar.set_label(colorbar_label)
    
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
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=dpi):
        for t in anim_t:

            # Potential (ZOH)
            v = potential_at(t)
            marker.set_data([t / 3600.0], [v])

            # Image (nearest)
            j = nearest_image_index(t)
            im.set_data(img_series[j])

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
    pixel_size=55,  # Pixel size in micrometers
    scalebar_length=5000,  # Length of scale bar in micrometers
    scale_label="5 mm",  # Full text to display on scale bar
    colorbar_label="Li thickness",  # Label for colorbar
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
    pixel_size : float
        Pixel size in micrometers
    scalebar_length : float
        Length of scale bar in micrometers
    scale_label : str
        Full label to display on scale bar (e.g., "5 mm")
    colorbar_label : str
        Label for colorbar
    """

    image_kwargs = image_kwargs or {}

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    img_dt, img_series, _ = load_images(image_folder, **image_kwargs)
    echem_dt, potential, _ = load_echem_data(
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
    def potential_at(t):
        i = bisect_right(echem_t, t) - 1
        return potential[i] if i >= 0 else potential[0]

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

    echem_t_h = echem_t / 3600.0
    ax_v.plot(echem_t_h, potential, lw=1.5)
    marker, = ax_v.plot([], [], "ro", ms=6)
    ax_v.set_xlabel("Time (h)")
    ax_v.set_ylabel("Potential (V)")
    ax_v.set_title("Potential evolution")

    im = ax_i.imshow(
        img_series[0],
        cmap=cmap,
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
    cbar.set_label(colorbar_label)
    
    # Remove ticks
    ax_i.set_xticks([])
    ax_i.set_yticks([])
    
    # Add scale bar - correct ScaleBar usage
    # Determine the unit from the scale_label
    if "mm" in scale_label:
        units = "mm"
    elif "cm" in scale_label:
        units = "cm"
    elif "m" in scale_label:
        units = "m"
    elif "µm" in scale_label or "um" in scale_label:
        units = "µm"
    else:
        units = "m"  # default
    
    # Extract the numeric value from the scale_label or use the provided length
    import re
    numeric_match = re.search(r'(\d+(?:\.\d+)?)', scale_label)
    if numeric_match:
        fixed_value = float(numeric_match.group(1))
        # Convert to meters if needed
        if units == "mm":
            fixed_value = fixed_value * 1e-3
        elif units == "cm":
            fixed_value = fixed_value * 1e-2
        elif units == "µm":
            fixed_value = fixed_value * 1e-6
    else:
        fixed_value = scalebar_length * 1e-6  # Convert from micrometers to meters
    
    scalebar = ScaleBar(
        pixel_size * 1e-6,  # dx: pixel size in meters
        units,  # units for the scale bar
        location='lower right',
        frameon=True,
        color='white',
        box_color='black',
        box_alpha=0.6,
        scale_loc='top',
        border_pad=0.5,
        fixed_value=fixed_value,  # Fixed length in meters
        fixed_units='m',  # Units for the fixed value (always meters internally)
        label=scale_label  # Custom label text
    )
    ax_i.add_artist(scalebar)

    plt.tight_layout()

    # ------------------------------------------------------------
    # Animation
    # ------------------------------------------------------------
    writer = FFMpegWriter(fps=fps)

    with writer.saving(fig, output_path, dpi=dpi):
        for t in t_anim:

            marker.set_data(t / 3600.0, potential_at(t))

            j = nearest_image_index(t)
            im.set_data(img_series[j])
            ax_i.set_title(
                f"Image @ {img_dt[j].strftime('%Y-%m-%d %H:%M:%S')}"
            )

            writer.grab_frame()

    plt.close(fig)


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
):
    """
    Plot potential, electrochemical thickness, and image-derived average thickness
    on a common time axis (elapsed time from earliest overlapping time).

    Time ranges where only electrochemical data exists (before or after image data)
    are truncated.
    Optional event labels can be added on the potential trace via event_times.
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

    # ---- Plot ----
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    # Potential (left axis)  <-- this must NOT use idx
    ax1.plot(t_echem, potential_series, label="Potential (Ewe/V)")
    ax1.set_xlabel("Time elapsed (h)")
    ax1.set_ylabel("Potential (V)")
    ax1.tick_params(axis="y")
    if potential_ylim is not None:
        ax1.set_ylim(potential_ylim)

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


    # Thickness (right axis)
    if plot_thickness:
        ax2 = ax1.twinx()
        ax2.plot(t_echem, echem_thickness_series, label="EC thickness")
        ax2.plot(t_img, avg_thickness_series, label="Image avg thickness")
        ax2.set_ylabel("Thickness (a.u.)")

        # ---- Legend handling ----
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
    else:
        ax1.legend(loc="best")

    plt.tight_layout()
    plt.show()


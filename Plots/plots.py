# =============================================================================
# plot_images
# =============================================================================
# Plot a multi-panel scientific image figure with individual color bars and a
# single physical scale bar.
#
# Parameters
# ----------
# img_series : array-like
#     List or array of 2D images to plot.
#
# img_indices : list of int
#     Indices of images from `img_series` to display.
#
# pixel_size : float
#     Physical size of one pixel. Must be in the same units used for the
#     scale bar length.
#
# r0 : float or None, optional
#     Half-width of the square region of interest (ROI) in physical units.
#     If None, the full image is shown.
#
# x0, y0 : float or None, optional
#     Physical coordinates of the ROI center. If None, the image center is
#     used.
#
# conv_thickness : float or None, optional
#     If provided, image values are divided by this factor before plotting.
#
# cmap : str, optional
#     Matplotlib colormap used for all images.
#
# cbar_ranges : list of tuple, optional
#     List of (vmin, vmax) tuples specifying color bar ranges for each image.
#
# cbar_position : {"top", "bottom"}, optional
#     Position of the color bar relative to each image.
#
# cbar_length_frac : float, optional
#     Fraction of the image width occupied by the color bar.
#
# cbar_unit : str or list of str, optional
#     Unit label(s) appended to non-zero color bar tick labels.
#
# scalebar_position : {"left", "right"} or None, optional
#     Position of the scale bar outside the image grid.
#
# scalebar_length : float, optional
#     Physical length of the scale bar. Pixel length is computed as
#     scalebar_length / pixel_size.
#
# show_labels : bool, optional
#     If True, subplot labels ("a", "b", "c", ...) are displayed.
#
# Returns
# -------
# fig : matplotlib.figure.Figure
#     The created figure.
#
# axes : ndarray of matplotlib.axes.Axes
#     Axes corresponding to the plotted images.
#
# Notes
# -----
# - Color bars never overlap the images.
# - Scale bar is physically accurate and placed outside the image grid.
# - Units are not shown on zero-valued color bar ticks.
#
# =============================================================================
    
'''
Example Use:

cbar_ranges = [
    (0, 5),
    (0, 2),
    (0, 100),
    (0, 40)
]

fig, axes = plot_images(
    img_series,
    img_indices=[0, 1, 2, 3],
    pixel_size=11,
    show_labels=False,
    
    # Color bar
    cmap="gray",
    cbar_ranges=cbar_ranges,
    cbar_length_frac=0.8,
    cbar_position="bottom",
    cbar_fontsize=12,
    cbar_unit=["μm","μm","nm","nm"],

    # Scale bar
    scale_fontsize=12,
    scalebar_position="left",
    scalebar_length=5000,
    scale_label="5 mm",
    scalebar_width_frac=0.01,
    
    # Figure layout   
    spacing=(0.05, 0),
    figsize=(16, 4),
    ncols=4,
    nrows=1
)

plt.show()
'''

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

def plot_images(
    # Data & indexing
    img_series,
    img_indices,
    pixel_size,
    conv_thickness=1,

    # Region of interest (ROI)
    r0=None,
    x0=None,
    y0=None,

    # Image appearance
    cmap="inferno",

    # Color bar settings
    cbar_ranges=None,
    cbar_position="bottom",
    cbar_length_frac=0.8,      # fraction of image width
    cbar_unit=None,

    # Color bar fonts
    cbar_fontsize=8,
    cbar_fontname="Arial",
    cbar_fontweight="normal",

    # Scale bar settings
    scalebar_position="left",  # "left" or "right"
    scalebar_length=5,         # physical length (same units as x/y axis)
    scalebar_width_frac=0.02,  # fraction of figure width
    scalebar_color="black",
    scale_label=None,
    scale_fontsize=9,
    scale_fontname="Arial",
    scale_fontweight="bold",

    # Figure layout
    figsize=(8, 6),
    nrows=None,
    ncols=None,
    spacing=(0.0, 0.15),

    # Panel labels (a, b, c, ...)
    show_labels=True,
    label_fontsize=14,
    label_fontname="Arial",
    label_fontweight="bold",
):

    """
    Plot images with calibration bars outside, independent scale bar next to images
    with physical length, optional image letters, and units on ticks.
    """
    img_series = img_series / conv_thickness
    ny, nx = img_series[0].shape

    # -------------------------
    # ROI handling
    # -------------------------
    if r0 is None:
        x_min, x_max = 0, nx*pixel_size
        y_min, y_max = 0, ny*pixel_size
        def get_roi(img): return img
    else:
        if x0 is None: x0=0
        if y0 is None: y0=0
        x_min = x0-r0
        x_max = x0+r0
        y_min = y0-r0
        y_max = y0+r0
        ix_min = max(int(np.floor(x_min/pixel_size)),0)
        ix_max = min(int(np.ceil(x_max/pixel_size)), nx)
        iy_min = max(int(np.floor(y_min/pixel_size)),0)
        iy_max = min(int(np.ceil(y_max/pixel_size)), ny)
        if ix_max<=ix_min or iy_max<=iy_min:
            raise ValueError("ROI is empty — check r0, x0, y0, pixel_size")
        def get_roi(img): return img[iy_min:iy_max, ix_min:ix_max]
        x_min = ix_min*pixel_size
        x_max = ix_max*pixel_size
        y_min = iy_min*pixel_size
        y_max = iy_max*pixel_size

    # -------------------------
    # Colorbar ranges
    # -------------------------
    n_images = len(img_indices)
    if cbar_ranges is None:
        cbar_ranges = []
        for idx in img_indices:
            roi = get_roi(img_series[idx])
            cbar_ranges.append((roi.min(), roi.max()))
    if len(cbar_ranges)!=n_images:
        raise ValueError("cbar_ranges must match img_indices length")

    # -------------------------
    # Units
    # -------------------------
    if cbar_unit is None:
        cbar_units_list = [""]*n_images
    elif isinstance(cbar_unit,str):
        cbar_units_list = [cbar_unit]*n_images
    elif isinstance(cbar_unit,list):
        if len(cbar_unit)!=n_images:
            raise ValueError("cbar_unit list must match number of images")
        cbar_units_list = cbar_unit
    else:
        raise TypeError("cbar_unit must be None, str, or list of str")

    # -------------------------
    # Layout
    # -------------------------
    if nrows is None or ncols is None:
        ncols = int(math.ceil(math.sqrt(n_images)))
        nrows = int(math.ceil(n_images/ncols))
    else:
        if nrows*ncols<n_images:
            raise ValueError("nrows*ncols must be >= number of images")

    wspace, hspace = spacing
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             gridspec_kw={"wspace":wspace,"hspace":hspace})
    axes = np.atleast_1d(axes).ravel()

    # -------------------------
    # Plot images and calibration bars
    # -------------------------
    for j, img_idx in enumerate(img_indices):
        ax = axes[j]
        roi_img = get_roi(img_series[img_idx])
        vmin, vmax = cbar_ranges[j]

        im = ax.imshow(roi_img, cmap=cmap,
                       extent=[x_min,x_max,y_min,y_max],
                       origin="lower", vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

        # Panel label
        if show_labels:
            ax.text(0.05,0.95, chr(97+j),
                    transform=ax.transAxes,
                    fontsize=label_fontsize,
                    fontname=label_fontname,
                    fontweight=label_fontweight,
                    color="white",
                    ha="left", va="top")

        # Calibration bar outside image
        pos = ax.get_position()
        cbar_width = cbar_length_frac * pos.width
        cbar_height = 0.03  # fixed thickness
        if cbar_position.lower() == "bottom":
            cbar_y0 = pos.y0 - cbar_height
        else:
            cbar_y0 = pos.y0 + pos.height
        cbar_x0 = pos.x0 + (1 - cbar_length_frac)/2 * pos.width

        cax = fig.add_axes([cbar_x0, cbar_y0, cbar_width, cbar_height])
        cbar = plt.colorbar(im, cax=cax, orientation="horizontal")

        # Tick position
        if cbar_position.lower() == "top":
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")
        else:
            cbar.ax.xaxis.set_ticks_position("bottom")
            cbar.ax.xaxis.set_label_position("bottom")

        # Font settings
        cbar.ax.tick_params(axis="x", length=0, pad=1)
        for tick in cbar.ax.get_xticklabels():
            tick.set_fontsize(cbar_fontsize)
            tick.set_fontname(cbar_fontname)
            tick.set_fontweight(cbar_fontweight)

        # Tick labels: integer, no decimals, no unit on zero
        ticks = [vmin, 0.5*(vmin+vmax), vmax]
        tick_labels = [f"{int(round(t))}" if int(round(t))==0 else f"{int(round(t))} {cbar_units_list[j]}" for t in ticks]
        cbar.set_ticks(ticks)
        cbar.set_ticklabels(tick_labels)

    # -------------------------
    # Independent vertical scale bar (physical length)
    # -------------------------
    if scale_label is None:
        scale_label = f"{scalebar_length} mm"

    # Select reference axis
    if scalebar_position.lower() == "left":
        col_idx = 0
        row_idx = 0
        ref_ax = axes[row_idx * ncols + col_idx] if row_idx * ncols + col_idx < n_images else axes[0]
    else:
        col_idx = ncols - 1
        row_idx = 0
        idx = row_idx * ncols + col_idx
        if idx >= n_images:
            # find last axis in first row
            for i in range(min(ncols, n_images)-1, -1, -1):
                if i < n_images:
                    ref_ax = axes[i]
                    break
        else:
            ref_ax = axes[idx]

    ref_pos = ref_ax.get_position()
    scale_height_fig = (scalebar_length / (y_max - y_min)) * ref_pos.height

    if scalebar_position.lower() == "left":
        scale_x0 = ref_pos.x0 - scalebar_width_frac - 0.01
    else:
        scale_x0 = ref_pos.x1 + 0.01

    scale_y0 = ref_pos.y0
    ax_scale = fig.add_axes([scale_x0, scale_y0, scalebar_width_frac, scale_height_fig])
    ax_scale.add_patch(patches.Rectangle((0,0), 1, 1, facecolor=scalebar_color, transform=ax_scale.transAxes))
    ax_scale.set_xticks([])
    ax_scale.set_yticks([])
    ax_scale.set_xlim(0,1)
    ax_scale.set_ylim(0,1)

    # Add label above the bar using figure coordinates
    fig.text(scale_x0 + scalebar_width_frac/2,
             scale_y0 + scale_height_fig + 0.005,
             scale_label,
             ha="center", va="bottom",
             fontsize=scale_fontsize,
             fontname=scale_fontname,
             fontweight=scale_fontweight,
             color=scalebar_color)

    # Disable unused axes
    for k in range(n_images, len(axes)):
        axes[k].axis("off")

    return fig, axes


# Example Use

cbar_ranges = [
    (0, 5),
    (0, 2),
    (0, 100),
    (0, 40)
]

fig, axes = plot_images(
    img_series,
    img_indices=[0, 1, 2, 3],
    pixel_size=11,
    show_labels=False,
    
    # Color bar
    cmap="gray",
    cbar_ranges=cbar_ranges,
    cbar_length_frac=0.8,
    cbar_position="bottom",
    cbar_fontsize=12,
    cbar_unit=["μm","μm","nm","nm"],

    # Scale bar
    scale_fontsize=12,
    scalebar_position="left",
    scalebar_length=5000,
    scale_label="5 mm",
    scalebar_width_frac=0.01,
    
    # Figure layout   
    spacing=(0.05, 0),
    figsize=(16, 4),
    ncols=4,
    nrows=1
)

plt.show()

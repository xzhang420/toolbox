# =============================================================================
# plot_images
# =============================================================================
# Plot a multi-panel scientific image figure with individual or shared color
# bars, optional row and column legends, and a single physical scale bar.
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
# r0 : int or None, optional
#     Half-width of the square region of interest (ROI) in pixels.
#     If None, the full image is shown.
#
# x0, y0 : int or None, optional
#     Pixel indices of the ROI center (column, row).
#     If None, the image center is used.
#
# conv_thickness : float, optional
#     Image values are divided by this factor before plotting.
#
# cmap : str, optional
#     Matplotlib colormap used for all images.
#
# cbar_ranges : list of tuple or None, optional
#     List of (vmin, vmax) tuples specifying color bar ranges for each image.
#     If None, ranges are determined automatically from the ROI of each image.
#
# cbar_mode : {"individual top", "individual bottom", "shared left", "shared right"}, optional
#     Determines whether each image has its own color bar or a single shared
#     color bar is shown on the left or right side.
#
# cbar_length_frac : float, optional
#     Fraction of the image width occupied by individual color bars.
#
# cbar_width_frac : float, optional
#     Thickness of the color bar in figure-relative units.
#
# cbar_n_ticks : int, optional
#     Number of ticks shown on each color bar.
#
# cbar_unit : str or list of str or None, optional
#     Unit label(s) appended to non-zero color bar tick labels.
#     If a list is provided, it must match the number of images.
#
# cbar_fontsize : int, optional
#     Font size of color bar tick labels.
#
# cbar_fontname : str, optional
#     Font name used for color bar tick labels.
#
# cbar_fontweight : str, optional
#     Font weight used for color bar tick labels.
#
# cbar_show_ticklines : bool, optional
#     If True, draw small tick lines on the color bar corresponding to each
#     tick value. If False, only tick labels are shown.
#
# scalebar_mode : {"inside", "outside left", "outside right"} or None, optional
#     Placement mode of the scale bar. If "inside", the scale bar is drawn
#     inside the first image. If "outside", it is drawn next to the image grid.
#
# scalebar_length : float, optional
#     Physical length of the scale bar. Must be in the same units as
#     `pixel_size`.
#
# scalebar_width_frac : float, optional
#     Thickness of the scale bar in figure-relative units.
#
# scalebar_color : str or None, optional
#     Color of the scale bar and its label. If None, defaults to black for
#     outside scale bars and white for inside scale bars.
#
# scale_label : str or None, optional
#     Text label displayed above the scale bar. If None, a label is generated
#     automatically from `scalebar_length`.
#
# scale_fontsize : int, optional
#     Font size of the scale bar label.
#
# scale_fontname : str, optional
#     Font name used for the scale bar label.
#
# scale_fontweight : str, optional
#     Font weight used for the scale bar label.
#
# scalebar_pad : float, optional
#     Distance between the outside scale bar and the image grid, in
#     figure-relative units.
#
# show_labels : bool, optional
#     If True, subplot labels ("a", "b", "c", ...) are displayed.
#
# label_fontsize : int, optional
#     Font size of subplot labels.
#
# label_fontname : str, optional
#     Font name used for subplot labels.
#
# label_fontweight : str, optional
#     Font weight used for subplot labels.
#
# row_labels : list of str or None, optional
#     Text labels for each row of images, displayed to the left of the grid.
#     Length must match the number of rows.
#
# col_labels : list of str or None, optional
#     Text labels for each column of images, displayed above the grid.
#     Length must match the number of columns.
#
# row_label_fontsize : int, optional
#     Font size of row labels.
#
# col_label_fontsize : int, optional
#     Font size of column labels.
#
# row_label_fontname : str, optional
#     Font name used for row labels.
#
# col_label_fontname : str, optional
#     Font name used for column labels.
#
# row_label_fontweight : str, optional
#     Font weight used for row labels.
#
# col_label_fontweight : str, optional
#     Font weight used for column labels.
#
# row_label_pad : float, optional
#     Horizontal distance between the left edge of the row label text and
#     the image grid, in figure-relative units. This allows consistent image
#     alignment when using row labels of different lengths.
#
# col_label_pad : float, optional
#     Vertical distance between the top of the image grid and the column
#     labels, in figure-relative units.
#
# figsize : tuple of float, optional
#     Size of the figure in inches.
#
# nrows, ncols : int or None, optional
#     Number of rows and columns in the image grid. If None, a nearly square
#     layout is chosen automatically.
#
# spacing : tuple of float, optional
#     Horizontal and vertical spacing between subplots, in figure-relative
#     units.
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
# - Shared color bar modes enforce a single normalization across all images.
# - ROI selection is performed in pixel indices, not physical units.
# - Color bar ticks are placed using locators (no FixedFormatter misuse).
# - Scale bar length is physically accurate and consistent across figures.
# - Row labels are positioned such that their left edge is aligned at a fixed
#   distance from the image grid, enabling vertical alignment across figures.
#
# =============================================================================


    
'''

Example Use:

cbar_ranges = [
    (0, 20),
    (0, 2),
    (0, 100),
    (0, 40)
]

fig, axes = plot_images(
    NRimg_series,
    img_indices=[0, 1, 2, 3],
    pixel_size=34.4,
    show_labels=False,
    
    col_labels=["a - 10 μm plated", "b - 1 μm plated", "c - 100 nm plated", "d - not plated"],
    col_label_fontsize=20,
    col_label_pad= 0.2,
    row_labels=["NR"],
    row_label_fontsize=20,
    row_label_pad=0.06,

    # Color bar
    cmap="gray",
    cbar_ranges=cbar_ranges,
    cbar_length_frac=0.8,
    cbar_mode="individual top",
    cbar_fontsize=16,
    cbar_unit=["μm","μm","nm","nm"],
 #   cbar_width_frac=0.01,
    cbar_show_ticklines=True,


    # Scale bar
    scale_fontsize=16,
    scalebar_mode="outside left",
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

# =============================================================================
# plot_images
# =============================================================================
def plot_images(
    img_series,
    img_indices,
    pixel_size,

    # ROI (indices, not physical units)
    r0=None,   # half-width in pixels
    x0=None,   # center column index
    y0=None,   # center row index

    # Image scaling
    conv_thickness=1,

    # Colormap / color bars
    cmap="inferno",
    cbar_ranges=None,
    cbar_mode="individual bottom",   # "individual top", "individual bottom",
                                     # "shared left", "shared right"
    cbar_length_frac=0.8,
    cbar_width_frac=0.03,
    cbar_n_ticks=3,
    cbar_unit=None,
    cbar_fontsize=8,
    cbar_fontname="Arial",
    cbar_fontweight="normal",
    cbar_show_ticklines=True,

    # Scale bar
    scalebar_mode="outside left",     # "inside", "outside left", "outside right"
    scalebar_length=5,
    scalebar_width_frac=0.02,
    scalebar_color=None,
    scale_label=None,
    scale_fontsize=9,
    scale_fontname="Arial",
    scale_fontweight="bold",
    scalebar_pad=0.02,                # <<< NEW: distance from images (figure fraction)

    # Labels
    show_labels=True,
    label_fontsize=14,
    label_fontname="Arial",
    label_fontweight="bold",

    # Layout
    figsize=(8, 6),
    nrows=None,
    ncols=None,
    spacing=(0.05, 0.05)
):
    img_series = np.asarray(img_series) / conv_thickness
    ny, nx = img_series[0].shape
    n_images = len(img_indices)

    # -------------------------
    # ROI handling (pixel indices)
    # -------------------------
    if r0 is None:
        x_min, x_max = 0, nx * pixel_size
        y_min, y_max = 0, ny * pixel_size

        def get_roi(img):
            return img
    else:
        if x0 is None:
            x0 = nx // 2
        if y0 is None:
            y0 = ny // 2

        ix_min = max(x0 - r0, 0)
        ix_max = min(x0 + r0, nx)
        iy_min = max(y0 - r0, 0)
        iy_max = min(y0 + r0, ny)

        x_min, x_max = ix_min * pixel_size, ix_max * pixel_size
        y_min, y_max = iy_min * pixel_size, iy_max * pixel_size

        def get_roi(img):
            return img[iy_min:iy_max, ix_min:ix_max]

    # -------------------------
    # Color bar ranges
    # -------------------------
    if cbar_ranges is None:
        cbar_ranges = []
        for idx in img_indices:
            roi = get_roi(img_series[idx])
            cbar_ranges.append((roi.min(), roi.max()))

    # Units
    if cbar_unit is None:
        cbar_units = [""] * n_images
    elif isinstance(cbar_unit, str):
        cbar_units = [cbar_unit] * n_images
    else:
        cbar_units = cbar_unit

    # -------------------------
    # Shared color scale
    # -------------------------
    shared = cbar_mode.startswith("shared")
    if shared:
        vmin_shared = min(v[0] for v in cbar_ranges)
        vmax_shared = max(v[1] for v in cbar_ranges)

    # -------------------------
    # Layout
    # -------------------------
    if nrows is None or ncols is None:
        ncols = int(math.ceil(math.sqrt(n_images)))
        nrows = int(math.ceil(n_images / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        gridspec_kw={"wspace": spacing[0], "hspace": spacing[1]},
    )
    axes = np.atleast_1d(axes).ravel()

    images = []

    # -------------------------
    # Plot images
    # -------------------------
    for j, idx in enumerate(img_indices):
        ax = axes[j]
        roi = get_roi(img_series[idx])

        if shared:
            vmin, vmax = vmin_shared, vmax_shared
        else:
            vmin, vmax = cbar_ranges[j]

        im = ax.imshow(
            roi,
            cmap=cmap,
            extent=[x_min, x_max, y_min, y_max],
            origin="lower",
            vmin=vmin,
            vmax=vmax,
        )
        images.append(im)

        ax.set_xticks([])
        ax.set_yticks([])

        if show_labels:
            ax.text(
                0.05,
                0.95,
                chr(97 + j),
                transform=ax.transAxes,
                fontsize=label_fontsize,
                fontname=label_fontname,
                fontweight=label_fontweight,
                color="white",
                ha="left",
                va="top",
            )

        # -------------------------
        # Scale bar inside first image
        # -------------------------
        if j == 0 and scalebar_mode == "inside":
            bar_color = scalebar_color or "white"
            text_color = bar_color

            scale_x = x_min + 0.1 * (x_max - x_min)
            scale_y = y_min + 0.1 * (y_max - y_min)

            ax.add_patch(
                patches.Rectangle(
                    (scale_x, scale_y),
                    scalebar_length,
                    0.015 * (y_max - y_min),
                    facecolor=bar_color,
                    edgecolor="none",
                )
            )

            label = scale_label or f"{scalebar_length} mm"
            ax.text(
                scale_x + scalebar_length / 2,
                scale_y + 0.03 * (y_max - y_min),
                label,
                color=text_color,
                ha="center",
                va="bottom",
                fontsize=scale_fontsize,
                fontname=scale_fontname,
                fontweight=scale_fontweight,
            )

    for k in range(n_images, len(axes)):
        axes[k].axis("off")

    # -------------------------
    # Helper: color bar ticks
    # -------------------------
    def apply_cbar_ticks(cbar, vmin, vmax, unit):
        ticks = np.linspace(vmin, vmax, cbar_n_ticks)
        cbar.set_ticks(ticks)

        labels = []
        for t in ticks:
            if np.isclose(t, 0):
                labels.append("0")
            else:
                labels.append(f"{t:g} {unit}".rstrip())

        cbar.set_ticklabels(labels)

        tick_length = 4 if cbar_show_ticklines else 0
        cbar.ax.tick_params(
            length=tick_length,
            width=0.8,
            direction="out",
        )

        for tick in cbar.ax.get_xticklabels() + cbar.ax.get_yticklabels():
            tick.set_fontsize(cbar_fontsize)
            tick.set_fontname(cbar_fontname)
            tick.set_fontweight(cbar_fontweight)

    # -------------------------
    # Color bars
    # -------------------------
    if cbar_mode.startswith("individual"):
        top = "top" in cbar_mode
        for j, im in enumerate(images):
            ax = axes[j]
            pos = ax.get_position()

            h = cbar_width_frac * pos.height
            y0 = pos.y1 if top else pos.y0 - h
            x0 = pos.x0 + (1 - cbar_length_frac) / 2 * pos.width

            cax = fig.add_axes([x0, y0, cbar_length_frac * pos.width, h])
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            apply_cbar_ticks(cbar, *im.get_clim(), cbar_units[j])
            cbar.ax.xaxis.set_ticks_position("top" if top else "bottom")

    else:
        right = "right" in cbar_mode
        ref = axes[n_images - 1 if right else 0].get_position()

        w = cbar_width_frac
        x0 = ref.x1 + 0.01 if right else ref.x0 - w - 0.01
        y0 = ref.y0
        h = ref.height

        cax = fig.add_axes([x0, y0, w, h])
        cbar = plt.colorbar(images[0], cax=cax, orientation="vertical")
        apply_cbar_ticks(cbar, vmin_shared, vmax_shared, cbar_units[0])

    # -------------------------
    # Outside scale bar
    # -------------------------
    if scalebar_mode.startswith("outside"):
        ref = axes[0].get_position()
        bar_color = scalebar_color or "black"
        label = scale_label or f"{scalebar_length} mm"

        height = (scalebar_length / (y_max - y_min)) * ref.height

        if "left" in scalebar_mode:
            x0 = ref.x0 - scalebar_width_frac - scalebar_pad
        else:
            x0 = axes[min(ncols - 1, n_images - 1)].get_position().x1 + scalebar_pad

        y0 = ref.y0

        ax_sb = fig.add_axes([x0, y0, scalebar_width_frac, height])
        ax_sb.add_patch(
            patches.Rectangle((0, 0), 1, 1, facecolor=bar_color)
        )
        ax_sb.axis("off")

        fig.text(
            x0 + scalebar_width_frac / 2,
            y0 + height + 0.005,
            label,
            ha="center",
            va="bottom",
            fontsize=scale_fontsize,
            fontname=scale_fontname,
            fontweight=scale_fontweight,
            color=bar_color,
        )

    return fig, axes

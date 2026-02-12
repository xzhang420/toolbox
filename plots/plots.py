import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

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
    cbar_individual_length=0.8,      # Renamed: width of individual color bars (fraction of image width)
    cbar_shared_length=0.8,          # NEW: length/height of shared color bars (fraction of grid height)
    cbar_width_frac=0.015,
    cbar_n_ticks=3,
    cbar_unit=None,
    cbar_fontsize=8,
    cbar_fontname="Arial",
    cbar_fontweight="normal",
    cbar_show_ticklines=True,
    cbar_pad=0.01,                   # Distance between color bar and subplots for shared left/right

    # Scale bar
    scalebar_mode="outside left",     # "inside", "outside left", "outside right"
    scalebar_length=5,
    scalebar_width_frac=0.015,
    scalebar_color=None,
    scale_label=None,
    scale_fontsize=9,
    scale_fontname="Arial",
    scale_fontweight="bold",
    scalebar_pad=0.02,               # Distance from images (figure fraction)

    # Labels
    show_labels=True,
    label_start=0,
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
                chr(97 + label_start + j),
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
                    scalebar_width_frac * (y_max - y_min),
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
            x0 = pos.x0 + (1 - cbar_individual_length) / 2 * pos.width

            cax = fig.add_axes([x0, y0, cbar_individual_length * pos.width, h])
            cbar = plt.colorbar(im, cax=cax, orientation="horizontal")
            apply_cbar_ticks(cbar, *im.get_clim(), cbar_units[j])
            cbar.ax.xaxis.set_ticks_position("top" if top else "bottom")

    else:  # Shared color bars
        # Get positions of all image axes to find the vertical center of the grid
        positions = [axes[i].get_position() for i in range(min(n_images, nrows * ncols))]
        
        # Find the vertical center of the entire grid
        y0_min = min(pos.y0 for pos in positions)
        y1_max = max(pos.y1 for pos in positions)
        grid_center_y = (y0_min + y1_max) / 2
        grid_height = y1_max - y0_min
        
        w = cbar_width_frac
        # Set color bar height based on cbar_shared_length (fraction of grid height)
        h = grid_height * cbar_shared_length
        
        if "left" in cbar_mode:
            # Position color bar to the left of the grid
            x0 = min(pos.x0 for pos in positions) - w - cbar_pad
            # Set ticks on the left side
            tick_position = "left"
        else:  # "right" in cbar_mode
            # Position color bar to the right of the grid
            x0 = max(pos.x1 for pos in positions) + cbar_pad
            # Set ticks on the right side
            tick_position = "right"
            
        # Center the color bar vertically
        y0 = grid_center_y - h/2

        cax = fig.add_axes([x0, y0, w, h])
        cbar = plt.colorbar(images[0], cax=cax, orientation="vertical")
        
        # Apply ticks and formatting
        apply_cbar_ticks(cbar, vmin_shared, vmax_shared, cbar_units[0])
        
        # Set tick position based on color bar location
        if cbar_mode == "shared left":
            # Move ticks to the left side
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("left")
        elif cbar_mode == "shared right":
            # Keep ticks on the right side (default)
            cbar.ax.yaxis.set_ticks_position("right")
            cbar.ax.yaxis.set_label_position("right")

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

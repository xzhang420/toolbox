"""Generic 2D/3D image-panel plotting shared by LumaCam workflows."""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math


def _normalize_scalebar_length(scalebar_length, scale_label=None):
    """Use micrometers for image coordinates; accept legacy 5 + '5 mm' inputs."""
    length = float(scalebar_length)
    label = str(scale_label or "").lower()
    if "mm" in label and length < 100:
        return length * 1000.0
    return length


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
    cbar_label=None,
    cbar_unit=None,
    cbar_fontsize=8,
    cbar_fontname="Arial",
    cbar_fontweight="normal",
    cbar_show_ticklines=True,
    cbar_pad=0.01,                   # Distance between color bar and subplots for shared left/right

    # Scale bar
    scalebar_mode="outside left",     # "inside", "outside left", "outside right"
    scalebar_length=5000,
    scalebar_width_frac=0.015,
    scalebar_color=None,
    scale_label=None,
    scale_fontsize=9,
    scale_fontname="Arial",
    scale_fontweight="bold",
    scalebar_pad=0.02,               # Distance from images (figure fraction)
    scalebar_x_frac=0.1,             # Inside scalebar x position as fraction of image width
    scalebar_y_frac=0.14,            # Inside scalebar y position as fraction of image height

    # Labels
    show_labels=True,
    label_start=0,
    label_fontsize=14,
    label_fontname="Arial",
    label_fontweight="bold",
    lower_left_texts=None,
    lower_right_texts=None,
    annotation_fontsize=12,
    annotation_fontname="Arial",
    annotation_fontweight="bold",
    annotation_color="white",
    annotation_bbox=None,

    # Layout
    figsize=(8, 6),
    nrows=None,
    ncols=None,
    spacing=(0.05, 0.05),

    # 3D rendering
    render_3d=False,
    surface_stride=2,
    surface_elev=35,
    surface_azim=-60,
    surface_zlim=None,
    surface_zlabel=None,
    surface_axis_labels=False,
    electrode_roi=None,
    show_electrode_roi=False,
    electrode_roi_label="electrode area",
    electrode_roi_color="cyan",
    electrode_roi_linewidth=1.0,
):
    original_img_series = list(img_series)
    img_series = np.asarray(img_series) / conv_thickness
    scalebar_length = _normalize_scalebar_length(scalebar_length, scale_label)
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

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows,
        ncols,
        wspace=spacing[0],
        hspace=spacing[1],
    )
    axes = []
    for pos in range(nrows * ncols):
        row = pos // ncols
        col = pos % ncols
        if render_3d:
            axes.append(fig.add_subplot(gs[row, col], projection="3d"))
        else:
            axes.append(fig.add_subplot(gs[row, col]))
    axes = np.asarray(axes, dtype=object)

    images = []
    if annotation_bbox is None:
        annotation_bbox = {"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.5}

    def annotation_text(values, pos):
        if values is None:
            return None
        if isinstance(values, str):
            return values
        if pos >= len(values):
            return None
        value = values[pos]
        if value is None:
            return None
        return str(value)

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

        if render_3d:
            stride = max(1, int(surface_stride))
            roi_plot = roi[::stride, ::stride]
            x_plot = np.linspace(x_min, x_max, roi.shape[1])[::stride]
            y_plot = np.linspace(y_min, y_max, roi.shape[0])[::stride]
            x_grid, y_grid = np.meshgrid(x_plot, y_plot)
            im = ax.plot_surface(
                x_grid,
                y_grid,
                roi_plot,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                linewidth=0,
                antialiased=True,
                shade=True,
            )
            ax.view_init(elev=surface_elev, azim=surface_azim)
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            if surface_zlim is not None:
                ax.set_zlim(surface_zlim)
            else:
                ax.set_zlim(vmin, vmax)
            if surface_axis_labels:
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel(surface_zlabel or (cbar_units[j] if cbar_units[j] else "value"))
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_zlabel("")
            ax.grid(False)
            ax.set_box_aspect((1, 1, 0.45))
        else:
            im = ax.imshow(
                roi,
                cmap=cmap,
                extent=[x_min, x_max, y_min, y_max],
                origin="lower",
                vmin=vmin,
                vmax=vmax,
            )
        images.append(im)

        if not render_3d:
            ax.set_xticks([])
            ax.set_yticks([])

        if show_labels:
            text_func = ax.text2D if render_3d else ax.text
            text_func(
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

        if render_3d and show_electrode_roi and electrode_roi:
            roi_x0 = float(electrode_roi.get("x0", nx / 2))
            roi_y0 = float(electrode_roi.get("y0", ny / 2))
            roi_r0 = float(electrode_roi.get("r0", min(nx, ny) / 4))
            source_img = original_img_series[idx]
            crop_x0, crop_y0 = getattr(source_img, "crop_origin_xy", (0, 0))
            bin_factor = max(1, int(getattr(source_img, "spatial_bin_factor", 1)))
            center_x = (roi_x0 / bin_factor - float(crop_x0)) * pixel_size
            center_y = (roi_y0 / bin_factor - float(crop_y0)) * pixel_size
            radius = roi_r0 / bin_factor * pixel_size
            theta = np.linspace(0, 2 * np.pi, 181)
            x_circle = center_x + radius * np.cos(theta)
            y_circle = center_y + radius * np.sin(theta)
            if surface_zlim is not None:
                z_plane = surface_zlim[0]
            else:
                z_plane = vmin
            z_circle = np.full_like(theta, z_plane, dtype=float)
            ax.plot(
                x_circle,
                y_circle,
                z_circle,
                color=electrode_roi_color,
                linewidth=electrode_roi_linewidth,
                zorder=30,
            )

        lower_left_text = annotation_text(lower_left_texts, j)
        if lower_left_text:
            text_func = ax.text2D if render_3d else ax.text
            text_func(
                0.04,
                0.04,
                lower_left_text,
                transform=ax.transAxes,
                fontsize=annotation_fontsize,
                fontname=annotation_fontname,
                fontweight=annotation_fontweight,
                color=annotation_color,
                ha="left",
                va="bottom",
                bbox=annotation_bbox,
            )

        lower_right_text = annotation_text(lower_right_texts, j)
        if lower_right_text:
            text_func = ax.text2D if render_3d else ax.text
            text_func(
                0.96,
                0.04,
                lower_right_text,
                transform=ax.transAxes,
                fontsize=annotation_fontsize,
                fontname=annotation_fontname,
                fontweight=annotation_fontweight,
                color=annotation_color,
                ha="right",
                va="bottom",
                bbox=annotation_bbox,
            )

        # -------------------------
        # Scale bar inside first image
        # -------------------------
        if j == 0 and scalebar_mode == "inside":
            bar_color = scalebar_color or "white"
            text_color = bar_color

            scale_x = x_min + scalebar_x_frac * (x_max - x_min)
            scale_y = y_min + scalebar_y_frac * (y_max - y_min)
            if render_3d:
                z_base = vmin if surface_zlim is None else surface_zlim[0]
                ax.plot(
                    [scale_x, scale_x + scalebar_length],
                    [scale_y, scale_y],
                    [z_base, z_base],
                    color=bar_color,
                    linewidth=3,
                )
            else:
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
            if render_3d:
                ax.text2D(
                    0.12,
                    0.10,
                    label,
                    transform=ax.transAxes,
                    color=text_color,
                    ha="left",
                    va="bottom",
                    fontsize=scale_fontsize,
                    fontname=scale_fontname,
                    fontweight=scale_fontweight,
                )
            else:
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
                labels.append(f"{t:g}")

        cbar.set_ticklabels(labels)
        if cbar_label:
            label = cbar_label
            if unit:
                label = f"{label} ({unit})"
            cbar.set_label(label)

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

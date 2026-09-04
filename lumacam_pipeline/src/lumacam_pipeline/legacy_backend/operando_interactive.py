"""Established interactive helpers retained by LumaCam compatibility APIs."""

import html
import numpy as np
import matplotlib.pyplot as plt


def _try_enable_widget_backend():
    """Switch from inline to the ipympl widget backend when running in IPython."""
    backend_name = str(plt.get_backend()).lower()
    if "ipympl" in backend_name or "widget" in backend_name or "nbagg" in backend_name:
        return None
    try:
        import matplotlib
        import ipympl  # noqa: F401
        from IPython import get_ipython

        ipython = get_ipython()
        if ipython is not None:
            ipython.run_line_magic("matplotlib", "widget")
            matplotlib.use("module://ipympl.backend_nbagg", force=True)
            plt.switch_backend("module://ipympl.backend_nbagg")
            plt.close("all")
            return None
    except Exception as exc:
        return exc
    return None


def _nearest_index_by_datetime(target_time, datetimes):
    deltas = np.asarray(
        [abs((dt - target_time).total_seconds()) for dt in datetimes],
        dtype=float,
    )
    return int(np.nanargmin(deltas))


def launch_draggable_event_selector(
    *,
    operando_plots,
    event_context,
    echem_datetimes,
    potential_series,
    img_datetimes,
    initial_selected_image_indices=None,
    echem_thickness_series=None,
    avg_thickness_series=None,
    plot_thickness=False,
    potential_plot_kwargs=None,
    side_text=None,
    label_start=0,
    event_marker_size=6,
    event_label_fontsize=18,
    snap_markers_to_tips=True,
    tip_times=None,
    tip_snap_tolerance_seconds=None,
    tip_snap_mode="upper",
    display_func=None,
):
    """Launch a selector where markers can be dragged along the EC curve.

    Dragging and adding still select the nearest neutron image index. When
    ``snap_markers_to_tips`` is enabled, selected markers are drawn at the
    nearest local potential extremum if the selected image timestamp is close
    enough. This is a visual positioning aid only; the returned
    ``selected_image_indices`` values remain image indices for downstream
    plotting and saving.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except Exception as exc:
        raise RuntimeError(
            "ipywidgets is required for interactive selection. Install/enable it, "
            "or edit selected_image_indices manually."
        ) from exc
    if display_func is None:
        display_func = display

    widget_backend_error = _try_enable_widget_backend()

    img_datetimes = np.asarray(img_datetimes)
    echem_datetimes = np.asarray(echem_datetimes)
    potential_series = np.asarray(potential_series)

    if len(img_datetimes) == 0:
        raise ValueError("No image timestamps are available for interactive selection.")
    if len(echem_datetimes) == 0:
        raise ValueError("No electrochemical timestamps are available for interactive selection.")

    auto_image_indices = list(event_context.get("img_indices", []))
    if initial_selected_image_indices is None:
        initial_selected_image_indices = auto_image_indices

    initial_selected_image_indices = operando_plots.normalize_selected_image_indices(
        initial_selected_image_indices,
        len(img_datetimes),
    )
    selected_image_indices = list(initial_selected_image_indices)

    potential_plot_kwargs = {
        key: value for key, value in dict(potential_plot_kwargs or {}).items()
        if key != "plot_thickness"
    }

    top_text = side_text
    was_interactive = plt.isinteractive()
    plt.ioff()
    fig, ax1, ax2 = operando_plots.plot_potential_and_thickness(
        echem_datetime_series=echem_datetimes,
        potential_series=potential_series,
        echem_thickness_series=echem_thickness_series,
        img_datetime_series=img_datetimes,
        avg_thickness_series=avg_thickness_series,
        plot_thickness=plot_thickness,
        event_times=[],
        label_start=label_start,
        event_marker_size=event_marker_size,
        event_label_fontsize=event_label_fontsize,
        side_text=None,
        show=False,
        return_axes=True,
        **potential_plot_kwargs,
    )
    if was_interactive:
        plt.ion()

    t_start = img_datetimes.min()
    t_end = img_datetimes.max()
    echem_mask = (echem_datetimes >= t_start) & (echem_datetimes <= t_end)
    echem_plot_datetimes = echem_datetimes[echem_mask]
    potential_plot_series = potential_series[echem_mask]
    if echem_thickness_series is not None:
        echem_thickness_series = np.asarray(echem_thickness_series)
        echem_thickness_plot_series = echem_thickness_series[echem_mask]
    else:
        echem_thickness_plot_series = None

    if len(echem_plot_datetimes) == 0:
        echem_plot_datetimes = echem_datetimes
        potential_plot_series = potential_series
        if echem_thickness_series is not None:
            echem_thickness_plot_series = echem_thickness_series

    t0 = t_start
    image_x = np.asarray(
        [(dt - t0).total_seconds() / 3600.0 for dt in img_datetimes],
        dtype=float,
    )

    def nearest_image_index(x_value):
        return int(np.nanargmin(np.abs(image_x - float(x_value))))

    def echem_value_at_time(target_time, value_series):
        nearest_echem_idx = _nearest_index_by_datetime(target_time, echem_plot_datetimes)
        return float(value_series[nearest_echem_idx])

    def visible_y_value(y_value):
        """
        Keep interactive markers inside the current potential axis limits.
        """
        y_min, y_max = ax1.get_ylim()
        if not np.isfinite(y_value) or y_max <= y_min:
            return y_value, False
        margin = 0.025 * (y_max - y_min)
        clipped_y = float(np.clip(y_value, y_min + margin, y_max - margin))
        return clipped_y, not np.isclose(clipped_y, y_value)

    if tip_times is None:
        tip_times = list(event_context.get("event_times", []))
    else:
        tip_times = list(tip_times)
    tip_time_by_image_idx = {
        image_idx: event_time
        for image_idx, event_time in zip(
            event_context.get("img_indices", []),
            event_context.get("event_times", []),
        )
    }

    if tip_snap_tolerance_seconds is None:
        tip_snap_tolerance_seconds = event_context.get("match_warning_threshold_seconds", 120.0)
    tip_snap_tolerance_seconds = float(tip_snap_tolerance_seconds)
    tip_snap_mode = str(tip_snap_mode or "upper").lower()

    def potential_extremum_near_time(center_time, max_delta_seconds):
        deltas = np.asarray(
            [abs((dt - center_time).total_seconds()) for dt in echem_plot_datetimes],
            dtype=float,
        )
        window_indices = np.flatnonzero(deltas <= max_delta_seconds)
        if len(window_indices) == 0:
            return None, None

        candidate_indices = []
        for idx in window_indices:
            if idx <= 0 or idx >= len(potential_plot_series) - 1:
                continue
            dy_before = potential_plot_series[idx] - potential_plot_series[idx - 1]
            dy_after = potential_plot_series[idx + 1] - potential_plot_series[idx]
            is_peak = dy_before >= 0 and dy_after <= 0
            is_valley = dy_before <= 0 and dy_after >= 0
            if is_peak or is_valley:
                candidate_indices.append(idx)

        if tip_snap_mode in {"upper", "max", "maximum", "top"}:
            if candidate_indices:
                candidate_values = np.asarray(
                    [potential_plot_series[idx] for idx in candidate_indices],
                    dtype=float,
                )
                candidate_deltas = np.asarray(
                    [abs((echem_plot_datetimes[idx] - center_time).total_seconds()) for idx in candidate_indices],
                    dtype=float,
                )
                sort_order = np.lexsort((candidate_deltas, -candidate_values))
                best_idx = int(candidate_indices[int(sort_order[0])])
            else:
                best_idx = int(window_indices[int(np.nanargmax(potential_plot_series[window_indices]))])
            best_delta = abs((echem_plot_datetimes[best_idx] - center_time).total_seconds())
            return echem_plot_datetimes[best_idx], float(best_delta)

        if tip_snap_mode in {"lower", "min", "minimum", "bottom"}:
            if candidate_indices:
                candidate_values = np.asarray(
                    [potential_plot_series[idx] for idx in candidate_indices],
                    dtype=float,
                )
                candidate_deltas = np.asarray(
                    [abs((echem_plot_datetimes[idx] - center_time).total_seconds()) for idx in candidate_indices],
                    dtype=float,
                )
                sort_order = np.lexsort((candidate_deltas, candidate_values))
                best_idx = int(candidate_indices[int(sort_order[0])])
            else:
                best_idx = int(window_indices[int(np.nanargmin(potential_plot_series[window_indices]))])
            best_delta = abs((echem_plot_datetimes[best_idx] - center_time).total_seconds())
            return echem_plot_datetimes[best_idx], float(best_delta)

        if len(window_indices) >= 3:
            first_idx = int(window_indices[0])
            last_idx = int(window_indices[-1])
            x_window = np.asarray(
                [(echem_plot_datetimes[idx] - echem_plot_datetimes[first_idx]).total_seconds() for idx in window_indices],
                dtype=float,
            )
            y_window = potential_plot_series[window_indices]
            if x_window[-1] > x_window[0]:
                baseline = np.interp(
                    x_window,
                    [x_window[0], x_window[-1]],
                    [potential_plot_series[first_idx], potential_plot_series[last_idx]],
                )
                if candidate_indices:
                    candidate_pos = np.asarray(
                        [int(np.where(window_indices == idx)[0][0]) for idx in candidate_indices],
                        dtype=int,
                    )
                    candidate_prominence = np.abs(y_window[candidate_pos] - baseline[candidate_pos])
                    candidate_deltas = np.asarray(
                        [abs((echem_plot_datetimes[idx] - center_time).total_seconds()) for idx in candidate_indices],
                        dtype=float,
                    )
                    sort_order = np.lexsort((candidate_deltas, -candidate_prominence))
                    best_idx = int(candidate_indices[int(sort_order[0])])
                    return echem_plot_datetimes[best_idx], float(candidate_deltas[int(sort_order[0])])

                residual = np.abs(y_window - baseline)
                best_idx = int(window_indices[int(np.nanargmax(residual))])
                best_delta = abs((echem_plot_datetimes[best_idx] - center_time).total_seconds())
                return echem_plot_datetimes[best_idx], float(best_delta)

        if candidate_indices:
            candidate_deltas = np.asarray(
                [abs((echem_plot_datetimes[idx] - center_time).total_seconds()) for idx in candidate_indices],
                dtype=float,
            )
            candidate_values = np.asarray(
                [potential_plot_series[idx] for idx in candidate_indices],
                dtype=float,
            )
            center_value = echem_value_at_time(center_time, potential_plot_series)
            candidate_prominence = np.abs(candidate_values - center_value)
            sort_order = np.lexsort((candidate_deltas, -candidate_prominence))
            best_idx = candidate_indices[int(sort_order[0])]
            return echem_plot_datetimes[best_idx], float(candidate_deltas[int(sort_order[0])])

        nearest_idx = int(window_indices[int(np.nanargmin(deltas[window_indices]))])
        nearest_delta = abs((echem_plot_datetimes[nearest_idx] - center_time).total_seconds())
        return echem_plot_datetimes[nearest_idx], float(nearest_delta)

    def image_is_one_of_two_nearest_to_time(image_idx, target_time):
        deltas = np.asarray(
            [abs((dt - target_time).total_seconds()) for dt in img_datetimes],
            dtype=float,
        )
        nearest_count = min(2, len(deltas))
        nearest_image_indices = set(np.argsort(deltas)[:nearest_count].astype(int))
        return int(image_idx) in nearest_image_indices

    def nearest_tip_for_image_index(image_idx):
        image_time = img_datetimes[image_idx]

        center_times = []
        if image_idx in tip_time_by_image_idx:
            matched_tip_time = tip_time_by_image_idx[image_idx]
            if abs((matched_tip_time - image_time).total_seconds()) <= tip_snap_tolerance_seconds:
                center_times.append(matched_tip_time)
        center_times.append(image_time)

        for center_time in center_times:
            tip_time, tip_delta = potential_extremum_near_time(
                center_time,
                tip_snap_tolerance_seconds,
            )
            if tip_time is not None and image_is_one_of_two_nearest_to_time(image_idx, tip_time):
                return tip_time, tip_delta

        if not tip_times:
            return None, None

        deltas = np.asarray(
            [abs((event_time - image_time).total_seconds()) for event_time in tip_times],
            dtype=float,
        )
        nearest_tip_pos = int(np.nanargmin(deltas))
        nearest_delta = float(deltas[nearest_tip_pos])
        if nearest_delta <= tip_snap_tolerance_seconds:
            tip_time, tip_delta = potential_extremum_near_time(
                tip_times[nearest_tip_pos],
                tip_snap_tolerance_seconds,
            )
            if tip_time is not None and image_is_one_of_two_nearest_to_time(image_idx, tip_time):
                return tip_time, tip_delta
            return None, nearest_delta
        return None, nearest_delta

    def display_time_for_image_index(image_idx):
        if snap_tips_toggle.value:
            tip_time, _delta_seconds = nearest_tip_for_image_index(image_idx)
            if tip_time is not None:
                return tip_time, True
        return img_datetimes[image_idx], False

    def sort_image_indices_by_time(image_indices):
        return sorted(image_indices, key=lambda idx: img_datetimes[int(idx)])

    def point_arrays():
        if not selected_image_indices:
            return np.empty((0, 2), dtype=float), [], []

        points = []
        snapped_flags = []
        clipped_flags = []
        for image_idx in selected_image_indices:
            display_time, is_snapped = display_time_for_image_index(image_idx)
            y_value = echem_value_at_time(display_time, potential_plot_series)
            y_value, is_clipped = visible_y_value(y_value)
            points.append(
                (
                    (display_time - t0).total_seconds() / 3600.0,
                    y_value,
                )
            )
            snapped_flags.append(is_snapped)
            clipped_flags.append(is_clipped)
        return np.asarray(points, dtype=float), snapped_flags, clipped_flags

    state = {"active_pos": None, "dragging": False}
    label_artists = []

    marker_area = max(140, event_marker_size * 35)
    selected_scatter = ax1.scatter(
        [],
        [],
        s=marker_area,
        facecolors="tab:orange",
        edgecolors="black",
        linewidths=1.4,
        zorder=8,
    )

    status_html = widgets.HTML()
    help_html = widgets.HTML(
        "<b>Interactive selection:</b> drag orange markers along the EC curve; "
        "dragging still selects the nearest neutron image. Enable <b>Snap Tips</b> "
        "to draw markers at nearby local curve tips while keeping the selected "
        "image indices unchanged. Click a marker, then use <b>Left/Right</b> "
        "arrow keys to move it by one neutron image; <b>Up/Down</b> selects the "
        "previous/next marker."
    )
    top_text_html = widgets.HTML()
    if top_text:
        top_text_html.value = (
            "<pre style='white-space: pre-wrap; margin: 0 0 8px 0; "
            "font-family: monospace; font-size: 12px;'>"
            f"{html.escape(str(top_text))}"
            "</pre>"
        )
    backend_html = widgets.HTML()
    backend_name = str(plt.get_backend()).lower()
    if "inline" in backend_name:
        detail = ""
        if widget_backend_error is not None:
            detail = f" Automatic widget-backend switch failed: {html.escape(str(widget_backend_error))}."
        backend_html.value = (
            '<span style="color:#b45309;">Current matplotlib backend looks static '
            f"({plt.get_backend()}). If dragging does not respond, rerun this cell "
            "after enabling <code>%matplotlib widget</code>."
            f"{detail}</span>"
        )

    control_layout = widgets.Layout(width="150px")
    wide_control_layout = widgets.Layout(width="170px")
    add_mode = widgets.ToggleButton(description="Add Point", value=False, layout=control_layout)
    snap_tips_toggle = widgets.ToggleButton(
        description="Snap Tips",
        value=snap_markers_to_tips,
        layout=control_layout,
    )
    prev_marker_button = widgets.Button(description="Prev Marker", layout=control_layout)
    left_button = widgets.Button(description="← Image", layout=control_layout)
    right_button = widgets.Button(description="Image →", layout=control_layout)
    next_marker_button = widgets.Button(description="Next Marker", layout=control_layout)
    delete_button = widgets.Button(
        description="Delete Selected",
        button_style="warning",
        layout=wide_control_layout,
    )
    reset_button = widgets.Button(description="Reset Initial", layout=control_layout)
    auto_button = widgets.Button(description="Use All Auto", layout=control_layout)
    sort_button = widgets.Button(description="Sort by Time", layout=control_layout)
    clear_button = widgets.Button(description="Clear", button_style="warning", layout=control_layout)

    def active_position():
        active_pos = state.get("active_pos")
        if active_pos is None or active_pos >= len(selected_image_indices):
            return None
        return active_pos

    def label_for_position(pos):
        return chr(97 + int(label_start) + pos)

    def refresh_status(extra_message=None):
        pairs = []
        for pos, image_idx in enumerate(selected_image_indices):
            image_time = img_datetimes[image_idx].strftime("%Y-%m-%d %H:%M:%S")
            display_time, is_snapped = display_time_for_image_index(image_idx)
            snap_suffix = ""
            if is_snapped:
                snap_time = display_time.strftime("%H:%M:%S")
                snap_suffix = f" -> tip {snap_time}"
            prefix = "*" if pos == active_position() else ""
            pairs.append(f"{prefix}{label_for_position(pos)}:{image_idx} ({image_time}{snap_suffix})")

        if pairs:
            status = "<b>Selected image indices:</b> " + ", ".join(pairs)
        else:
            status = "<b>Selected image indices:</b> none"
        if extra_message:
            status += f"<br><span>{extra_message}</span>"
        status_html.value = status

    def refresh_plot(extra_message=None):
        points, snapped_flags, clipped_flags = point_arrays()
        selected_scatter.set_offsets(points)

        active_pos = active_position()
        if len(selected_image_indices) > 0:
            edgecolors = [
                "tab:red" if pos == active_pos else "tab:purple" if clipped_flags[pos] else "tab:cyan" if snapped_flags[pos] else "black"
                for pos in range(len(selected_image_indices))
            ]
            linewidths = [
                2.6 if pos == active_pos else 1.4
                for pos in range(len(selected_image_indices))
            ]
            selected_scatter.set_edgecolors(edgecolors)
            selected_scatter.set_linewidths(linewidths)

        for artist in label_artists:
            artist.remove()
        label_artists.clear()
        for pos, (x_val, y_val) in enumerate(points):
            label_artists.append(
                ax1.annotate(
                    label_for_position(pos),
                    (x_val, y_val),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=event_label_fontsize,
                    fontweight="bold",
                    color="tab:red" if pos == active_pos else "black",
                    zorder=9,
                )
            )

        refresh_status(extra_message=extra_message)
        fig.canvas.draw_idle()

    def set_selected_indices(new_indices, active_pos=None, active_image_idx=None, extra_message=None):
        normalized = operando_plots.normalize_selected_image_indices(
            new_indices,
            len(img_datetimes),
        )
        normalized = sort_image_indices_by_time(normalized)
        selected_image_indices[:] = normalized
        if active_image_idx is not None and active_image_idx in selected_image_indices:
            state["active_pos"] = selected_image_indices.index(active_image_idx)
        elif active_pos is None or not selected_image_indices:
            state["active_pos"] = None
        else:
            state["active_pos"] = max(0, min(int(active_pos), len(selected_image_indices) - 1))
        refresh_plot(extra_message=extra_message)

    def marker_hit_position(event, max_pixels=16):
        if event.x is None or event.y is None or not selected_image_indices:
            return None
        points, _snapped_flags, _clipped_flags = point_arrays()
        display_points = ax1.transData.transform(points)
        distances = np.hypot(display_points[:, 0] - event.x, display_points[:, 1] - event.y)
        nearest_pos = int(np.nanargmin(distances))
        if distances[nearest_pos] <= max_pixels:
            return nearest_pos
        return None

    def move_active_to_x(x_value):
        active_pos = active_position()
        if active_pos is None:
            return
        new_image_idx = nearest_image_index(x_value)
        current_idx = selected_image_indices[active_pos]
        if new_image_idx != current_idx and new_image_idx in selected_image_indices:
            refresh_plot(extra_message=f"Image index {new_image_idx} is already selected.")
            return
        new_indices = list(selected_image_indices)
        new_indices[active_pos] = new_image_idx
        set_selected_indices(new_indices, active_image_idx=new_image_idx)

    def move_active_by_images(step):
        active_pos = active_position()
        if active_pos is None:
            refresh_plot(extra_message="Click a marker first, then use the arrow keys.")
            return
        current_idx = selected_image_indices[active_pos]
        new_image_idx = max(0, min(len(img_datetimes) - 1, current_idx + int(step)))
        if new_image_idx == current_idx:
            refresh_plot(extra_message=f"Image index {current_idx} is already at the boundary.")
            return
        if new_image_idx in selected_image_indices:
            refresh_plot(extra_message=f"Image index {new_image_idx} is already selected.")
            return
        old_label = label_for_position(active_pos)
        new_indices = list(selected_image_indices)
        new_indices[active_pos] = new_image_idx
        set_selected_indices(
            new_indices,
            active_image_idx=new_image_idx,
            extra_message=f"Moved {old_label} to image index {new_image_idx}. Labels were reordered by time.",
        )

    def select_marker_by_step(step):
        if not selected_image_indices:
            refresh_plot(extra_message="No markers are selected.")
            return
        active_pos = active_position()
        if active_pos is None:
            active_pos = 0
        else:
            active_pos = (active_pos + int(step)) % len(selected_image_indices)
        state["active_pos"] = active_pos
        refresh_plot(extra_message=f"Active marker: {label_for_position(active_pos)}.")

    def add_point_at_x(x_value):
        image_idx = nearest_image_index(x_value)
        if image_idx in selected_image_indices:
            state["active_pos"] = selected_image_indices.index(image_idx)
            refresh_plot(extra_message=f"Image index {image_idx} is already selected.")
            return
        set_selected_indices(
            list(selected_image_indices) + [image_idx],
            active_image_idx=image_idx,
            extra_message="Added point. Labels were reordered by time.",
        )

    def on_press(event):
        if event.inaxes is not ax1 or event.xdata is None:
            return
        hit_pos = marker_hit_position(event)
        if hit_pos is not None:
            state["active_pos"] = hit_pos
            state["dragging"] = True
            refresh_plot()
            return
        if add_mode.value:
            add_point_at_x(event.xdata)

    def on_motion(event):
        if not state.get("dragging") or event.inaxes is not ax1 or event.xdata is None:
            return
        move_active_to_x(event.xdata)

    def on_release(event):
        if state.get("dragging") and event.inaxes is ax1 and event.xdata is not None:
            move_active_to_x(event.xdata)
        state["dragging"] = False

    def on_delete(_button):
        active_pos = active_position()
        if active_pos is None:
            refresh_plot(extra_message="No active marker selected.")
            return
        new_indices = list(selected_image_indices)
        del new_indices[active_pos]
        next_active = min(active_pos, len(new_indices) - 1) if new_indices else None
        set_selected_indices(new_indices, active_pos=next_active)

    def on_reset(_button):
        set_selected_indices(initial_selected_image_indices)

    def on_auto(_button):
        set_selected_indices(auto_image_indices)

    def on_sort(_button):
        set_selected_indices(sorted(selected_image_indices), active_pos=None)

    def on_clear(_button):
        set_selected_indices([])

    def on_snap_tips(change):
        if change.get("name") == "value":
            refresh_plot()

    def on_key_press(event):
        key = (event.key or "").lower()
        if key in {"left", "ctrl+left", "cmd+left", "j", ","}:
            move_active_by_images(-1)
        elif key in {"right", "ctrl+right", "cmd+right", "l", "."}:
            move_active_by_images(1)
        elif key in {"down", "tab"}:
            select_marker_by_step(1)
        elif key in {"up", "shift+tab"}:
            select_marker_by_step(-1)
        elif key in {"delete", "backspace"}:
            on_delete(None)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    delete_button.on_click(on_delete)
    prev_marker_button.on_click(lambda _button: select_marker_by_step(-1))
    left_button.on_click(lambda _button: move_active_by_images(-1))
    right_button.on_click(lambda _button: move_active_by_images(1))
    next_marker_button.on_click(lambda _button: select_marker_by_step(1))
    reset_button.on_click(on_reset)
    auto_button.on_click(on_auto)
    sort_button.on_click(on_sort)
    clear_button.on_click(on_clear)
    snap_tips_toggle.observe(on_snap_tips, names="value")

    controls_row_1 = widgets.HBox(
        [
            add_mode,
            snap_tips_toggle,
            prev_marker_button,
            left_button,
            right_button,
            next_marker_button,
        ],
        layout=widgets.Layout(flex_flow="row wrap", gap="6px 8px"),
    )
    controls_row_2 = widgets.HBox(
        [
            delete_button,
            reset_button,
            auto_button,
            sort_button,
            clear_button,
        ],
        layout=widgets.Layout(flex_flow="row wrap", gap="6px 8px"),
    )
    controls = widgets.VBox([controls_row_1, controls_row_2], layout=widgets.Layout(gap="6px"))
    display_items = [help_html]
    if top_text_html.value:
        display_items.append(top_text_html)
    display_items.extend([backend_html, status_html, controls])
    canvas_displayed = False
    if isinstance(fig.canvas, widgets.Widget):
        display_items.append(fig.canvas)
        canvas_displayed = True
    display_func(widgets.VBox(display_items))

    refresh_plot()
    if canvas_displayed:
        fig.canvas.draw_idle()
    else:
        display_func(fig)
    if not canvas_displayed:
        plt.close(fig)

    return {
        "selected_image_indices": selected_image_indices,
        "initial_selected_image_indices": initial_selected_image_indices,
        "candidate_image_indices": auto_image_indices,
        "tip_times": tip_times,
        "tip_snap_tolerance_seconds": tip_snap_tolerance_seconds,
        "figure": fig,
        "widgets": {
            "status_html": status_html,
            "help_html": help_html,
            "backend_html": backend_html,
            "add_mode": add_mode,
            "snap_tips_toggle": snap_tips_toggle,
            "delete_button": delete_button,
            "reset_button": reset_button,
            "auto_button": auto_button,
            "sort_button": sort_button,
            "clear_button": clear_button,
        },
    }

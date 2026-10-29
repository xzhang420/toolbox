"""Paired Markdown/PDF overviews of recorded results; never run pipeline sections.

The Markdown contains numerical results and relative image links. The PDF embeds
the same figures. Only report-owned files are refreshed, not source data, exports
or settings. Matplotlib is already a core dependency; no PDF service is needed.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
import textwrap
from urllib.parse import quote

import numpy as np

from .config import load_config
from .handover import MANIFEST, contained_path, file_digest
from .image_io import read_image
from .report_interpretation import load_interpretation, experiment_interpretation, interpretation_pdf_pages

MARKER = "<!-- lumacam-overview schema=1 -->"


@dataclass
class ResultReport:
    root: Path
    title: str
    facts: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    links: list = field(default_factory=list)
    figures: list = field(default_factory=list)
    states: list = field(default_factory=list)
    count: int = 0
    tables: list = field(default_factory=list)
    highlights: list = field(default_factory=list)


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _text(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _settings_facts(report, config, sections, prefix=""):
    for name, keys in sections.items():
        settings = config.get(name, {})
        details = [f"{key}={_text(settings[key])}" for key in keys if key in settings]
        if details:
            report.facts.append((prefix + name.replace("_", " "), "; ".join(details)))


def _figure(path, title, caption=""):
    return {"path": Path(path), "title": title, "caption": caption}


def _new_figure(size):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    figure = Figure(figsize=size, constrained_layout=True)
    FigureCanvasAgg(figure)
    return figure


def _preview(report, paths, work, config=None):
    """Bounded QC only: at most three stored frames, never averaging a new image."""
    if not paths:
        return
    indices = sorted(set(np.linspace(0, len(paths)-1, min(3, len(paths)), dtype=int)))
    images = [read_image(paths[i], fits_hdu=(config or {}).get("fits_hdu"))[0] for i in indices]
    finite = [image[np.isfinite(image)] for image in images]
    finite = [a for a in finite if a.size]
    low = min(float(a.min()) for a in finite) if finite else 0.
    high = max(float(a.max()) for a in finite) if finite else 1.
    fig = _new_figure((10, 3.8))
    axes = fig.subplots(1, len(images), squeeze=False)
    for axis, index, image in zip(axes.flat, indices, images):
        artist = axis.imshow(image, origin="upper", cmap="inferno", vmin=low, vmax=high if high > low else low+1)
        axis.set_title(f"Frame {index}")
        axis.set_xlabel("x (pixel)"); axis.set_ylabel("y (pixel)")
    unit = dict(report.facts).get("Image unit", "unknown / a.u.")
    fig.colorbar(artist, ax=list(axes.flat), label=unit)
    path = work / "preview.png"
    fig.savefig(path, dpi=130)
    report.figures.append(_figure(path, "Representative stored images",
        "First, middle and last frames (or all frames for a shorter set); zero-based indices. "
        "Common display limits use only these previews. No filtering or averaging was applied."))


def _measurements(report, path, work, *, plot=True):
    import pandas as pd
    table = pd.read_csv(path)
    if not {"frame", "value", "label"}.issubset(table.columns):
        raise ValueError(f"Unrecognized measurement table: {path}")
    values = pd.to_numeric(table.value, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(values)
    label = str(table.label.iloc[0]) if len(table) else "Image measurement"
    report.facts.append(("Measurement", label))
    report.facts.append(("Measured frames", f"{int(valid.sum())}/{len(table)} finite values"))
    if "quality_flags" in table:
        flags = table.quality_flags.fillna("").astype(str)
        flagged = int(flags.ne("").sum())
        if flagged:
            report.warnings.append(f"{flagged} frames have measurement quality flags; inspect quality_flags and pixel counts in the table.")
    if not valid.all():
        report.warnings.append(f"{int((~valid).sum())} nonfinite measurements are retained in the table.")
    if not valid.any():
        return
    peak = int(np.nanargmax(np.where(valid, values, np.nan)))
    report.facts.extend([("Minimum / maximum", f"{values[valid].min():.6g} / {values[valid].max():.6g} ({label})"),
                         ("Peak frame", str(table.frame.iloc[peak])),
                         ("Last finite value", f"{values[valid][-1]:.6g} ({label})")])
    zeros = table.frame[valid & (values == 0)].tolist()
    if zeros:
        mean = "primary_method" not in table or table.primary_method.iloc[0] == "mean"
        phrase = "Zero-mean measurement frames" if mean else "Zero-valued primary measurements"
        report.warnings.append(f"{phrase} (zero-based): {zeros}. "
                               "Review their source images; a zero measurement does not prove the entire image is empty. Values were retained.")
    times = pd.to_datetime(table.get("time", pd.Series([None]*len(table))), errors="coerce", utc=True)
    timed = times.notna().all()
    x = (times-times.iloc[0]).dt.total_seconds().to_numpy()/3600 if timed else table.frame.to_numpy()
    if timed:
        report.facts.extend([("Image time range (UTC)", f"{times.min().isoformat()} to {times.max().isoformat()}"),
                             ("Duration (hours)", f"{(times.max()-times.min()).total_seconds()/3600:.6g}")])
    if not plot:
        return
    fig = _new_figure((10, 4.5)); axis = fig.subplots()
    # Do not draw connecting lines across missing measurements or long acquisition gaps.
    xx, yy = [], []
    for i, (a, b) in enumerate(zip(x, values)):
        if timed and i and a-x[i-1] > 5/60:
            xx.append(np.nan); yy.append(np.nan)
        xx.append(a); yy.append(b if valid[i] else np.nan)
    axis.plot(xx, yy, color="tab:orange", linewidth=1.2)
    axis.set(xlabel="Hours since first image" if timed else "Frame index (zero-based)", ylabel=label)
    axis.grid(alpha=.2)
    path = work / "measurements.png"; fig.savefig(path, dpi=150)
    report.figures.append(_figure(path, "Image measurements",
        "Recorded primary measurements, without additional temporal smoothing or point rejection. "
        + ("Lines break across acquisition gaps longer than five minutes." if timed else "Acquisition times unavailable; frame order is used.")))


def _processing_history(report, source, payload):
    if payload:
        report.links.append(("Source processing manifest", source / MANIFEST))
        report.links.append(("Source processing settings", source / "processing/config_used.ini"))
        _settings_facts(report, payload.get("processing_settings", {}), {
            "corrections": ("prepared_reference_path", "reference_dir", "beamhistory_path", "reference_thickness_um"),
            "background": ("enabled", "source", "background_path", "first_images"),
            "images": ("white_spot_enabled", "frames_per_merge", "spatial_pixel_merge")})
    elif source.name in {"corrected_frames", "processed_frames"} and source.parent.name == "images":
        records = source.parent.parent / "processing/legacy_records"
        metadata = records / "processing_metadata.json"
        if metadata.is_file():
            sections = _json(metadata).get("sections", {})
            parameters = sections.get("obint_correction", {}).get("parameters", {})
            reference = parameters.get("reference_dir")
            if reference:
                report.facts.append(("Historical reference", Path(reference).name))
            if "reference_thickness_um" in parameters:
                report.facts.append(("Historical reference thickness (µm)", str(parameters["reference_thickness_um"])))
            if parameters.get("beamhistory_path"):
                report.facts.append(("Historical beam-history input", Path(parameters["beamhistory_path"]).name))
            bg = sections.get("background_processing", {})
            used = bg.get("background_image", {}).get("extra", {}).get("valid_images_used")
            if used is not None:
                report.facts.append(("Historical BG frames (processing metadata)", str(used)))
            if bg.get("background_subtraction"):
                report.facts.append(("Historical background subtraction", "Recorded; existing final values retained"))
            report.links.append(("Original processing metadata (historical paths)", metadata))
            if (records / "operando_config.ini").is_file():
                report.links.append(("Original operando configuration (historical record)", records / "operando_config.ini"))


def _tof_result(report, path, work):
    if path.name == "tof_global_spectrum.csv":
        import pandas as pd
        table = pd.read_csv(path)
        fig = _new_figure((10, 4.5)); axis = fig.subplots()
        axis.plot(table.iloc[:, 0].to_numpy(), table.iloc[:, 1].to_numpy())
        axis.set(xlabel=str(table.columns[0]), ylabel="Recorded count", title="Global ToF spectrum")
        target = work / "tof_spectrum.png"; fig.savefig(target, dpi=150)
        report.figures.append(_figure(target, "Global ToF spectrum", "Saved spectrum; axis label and calibration retained from the export."))
        report.facts.append(("Spectral channels", str(len(table))))
    elif path.name == "tof_integrated_counts.tiff":
        image, _ = read_image(path)
        fig = _new_figure((6, 5)); axis = fig.subplots()
        artist = axis.imshow(image, origin="upper", cmap="inferno")
        fig.colorbar(artist, ax=axis, label="Recorded count")
        target = work / "tof_integrated.png"; fig.savefig(target, dpi=130)
        report.figures.append(_figure(target, "Integrated ToF image", "Existing integrated image; the cube was not reduced or recomputed."))


def _profile(root, work):
    preset, config = load_config(root / "settings.ini")
    identity = _json(root / "dataset.json")
    notes = _json(root / "source_notes.json") if (root / "source_notes.json").is_file() else {}
    source = (root / str(config.get("input", {}).get("path", "."))).resolve()
    input_name = source.stem if source.is_file() else source.name
    if source.is_file():
        source = source.parent.parent if source.name == "processed_dataset.json" else source.parent
    manifest = source / MANIFEST
    payload = _json(manifest) if manifest.is_file() else {}
    name = notes.get("plan", {}).get("experiment") or payload.get("experiment") or input_name
    report = ResultReport(root, str(name), count=len(identity.get("inputs", [])))
    report.facts = [("Stage", "Analysis & visualization of existing processed inputs"),
        ("Mode", preset), ("Profile", root.name), ("Frames", str(report.count)),
        ("Dataset ID", identity.get("dataset_id", "unavailable")),
        ("Image unit", config.get("input", {}).get("unit") or payload.get("unit") or "unknown / a.u."),
        ("Variant", payload.get("variant") or "not recorded in this profile")]
    report.links = [("Editable visualization settings", root / "settings.ini"), ("Input identity", root / "dataset.json")]
    _processing_history(report, source, payload)
    if config.get("input", {}).get("cube"):
        report.facts.append(("Input type", "Reconstructed ToF cube; frame count above counts input files, not spectral channels"))
    report.warnings.append("Report assembly did not run reduction, corrections or new scientific measurements. "
                           "Numbers and plots describe recorded results, not a new validation of calibration.")
    latest = {}
    for path in sorted((root / "records").glob("*/export.json")):
        row = _json(path)
        if row.get("dataset_id") == identity.get("dataset_id"):
            latest[row["section"]] = (path, row)
    if not latest:
        report.warnings.append("No successful section exports recorded yet. Only input/settings information is available.")
    combined_available = any(
        section in {"combined_plot", "nci_event_outputs"}
        and any(Path(item["path"]).suffix.lower() == ".png"
                and (section == "combined_plot" or Path(item["path"]).name.startswith(("potential_", "electrochemistry_")))
                and contained_path(root, item["path"]).is_file()
                and file_digest(contained_path(root, item["path"])) == item.get("sha256")
                for item in row.get("outputs", []))
        for section, (_, row) in latest.items())
    # The combined NCI action and individual figure buttons can produce the
    # same presentation. Keep only the newest successful figure of each role.
    presentation = {}
    for section, (record, row) in sorted(latest.items(), key=lambda pair: pair[1][0].parent.name):
        if section not in {"image_plot", "combined_plot", "nci_event_outputs"}:
            continue
        for item in row.get("outputs", []):
            path = contained_path(root, item["path"])
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg"} or not path.is_file():
                continue
            if file_digest(path) != item.get("sha256"):
                continue
            role = ("curve" if section == "combined_plot" or path.name.startswith(("potential_", "electrochemistry_")) else "panel")
            presentation[role] = (section, path)
    for section, (record, row) in latest.items():
        report.states.append((section, "recorded successful export", record.parent.name))
        for item in row.get("outputs", []):
            path = contained_path(root, item["path"])
            if not path.is_file() or (item.get("sha256") and file_digest(path) != item["sha256"]):
                report.warnings.append(f"Missing or changed export excluded: {item['path']}")
                continue
            if section != "individual_image_plots":
                report.links.append((path.name, path))
            if section == "tof_cube":
                _tof_result(report, path, work)
            if section == "inhomogeneity" and path.suffix.lower() == ".png":
                radial = path.stem == 'radial_profiles'
                report.figures.append(_figure(path, 'Radial deposition profiles' if radial else "Plating pattern variation",
                    'Annular means in the saved circular ROI; letters match selected images. ROI sensitivity is diagnostic only.' if radial else
                    "CV and percentile spread use signed finite ROI values. Red open markers flag bright outliers or background-dominated regions. "
                    "Without a validated signal floor, interpret CV cautiously near zero."))
            if section == 'quality' and path.name == 'image_quality.json':
                quality = _json(path)
                if not quality.get('electrode_boundary_confirmed'):
                    report.warnings.append('Electrode boundary is unconfirmed; inspect the recorded ROI. Cross-experiment uniformity rankings require geometry validation.')
                fit=quality.get('centre_estimation') or {}
                if fit:
                    report.highlights.append(f"Diameter {fit['diameter_mm']:g} mm; "
                        + ('image-supported centre estimate applied' if fit.get('accepted') else 'centre remains uncertain; historical centre retained')+'.')
                if quality.get('suspected_outlier_frames'):
                    report.warnings.append(f"Suspected bright-outlier frames: {quality['suspected_outlier_frames']}. Inspect before interpreting CV.")
                report.highlights.append(f"Image quality: {len(quality.get('flagged_frames',[]))} flagged frame(s).")
                if quality.get('baseline_indices'):
                    report.facts.append(('Selected baseline frames', quality['baseline_indices']))
                    report.facts.append(('Baseline ROI mean / temporal SD', f"{quality.get('baseline_mean')} / {quality.get('baseline_temporal_sd')}"))
                    if not quality.get('baseline_is_independent'):
                        report.warnings.append('Selected baseline frames also informed background preparation; baseline variability is not an independent detection limit.')
            if section == 'quality' and path.suffix.lower() == '.png':
                report.figures.append(_figure(path, 'Measurement-region diagnostic',
                    'Cyan outlines the measurement ROI; display contrast is diagnostic only. Confirm the electrode outline independently of bright deposits.'))
            if section == 'cycle_summary' and path.name == 'phase_endpoints.csv':
                import pandas as pd
                endpoints = pd.read_csv(path)
                headers = ['Image','Phase','Mean','CV','Relative to OCV','Status']
                rows = [[str(r.label), str(r.caption), f'{r.value:.3f}',
                         f'{r.cv:.2f}' if np.isfinite(getattr(r,"cv",np.nan)) else '—',
                         f'{r.value_minus_initial_ocv:.3f}', str(r.status)]
                        for r in endpoints.itertuples()]
                report.tables.append(('Recorded phase endpoints (image units; thickness in µm when calibrated)', headers, rows))
                stripped = endpoints.loc[endpoints.phase == 'stripping']
                if not stripped.empty:
                    report.highlights.append(f"Stripping-end signal: {stripped.value.iloc[0]:.3f} → {stripped.value.iloc[-1]:.3f} "
                                             f"{endpoints.unit.iloc[0]}; final endpoint status: {stripped.status.iloc[-1]}.")
                if endpoints.timing_basis.astype(str).str.contains('unavailable').any():
                    report.warnings.append('Exposure intervals are unavailable for these legacy images; endpoint matching uses representative timestamps.')
            if section == 'cycle_summary' and path.suffix.lower() == '.png':
                report.figures.append(_figure(path, 'Cycle endpoints and residual signal',
                    'Recorded phase endpoints and stripping-end signal relative to initial OCV. This is not a coulombic efficiency or a chemical identification of residual lithium.'))
            if section == "image_measurements" and path.name == "image_measurements.csv":
                _measurements(report, path, work, plot=not combined_available)
                _, measured = load_config(record.parent / "settings.ini")
                _settings_facts(report, measured, {"image_measurements": ("region", "roi", "methods", "primary_method",
                    "electrode_area_mm2", "active_threshold", "trim_fraction", "label"),
                    "input": ("pixel_size_um", "unit", "timestamp_timezone")}, prefix="Recorded measurement: ")
            if section == "synchronization" and path.name == "synchronized.csv":
                import pandas as pd
                table = pd.read_csv(path)
                matched = int(table.echem_time.notna().sum())
                report.facts.append(("Electrochemistry matches", f"{matched}/{len(table)}; {len(table)-matched} unmatched"))
                if matched < len(table):
                    report.warnings.append("Unmatched images remain in the image measurements. "
                                           "Missing electrochemistry matches are not evidence of missing images.")
            if section in {"image_plot", "combined_plot", "nci_event_outputs", "tof_cube"} and path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                if section != "tof_cube" and (section, path) not in presentation.values():
                    continue
                panel = section == "image_plot" or (section == "nci_event_outputs" and not path.name.startswith(("potential_", "electrochemistry_")))
                caption = (("Letters match the marked curve points. " if combined_available else "Letters identify selected images. ")
                           + "When shown, panel values use the recorded primary measurement method; the colour scale and first-panel scale bar apply throughout."
                           if panel else "Potential and image-derived ROI measurements share the time axis. Letters identify the selected images; image and event times may differ within the saved tolerance.")
                title = "Image panel" if panel else "Electrochemistry and image measurements"
                if section == "tof_cube":
                    title, caption = "Time-of-flight result", "Recorded reconstructed-cube analysis; no new reduction."
                report.figures.append(_figure(path, title, caption))
        if section == "individual_image_plots":
            report.facts.append(("Individual image figures", str(len(row.get("outputs", [])))))
        if section in {"csv_export", "image_plot", "combined_plot", "image_video", "combined_video", "nci_event_outputs"}:
            _, exported = load_config(record.parent / "settings.ini")
            _settings_facts(report, exported, {"time_window": ("enabled", "mode", "start", "end", "apply_figures", "apply_panels", "apply_videos"),
                "csv_export": ("scope", "tolerance_seconds") if section == "csv_export" else (),
                "combined_plot": ("image_methods",) if section == "combined_plot" else ()}, prefix=f"{section}: ")
        report.links.append((f"{section}: settings used", record.parent / "settings.ini"))
    _settings_facts(report, config, {"input": ("pixel_size_um", "unit", "timestamp_timezone"),
        "image_plot": ("vmin", "vmax", "cmap"),
        "electrochemistry": ("path", "timestamp_timezone", "capacity_to_thickness", "thickness_capacity_mode"),
        "synchronization": ("direction", "tolerance_seconds", "allow_extrapolation"),
        "image_video": ("fps",), "combined_video": ("fps", "dt_seconds", "limit_to_overlap")}, prefix="Current settings: ")
    report.warnings.append("Figures use the latest successful export of each section. Their settings may differ; "
                           "use the linked per-export settings, not only the current editable INI, to reproduce a figure.")
    if "electrochemistry" not in latest:
        report.warnings.append("Electrochemistry was not included in the recorded exports; image-only results remain valid.")
    if notes.get("unit_basis"):
        report.facts.append(("Calibration provenance", notes["unit_basis"]))
    for key in ("hot_spot_note", "processing_note"):
        if notes.get(key):
            report.warnings.append(notes[key])
    report.warnings.extend(notes.get("quality_notes", []))
    if (root / "verification.json").is_file():
        checks = _json(root / "verification.json")
        if "source_frames_sha256_unchanged" in checks:
            report.facts.append(("Prior run source checksum check", f"{checks['source_frames_sha256_unchanged']} unchanged frames; not rechecked by report assembly"))
        report.links.append(("Prior run verification", root / "verification.json"))
    if not any(item["title"] == "Image panel" for item in report.figures) and not config.get("input", {}).get("cube"):
        try:
            paths = [contained_path(source, row["path"]) for row in identity.get("inputs", [])]
            _preview(report, paths, work, config.get("input", {}))
        except (OSError, ValueError) as exc:
            report.warnings.append(f"Source preview unavailable: {exc}")
    order = {'Image panel':0, 'Electrochemistry and image measurements':1, 'Cycle endpoints and residual signal':2,
             'Plating pattern variation':3, 'Radial deposition profiles':4, 'Measurement-region diagnostic':5}
    report.figures.sort(key=lambda item: order.get(item['title'], 10))
    return report


def _processed(root, work):
    payload = _json(root / MANIFEST)
    report = ResultReport(root, payload.get("experiment", root.name), count=payload["frame_count"])
    report.facts = [("Stage", "Data processing; analysis & visualization has not been run by this report"),
        ("Frames", str(report.count)), ("Source TIFF count", str(payload.get("source_image_count", "unknown"))),
        ("Variant", payload.get("variant", "unknown")), ("Image unit", payload.get("unit") or "unknown / a.u."),
        ("Pixel size (µm)", str(payload.get("pixel_size_um") or "unknown")),
        ("Dataset ID", payload.get("dataset_id", "unknown"))]
    report.links = [("Processed dataset", root / MANIFEST), ("Processing settings used", root / "processing/config_used.ini")]
    recipes = _json(root / "processing/recipes.json") if (root / "processing/recipes.json").is_file() else {}
    report.states = [(key, "recorded completed recipe", "processing/recipes.json") for key in recipes]
    config = payload.get("processing_settings", {})
    _settings_facts(report, config, {"corrections": ("enabled", "prepared_reference_path", "reference_dir", "beamhistory_path", "reference_thickness_um"),
        "background": ("enabled", "source", "background_path", "first_images"),
        "images": ("white_spot_enabled", "white_spot_method", "white_spot_threshold", "frames_per_merge", "spatial_pixel_merge", "roi")})
    paths = [contained_path(root, row["path"]) for row in payload["frames"]]
    _preview(report, paths, work)
    average = payload.get("average")
    if average:
        path = contained_path(root, average)
        image, _ = read_image(path)
        fig = _new_figure((6, 5)); axis = fig.subplots()
        artist = axis.imshow(image, origin="upper", cmap="inferno")
        fig.colorbar(artist, ax=axis, label=payload.get("unit") or "Image intensity (a.u.)")
        axis.set(xlabel="x (pixel)", ylabel="y (pixel)")
        target = work / "average.png"; fig.savefig(target, dpi=130)
        report.figures.append(_figure(target, "Saved experiment-average image", "Display of the existing average TIFF; no new average was calculated."))
        report.links.append(("Average TIFF", path))
        report.facts.append(("Frames contributing to saved average", str(report.count)))
        finite = image[np.isfinite(image)]
        if finite.size:
            report.facts.append(("Average-image pixel min / mean / max", f"{finite.min():.6g} / {finite.mean():.6g} / {finite.max():.6g} (stored image units; whole-image QC)"))
    else:
        report.warnings.append("No saved average is recorded; representative frames are shown instead.")
    report.warnings.append("Processing QC previews are not scientific ROI measurements. "
                           "Open the visualization notebook for thickness/ROI measurements and electrochemistry correlation.")
    return report


def _relative(path, root):
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:  # Different Windows drives.
        return Path(path).as_uri()


def _cell(value):
    return _text(value).replace("|", "\\|").replace("\n", " ")


def _write_pdf(path, title, reports, notes, timestamp, output, interpretation=None):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    import matplotlib.image as mpimg
    page = 0
    with PdfPages(path, metadata={"Title": title, "Author": "LumaCam pipeline", "Subject": "Recorded results and quality overview"}) as pdf:
        def save(fig):
            nonlocal page
            page += 1
            fig.text(.07, .025, f"LumaCam | {timestamp} | Page {page}", fontsize=8, color="0.4")
            pdf.savefig(fig)
        def prose(heading, lines, compact=False):
            wrapped = []
            for line in lines:
                wrapped.extend(textwrap.wrap(str(line), width=103, break_long_words=True) or [""])
                if not compact:
                    wrapped.append("")
            chunks = [wrapped[i:i+57] for i in range(0, len(wrapped), 57)] or [[]]
            for i, chunk in enumerate(chunks):
                fig = Figure(figsize=(8.27, 11.69))
                fig.text(.07, .95, "\n".join(textwrap.wrap(heading + (" (continued)" if i else ""), 65)),
                         va="top", fontsize=14, weight="bold")
                fig.text(.07, .87, "\n".join(chunk), va="top", fontsize=9, linespacing=1.25)
                save(fig)
        interpretation_pdf_pages(interpretation or {}, title, save)
        prose(title, [f"Generated (UTC): {timestamp}", f"Datasets: {len(reports)}; recorded frames: {sum(r.count for r in reports)}.",
            "This report embeds results and summarises saved settings and quality notes. "
            "It does not rerun reduction or corrections. No causal interpretation is inferred."] + list(notes) +
            [f"{r.title}: {r.count} frames; variant {dict(r.facts).get('Variant', 'unknown')}" for r in reports])
        for report in reports:
            human_keys = {"Stage", "Frames", "Variant", "Image unit", "Measurement", "Duration (hours)",
                          "Minimum / maximum", "Electrochemistry matches", "Historical reference thickness (µm)",
                          "Historical BG frames (processing metadata)", "Calibration provenance"}
            lines = [f"{key}: {value}" for key, value in report.facts if key in human_keys]
            finding = experiment_interpretation(interpretation or {}, report)
            lines = [f'{key.title()}: {value}' for key, value in finding.items()] + lines
            lines += ["", "QUALITY AND LIMITATIONS"] + report.warnings
            recorded = [key.replace("_", " ") for key, state, _ in report.states if state == "recorded successful export"]
            lines += ["", "RECORDED SECTIONS"]
            if recorded:
                lines += ["Successful exports: " + "; ".join(recorded) + "."]
            lines += [f"{key}: {state}" for key, state, _ in report.states if state != "recorded successful export"]
            lines += ["", "Full coordinates, timestamps, input paths and exact settings are in overview.md and the linked CSVs/INIs."]
            prose(report.title, lines, compact=True)
            for item in report.figures:
                fig = Figure(figsize=(8.27, 11.69))
                fig.text(.07, .96, "\n".join(textwrap.wrap(report.title, 65)), fontsize=12, va="top", weight="bold")
                fig.text(.07, .90, item["title"], fontsize=11, va="top")
                axis = fig.add_axes([.06, .18, .88, .68])
                axis.imshow(mpimg.imread(item["path"])); axis.axis("off")
                fig.text(.07, .14, "\n".join(textwrap.wrap(item["caption"], 104)), fontsize=9, va="top")
                save(fig)


def _write_concise_pdf(path, title, reports, notes, timestamp, output, interpretation=None):
    """Figure-led pages; full settings/history remain in the companion Markdown."""
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.figure import Figure
    import matplotlib.image as mpimg
    with PdfPages(path, metadata={'Title':title, 'Author':'LumaCam pipeline'}) as pdf:
        page = 0
        def new(heading):
            fig = Figure(figsize=(8.27,11.69))
            fig.text(.07,.96,'\n'.join(textwrap.wrap(heading,64)),fontsize=12,weight='bold',va='top')
            return fig
        def save(fig):
            nonlocal page
            page += 1
            fig.text(.07,.025,f'LumaCam | {timestamp} | {page}',fontsize=7,color='0.4')
            pdf.savefig(fig)
        interpretation = interpretation or {}
        interpretation_pdf_pages(interpretation, title, save)
        fig = new(title); y = .87
        intro = [f'{len(reports)} experiment(s) | {sum(r.count for r in reports)} recorded frames',
                 'Results use recorded exports. Exact settings, CSVs and quality flags are in overview.md.'] + list(notes)
        for line in intro:
            wrapped = textwrap.wrap(str(line),96)
            fig.text(.07,y,'\n'.join(wrapped),fontsize=9,va='top'); y -= .021*len(wrapped)+.015
        for report in reports:
            facts = dict(report.facts)
            lines = [report.title, f"{report.count} frames | EC matches: {facts.get('Electrochemistry matches','not loaded')}"] + report.highlights
            for line in lines:
                wrapped = textwrap.wrap(str(line),96)
                if y-.021*len(wrapped)<.08: save(fig); fig=new(title); y=.88
                fig.text(.07,y,'\n'.join(wrapped),fontsize=9,va='top'); y-=.021*len(wrapped)+.008
            y-=.025
        if not interpretation.get('summary'):
            save(fig)
        for report in reports:
            by_title={item['title']:item for item in report.figures}
            groups=[['Image panel'],['Electrochemistry and image measurements','Cycle endpoints and residual signal'],
                    ['Plating pattern variation','Radial deposition profiles']]
            used={name for group in groups for name in group}
            remaining=[item['title'] for item in report.figures if item['title'] not in used
                       and item['title']!='Measurement-region diagnostic']
            if remaining: groups.append(remaining)
            groups=[[by_title[name] for name in group if name in by_title] for group in groups]
            groups=[group for group in groups if group]
            if not groups: groups=[list(report.figures)]
            for group_index,items in enumerate(groups):
                fig=new(report.title); y=.89
                if group_index==0:
                    finding = experiment_interpretation(interpretation, report)
                    for label in ('observation', 'conclusion'):
                        if finding.get(label):
                            wrapped = textwrap.wrap(label.title()+': '+finding[label], 108)
                            fig.text(.07,y,'\n'.join(wrapped),fontsize=8,va='top',
                                     color='#183b4d',bbox=dict(facecolor='#eaf2f6',edgecolor='none',pad=4))
                            y-=.015*len(wrapped)+.012
                    # A brief quality note plus the actual image panel and
                    # endpoint numbers; no separate methods-only page.
                    warnings=[] if finding else [s for s in report.warnings if any(word in s.lower() for word in
                              ('boundary is unconfirmed','suspected bright-outlier','cv has no'))][:2]
                    for line in warnings:
                        wrapped=textwrap.wrap(line,105)
                        fig.text(.07,y,'\n'.join(wrapped),fontsize=7.5,va='top',color='0.3')
                        y-=.014*len(wrapped)+.006
                for item in items:
                    array=mpimg.imread(item['path'])
                    cap=.39 if group_index==0 and report.tables else .40 if item['title']=='Electrochemistry and image measurements' else .44
                    height=min(cap,.86*8.27*(array.shape[0]/array.shape[1])/11.69)
                    caption=textwrap.wrap(item['caption'],110)
                    needed=height+.03+.012*len(caption)+.018
                    if y-needed<.06: save(fig);fig=new(report.title);y=.89
                    fig.text(.07,y,item['title'],fontsize=9,weight='bold',va='top');y-=.023
                    axis=fig.add_axes([.07,y-height,.86,height]);axis.imshow(array);axis.axis('off');y-=height+.008
                    fig.text(.07,y,'\n'.join(caption),fontsize=7.5,va='top');y-=.012*len(caption)+.018
                if group_index==0:
                    for heading,headers,body in report.tables:
                        height=.0185*(len(body)+1)
                        if y-height-.035<.06:save(fig);fig=new(report.title);y=.89
                        fig.text(.07,y,heading,fontsize=7.5,va='top');y-=.024
                        axis=fig.add_axes([.07,y-height,.86,height]);axis.axis('off')
                        tab=axis.table(cellText=body,colLabels=headers,cellLoc='left',loc='center',
                                       colWidths=[.07,.16,.1,.09,.18,.4])
                        tab.auto_set_font_size(False);tab.set_fontsize(6.5);tab.scale(1,1.2)
                        y-=height+.025
                if not items:fig.text(.07,y,'No recorded figures available.',fontsize=9)
                save(fig)


def create_report(sources, *, output_dir=None, title="LumaCam results overview", variant=None, notes=(), states=None, style='detailed'):
    """Report one or several processed-result/profile folders using saved outputs.

    ``variant`` is an explicit, user-confirmed label override, never inferred
    from filenames. It changes this report only, not datasets or configurations.
    """
    if style not in {'concise','detailed'}: raise ValueError('Report style must be concise or detailed.')
    if isinstance(sources, (str, Path)):
        sources = [sources]
    roots = [Path(p).expanduser().resolve() for p in sources]
    if not roots or len(set(roots)) != len(roots):
        raise ValueError("Select one or more distinct result/profile folders.")
    if len(roots) > 1 and output_dir is None:
        raise ValueError("A combined report needs an explicit output directory.")
    output = Path(output_dir).expanduser().resolve() if output_dir else roots[0]
    interpretation = load_interpretation(output, roots)
    if interpretation.get('warning'):
        notes = list(notes) + [interpretation['warning']]
    if (output / "report_assets").is_symlink():
        raise ValueError("Report assets must not be a symlink to another directory.")
    marker = output / "overview.md"
    if marker.exists() and not marker.read_text(encoding="utf-8").startswith(MARKER):
        raise ValueError("Existing overview.md is not a generated LumaCam report; choose another output folder.")
    if not marker.exists() and (output / "overview.pdf").exists():
        raise ValueError("Existing overview.pdf has no report marker; choose another output folder.")
    for root in roots:
        if not (root / MANIFEST).is_file() and not ((root / "dataset.json").is_file() and (root / "settings.ini").is_file()):
            raise ValueError(f"Select a published processing result or a saved visualization profile: {root}")
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".overview-", dir=output) as temporary:
        stage = Path(temporary)
        reports = []
        for i, root in enumerate(roots):
            work = stage / str(i); work.mkdir()
            report = _profile(root, work) if (root / "dataset.json").is_file() else _processed(root, work)
            if variant:
                report.facts = [(key, value) for key, value in report.facts if key != "Variant"]
                report.facts.append(("Variant", f"{variant} (user-confirmed; folder names unchanged)"))
            if states and len(roots) == 1:
                report.states = [(row["section"], row["state"], row.get("message", "")) for row in states]
            reports.append(report)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        lines = [MARKER, f"# {title}", "", f"Generated: {timestamp}. Datasets: {len(reports)}; recorded frames: {sum(r.count for r in reports)}.", "",
                 "Report assembly only: no reduction, correction, filtering, averaging or new scientific measurement was run.", "",
                 "[Human-readable PDF](overview.pdf)", ""]
        for note in notes:
            lines += [str(note), ""]
        assets = output / "report_assets"
        assets.mkdir(exist_ok=True)
        if interpretation.get('summary'):
            lines += ['## Observations and conclusions', '', interpretation.get('scope',''), '']
            for item in interpretation['summary']:
                lines += [f"### {item['heading']}", '', f"Observation: {item['observation']}", '',
                          f"> **Conclusion:** {item['conclusion']}", '']
            lines += ['### Interpretation limits', ''] + ['- '+v for v in interpretation.get('limitations',[])] + ['']
            for page in interpretation.get('comparison_pages', []):
                figure = page['figure']
                name = file_digest(figure['path'])[:16] + figure['path'].suffix.lower()
                destination = assets / name
                if destination.is_symlink(): raise ValueError('Report image must not be a symlink.')
                if not destination.exists(): shutil.copyfile(figure['path'], destination)
                elif file_digest(destination) != file_digest(figure['path']): raise ValueError('Report image changed.')
                lines += ['### '+page['title'], '', 'Observation: '+page['observation'], '',
                          f"![{page['title']}](report_assets/{name})", '',
                          '> **Conclusion:** '+page['conclusion'], '', figure.get('caption',''), '', page.get('methods',''), '']
            lines += ['[Interpretation and evidence checksums](interpretation.json)', '']
        for report in reports:
            lines += [f"## {report.title}", ""]
            finding = experiment_interpretation(interpretation, report)
            if finding:
                lines += ['Observation: '+finding.get('observation',''), '',
                          '> **Conclusion:** '+finding.get('conclusion',''), '']
            lines += [f'- {line}' for line in report.highlights] + ['']
            for heading, headers, body in report.tables:
                lines += [f'### {heading}', '', '| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']
                lines += ['| '+' | '.join(_cell(value) for value in row)+' |' for row in body] + ['']
            if style == 'concise': lines += ['<details><summary>Settings and numerical audit</summary>', '']
            lines += ["| Item | Result / setting |", "| --- | --- |"]
            lines += [f"| {_cell(key)} | {_cell(value)} |" for key, value in report.facts]
            if style == 'concise': lines += ['', '</details>']
            lines += ["", "### Quality and limitations", ""] + [f"- {warning}" for warning in report.warnings]
            lines += ["", "### Results", ""]
            for item in report.figures:
                digest = file_digest(item["path"])[:16]
                name = f"{digest}{item['path'].suffix.lower()}"
                destination = assets / name
                if destination.is_symlink():
                    raise ValueError(f"Report image must not be a symlink: {destination}")
                if not destination.exists():
                    shutil.copyfile(item["path"], destination)
                elif file_digest(destination) != file_digest(item["path"]):
                    raise ValueError(f"Report image was modified; refusing to overwrite: {destination}")
                lines += [f"#### {item['title']}", "", f"![{item['title']}](report_assets/{name})", "", item["caption"], ""]
            if not report.figures:
                lines += ["No figures are available yet; no results were invented.", ""]
            lines += ["### Section status", "", "| Section | State | Detail |", "| --- | --- | --- |"]
            lines += [f"| {_cell(key)} | {_cell(state)} | {_cell(detail)} |" for key, state, detail in report.states]
            lines += ["", "### Files and reproducibility", ""]
            lines += [f"- [{label}]({quote(_relative(path, output), safe='/.:')})" for label, path in report.links]
            lines += [""]
        lines += ["## Reading this report with AI", "", "Read the numerical tables and quality notes first, then inspect the linked plot images. "
                  "Markdown image links alone do not give a text-only AI access to pixel content. "
                  "Use the linked CSVs and per-export settings for quantitative follow-up. "
                  "Do not infer missing calibration, timing or correction history. "
                  "Report generation does not authorize rerunning earlier workflow stages.", ""]
        (stage / "overview.md").write_text("\n".join(lines), encoding="utf-8")
        (_write_concise_pdf if style == 'concise' else _write_pdf)(stage / "overview.pdf", title, reports, notes, timestamp, output, interpretation)
        # Preserve previous reports if assembly fails. Never overwrite existing plots/settings.
        os.replace(stage / "overview.pdf", output / "overview.pdf")
        os.replace(stage / "overview.md", marker)
    return {"markdown": marker, "pdf": output / "overview.pdf"}


def refresh_session_report(session, section, outcome):
    """Best-effort reporting: failures stay visible without invalidating good data."""
    role = session.config.get("workflow", {}).get("role")
    if role not in {"processing", "visualization"} or not session.config.get("reports", {}).get("enabled", True):
        return
    root = session.artifacts.get("results_dir")
    if not root or (role == "processing" and not (Path(root) / MANIFEST).is_file()):
        return
    if role == "processing" and section not in {"images", "average"}:
        return
    if role == "visualization" and not (Path(root) / "dataset.json").is_file():
        return
    try:
        paths = create_report(root, states=session.summary(), style=session.config.get('reports',{}).get('style','concise'))
        options=session.config.get('reports',{})
        profiles=options.get('combined_profiles') or []
        if profiles:
            parent=(Path(root)/options.get('combined_output','..')).resolve()
            if any(Path(name).name != name or name in {'.','..'} for name in profiles):
                raise ValueError('Combined report profiles must be simple folder names.')
            profiles_root=(parent/options.get('combined_profiles_root','.')).resolve()
            sources=[profiles_root/name for name in profiles]
            if all((path/'dataset.json').is_file() for path in sources):
                combined=create_report(sources,output_dir=parent,style=options.get('style','concise'),
                    title=options.get('title','LumaCam results overview'))
                session.artifacts.put('combined_overview_paths',combined,'reports')
        session.artifacts.put("overview_paths", paths, "reports")
        session.artifacts.put("overview_error", "", "reports")
        outcome.message += f" Overview: {paths['markdown']} and overview.pdf."
    except Exception as exc:
        message = f"Overview report failed (existing results retained): {type(exc).__name__}: {exc}"
        session.artifacts.put("overview_error", message, "reports")
        outcome.message += " " + message

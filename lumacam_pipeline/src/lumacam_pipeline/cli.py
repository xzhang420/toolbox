"""Command-line entry point for inspecting or running selected sections."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .batch import DEFAULT_BATCH_SECTIONS, run_batch, save_batch_plan, validate_batch
from .config import load_config
from .presets import PRESETS, create_session
from pathlib import Path
from contextlib import redirect_stdout
import sys


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="INI file with one section per heading")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="operando_nci")
    parser.add_argument("--data", help="Override project.data_path")
    parser.add_argument("--experiment", help="Override project.experiment")
    parser.add_argument("--variant", help="Override project.variant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show current section readiness")
    subparsers.add_parser("doctor", help="Report dependency and FFmpeg availability as JSON")
    subparsers.add_parser("inspect", help="Inspect proposal layout and suggest paths; no writes")
    report = subparsers.add_parser("report", help="Write Markdown/PDF with recorded results and embedded plots; no processing")
    report.add_argument("--input", nargs="+", required=True, help="Saved visualization profiles or published processing results")
    report.add_argument("--output", help="Report directory; required for multiple inputs")
    report.add_argument("--title", default="LumaCam results overview")
    report.add_argument("--note", action="append", default=[], help="Explicit contextual note; repeat as needed")
    report.add_argument("--style", choices=['detailed', 'concise'], default='concise')
    derive = subparsers.add_parser('derive-background', help='Create a new signed-background variant from existing beam/OB-corrected images')
    derive.add_argument('--input', required=True)
    derive.add_argument('--background', required=True)
    derive.add_argument('--output', required=True)
    derive.add_argument('--name', required=True, help='Full experiment name')
    derive.add_argument('--product-variant', required=True)
    derive.add_argument('--timezone', default='')
    derive.add_argument('--pixel-size-um', type=float)
    derive.add_argument('--unit', default='')
    derive.add_argument('--white-spot-absolute-threshold', type=float)
    process = subparsers.add_parser("process", help="Run processing only and publish portable frames")
    process.add_argument("--sections", nargs="+")
    process.add_argument("--validate-only", action="store_true", help="Inspect selected inputs without processing")
    render = subparsers.add_parser("visualize", help="Analyze or render processed TIFF/FITS images")
    render.add_argument("--input", help="Experiment result directory, manifest, or TIFF/FITS folder/file")
    render.add_argument("--profile", default="overview")
    render.add_argument("--output-root")
    render.add_argument("--cube", action="store_true", help="Open reconstructed 3-D ToF cube")
    render.add_argument("--sections", nargs="+", default=["selection", "image_plot"])
    render.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE",
                        help="Override a setting; JSON numbers/booleans/lists/objects are accepted")
    process_batch = subparsers.add_parser("process-batch", help="Run a processing-only batch")
    process_batch.add_argument("--experiments", nargs="+", required=True)
    process_batch.add_argument("--sections", nargs="+")
    process_batch.add_argument("--name", default="")
    process_batch.add_argument("--validate-only", action="store_true")
    run = subparsers.add_parser("run", help="Run only the named sections")
    run.add_argument("sections", nargs="+", help="Section ids in the desired order")
    batch = subparsers.add_parser("batch", help="Validate or run several experiments")
    batch.add_argument("--experiments", nargs="*", help="Experiment names or short IDs")
    batch.add_argument("--sections", nargs="*", help="Selected batch section IDs")
    batch.add_argument("--name", default="", help="Reusable batch name")
    batch.add_argument("--overwrite", action="store_true", help="Re-run completed jobs")
    batch.add_argument("--validate-only", action="store_true", help="Write the plan without processing")
    return parser


def _main(argv=None):
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        from .diagnostics import doctor
        report = doctor()
        print(json.dumps(report, indent=2))
        return 0 if report["core_ready"] else 1
    if args.command == "report":
        from .reports import create_report
        result = create_report(args.input, output_dir=args.output, title=args.title,
                               variant=args.variant, notes=args.note, style=args.style)
        print(json.dumps(result, default=str, indent=2))
        return 0
    if args.command == 'derive-background':
        from .processed_variant import derive_background_variant
        threshold = args.white_spot_absolute_threshold
        settings = {'white_spot_enabled': threshold is not None, 'white_spot_method': 'abs',
                    'white_spot_threshold': threshold if threshold is not None else 8}
        result = derive_background_variant(args.input, args.background, args.output, experiment=args.name,
            variant=args.product_variant, timestamp_timezone=args.timezone, pixel_size_um=args.pixel_size_um,
            unit=args.unit, white_spots=settings)
        print(json.dumps({'processed_dataset': str(result / 'processing/processed_dataset.json')}))
        return 0
    if args.command == "visualize":
        from .visualization import open_visualization, load_visualization_settings, save_visualization_settings
        values = {}
        for expression in args.set:
            key, value = expression.split("=", 1)
            section, name = key.split(".", 1)
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass
            values.setdefault(section, {})[name] = value
        if args.config:
            session = load_visualization_settings(args.config)
            if args.input:
                raise ValueError("Use either --input or a saved visualization config, not both.")
            for section, updates in values.items():
                if section == "input":
                    raise ValueError("To change input decoding, edit input settings and reload the visualization config.")
                session.configure(section, **updates)
        else:
            if not args.input:
                raise ValueError("Provide visualize --input or --config pointing to visualization settings.ini.")
            session = open_visualization(args.input, profile=args.profile, output_root=args.output_root,
                                         preset=args.preset, config=values, cube=args.cube)
        reports = []
        for section in args.sections:
            if section not in session.sections:
                raise ValueError(f"Unknown analysis section: {section}")
            if section != "time_window":
                session.configure(section, enabled=True)
            with redirect_stdout(sys.stderr):
                outcome = session.run(section)
            reports.append({"section": section, "state": outcome.state.value, "message": outcome.message})
            if outcome.state.value != "completed":
                break
        print(json.dumps({"results": reports, "profile": str(session.artifacts.require("results_dir"))}, indent=2))
        return 0 if reports and all(row["state"] == "completed" for row in reports) else 1
    if args.config:
        preset, config = load_config(args.config)
        if args.data is not None:
            config.setdefault("project", {})["data_path"] = str(Path(args.data).expanduser().resolve())
        if args.command in {"inspect", "process", "process-batch"}:
            from .processing import resolve_processing_paths
            config = resolve_processing_paths(config, args.config)
    else:
        preset, config = args.preset, {}
    project = config.setdefault("project", {})
    for name in ("data", "experiment", "variant"):
        value = getattr(args, name)
        if value is not None:
            project["data_path" if name == "data" else name] = value
    if args.command == "inspect" or (args.command == "process" and args.validate_only):
        from .inspection import inspect_layout
        report = inspect_layout(project.get("data_path", ""), config=config)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    if args.command in {"process", "process-batch"}:
        from .processing import create_processing_session, PROCESSING_SECTIONS
        from .inspection import inspect_layout
        from .config import save_config
        session = create_processing_session(preset, config)
        if args.sections is None:
            args.sections = ["project"] if args.command == "process" else []
            if session.section_config("corrections").get("enabled", True):
                args.sections.append("corrections")
            if session.section_config("background").get("enabled", False):
                args.sections.append("background")
            args.sections.extend(["images", "average"])
        if set(args.sections) - PROCESSING_SECTIONS:
            raise ValueError("Processing actions cannot include analysis/visualization sections.")
        if args.command == "process-batch":
            with redirect_stdout(sys.stderr):
                report = validate_batch(project["data_path"], args.experiments, preset=preset,
                                        config=session.config, sections=args.sections)
            if args.validate_only or not report.valid:
                print(json.dumps({"valid": report.valid, "errors": report.common_errors,
                                  "experiments": [asdict(row) for row in report.rows]}, default=str, indent=2))
                return 0 if report.valid else 1
            main_config = Path(report.proposal_root)
            from .proposal_paths import analysis_root_for_proposal
            save_config(analysis_root_for_proposal(main_config) / "lumacam_pipeline.ini", preset, session.config)
            with redirect_stdout(sys.stderr):
                summary = run_batch(project["data_path"], args.experiments, preset=preset,
                                    config=session.config, sections=args.sections, batch_name=args.name)
            print(json.dumps({"failed": summary.failed_count, "status_path": str(summary.status_path),
                              "report_error": summary.report_error}, indent=2))
            return 1 if summary.failed_count else 0
        reports = []
        for section in args.sections:
            with redirect_stdout(sys.stderr):
                outcome = session.run(section)
            reports.append({"section": section, "state": outcome.state.value, "message": outcome.message})
            if outcome.state.value != "completed":
                break
            if section == "project":
                save_config(Path(session.artifacts.require("analysis_root")) / "lumacam_pipeline.ini", preset, session.config)
        print(json.dumps({"results": reports, "processed_dataset": str(session.artifacts.get("processed_dataset_path", ""))}, indent=2))
        return 0 if reports and all(row["state"] == "completed" for row in reports) else 1
    session = create_session(preset, config)
    if args.command == "status":
        print(json.dumps(session.summary(), indent=2))
        return 0
    if args.command == "batch":
        batch_values = config.pop("batch", {})
        experiments = args.experiments or batch_values.get("experiments", ())
        sections = args.sections or batch_values.get("sections", DEFAULT_BATCH_SECTIONS)
        report = validate_batch(
            project.get("data_path", ""),
            experiments,
            preset=preset,
            config=config,
            sections=sections,
        )
        print(
            json.dumps(
                {
                    "valid": report.valid,
                    "common_errors": list(report.common_errors),
                    "sections": list(report.sections),
                    "experiments": [
                        {**asdict(row), "result_dir": str(row.result_dir)} for row in report.rows
                    ],
                },
                indent=2,
            )
        )
        if not report.valid:
            return 1
        if args.validate_only:
            target = save_batch_plan(
                report, preset=preset, config=config, batch_name=args.name or batch_values.get("name", "")
            )
            print(f"Batch plan saved: {target}")
            return 0
        summary = run_batch(
            project.get("data_path", ""),
            experiments,
            preset=preset,
            config=config,
            sections=sections,
            batch_name=args.name or batch_values.get("name", ""),
            overwrite=args.overwrite,
        )
        print(f"Batch status: {summary.status_path}")
        if summary.report_error:
            print(summary.report_error)
        return 1 if summary.failed_count else 0
    exit_code = 0
    for section_id in args.sections:
        outcome = session.run(section_id)
        print(f"{section_id}: {outcome.state.value}: {outcome.message}")
        if outcome.state.value == "failed":
            exit_code = 1
            break
    return exit_code


def main(argv=None):
    try:
        return _main(argv)
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}))
        return 1

"""Command-line entry point for inspecting or running selected sections."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .batch import DEFAULT_BATCH_SECTIONS, run_batch, save_batch_plan, validate_batch
from .config import load_config
from .presets import PRESETS, create_session


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="INI file with one section per heading")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="operando_nci")
    parser.add_argument("--data", help="Override project.data_path")
    parser.add_argument("--experiment", help="Override project.experiment")
    parser.add_argument("--variant", help="Override project.variant")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="Show current section readiness")
    run = subparsers.add_parser("run", help="Run only the named sections")
    run.add_argument("sections", nargs="+", help="Section ids in the desired order")
    batch = subparsers.add_parser("batch", help="Validate or run several experiments")
    batch.add_argument("--experiments", nargs="*", help="Experiment names or short IDs")
    batch.add_argument("--sections", nargs="*", help="Selected batch section IDs")
    batch.add_argument("--name", default="", help="Reusable batch name")
    batch.add_argument("--overwrite", action="store_true", help="Re-run completed jobs")
    batch.add_argument("--validate-only", action="store_true", help="Write the plan without processing")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    if args.config:
        preset, config = load_config(args.config)
    else:
        preset, config = args.preset, {}
    project = config.setdefault("project", {})
    for name in ("data", "experiment", "variant"):
        value = getattr(args, name)
        if value is not None:
            project["data_path" if name == "data" else name] = value
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
        return 1 if summary.failed_count else 0
    exit_code = 0
    for section_id in args.sections:
        outcome = session.run(section_id)
        print(f"{section_id}: {outcome.state.value}: {outcome.message}")
        if outcome.state.value == "failed":
            exit_code = 1
            break
    return exit_code

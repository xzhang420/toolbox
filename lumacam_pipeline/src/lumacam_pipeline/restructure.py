"""Audited migration of earlier analysis outputs; never touches acquisition data.

Create a plan with --proposal PATH --plan /tmp/plan.json; inspect the report,
then use --apply-plan /tmp/plan.json. Original records and removed duplicate
backgrounds are archived under the proposal's .trash before any move.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from .proposal_paths import analysis_root_for_proposal

from pathlib import Path
import re
import tarfile
import time

from .config import load_config, save_config
from .presets import create_session
from .provenance import recipe_for
from .result_layout import average_filename, result_directory, shared_directory, snapshot_path

TEXT_SUFFIXES = {".json", ".ini", ".csv", ".log", ".txt", ".ipynb", ".md"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mark_relocated_batches(plan):
    """Retain resume eligibility only with a verified relocated config snapshot."""
    root = analysis_root_for_proposal(Path(plan["proposal"]).resolve())
    count = 0
    for record in plan["experiments"]:
        result = Path(record["new"]).resolve()
        if not result.is_relative_to(root) or not snapshot_path(result).is_file():
            raise ValueError(f"Missing or invalid relocated snapshot: {result}")
        for path in (result / "processing").glob("batch_*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("status") == "completed":
                payload["relocated_config_snapshot"] = str(snapshot_path(result))
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                count += 1
    return count


def make_plan(proposal):
    proposal = Path(proposal).resolve(strict=True)
    root = analysis_root_for_proposal(proposal)
    old_summary = root / "proposal_summary"
    main = old_summary / "lumacam_pipeline.ini"
    if (root / "lumacam_pipeline.ini").exists():
        raise ValueError("The new proposal config already exists; do not run this legacy migration twice.")
    _, main_config = load_config(main)
    shared = shared_directory(proposal, main_config)
    reference = Path(main_config["shared_setup"]["prepared_reference_path"])
    background = Path(main_config["shared_setup"]["prepared_background_path"])
    if not reference.is_relative_to(root) or not background.is_relative_to(root):
        raise ValueError("Shared correction inputs must be inside this analysis folder.")
    if len(list((old_summary / "reference_cache").glob("*/reference_cache.json"))) != 1:
        raise ValueError("Multiple reference caches need an explicit reference-set mapping.")
    if len(list((old_summary / "shared_inputs" / "background_cache").glob("*/shared_background.json"))) != 1:
        raise ValueError("Multiple background caches need an explicit reference-set mapping.")
    background_metadata = json.loads((background.parent / "shared_background.json").read_text())
    bg_name = background_metadata["background_experiment"]
    variant = main_config["project"]["variant"]
    old_work = background.parent / "work" / bg_name
    new_work = result_directory(shared / "processing" / "work", bg_name, variant)
    mapping = {
        str(main): str(root / "lumacam_pipeline.ini"),
        str(old_summary / "batches"): str(root / "batches"),
        str(reference): str(shared / "ob.tiff"),
        str(reference.parent / "reference_cache.json"): str(shared / "ob_preparation.json"),
        str(background): str(shared / "background.tiff"),
        str(background.parent / "shared_background.json"): str(shared / "preparation.json"),
        str(background.parent / "background_calculation.json"): str(shared / "processing" / "background_calculation.json"),
        str(old_summary / "shared_inputs" / "preparation.log"): str(shared / "processing" / "preparation.log"),
        str(old_work): str(new_work),
        str(old_work / "images" / f"obint_corrected_{bg_name}"): str(new_work / "processing" / "intermediate" / "beam_ob_corrected"),
        str(old_work / "images" / f"uncorrected_{bg_name}"): str(new_work / "processing" / "intermediate" / "uncorrected"),
    }
    experiment_records = []
    duplicate_sources = set()
    shared_digest = sha256(background)
    for old in sorted(root.glob("exp*")):
        if not old.is_dir():
            continue
        preset, config = load_config(old / "lumacam_pipeline.ini")
        variant = config["project"]["variant"]
        new = result_directory(root, old.name, variant)
        if new.exists():
            raise FileExistsError(new)
        mapping[str(old)] = str(new)
        mapping[str(old / "lumacam_pipeline.ini")] = str(snapshot_path(new))
        for average in (old / "images").glob("average_*.tif*"):
            mapping[str(average)] = str(new / "images" / average_filename(old.name, variant))
        corrected = old / "images" / f"obint_corrected_{old.name}"
        final = old / "images" / f"final_{old.name}"
        stage = "background" if final.is_dir() else "corrections" if corrected.is_dir() else "source"
        mapping[str(old / "images" / f"uncorrected_{old.name}")] = str(new / "processing" / "intermediate" / "uncorrected")
        mapping[str(corrected)] = str(new / "processing" / "intermediate" / "beam_ob_corrected" if stage == "background" else new / "images" / "corrected_frames")
        mapping[str(final)] = str(new / "images" / "corrected_frames")
        for candidate in (old / "processing").glob("background_*.tif*"):
            if sha256(candidate) == shared_digest:
                mapping[str(candidate)] = str(shared / "background.tiff")
                duplicate_sources.add(str(candidate))
            else:
                mapping[str(candidate)] = str(new / "processing" / "background.tiff")
        experiment_records.append({"old": str(old), "new": str(new), "preset": preset, "stage": stage})

    # These are custom exp101 analyses, not pipeline intermediates.
    varia = root / "varia"
    matches = sorted(root.glob("exp101_*"))
    if varia.exists() and len(matches) == 1:
        noise = matches[0] / "3to1000" / "noise_analysis"
        for item in varia.iterdir():
            if item.name.startswith("exp101_") and "temporal_noise_3to1000" in item.name:
                target = (noise / item.name if item.is_file() else
                          noise / ("ob_normalized" if "ob_normalized" in item.name else "uncorrected"))
                mapping[str(item)] = str(target)

    keys = sorted(mapping, key=len, reverse=True)
    files = []
    targets = set()
    for source in sorted(root.rglob("*")):
        if source.is_symlink():
            raise ValueError(f"Inspect symlink before migration: {source}")
        if not source.is_file():
            continue
        if not source.resolve().is_relative_to(root):
            raise ValueError(f"Source escapes analysis: {source}")
        old = str(source)
        prefix = next((key for key in keys if old == key or old.startswith(key + "/")), None)
        target = Path(mapping[prefix] + old[len(prefix):]) if prefix else source
        duplicate = old in duplicate_sources
        if target != source and target.exists():
            raise FileExistsError(target)
        if str(target) in targets and not duplicate and target != shared / "background.tiff":
            raise ValueError(f"Two files would collide at {target}")
        targets.add(str(target))
        stat = source.stat()
        if time.time() - stat.st_mtime < 120:
            raise ValueError(f"Recently written file; wait until processing is idle: {source}")
        files.append({"source": old, "target": str(target), "sha256": sha256(source),
                      "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                      "duplicate": duplicate})
    return {"schema": "lumacam-analysis-restructure-1", "proposal": str(proposal),
            "created_at": datetime.now(timezone.utc).isoformat(), "mapping": mapping,
            "experiments": experiment_records, "files": files}


def apply_plan(plan):
    if plan.get("schema") != "lumacam-analysis-restructure-1":
        raise ValueError("Unsupported restructure plan.")
    proposal = Path(plan["proposal"]).resolve(strict=True)
    root = analysis_root_for_proposal(proposal)
    if (root / "lumacam_pipeline.ini").exists():
        raise ValueError("New analysis config already exists; inspect previous migration before continuing.")
    for item in plan["files"]:
        source, target = Path(item["source"]), Path(item["target"])
        if not source.resolve().is_relative_to(root) or not target.resolve().is_relative_to(root):
            raise ValueError("A plan path escapes this proposal's analysis folder.")
        stat = source.stat()
        if stat.st_size != item["size"] or stat.st_mtime_ns != item["mtime_ns"] or sha256(source) != item["sha256"]:
            raise ValueError(f"Source changed since audit: {source}")
        if target != source and target.exists():
            raise FileExistsError(target)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = proposal / ".trash" / f"analysis_restructure_{stamp}"
    backup.mkdir(parents=True, exist_ok=False)
    (backup / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    archive_path = backup / "original_records.tar.gz"
    records = [item for item in plan["files"] if item["duplicate"] or Path(item["source"]).suffix in TEXT_SUFFIXES]
    with tarfile.open(archive_path, "w:gz") as archive:
        for item in records:
            archive.add(item["source"], arcname=str(Path(item["source"]).relative_to(root)), recursive=False)
    expected = {str(Path(item["source"]).relative_to(root)): item["sha256"] for item in records}
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            handle = archive.extractfile(member)
            if handle is None or hashlib.sha256(handle.read()).hexdigest() != expected.pop(member.name):
                raise RuntimeError(f"Backup verification failed: {member.name}")
    if expected:
        raise RuntimeError("Backup is missing files.")
    print(f"Verified recovery archive: {archive_path}", flush=True)
    old_directories = sorted((path for path in root.rglob("*") if path.is_dir()), key=lambda path: len(path.parts), reverse=True)
    with (backup / "operations.jsonl").open("a", encoding="utf-8") as log:
        for item in sorted(plan["files"], key=lambda item: item["duplicate"]):
            source, target = Path(item["source"]), Path(item["target"])
            if item["duplicate"]:
                if sha256(target) != item["sha256"]:
                    raise RuntimeError(f"Shared background is not identical: {source}")
                source.unlink()  # verified in the recovery archive and at the shared target
            elif source != target:
                target.parent.mkdir(parents=True, exist_ok=True)
                source.rename(target)
            log.write(json.dumps(item) + "\n")
            log.flush()
    for item in plan["files"]:
        if sha256(item["target"]) != item["sha256"]:
            raise RuntimeError(f"Post-move verification failed: {item['target']}")

    mapping = plan["mapping"]
    pattern = re.compile("|".join(re.escape(key) for key in sorted(mapping, key=len, reverse=True)))
    for target in {Path(item["target"]) for item in plan["files"]}:
        if target.suffix in TEXT_SUFFIXES:
            original = target.read_text(encoding="utf-8")
            updated = pattern.sub(lambda match: mapping[match.group(0)], original)
            if original != updated:
                target.write_text(updated, encoding="utf-8")
    # Import the actual completed processing stages, not merely enabled options.
    for experiment in plan["experiments"]:
        result = Path(experiment["new"])
        preset, config = load_config(snapshot_path(result))
        session = create_session(preset, config)
        session.artifacts.put("results_dir", result, "project")
        stage = experiment["stage"]
        recipes = {}
        if stage != "source":
            session.artifacts.put("processed_image_dir", result / "processing" / "intermediate" / "beam_ob_corrected"
                                  if stage == "background" else result / "images" / "corrected_frames", "corrections")
            recipes["corrections"] = recipe_for(session, "corrections")
        if stage == "background":
            session.artifacts.put("analysis_image_dir", result / "images" / "corrected_frames", "background")
            recipes["background"] = recipe_for(session, "background")
        if list((result / "images").glob("average_*.tif*")):
            recipes["average"] = recipe_for(session, "average")
        (result / "processing" / "recipes.json").write_text(json.dumps(recipes, indent=2, sort_keys=True), encoding="utf-8")
        save_config(snapshot_path(result), preset, session.config)
    preset, config = load_config(root / "lumacam_pipeline.ini")
    save_config(root / "lumacam_pipeline.ini", preset, create_session(preset, config).config)
    mark_relocated_batches(plan)
    for directory in old_directories:
        if directory.exists() and not any(directory.iterdir()):
            directory.rmdir()
    # Recheck every TIFF after all text/config updates; no pixel data were edited.
    tiffs = [item for item in plan["files"] if Path(item["source"]).suffix.lower() in {".tif", ".tiff"}]
    for item in tiffs:
        if sha256(item["target"]) != item["sha256"]:
            raise RuntimeError(f"TIFF verification failed: {item['target']}")
    report = {"completed_at": datetime.now(timezone.utc).isoformat(), "recovery": str(backup),
              "experiment_count": len(plan["experiments"]), "tiffs_verified": len(tiffs),
              "duplicate_backgrounds_archived": sum(item["duplicate"] for item in plan["files"]),
              "acquisition_data_changed": False}
    (root / "proposal_summary").mkdir(exist_ok=True)
    (root / "proposal_summary" / "restructure_record.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal")
    parser.add_argument("--plan")
    parser.add_argument("--apply-plan")
    args = parser.parse_args()
    if args.apply_plan:
        apply_plan(json.loads(Path(args.apply_plan).read_text(encoding="utf-8")))
    elif args.proposal and args.plan:
        plan = make_plan(args.proposal)
        Path(args.plan).write_text(json.dumps(plan, indent=2), encoding="utf-8")
        print(f"Audit only: {len(plan['experiments'])} experiments, {len(plan['files'])} files; "
              f"{sum(item['duplicate'] for item in plan['files'])} identical background copies to archive. Plan: {args.plan}")
    else:
        parser.error("Choose --proposal PATH --plan FILE or --apply-plan FILE")


if __name__ == "__main__":
    main()

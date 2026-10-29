"""Optional image correction sections."""

from __future__ import annotations

from ..proposal_paths import raw_root_for_layout

from pathlib import Path

from ..background import background_frame_count, shared_background_path

import numpy as np
import tifffile

from .. import backends
from ..products import is_all_runs_sum
from ..prepared_inputs import uses_prepared_ob, validate_prepared_ob, prepared_ob_integral
from ..section import PipelineSection, SectionOutcome, SectionState


def _optional_path(value):
    text = str(value or "").strip()
    return str(Path(text).expanduser().resolve()) if text else None


class CorrectionSection(PipelineSection):
    id = "corrections"
    title = "Corrections"
    requires = ("dataset", "results_dir")
    produces = ("processed_image_dir", "correction_result", "nci_calibrated")

    def run(self, session, config):
        dataset = session.artifacts.require("dataset")
        results_dir = Path(session.artifacts.require("results_dir"))
        reference_dir = _optional_path(config.get("reference_dir"))
        beamhistory_path = _optional_path(config.get("beamhistory_path"))
        reference_beamhistory_path = _optional_path(config.get("reference_beamhistory_path"))
        prepared_reference_path = _optional_path(config.get("prepared_reference_path"))
        reference_integral = config.get("reference_beam_integral_per_minute") or None
        if uses_prepared_ob(config):
            reference_dir = None
            reference_beamhistory_path = None
            if prepared_reference_path:
                validate_prepared_ob(prepared_reference_path)
                if reference_integral in (None, ""):
                    reference_integral = prepared_ob_integral(prepared_reference_path)
        file_filter = str(config.get("file_filter") or dataset.selected_variant or "").strip() or None
        processing = session.config.get("workflow", {}).get("role") == "processing"
        normalized = None
        if processing:
            from ..processing import available_beamhistory
            from ..inspection import processing_layout
            beamhistory_path = available_beamhistory(beamhistory_path)
            reference_beamhistory_path = available_beamhistory(reference_beamhistory_path)
            normalized = processing_layout(dataset, session.artifacts.require("source_products"))
            if beamhistory_path:
                module = backends.nci_data()
                missing = [run["name"] for run in normalized["runs"]
                           if not module.get_run_exposure_windows(run, tpx_time=float(config.get("tpx_time", 2)))]
                if missing:
                    raise ValueError(f"Beam correction requires timing metadata for every selected run; missing: {missing}")
                if reference_dir and not reference_beamhistory_path:
                    reference_beamhistory_path = beamhistory_path
                if reference_integral is not None and (not np.isfinite(float(reference_integral)) or float(reference_integral) <= 0):
                    raise ValueError("Reference beam integral must be finite and positive.")
                if reference_integral is None and (prepared_reference_path or not reference_dir):
                    raise ValueError("Beam correction needs an explicit reference beam integral or reference measurement.")
            if reference_dir and reference_beamhistory_path:
                # Validate JSON timing before the legacy reference calculation
                # can fall back to raw file timestamps.
                module = backends.nci_data()
                reference_layout = module.resolve_nci_experiment(reference_dir)
                missing = [run["name"] for run in reference_layout["runs"]
                           if not module.get_exposure_windows(metadata_dir=run.get("metadata_dir"), tpx_folder=None,
                                                              tpx_time=float(config.get("tpx_time", 2)))]
                if missing:
                    raise ValueError(f"Reference beam correction requires JSON timing metadata; missing: {missing}")
        results_dir.mkdir(parents=True, exist_ok=True)
        intermediate = results_dir / "processing" / "intermediate"
        corrected_dir = (
            intermediate / "beam_ob_corrected"
            if session.section_config("background").get("enabled", False)
            else results_dir / "images" / "corrected_frames"
        )
        uncorrected_dir = intermediate / "uncorrected"

        common = {
            "beamhistory_path": beamhistory_path,
            "tpx_time": float(config.get("tpx_time", 2)),
            "n_tpx_per_image": int(config.get("n_tpx_per_image", 30)),
            "file_filter": file_filter,
            "reference_beam_integral_per_minute": reference_integral,
            "reference_thickness_um": float(config.get("reference_thickness_um", 1.0)),
        }
        if dataset.layout_type == "flat" and not processing:
            module = backends.exsitu_pipeline()
            result = module.run_exsitu_pipeline(
                str(dataset.input_path),
                ob_path=prepared_reference_path or reference_dir,
                results_dir=str(results_dir),
                ob_bin_y=1 if prepared_reference_path else int(config.get("reference_bin_y", 2)),
                ob_bin_x=1 if prepared_reference_path else int(config.get("reference_bin_x", 2)),
                background_correction=False,
                verbose=bool(config.get("verbose", True)),
                organized_output=True,
                output_paths={"corrected": corrected_dir, "uncorrected": uncorrected_dir},
                write_average=False,
                **common,
            )
            processed_dir = result["obint_corrected_dir"]
            calibrated = bool(result.get("ob_corrected"))
        else:
            module = backends.nci_data()
            image_root = results_dir / "images"
            processing_root = results_dir / "processing"
            experiment_input = dataset.input_path
            if dataset.layout_root is not None:
                candidates = (
                    dataset.layout_root / "metadata" / dataset.experiment,
                    dataset.layout_root / "final" / dataset.experiment,
                    raw_root_for_layout(dataset.layout_root) / dataset.experiment,
                )
                experiment_input = next(
                    (path for path in candidates if path.is_dir()), dataset.input_path
                )
            result = module.run_operando_pipeline(
                exp_dir=normalized if normalized is not None else str(experiment_input),
                reference_dir=reference_dir,
                reference_beamhistory_path=reference_beamhistory_path,
                reference_bin_y=int(config.get("reference_bin_y", 2)),
                reference_bin_x=int(config.get("reference_bin_x", 2)),
                reference_final_subfolder=str(config.get("reference_final_subfolder", "final")),
                clip_ob_to_small_roi_p5=bool(config.get("clip_reference", False)),
                ob_clip_roi=config.get("reference_clip_roi"),
                ob_clip_percentile=float(config.get("reference_clip_percentile", 5.0)),
                results_dir=str(results_dir),
                corrected_dir=str(corrected_dir),
                uncorrected_dir=str(uncorrected_dir),
                metadata_path=str(processing_root / "processing_metadata.json"),
                processed_reference_path=prepared_reference_path,
                **common,
            )
            processed_dir = result["corrected_dir"]
            calibrated = bool(result.get("reference_corrected"))

        applied = []
        if result.get("intensity_corrected"):
            applied.append("beam intensity")
        if result.get("reference_corrected") or result.get("ob_corrected"):
            applied.append("reference")
        message = "Prepared image stack"
        message += f" with {', '.join(applied)} correction." if applied else " without optional beam/reference correction."
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            message,
            {
                "processed_image_dir": Path(processed_dir),
                "correction_result": result,
                "nci_calibrated": calibrated,
            },
        )


class BackgroundSection(PipelineSection):
    id = "background"
    title = "Background correction"
    requires = ("processed_image_dir", "results_dir", "experiment")
    produces = ("analysis_image_dir", "background_path", "background_result")

    def is_configured(self, config):
        return bool(config.get("enabled", False))

    def run(self, session, config):
        module = backends.nci_data()
        input_dir = Path(session.artifacts.require("processed_image_dir"))
        results_dir = Path(session.artifacts.require("results_dir"))
        experiment = session.artifacts.require("experiment")
        processing_dir = results_dir / "processing"
        output_dir = results_dir / "images" / "corrected_frames"
        processing_dir.mkdir(parents=True, exist_ok=True)
        background_path = processing_dir / "background.tiff"
        external_path = _optional_path(shared_background_path(config))
        if external_path:
            external = Path(external_path)
            if external.is_file():
                if is_all_runs_sum(external):
                    raise ValueError(f"All-runs sum is a preview, not an individual BG input: {external}")
                background = tifffile.imread(external)
                background_path = external
            elif external.is_dir():
                background = module.compute_background_image(
                    str(external),
                    n_images=background_frame_count(config),
                    compute_dtype=np.float32,
                )
            else:
                raise ValueError(f"Background path does not exist: {external}")
        else:
            background = module.compute_background_image(
                str(input_dir),
                n_images=background_frame_count(config),
                compute_dtype=np.float32,
                metadata_path=str(processing_dir / "background_metadata.json"),
            )
        if not external_path or not Path(external_path).is_file():
            tifffile.imwrite(background_path, np.asarray(background, dtype=np.float32))
        # Enabling BG after a no-BG run must not subtract in-place. Keep the
        # original corrected frames and update their provenance paths.
        if input_dir == output_dir:
            intermediate = processing_dir / "intermediate" / "beam_ob_corrected"
            if intermediate.exists():
                raise ValueError("Pre-background frames already exist. Use a new Analysis name.")
            intermediate.parent.mkdir(parents=True, exist_ok=True)
            input_dir.rename(intermediate)
            for metadata in processing_dir.glob("*.json"):
                text = metadata.read_text(encoding="utf-8")
                metadata.write_text(text.replace(str(input_dir), str(intermediate)), encoding="utf-8")
            result = session.artifacts.get("correction_result", {})
            for key in ("corrected_dir", "obint_corrected_dir"):
                if result.get(key) == str(input_dir):
                    result[key] = str(intermediate)
            input_dir = intermediate
            session.artifacts.put("processed_image_dir", input_dir, "corrections")
        result = module.background_subtract_directory(
            str(input_dir),
            str(output_dir),
            str(background_path),
            clip_negatives=bool(config.get("clip_negatives", True)),
            skip_existing=False,
            metadata_path=str(processing_dir / "background_subtraction_metadata.json"),
        )
        return SectionOutcome(
            self.id,
            SectionState.COMPLETED,
            f"Background-corrected images are in {output_dir}",
            {
                "analysis_image_dir": output_dir,
                "background_path": background_path,
                "background_result": result,
            },
        )

"""Quick ex-situ neutron image processing helpers."""

__all__ = ["run_exsitu_pipeline", "launch_exsitu_gui", "ExSituPipelineGUI"]


def __getattr__(name):
    if name == "run_exsitu_pipeline":
        from .exsitu_pipeline import run_exsitu_pipeline

        return run_exsitu_pipeline
    if name in {"launch_exsitu_gui", "ExSituPipelineGUI"}:
        from .exsitu_gui import ExSituPipelineGUI, launch_exsitu_gui

        return {
            "launch_exsitu_gui": launch_exsitu_gui,
            "ExSituPipelineGUI": ExSituPipelineGUI,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

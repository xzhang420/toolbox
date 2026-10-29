"""Paths shared by acquisition-layout discovery and analysis output writers."""

from pathlib import Path


def split_proposal_root(layout_root):
    root = Path(layout_root)
    if (root.name in {"experiment", "experiments"} and root.parent.name == "data"
            and root.parent.parent.name == "reduced"):
        return root.parent.parent.parent
    return None


def proposal_root_for_layout(layout_root):
    root = Path(layout_root)
    outer = split_proposal_root(root)
    if outer is not None:
        return outer
    return root.parent.parent if root.parent.name == "data" else root.parent


def raw_root_for_layout(layout_root):
    root = Path(layout_root)
    outer = split_proposal_root(root)
    return outer / "raw" if outer is not None else root / "tpx3Files"


def reduced_root_for_proposal(proposal):
    root = Path(proposal)
    reduced = root / "reduced"
    if any((reduced / "data" / name).is_dir() for name in ("experiment", "experiments")):
        return reduced
    return root


def analysis_root_for_proposal(proposal):
    return reduced_root_for_proposal(proposal) / "analysis"

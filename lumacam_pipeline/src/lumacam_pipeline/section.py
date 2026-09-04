"""Dependency-aware execution primitives for selectable pipeline sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

from .models import Artifact


class SectionState(str, Enum):
    READY = "ready"
    NOT_CONFIGURED = "not configured"
    WAITING = "waiting for input"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SectionOutcome:
    section_id: str
    state: SectionState
    message: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class ArtifactStore:
    def __init__(self) -> None:
        self._artifacts: dict[str, Artifact] = {}

    def put(self, name: str, value: Any, producer: str, **metadata: Any) -> None:
        self._artifacts[name] = Artifact(name, value, producer, metadata)

    def get(self, name: str, default: Any = None) -> Any:
        artifact = self._artifacts.get(name)
        return default if artifact is None else artifact.value

    def require(self, name: str) -> Any:
        if name not in self._artifacts:
            raise KeyError(f"Required artifact is unavailable: {name}")
        return self._artifacts[name].value

    def has(self, name: str) -> bool:
        return name in self._artifacts

    def discard_by_producer(self, producer: str) -> None:
        for name in [name for name, artifact in self._artifacts.items() if artifact.producer == producer]:
            del self._artifacts[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._artifacts))


class PipelineSection:
    id = "section"
    title = "Section"
    requires: tuple[str, ...] = ()
    optional_inputs: tuple[str, ...] = ()
    produces: tuple[str, ...] = ()

    def is_configured(self, config: Mapping[str, Any]) -> bool:
        return bool(config.get("enabled", True))

    def run(self, session: "PipelineSession", config: Mapping[str, Any]) -> SectionOutcome:
        raise NotImplementedError


class PipelineSession:
    """Run individual sections without imposing one global linear workflow."""

    def __init__(
        self,
        sections: Iterable[PipelineSection],
        config: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        section_list = list(sections)
        self.sections = {section.id: section for section in section_list}
        if len(self.sections) != len(section_list):
            raise ValueError("Section ids must be unique")
        self.config = {name: dict(values) for name, values in (config or {}).items()}
        self.artifacts = ArtifactStore()
        self.outcomes: dict[str, SectionOutcome] = {}

    def section_config(self, section_id: str) -> dict[str, Any]:
        return self.config.setdefault(section_id, {})

    def configure(self, section_id: str, **values: Any) -> None:
        if section_id not in self.sections:
            raise KeyError(f"Unknown section: {section_id}")
        current = self.section_config(section_id)
        changed = any(current.get(name) != value for name, value in values.items())
        if not changed:
            return
        current.update(values)
        self.outcomes.pop(section_id, None)
        self.artifacts.discard_by_producer(section_id)
        self._invalidate_consumers(self.sections[section_id].produces)

    def _invalidate_consumers(self, artifact_names, excluded=()):
        pending = set(artifact_names)
        excluded_ids = set(excluded)
        while pending:
            produced_name = pending.pop()
            for consumer_id, consumer in self.sections.items():
                if consumer_id in excluded_ids:
                    continue
                inputs = set(consumer.requires).union(consumer.optional_inputs)
                if produced_name not in inputs:
                    continue
                excluded_ids.add(consumer_id)
                self.outcomes.pop(consumer_id, None)
                self.artifacts.discard_by_producer(consumer_id)
                pending.update(consumer.produces)

    def missing_requirements(self, section_id: str) -> tuple[str, ...]:
        return tuple(name for name in self.sections[section_id].requires if not self.artifacts.has(name))

    def status(self, section_id: str) -> SectionState:
        if section_id in self.outcomes:
            return self.outcomes[section_id].state
        section = self.sections[section_id]
        if not section.is_configured(self.section_config(section_id)):
            return SectionState.NOT_CONFIGURED
        if self.missing_requirements(section_id):
            return SectionState.WAITING
        return SectionState.READY

    def run(self, section_id: str) -> SectionOutcome:
        if section_id not in self.sections:
            raise KeyError(f"Unknown section: {section_id}")
        section = self.sections[section_id]
        config = self.section_config(section_id)
        if not section.is_configured(config):
            outcome = SectionOutcome(section_id, SectionState.NOT_CONFIGURED, "Section is disabled or has no input.")
            self.outcomes[section_id] = outcome
            return outcome
        missing = self.missing_requirements(section_id)
        if missing:
            outcome = SectionOutcome(
                section_id,
                SectionState.WAITING,
                "Missing artifacts: " + ", ".join(missing),
            )
            self.outcomes[section_id] = outcome
            return outcome
        try:
            outcome = section.run(self, config)
            if outcome.state == SectionState.COMPLETED:
                self._invalidate_consumers(section.produces, excluded=(section_id,))
                for name, value in outcome.outputs.items():
                    self.artifacts.put(name, value, section_id, **outcome.metadata)
        except Exception as exc:
            outcome = SectionOutcome(section_id, SectionState.FAILED, f"{type(exc).__name__}: {exc}")
        self.outcomes[section_id] = outcome
        return outcome

    def summary(self) -> list[dict[str, Any]]:
        rows = []
        for section_id, section in self.sections.items():
            outcome = self.outcomes.get(section_id)
            rows.append(
                {
                    "section": section_id,
                    "title": section.title,
                    "state": self.status(section_id).value,
                    "requires": list(section.requires),
                    "produces": list(section.produces),
                    "missing": list(self.missing_requirements(section_id)),
                    "message": outcome.message if outcome is not None else "",
                }
            )
        return rows

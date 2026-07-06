"""Typed simulation events emitted by the world."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EventType(StrEnum):
    BIRTH = "birth"
    DEATH = "death"
    PREDATION = "predation"
    EXTINCTION = "extinction"
    RESOURCE_EATEN = "resource_eaten"


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """Base class for timeline events."""

    tick: int

    @property
    def type(self) -> EventType:
        raise NotImplementedError

    @property
    def subject_id(self) -> int | None:
        return None

    @property
    def details(self) -> dict[str, object]:
        return {}


@dataclass(frozen=True, slots=True)
class BirthEvent(SimulationEvent):
    parent_id: int
    child_id: int
    species: str
    generation: int

    @property
    def type(self) -> EventType:
        return EventType.BIRTH

    @property
    def subject_id(self) -> int:
        return self.child_id

    @property
    def details(self) -> dict[str, object]:
        return {"parent_id": self.parent_id, "generation": self.generation}


@dataclass(frozen=True, slots=True)
class DeathEvent(SimulationEvent):
    agent_id: int
    species: str
    cause: str

    @property
    def type(self) -> EventType:
        return EventType.DEATH

    @property
    def subject_id(self) -> int:
        return self.agent_id

    @property
    def details(self) -> dict[str, object]:
        return {"cause": self.cause}


@dataclass(frozen=True, slots=True)
class PredationEvent(SimulationEvent):
    predator_id: int
    prey_id: int
    energy_gained: float
    species: str = "prey"

    @property
    def type(self) -> EventType:
        return EventType.PREDATION

    @property
    def subject_id(self) -> int:
        return self.prey_id

    @property
    def details(self) -> dict[str, object]:
        return {"predator_id": self.predator_id, "energy_gained": self.energy_gained}


@dataclass(frozen=True, slots=True)
class ExtinctionEvent(SimulationEvent):
    species: str

    @property
    def type(self) -> EventType:
        return EventType.EXTINCTION

    @property
    def details(self) -> dict[str, object]:
        return {"species": self.species}


@dataclass(frozen=True, slots=True)
class ResourceEatenEvent(SimulationEvent):
    agent_id: int
    resource_id: int
    energy_gained: float
    species: str

    @property
    def type(self) -> EventType:
        return EventType.RESOURCE_EATEN

    @property
    def subject_id(self) -> int:
        return self.agent_id

    @property
    def details(self) -> dict[str, object]:
        return {"resource_id": self.resource_id, "energy_gained": self.energy_gained}

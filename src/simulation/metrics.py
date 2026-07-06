"""Population, trait, and event metrics for analysis."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from src.simulation.agent import Agent
from src.simulation.species import PREDATOR, PREY

if TYPE_CHECKING:
    from src.simulation.world import World


class MetricsRecorder:
    """Collect cumulative events and time-series ecosystem metrics."""

    fieldnames = [
        "tick",
        "prey_population",
        "predator_population",
        "resource_count",
        "avg_prey_speed",
        "avg_predator_speed",
        "avg_prey_vision",
        "avg_predator_vision",
        "births_total",
        "deaths_total",
        "predation_total",
        "prey_extinct",
        "predators_extinct",
    ]

    def __init__(self) -> None:
        self.births = 0
        self.deaths = 0
        self.predation_events = 0
        self.extinction_events = 0
        self.records: list[dict[str, float | int]] = []

    def reset(self) -> None:
        """Clear all cumulative and time-series metrics."""

        self.births = 0
        self.deaths = 0
        self.predation_events = 0
        self.extinction_events = 0
        self.records.clear()

    def record_birth(self) -> None:
        self.births += 1

    def record_death(self) -> None:
        self.deaths += 1

    def record_predation(self) -> None:
        self.predation_events += 1

    def record_extinction(self) -> None:
        self.extinction_events += 1

    def record(self, world: "World") -> None:
        """Append a metrics row for the current world state."""

        agents = _world_agents(world)
        prey = [agent for agent in agents if agent.species == PREY and agent.alive]
        predators = [agent for agent in agents if agent.species == PREDATOR and agent.alive]

        self.records.append(
            {
                "tick": world.tick_count,
                "prey_population": len(prey),
                "predator_population": len(predators),
                "resource_count": len(world.resources),
                "avg_prey_speed": self.average_trait(prey, "speed"),
                "avg_predator_speed": self.average_trait(predators, "speed"),
                "avg_prey_vision": self.average_trait(prey, "vision_radius"),
                "avg_predator_vision": self.average_trait(predators, "vision_radius"),
                "births_total": self.births,
                "deaths_total": self.deaths,
                "predation_total": self.predation_events,
                "prey_extinct": int(len(prey) == 0),
                "predators_extinct": int(len(predators) == 0),
            }
        )

    def population_counts(self, agents: Iterable[Agent]) -> dict[str, int]:
        """Calculate live population counts by species."""

        return {
            PREY: sum(1 for agent in agents if agent.species == PREY and agent.alive),
            PREDATOR: sum(1 for agent in agents if agent.species == PREDATOR and agent.alive),
        }

    def latest_record(self) -> dict[str, float | int] | None:
        """Return the newest recorded row, if one exists."""

        return self.records[-1] if self.records else None

    def save_csv(self, path: str | Path) -> Path:
        """Write recorded metrics to CSV and return the output path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.records)
        return output_path

    @staticmethod
    def average_trait(agents: Iterable[Agent], trait: str) -> float:
        """Average one genome trait over a collection of agents."""

        agent_list = list(agents)
        if not agent_list:
            return 0.0
        return sum(getattr(agent.genome, trait) for agent in agent_list) / len(agent_list)

    _average_trait = average_trait


def _world_agents(world: "World") -> list[Agent]:
    agents = world.agents
    if hasattr(agents, "values"):
        return list(agents.values())
    return list(agents)

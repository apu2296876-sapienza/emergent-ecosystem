"""Creature model for prey and predator agents."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable

from src.simulation.genetics import Genome
from src.simulation.neural import Brain, inherit_brain
from src.simulation.species import PREDATOR, PREY, Species, normalize_species
from src.utils.constants import WANDER
from src.utils.random_utils import (
    Vec2,
    add,
    clamp_position,
    distance,
    jittered_position,
    multiply,
    normalize,
    subtract,
)


@dataclass(slots=True, init=False)
class Agent:
    """Autonomous creature with state, genome, and local behavior helpers."""

    id: int
    species: str
    x: float
    y: float
    genome: Genome
    energy: float
    age: float
    alive: bool
    cooldowns: dict[str, int]
    current_action: str
    parent_id: int | None
    generation: int
    heading: Vec2
    brain: Brain | None

    def __init__(
        self,
        id: int,
        species: Species | str,
        x: float | Vec2 = 0.0,
        y: float | Genome | None = None,
        genome: Genome | None = None,
        energy: float = 0.0,
        age: float = 0.0,
        alive: bool = True,
        cooldowns: dict[str, int] | None = None,
        current_action: str = WANDER,
        parent_id: int | None = None,
        generation: int = 0,
        heading: Vec2 = (1.0, 0.0),
        reproduction_cooldown: int = 0,
        brain: Brain | None = None,
        *,
        position: Vec2 | None = None,
    ) -> None:
        """Create an agent.

        The constructor accepts both the explicit `x, y, genome` shape and the
        previous `position, genome` tuple shape used by the renderer/tests.
        """

        # The block below just lets callers build an Agent two ways:
        #   Agent(id, species, x, y, genome=...)          <- explicit coordinates
        #   Agent(id, species, position=(x, y), genome=...) <- position tuple
        # Older renderer/test code passed `(x, y)` as a single tuple in the `x`
        # slot with the genome in the `y` slot, so we untangle that case first.
        if isinstance(x, tuple) and isinstance(y, Genome) and genome is None:
            genome = y
            y = None

        if position is not None:
            x, y = position
        elif isinstance(x, tuple):
            x, y = x
        elif y is None or isinstance(y, Genome):
            y = 0.0

        if genome is None:
            raise ValueError("Agent requires a genome")

        self.id = id
        self.species = normalize_species(species)
        self.x = float(x)
        self.y = float(y)
        self.genome = genome.clamped()
        self.energy = float(energy)
        self.age = float(age)
        self.alive = bool(alive)
        self.cooldowns = dict(cooldowns or {})
        self.cooldowns.setdefault("reproduction", int(reproduction_cooldown))
        self.current_action = current_action
        self.parent_id = parent_id
        self.generation = int(generation)
        self.heading = normalize(heading) or (1.0, 0.0)
        self.brain = brain

    @property
    def position(self) -> Vec2:
        """Tuple position alias for vector helpers and renderers."""

        return (self.x, self.y)

    @position.setter
    def position(self, value: Vec2) -> None:
        self.x = float(value[0])
        self.y = float(value[1])

    @property
    def reproduction_cooldown(self) -> int:
        """Compatibility alias for the reproduction cooldown entry."""

        return self.cooldowns.get("reproduction", 0)

    @reproduction_cooldown.setter
    def reproduction_cooldown(self, value: int) -> None:
        self.cooldowns["reproduction"] = max(0, int(value))

    @property
    def is_prey(self) -> bool:
        return self.species == PREY

    @property
    def is_predator(self) -> bool:
        return self.species == PREDATOR

    def distance_to(self, target: object) -> float:
        """Return Euclidean distance to another entity or `(x, y)` tuple."""

        return distance(self.position, _target_position(target))

    def move_towards(
        self,
        target: object,
        dt: float,
        *,
        speed_multiplier: float = 1.0,
        bounds: tuple[float, float] | None = None,
        margin: float = 0.0,
    ) -> float:
        """Move toward a target and return distance travelled."""

        return self._move(subtract(_target_position(target), self.position), dt, speed_multiplier, bounds, margin)

    def move_away_from(
        self,
        target: object,
        dt: float,
        *,
        speed_multiplier: float = 1.0,
        bounds: tuple[float, float] | None = None,
        margin: float = 0.0,
    ) -> float:
        """Move away from a target and return distance travelled."""

        return self._move(subtract(self.position, _target_position(target)), dt, speed_multiplier, bounds, margin)

    def consume_energy(self, amount: float) -> bool:
        """Subtract energy and mark the agent dead if energy is depleted."""

        self.energy -= max(0.0, amount)
        if self.energy <= 0.0:
            self.energy = 0.0
            self.die()
        return self.alive

    def can_reproduce(self) -> bool:
        """Return true when energy and cooldown permit reproduction.

        All three must hold: the agent is alive, its post-birth cooldown has
        elapsed, and it has banked at least its (genetic) reproduction threshold
        of energy. This is checked by every decision policy before offering
        REPRODUCE as an option.
        """

        return (
            self.alive
            and self.reproduction_cooldown <= 0
            and self.energy >= self.genome.reproduction_threshold
        )

    def reproduce(
        self,
        *,
        new_id: int,
        rng: random.Random,
        mutation_rate: float,
        mutation_strength: float,
        energy_fraction: float,
        cooldown_ticks: int,
        jitter_radius: float,
        bounds: tuple[float, float],
        margin: float = 0.0,
        brain_mutation_rate: float = 0.25,
        brain_mutation_strength: float = 0.30,
    ) -> "Agent | None":
        """Create an offspring agent with inherited mutated traits.

        Physiological traits mutate through ``Genome.inherit``; if the parent
        carries a neural controller, its weight vector is inherited through
        ``inherit_brain`` under the same mutate-on-birth scheme, which is what
        makes the neural policy *evolve* across generations.
        """

        if not self.can_reproduce():
            return None

        # Reproduction has a cost: the parent hands a fraction of its energy to
        # the child and then goes on cooldown so it can't breed again instantly.
        child_energy = self.energy * energy_fraction
        self.energy -= child_energy
        self.reproduction_cooldown = cooldown_ticks

        # Inherit the BRAIN (the evolved neural weights) with mutation, if this
        # agent has one. This is the line that lets the neural policy evolve:
        # the child's network is a slightly-mutated copy of the parent's.
        child_brain = None
        if self.brain is not None:
            child_brain = inherit_brain(
                self.brain,
                mutation_rate=brain_mutation_rate,
                mutation_strength=brain_mutation_strength,
                rng=rng,
            )

        child_position = jittered_position(
            self.position,
            jitter_radius,
            bounds[0],
            bounds[1],
            rng,
            margin,
        )
        child = Agent(
            id=new_id,
            species=self.species,
            position=child_position,
            # Inherit the BODY traits with mutation too (same idea as the brain).
            genome=Genome.inherit(
                self.genome,
                mutation_rate=mutation_rate,
                mutation_strength=mutation_strength,
                rng=rng,
            ),
            energy=child_energy,
            parent_id=self.id,
            generation=self.generation + 1,   # track lineage depth
            reproduction_cooldown=cooldown_ticks,
            brain=child_brain,
        )
        return child

    def die(self) -> None:
        """Mark the agent as dead."""

        self.alive = False

    def update_age(self, dt: float, max_age: float | None = None) -> bool:
        """Advance age/cooldowns and return whether the agent remains alive."""

        self.age += dt
        for key, value in list(self.cooldowns.items()):
            self.cooldowns[key] = max(0, value - 1)
        if max_age is not None and self.age > max_age:
            self.die()
        return self.alive

    def get_perceived_agents(
        self,
        agents: Iterable["Agent"],
        *,
        species: Species | str | None = None,
    ) -> list["Agent"]:
        """Return living nearby agents inside this agent's vision radius."""

        species_key = normalize_species(species) if species is not None else None
        perceived: list[Agent] = []
        for other in agents:
            if other.id == self.id or not other.alive:
                continue
            if species_key is not None and other.species != species_key:
                continue
            if self.distance_to(other) <= self.genome.vision_radius:
                perceived.append(other)
        return perceived

    def get_perceived_resources(self, resources: Iterable[object]) -> list[object]:
        """Return nearby resources inside this agent's vision radius."""

        return [
            resource
            for resource in resources
            if self.distance_to(resource) <= self.genome.vision_radius
        ]

    def _move(
        self,
        direction: Vec2,
        dt: float,
        speed_multiplier: float,
        bounds: tuple[float, float] | None,
        margin: float,
    ) -> float:
        normalized = normalize(direction)
        if normalized == (0.0, 0.0):
            return 0.0

        start = self.position
        step = self.genome.speed * speed_multiplier * dt
        next_position = add(start, multiply(normalized, step))
        if bounds is not None:
            next_position = clamp_position(next_position, bounds[0], bounds[1], margin)
        self.position = next_position
        self.heading = normalized
        return distance(start, self.position)


def _target_position(target: object) -> Vec2:
    if isinstance(target, tuple):
        return (float(target[0]), float(target[1]))
    if hasattr(target, "position"):
        position = getattr(target, "position")
        return (float(position[0]), float(position[1]))
    if hasattr(target, "x") and hasattr(target, "y"):
        return (float(getattr(target, "x")), float(getattr(target, "y")))
    raise TypeError(f"Cannot read a 2D position from {target!r}")

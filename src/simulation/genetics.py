"""Genetic traits, default factories, inheritance, and mutation.

This is the *physiological* genome: the six body/behaviour numbers every agent
is born with (speed, eyesight, metabolism, and so on). It works exactly like the
neural genome in ``neural.py`` -- children inherit their parents' values with
small random mutation -- so both the body and the brain evolve under the same
selection pressure. Keeping the two genomes structurally parallel is intentional.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import ClassVar

from src.simulation.species import PREDATOR, PREY, Species, normalize_species
from src.utils.random_utils import clamp

# Hard limits for each trait. Every genome is clamped into these after creation
# or mutation, so evolution can never drift into impossible values (e.g. negative
# speed, or eyesight larger than the world). These are the absolute bounds.
TRAIT_BOUNDS: dict[str, tuple[float, float]] = {
    "speed": (0.55, 4.60),
    "vision_radius": (32.0, 185.0),
    "metabolism": (0.035, 0.340),
    "reproduction_threshold": (45.0, 165.0),
    "fear_sensitivity": (0.0, 1.55),
    "aggression": (0.0, 1.55),
}

# Narrower *starting* ranges used only when spawning the very first generation,
# so prey begin fearful and fast-ish while predators begin aggressive and hungry.
# After birth, mutation can push traits anywhere inside TRAIT_BOUNDS above.
SPECIES_TRAIT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    PREY: {
        "speed": (1.25, 3.35),
        "vision_radius": (58.0, 145.0),
        "metabolism": (0.070, 0.200),
        "reproduction_threshold": (58.0, 108.0),
        "fear_sensitivity": (0.65, 1.35),
        "aggression": (0.0, 0.42),
    },
    PREDATOR: {
        "speed": (1.05, 2.90),
        "vision_radius": (72.0, 172.0),
        "metabolism": (0.110, 0.290),
        "reproduction_threshold": (82.0, 152.0),
        "fear_sensitivity": (0.0, 0.38),
        "aggression": (0.70, 1.42),
    },
}


@dataclass(frozen=True, slots=True)
class Genome:
    """Heritable physiological and behavioral traits for one creature."""

    speed: float
    vision_radius: float
    metabolism: float
    reproduction_threshold: float
    fear_sensitivity: float
    aggression: float

    trait_names: ClassVar[tuple[str, ...]] = tuple(TRAIT_BOUNDS.keys())

    @classmethod
    def random_for_species(
        cls,
        species: Species | str,
        rng: random.Random | None = None,
    ) -> "Genome":
        """Create a species-appropriate random genome."""

        species_key = normalize_species(species)
        rng = rng or random.Random()
        values = {
            trait: rng.uniform(*bounds)
            for trait, bounds in SPECIES_TRAIT_RANGES[species_key].items()
        }
        return cls(**values).clamped()

    @classmethod
    def inherit(
        cls,
        parent_a: "Genome",
        parent_b: "Genome | None" = None,
        *,
        mutation_rate: float = 0.12,
        mutation_strength: float = 0.08,
        rng: random.Random | None = None,
    ) -> "Genome":
        """Create an offspring genome from one or two parents."""

        return inherit_genome(
            parent_a,
            parent_b,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            rng=rng,
        )

    def mutated(
        self,
        *,
        mutation_rate: float = 0.12,
        mutation_strength: float = 0.08,
        rng: random.Random | None = None,
    ) -> "Genome":
        """Return a mutated copy of this genome."""

        return mutate_genome(
            self,
            mutation_rate=mutation_rate,
            mutation_strength=mutation_strength,
            rng=rng,
        )

    def clamped(self) -> "Genome":
        """Return a genome with every trait inside biological bounds."""

        return Genome(
            **{
                trait: clamp(getattr(self, trait), *TRAIT_BOUNDS[trait])
                for trait in self.trait_names
            }
        )

    def as_dict(self) -> dict[str, float]:
        """Serialize traits for tests, logs, or reports."""

        return {trait: getattr(self, trait) for trait in self.trait_names}


def default_prey_genome(rng: random.Random | None = None) -> Genome:
    """Factory for a random prey genome drawn from prey defaults."""

    return Genome.random_for_species(PREY, rng)


def default_predator_genome(rng: random.Random | None = None) -> Genome:
    """Factory for a random predator genome drawn from predator defaults."""

    return Genome.random_for_species(PREDATOR, rng)


def mutate_genome(
    genome: Genome,
    *,
    mutation_rate: float = 0.12,
    mutation_strength: float = 0.08,
    rng: random.Random | None = None,
) -> Genome:
    """Mutate each trait independently with small Gaussian noise."""

    rng = rng or random.Random()
    values: dict[str, float] = {}
    for trait in Genome.trait_names:
        value = getattr(genome, trait)
        # Each trait independently has a `mutation_rate` chance of changing.
        if rng.random() < mutation_rate:
            lower, upper = TRAIT_BOUNDS[trait]
            # The nudge is scaled by the trait's own range, so `mutation_strength`
            # means the same *relative* amount for every trait -- a step for
            # `speed` (range ~4) and one for `vision_radius` (range ~150) are
            # proportional rather than one dominating the other.
            value += rng.gauss(0.0, (upper - lower) * mutation_strength)
        # Always clamp back into the legal range after mutating.
        values[trait] = clamp(value, *TRAIT_BOUNDS[trait])
    return Genome(**values)


def inherit_genome(
    parent_a: Genome,
    parent_b: Genome | None = None,
    *,
    mutation_rate: float = 0.12,
    mutation_strength: float = 0.08,
    rng: random.Random | None = None,
) -> Genome:
    """Inherit parent traits, optionally averaging two parents, then mutate."""

    values: dict[str, float] = {}
    for trait in Genome.trait_names:
        if parent_b is None:
            values[trait] = getattr(parent_a, trait)
        else:
            values[trait] = (getattr(parent_a, trait) + getattr(parent_b, trait)) / 2.0

    inherited = Genome(**values).clamped()
    return mutate_genome(
        inherited,
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        rng=rng,
    )

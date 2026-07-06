from __future__ import annotations

import random

from src.simulation.genetics import (
    Genome,
    TRAIT_BOUNDS,
    default_predator_genome,
    default_prey_genome,
    inherit_genome,
    mutate_genome,
)


def assert_genome_inside_bounds(genome: Genome) -> None:
    for trait, value in genome.as_dict().items():
        lower, upper = TRAIT_BOUNDS[trait]
        assert lower <= value <= upper, f"{trait}={value} outside [{lower}, {upper}]"


def stable_parent_genome() -> Genome:
    return Genome(
        speed=2.0,
        vision_radius=100.0,
        metabolism=0.12,
        reproduction_threshold=80.0,
        fear_sensitivity=0.8,
        aggression=0.6,
    )


def test_default_prey_genome_values_are_inside_allowed_bounds() -> None:
    genome = default_prey_genome(random.Random(1))

    assert_genome_inside_bounds(genome)


def test_default_predator_genome_values_are_inside_allowed_bounds() -> None:
    genome = default_predator_genome(random.Random(2))

    assert_genome_inside_bounds(genome)


def test_mutation_keeps_values_inside_bounds() -> None:
    rng = random.Random(3)
    parent = Genome(
        speed=TRAIT_BOUNDS["speed"][1],
        vision_radius=TRAIT_BOUNDS["vision_radius"][0],
        metabolism=TRAIT_BOUNDS["metabolism"][1],
        reproduction_threshold=TRAIT_BOUNDS["reproduction_threshold"][0],
        fear_sensitivity=TRAIT_BOUNDS["fear_sensitivity"][1],
        aggression=TRAIT_BOUNDS["aggression"][0],
    )

    for _ in range(100):
        mutated = mutate_genome(parent, mutation_rate=1.0, mutation_strength=0.45, rng=rng)
        assert_genome_inside_bounds(mutated)


def test_inheritance_returns_valid_genome() -> None:
    rng = random.Random(4)
    parent_a = stable_parent_genome()
    parent_b = Genome(
        speed=2.8,
        vision_radius=130.0,
        metabolism=0.16,
        reproduction_threshold=95.0,
        fear_sensitivity=1.1,
        aggression=1.0,
    )

    child = inherit_genome(parent_a, parent_b, mutation_rate=0.2, mutation_strength=0.05, rng=rng)

    assert isinstance(child, Genome)
    assert_genome_inside_bounds(child)


def test_forced_mutation_can_change_at_least_one_trait() -> None:
    parent = stable_parent_genome()

    mutated = mutate_genome(
        parent,
        mutation_rate=1.0,
        mutation_strength=0.08,
        rng=random.Random(10),
    )

    assert any(
        getattr(mutated, trait) != getattr(parent, trait)
        for trait in Genome.trait_names
    )

"""Species definitions and display metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Species(StrEnum):
    """Supported creature species."""

    PREY = "prey"
    PREDATOR = "predator"


PREY = Species.PREY.value
PREDATOR = Species.PREDATOR.value


@dataclass(frozen=True, slots=True)
class SpeciesProfile:
    """Lightweight metadata used by renderers and reports."""

    name: str
    display_name: str
    color: tuple[int, int, int]


PREY_PROFILE = SpeciesProfile(PREY, "Prey", (76, 166, 255))
PREDATOR_PROFILE = SpeciesProfile(PREDATOR, "Predator", (235, 92, 82))

SPECIES_PROFILES = {
    PREY: PREY_PROFILE,
    PREDATOR: PREDATOR_PROFILE,
}


def normalize_species(species: Species | str) -> str:
    """Return a stable string value for a species enum or string."""

    value = species.value if isinstance(species, Species) else str(species)
    if value not in SPECIES_PROFILES:
        raise ValueError(f"Unknown species: {species}")
    return value


def get_profile(species: Species | str) -> SpeciesProfile:
    """Return visual/profile metadata for a species."""

    return SPECIES_PROFILES[normalize_species(species)]

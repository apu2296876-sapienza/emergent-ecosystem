"""Small math and randomness helpers used by the simulation."""

from __future__ import annotations

import math
import random
from typing import TypeAlias

Vec2: TypeAlias = tuple[float, float]


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp a number into an inclusive range."""

    return max(lower, min(upper, value))


def clamp_position(position: Vec2, width: float, height: float, margin: float = 0.0) -> Vec2:
    """Keep a position inside the playable world rectangle."""

    return (
        clamp(position[0], margin, width - margin),
        clamp(position[1], margin, height - margin),
    )


def add(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] + b[0], a[1] + b[1])


def subtract(a: Vec2, b: Vec2) -> Vec2:
    return (a[0] - b[0], a[1] - b[1])


def multiply(vector: Vec2, scalar: float) -> Vec2:
    return (vector[0] * scalar, vector[1] * scalar)


def length(vector: Vec2) -> float:
    return math.hypot(vector[0], vector[1])


def distance(a: Vec2, b: Vec2) -> float:
    return length(subtract(a, b))


def normalize(vector: Vec2) -> Vec2:
    magnitude = length(vector)
    if magnitude <= 1e-9:
        return (0.0, 0.0)
    return (vector[0] / magnitude, vector[1] / magnitude)


def random_unit_vector(rng: random.Random) -> Vec2:
    angle = rng.uniform(0.0, math.tau)
    return (math.cos(angle), math.sin(angle))


def random_position(width: float, height: float, rng: random.Random, margin: float = 0.0) -> Vec2:
    return (
        rng.uniform(margin, width - margin),
        rng.uniform(margin, height - margin),
    )


def jittered_position(
    origin: Vec2,
    spread: float,
    width: float,
    height: float,
    rng: random.Random,
    margin: float = 0.0,
) -> Vec2:
    """Return a random point near an origin while staying inside the world."""

    offset = multiply(random_unit_vector(rng), rng.uniform(0.0, spread))
    return clamp_position(add(origin, offset), width, height, margin)

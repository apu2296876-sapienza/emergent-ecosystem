"""Shared constants and action names."""

from __future__ import annotations

PREY = "prey"
PREDATOR = "predator"

WANDER = "wander"
SEEK_FOOD = "seek_food"
FLEE = "flee"
CHASE_PREY = "chase_prey"
EAT = "eat"
REPRODUCE = "reproduce"
REST = "rest"

ALL_ACTIONS = (WANDER, SEEK_FOOD, FLEE, CHASE_PREY, EAT, REPRODUCE, REST)

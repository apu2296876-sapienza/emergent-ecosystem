from __future__ import annotations

from src.simulation.agent import Agent
from src.simulation.decision import (
    Decision,
    FSMDecisionSystem,
    Perception,
    UtilityDecisionSystem,
    VisibleEntity,
)
from src.simulation.genetics import Genome
from src.simulation.species import PREDATOR, PREY
from src.utils.constants import ALL_ACTIONS, CHASE_PREY, FLEE, REPRODUCE, SEEK_FOOD


def make_genome(**overrides: float) -> Genome:
    values = {
        "speed": 2.0,
        "vision_radius": 100.0,
        "metabolism": 0.1,
        "reproduction_threshold": 70.0,
        "fear_sensitivity": 1.0,
        "aggression": 1.0,
    }
    values.update(overrides)
    return Genome(**values)


def make_agent(species: str, *, energy: float, genome: Genome | None = None) -> Agent:
    return Agent(
        id=1,
        species=species,
        position=(40.0, 40.0),
        genome=genome or make_genome(),
        energy=energy,
    )


def test_prey_chooses_flee_when_predator_is_nearby() -> None:
    prey = make_agent(PREY, energy=45.0)
    perception = Perception(
        nearest_predator=VisibleEntity(2, (45.0, 40.0), distance=5.0),
        visible_predators=1,
    )

    decision = UtilityDecisionSystem().choose_action(prey, perception)

    assert decision.action == FLEE
    assert decision.target == perception.nearest_predator


def test_prey_chooses_seek_food_when_hungry_and_food_is_visible() -> None:
    prey = make_agent(PREY, energy=18.0)
    perception = Perception(
        nearest_resource=VisibleEntity(2, (80.0, 40.0), distance=40.0, energy=30.0),
        visible_resources=1,
    )

    decision = UtilityDecisionSystem(eat_distance=8.0).choose_action(prey, perception)

    assert decision.action == SEEK_FOOD
    assert decision.target == perception.nearest_resource


def test_prey_chooses_reproduce_when_energy_is_high_and_no_threat_exists() -> None:
    prey = make_agent(
        PREY,
        energy=135.0,
        genome=make_genome(reproduction_threshold=65.0),
    )

    decision = UtilityDecisionSystem().choose_action(prey, Perception())

    assert decision.action == REPRODUCE
    assert decision.target is None


def test_predator_chooses_chase_prey_when_prey_is_visible() -> None:
    predator = make_agent(
        PREDATOR,
        energy=45.0,
        genome=make_genome(reproduction_threshold=120.0, aggression=1.2),
    )
    perception = Perception(
        nearest_prey=VisibleEntity(2, (90.0, 40.0), distance=50.0, energy=40.0),
        visible_prey=1,
    )

    decision = UtilityDecisionSystem(capture_distance=10.0).choose_action(predator, perception)

    assert decision.action == CHASE_PREY
    assert decision.target == perception.nearest_prey


def test_predator_chooses_reproduce_when_energy_is_high_and_no_prey_urgency_dominates() -> None:
    predator = make_agent(
        PREDATOR,
        energy=190.0,
        genome=make_genome(reproduction_threshold=100.0, aggression=1.0),
    )

    decision = UtilityDecisionSystem().choose_action(predator, Perception())

    assert decision.action == REPRODUCE


def test_utility_decision_returns_valid_action_names() -> None:
    prey = make_agent(PREY, energy=45.0)
    predator = make_agent(PREDATOR, energy=45.0)
    decision_system = UtilityDecisionSystem()

    decisions = [
        decision_system.choose_action(prey, Perception()),
        decision_system.choose_action(
            prey,
            Perception(nearest_predator=VisibleEntity(2, (45.0, 40.0), distance=5.0)),
        ),
        decision_system.choose_action(
            predator,
            Perception(nearest_prey=VisibleEntity(3, (90.0, 40.0), distance=50.0)),
        ),
    ]

    assert all(isinstance(decision, Decision) for decision in decisions)
    assert all(decision.action in ALL_ACTIONS for decision in decisions)


def test_fsm_returns_valid_actions() -> None:
    decision_system = FSMDecisionSystem()
    prey = make_agent(PREY, energy=18.0)
    predator = make_agent(PREDATOR, energy=45.0)

    decisions = [
        decision_system.choose_action(prey, Perception()),
        decision_system.choose_action(
            prey,
            Perception(nearest_predator=VisibleEntity(2, (42.0, 40.0), distance=2.0)),
        ),
        decision_system.choose_action(
            predator,
            Perception(nearest_prey=VisibleEntity(3, (80.0, 40.0), distance=40.0)),
        ),
    ]

    assert all(isinstance(decision, Decision) for decision in decisions)
    assert all(decision.action in ALL_ACTIONS for decision in decisions)

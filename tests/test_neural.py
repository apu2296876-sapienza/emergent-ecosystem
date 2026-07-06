from __future__ import annotations

import random
from pathlib import Path

from config import SimulationConfig
from src.analysis.experiment_runner import run_experiment
from src.simulation.agent import Agent
from src.simulation.decision import NeuralDecisionSystem, Perception, VisibleEntity
from src.simulation.genetics import Genome
from src.simulation.neural import (
    N_HIDDEN,
    N_INPUTS,
    N_OUTPUTS,
    N_PARAMS,
    WEIGHT_BOUND,
    Brain,
    forward,
    inherit_brain,
    random_brain,
)
from src.simulation.species import PREDATOR, PREY
from src.simulation.world import World
from src.utils.constants import ALL_ACTIONS, EAT, FLEE, REST, WANDER


def make_genome(**overrides: float) -> Genome:
    values = {
        "speed": 2.0,
        "vision_radius": 95.0,
        "metabolism": 0.08,
        "reproduction_threshold": 75.0,
        "fear_sensitivity": 1.0,
        "aggression": 1.0,
    }
    values.update(overrides)
    return Genome(**values)


def zero_brain() -> Brain:
    return Brain(tuple(0.0 for _ in range(N_PARAMS)))


def brain_favoring(action: str, score: float = 5.0) -> Brain:
    """Zero brain whose only nonzero weight is the output bias of `action`."""

    weights = [0.0] * N_PARAMS
    bias_start = N_HIDDEN * N_INPUTS + N_HIDDEN + N_OUTPUTS * N_HIDDEN
    weights[bias_start + ALL_ACTIONS.index(action)] = score
    return Brain(tuple(weights))


def perception_with(**overrides: object) -> Perception:
    return Perception(**overrides)


def visible(distance: float, entity_id: int = 99) -> VisibleEntity:
    return VisibleEntity(id=entity_id, position=(0.0, 0.0), distance=distance)


def test_forward_shape_and_determinism() -> None:
    brain = random_brain(random.Random(3))
    inputs = [0.5] * N_INPUTS

    scores_a = forward(brain, inputs)
    scores_b = forward(brain, inputs)

    assert len(scores_a) == N_OUTPUTS == len(ALL_ACTIONS)
    assert scores_a == scores_b


def test_zero_brain_defaults_to_first_valid_action() -> None:
    system = NeuralDecisionSystem()
    agent = Agent(id=1, species=PREY, position=(10.0, 10.0),
                  genome=make_genome(), energy=30.0, brain=zero_brain())

    decision = system.choose_action(agent, perception_with())

    # All scores tie at zero and nothing is visible, so the argmax over the
    # always-valid pair {WANDER, REST} resolves to WANDER (first in order).
    assert decision.action == WANDER


def test_masking_blocks_invalid_actions_even_when_preferred() -> None:
    system = NeuralDecisionSystem(capture_distance=10.0)
    predator = Agent(id=2, species=PREDATOR, position=(10.0, 10.0),
                     genome=make_genome(), energy=30.0,
                     brain=brain_favoring(FLEE))
    perception = perception_with(nearest_prey=visible(5.0), visible_prey=1)

    decision = system.choose_action(predator, perception)

    # Predators can never FLEE; the network's favourite is masked out.
    assert decision.action != FLEE
    assert decision.utilities[FLEE] < -9.0


def test_eat_requires_target_in_range() -> None:
    system = NeuralDecisionSystem(eat_distance=8.0)
    prey = Agent(id=3, species=PREY, position=(10.0, 10.0),
                 genome=make_genome(reproduction_threshold=150.0), energy=30.0,
                 brain=brain_favoring(EAT))

    far = system.choose_action(prey, perception_with(nearest_resource=visible(50.0), visible_resources=1))
    near = system.choose_action(prey, perception_with(nearest_resource=visible(4.0), visible_resources=1))

    assert far.action != EAT
    assert near.action == EAT
    assert near.target is not None and near.target.distance == 4.0


def test_rest_can_win_when_network_prefers_it() -> None:
    system = NeuralDecisionSystem()
    agent = Agent(id=4, species=PREY, position=(10.0, 10.0),
                  genome=make_genome(), energy=30.0,
                  brain=brain_favoring(REST))

    assert system.choose_action(agent, perception_with()).action == REST


def test_inherit_brain_mutates_and_clamps() -> None:
    rng = random.Random(11)
    parent = random_brain(rng)

    identical = inherit_brain(parent, mutation_rate=0.0, mutation_strength=1.0, rng=rng)
    mutated = inherit_brain(parent, mutation_rate=1.0, mutation_strength=0.5, rng=rng)

    assert identical.weights == parent.weights
    assert mutated.weights != parent.weights
    assert len(mutated.weights) == N_PARAMS
    assert all(-WEIGHT_BOUND <= w <= WEIGHT_BOUND for w in mutated.weights)


def test_world_neural_mode_runs_and_children_inherit_brains(tmp_path: Path) -> None:
    config = SimulationConfig(
        width=240, height=180, seed=9, decision_mode="neural",
        initial_prey=3, initial_predators=1, initial_resources=6,
        resource_regen_probability=0.0, metrics_interval=1,
    )
    world = World(config)
    world.populate_initial()
    for _ in range(5):
        world.update()
    assert world.metrics.latest_record() is not None
    assert all(agent.brain is not None for agent in world.agents)

    # Direct reproduction path: the child must carry an inherited controller.
    parent = Agent(id=500, species=PREY, position=(50.0, 50.0),
                   genome=make_genome(reproduction_threshold=40.0), energy=120.0,
                   brain=random_brain(random.Random(1)))
    child = parent.reproduce(
        new_id=501, rng=random.Random(2), mutation_rate=0.0, mutation_strength=0.0,
        energy_fraction=0.4, cooldown_ticks=5, jitter_radius=10.0,
        bounds=(240.0, 180.0),
    )
    assert child is not None and child.brain is not None
    assert len(child.brain.weights) == N_PARAMS


def test_experiment_runner_supports_neural_mode(tmp_path: Path) -> None:
    config = SimulationConfig(
        width=240, height=180, seed=5,
        initial_prey=2, initial_predators=1, initial_resources=4,
        resource_regen_probability=0.0, metrics_interval=1,
    )

    row = run_experiment("baseline", config, decision_mode="neural",
                         ticks=5, output_dir=tmp_path)

    assert row["decision_mode"] == "neural"
    assert (tmp_path / "baseline_neural.csv").exists()

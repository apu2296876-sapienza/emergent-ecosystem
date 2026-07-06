from __future__ import annotations

from pathlib import Path

from config import SimulationConfig
from src.analysis.experiment_runner import run_experiment
from src.simulation.genetics import Genome
from src.simulation.species import PREDATOR, PREY
from src.simulation.world import World


def make_config(**overrides: object) -> SimulationConfig:
    values = {
        "width": 240,
        "height": 180,
        "seed": 5,
        "initial_prey": 0,
        "initial_predators": 0,
        "initial_resources": 0,
        "max_resources": 30,
        "resource_regen_probability": 0.0,
        "metrics_interval": 1,
        "mutation_rate": 0.0,
        "mutation_strength": 0.0,
        "reproduction_cooldown_ticks": 5,
        "resource_energy": 35.0,
        "starting_energy_prey": 80.0,
        "starting_energy_predator": 110.0,
        "max_agent_energy": 200.0,
        "eat_distance": 9.0,
        "capture_distance": 10.0,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def stable_genome(**overrides: float) -> Genome:
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


def agent_snapshot(world: World) -> list[tuple[str, tuple[float, float], dict[str, float]]]:
    return [
        (agent.species, agent.position, agent.genome.as_dict())
        for agent in world.agents
    ]


def test_world_initializes_with_requested_numbers_of_agents_and_resources() -> None:
    config = make_config(initial_prey=4, initial_predators=2, initial_resources=7)
    world = World(config)

    world.populate_initial()

    assert world.population(PREY) == 4
    assert world.population(PREDATOR) == 2
    assert len(world.resources) == 7


def test_agents_lose_energy_over_time() -> None:
    world = World(make_config())
    prey = world.add_prey(
        position=(40.0, 40.0),
        genome=stable_genome(reproduction_threshold=150.0),
        energy=50.0,
    )
    starting_energy = prey.energy

    world.update()

    assert prey.energy < starting_energy


def test_agent_dies_when_energy_reaches_zero() -> None:
    world = World(make_config())
    world.add_prey(position=(40.0, 40.0), genome=stable_genome(), energy=0.0)

    world.update()

    assert world.population(PREY) == 0
    assert world.metrics.deaths == 1


def test_prey_eating_resource_increases_energy_and_removes_resource() -> None:
    world = World(make_config())
    prey = world.add_prey(
        position=(50.0, 50.0),
        genome=stable_genome(reproduction_threshold=150.0),
        energy=20.0,
    )
    world.add_resource(position=(50.0, 50.0), energy_value=35.0)

    world.update()

    assert prey.energy > 20.0
    assert len(world.resources) == 0


def test_predator_eating_prey_increases_energy_and_kills_prey() -> None:
    world = World(make_config())
    predator = world.add_predator(
        position=(70.0, 70.0),
        genome=stable_genome(reproduction_threshold=150.0, aggression=1.2),
        energy=40.0,
    )
    starting_energy = predator.energy
    world.add_prey(
        position=(70.0, 70.0),
        genome=stable_genome(reproduction_threshold=150.0),
        energy=45.0,
    )

    world.update()

    assert predator.energy > starting_energy
    assert world.population(PREY) == 0
    assert world.population(PREDATOR) == 1
    assert world.metrics.predation_events == 1


def test_reproduction_creates_an_offspring() -> None:
    world = World(make_config())
    parent_genome = stable_genome(reproduction_threshold=55.0)
    parent = world.add_prey(position=(90.0, 90.0), genome=parent_genome, energy=130.0)

    world.update()

    assert world.population(PREY) == 2
    assert world.metrics.births == 1
    offspring = [agent for agent in world.agents if agent.id != parent.id][0]
    assert offspring.parent_id == parent.id
    assert offspring.generation == parent.generation + 1
    assert offspring.genome.as_dict() == parent_genome.as_dict()


def test_metrics_recorder_logs_population_values(tmp_path: Path) -> None:
    world = World(make_config())
    world.add_prey(position=(40.0, 40.0), genome=stable_genome(), energy=40.0)
    world.add_predator(position=(180.0, 120.0), genome=stable_genome(), energy=40.0)
    world.add_resource(position=(100.0, 90.0))

    world.update()
    record = world.metrics.latest_record()
    output_path = world.metrics.save_csv(tmp_path / "nested" / "metrics.csv")

    assert record is not None
    assert record["tick"] == 1
    assert record["prey_population"] == 1
    assert record["predator_population"] == 1
    assert record["resource_count"] == 1
    assert output_path.exists()


def test_world_update_does_not_crash_when_prey_or_predators_go_extinct() -> None:
    prey_only = World(make_config())
    prey_only.add_prey(position=(50.0, 50.0), genome=stable_genome(), energy=25.0)
    predator_only = World(make_config())
    predator_only.add_predator(position=(70.0, 70.0), genome=stable_genome(), energy=25.0)
    empty = World(make_config())

    for world in (prey_only, predator_only, empty):
        for _ in range(3):
            world.update()
        assert world.metrics.latest_record() is not None


def test_same_seed_produces_same_initial_positions_and_genomes() -> None:
    config_a = make_config(
        seed=123,
        initial_prey=5,
        initial_predators=3,
        initial_resources=4,
    )
    config_b = make_config(
        seed=123,
        initial_prey=5,
        initial_predators=3,
        initial_resources=4,
    )
    world_a = World(config_a)
    world_b = World(config_b)

    world_a.populate_initial()
    world_b.populate_initial()

    assert agent_snapshot(world_a) == agent_snapshot(world_b)
    assert [resource.position for resource in world_a.resources] == [
        resource.position for resource in world_b.resources
    ]


def test_utility_and_fsm_modes_both_run_without_crashing() -> None:
    for decision_mode in ("utility", "fsm"):
        world = World(
            make_config(
                decision_mode=decision_mode,
                initial_prey=3,
                initial_predators=1,
                initial_resources=5,
            )
        )
        world.populate_initial()

        for _ in range(5):
            world.update()

        assert world.metrics.latest_record() is not None


def test_experiment_runner_supports_both_decision_modes(tmp_path: Path) -> None:
    config = make_config(
        initial_prey=2,
        initial_predators=1,
        initial_resources=4,
    )

    utility_row = run_experiment(
        "baseline",
        config,
        decision_mode="utility",
        ticks=5,
        output_dir=tmp_path,
    )
    fsm_row = run_experiment(
        "baseline",
        config,
        decision_mode="fsm",
        ticks=5,
        output_dir=tmp_path,
    )

    assert utility_row["decision_mode"] == "utility"
    assert fsm_row["decision_mode"] == "fsm"
    assert (tmp_path / "baseline_utility.csv").exists()
    assert (tmp_path / "baseline_fsm.csv").exists()

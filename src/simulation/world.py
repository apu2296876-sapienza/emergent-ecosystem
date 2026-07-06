"""World state, update loop, collisions, reproduction, and regeneration."""

from __future__ import annotations

from dataclasses import replace
import random
from typing import Generic, TypeVar

from config import SimulationConfig
from src.simulation.agent import Agent
from src.simulation.decision import Decision, Perception, VisibleEntity, create_decision_system
from src.simulation.events import (
    BirthEvent,
    DeathEvent,
    ExtinctionEvent,
    PredationEvent,
    ResourceEatenEvent,
    SimulationEvent,
)
from src.simulation.genetics import Genome, default_predator_genome, default_prey_genome
from src.simulation.metrics import MetricsRecorder
from src.simulation.neural import random_brain
from src.simulation.resource import Resource
from src.simulation.species import PREDATOR, PREY, Species, normalize_species
from src.utils.constants import CHASE_PREY, EAT, FLEE, REPRODUCE, REST, SEEK_FOOD, WANDER
from src.utils.random_utils import (
    Vec2,
    add,
    distance,
    jittered_position,
    multiply,
    normalize,
    random_position,
    random_unit_vector,
)

T = TypeVar("T")


class EntityList(list[T], Generic[T]):
    """List with small dictionary-style helpers for id-addressed entities."""

    def values(self) -> list[T]:
        return list(self)

    def get(self, entity_id: int, default: T | None = None) -> T | None:
        for entity in self:
            if getattr(entity, "id", None) == entity_id:
                return entity
        return default

    def remove_id(self, entity_id: int) -> T | None:
        entity = self.get(entity_id)
        if entity is not None:
            self.remove(entity)
        return entity


class World:
    """A continuous 2D artificial-life ecosystem."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        *,
        width: int | None = None,
        height: int | None = None,
        seed: int | None = None,
    ) -> None:
        self.config = replace(config) if config is not None else SimulationConfig()
        if width is not None:
            self.config.width = width
        if height is not None:
            self.config.height = height
        if seed is not None:
            self.config.seed = seed

        self.width = self.config.width
        self.height = self.config.height
        self.rng = random.Random(self.config.seed)
        self.agents: EntityList[Agent] = EntityList()
        self.resources: EntityList[Resource] = EntityList()
        self.events: list[SimulationEvent] = []
        self.metrics = MetricsRecorder()
        self.decision_system = create_decision_system(
            self.config.decision_mode,
            eat_distance=self.config.eat_distance,
            capture_distance=self.config.capture_distance,
        )
        self.tick_count = 0
        self._next_agent_id = 1
        self._next_resource_id = 1
        self._death_recorded: set[int] = set()
        self._last_population = {PREY: 0, PREDATOR: 0}

    @property
    def tick(self) -> int:
        """Alias for the current simulation tick."""

        return self.tick_count

    @tick.setter
    def tick(self, value: int) -> None:
        self.tick_count = int(value)

    def reset(self, seed: int | None = None) -> None:
        """Clear all entities and repopulate from configuration."""

        if seed is not None:
            self.config.seed = seed
        self.rng = random.Random(self.config.seed)
        self.agents.clear()
        self.resources.clear()
        self.events.clear()
        self.metrics.reset()
        self.tick_count = 0
        self._next_agent_id = 1
        self._next_resource_id = 1
        self._death_recorded.clear()
        self._last_population = {PREY: 0, PREDATOR: 0}
        self.populate_initial()

    def populate_initial(self) -> None:
        """Create the configured starting population and resources."""

        for _ in range(self.config.initial_resources):
            self.add_resource()
        for _ in range(self.config.initial_prey):
            self.add_prey()
        for _ in range(self.config.initial_predators):
            self.add_predator()
        self._last_population = {
            PREY: self.population(PREY),
            PREDATOR: self.population(PREDATOR),
        }

    def update(self, dt: float | None = None) -> None:
        """Advance the ecosystem by one tick (one simulation step).

        The order of operations each tick is:
          1. age and regrow food,
          2. let every living agent sense, decide, and act (this is where each
             agent's decision policy runs and where births/deaths happen),
          3. clear out anything that died this tick,
          4. check whether a whole species has gone extinct,
          5. periodically record metrics for the CSV/plots.
        """

        dt = self.config.dt if dt is None else dt
        self.tick_count += 1

        self._age_resources(dt)          # food patches get older...
        self._regenerate_resources()     # ...and new food may appear
        # Iterate over a COPY of the list: agents can be born or die during the
        # loop, and we must not modify the list we're iterating over.
        for agent in list(self.agents):
            if agent.alive:
                self._update_agent(agent, dt)   # perceive -> decide -> act

        self._remove_dead_agents()
        self._check_extinctions()
        if self.tick_count % self.config.metrics_interval == 0:
            self.metrics.record(self)

    def add_prey(
        self,
        position: Vec2 | None = None,
        genome: Genome | None = None,
        energy: float | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
        parent_id: int | None = None,
        generation: int = 0,
    ) -> Agent:
        """Spawn a prey agent."""

        return self._add_agent(
            PREY,
            position=position,
            genome=genome or default_prey_genome(self.rng),
            energy=energy if energy is not None else self.config.starting_energy_prey,
            x=x,
            y=y,
            parent_id=parent_id,
            generation=generation,
        )

    def add_predator(
        self,
        position: Vec2 | None = None,
        genome: Genome | None = None,
        energy: float | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
        parent_id: int | None = None,
        generation: int = 0,
    ) -> Agent:
        """Spawn a predator agent."""

        return self._add_agent(
            PREDATOR,
            position=position,
            genome=genome or default_predator_genome(self.rng),
            energy=energy if energy is not None else self.config.starting_energy_predator,
            x=x,
            y=y,
            parent_id=parent_id,
            generation=generation,
        )

    def spawn_prey(self, *args: object, **kwargs: object) -> Agent:
        return self.add_prey(*args, **kwargs)

    def spawn_predator(self, *args: object, **kwargs: object) -> Agent:
        return self.add_predator(*args, **kwargs)

    def add_resource(
        self,
        position: Vec2 | None = None,
        energy: float | None = None,
        radius: float | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
        energy_value: float | None = None,
        regrowth_timer: int = 0,
    ) -> Resource:
        """Spawn an edible resource patch."""

        resource_position = self._resolve_position(position, x, y)
        resource = Resource(
            id=self._next_resource_id,
            position=resource_position,
            energy_value=energy_value if energy_value is not None else energy if energy is not None else self.config.resource_energy,
            radius=radius if radius is not None else self.config.resource_radius,
            regrowth_timer=regrowth_timer,
        )
        self.resources.append(resource)
        self._next_resource_id += 1
        return resource

    def spawn_resource(self, *args: object, **kwargs: object) -> Resource:
        return self.add_resource(*args, **kwargs)

    def add_resource_patch(self, position: Vec2 | None = None, count: int | None = None) -> list[Resource]:
        """Create a clustered food patch."""

        center = position or random_position(
            self.width,
            self.height,
            self.rng,
            self.config.bounds_margin,
        )
        patch_size = self.config.resource_patch_size if count is None else count
        resources: list[Resource] = []
        for _ in range(patch_size):
            if len(self.resources) >= self.config.max_resources:
                break
            resources.append(
                self.add_resource(
                    position=jittered_position(
                        center,
                        self.config.resource_cluster_spread,
                        self.width,
                        self.height,
                        self.rng,
                        self.config.bounds_margin,
                    )
                )
            )
        return resources

    def population(self, species: Species | str) -> int:
        """Return the number of living agents for one species."""

        species_key = normalize_species(species)
        return sum(1 for agent in self.agents if agent.species == species_key and agent.alive)

    def perceive(self, agent: Agent) -> Perception:
        """Build a local perception snapshot for an agent."""

        visible_resources = agent.get_perceived_resources(self.resources)
        visible_agents = agent.get_perceived_agents(self.agents)
        visible_prey = [other for other in visible_agents if other.species == PREY]
        visible_predators = [other for other in visible_agents if other.species == PREDATOR]
        same_species = [other for other in visible_agents if other.species == agent.species]

        return Perception(
            nearest_resource=self._nearest_visible(agent, visible_resources),
            nearest_prey=self._nearest_visible(agent, visible_prey),
            nearest_predator=self._nearest_visible(agent, visible_predators),
            visible_resources=len(visible_resources),
            visible_prey=len(visible_prey),
            visible_predators=len(visible_predators),
            same_species_nearby=len(same_species),
        )

    def _add_agent(
        self,
        species: Species | str,
        *,
        position: Vec2 | None,
        genome: Genome,
        energy: float,
        x: float | None,
        y: float | None,
        parent_id: int | None,
        generation: int,
    ) -> Agent:
        agent = Agent(
            id=self._next_agent_id,
            species=species,
            position=self._resolve_position(position, x, y),
            genome=genome,
            energy=energy,
            heading=random_unit_vector(self.rng),
            parent_id=parent_id,
            generation=generation,
            # Every agent carries a generation-zero random controller so the
            # decision mode can be switched (even live) without respawning;
            # utility and FSM policies simply ignore it.
            brain=random_brain(self.rng),
        )
        self.agents.append(agent)
        self._next_agent_id += 1
        return agent

    def _resolve_position(
        self,
        position: Vec2 | None,
        x: float | None,
        y: float | None,
    ) -> Vec2:
        if position is not None:
            return position
        if x is not None and y is not None:
            return (float(x), float(y))
        return random_position(self.width, self.height, self.rng, self.config.bounds_margin)

    def _update_agent(self, agent: Agent, dt: float) -> None:
        # This is the full life of one agent in one tick.

        # Die first if already out of energy or too old.
        if agent.energy <= 0.0:
            self._kill_agent(agent, cause="starvation")
            return
        if not agent.update_age(dt, self._max_age(agent)):
            self._kill_agent(agent, cause="old_age")
            return

        # 1. SENSE: gather what this agent can see nearby.
        perception = self.perceive(agent)
        # 2. DECIDE: run whichever policy is active (utility / FSM / neural).
        #    This single call is the only line that differs between the three
        #    architectures being compared.
        decision = self.decision_system.choose_action(agent, perception)
        agent.current_action = decision.action

        # 3. PAY: simply being alive burns energy (metabolism); resting burns
        #    less. If that cost empties the tank, the agent starves.
        metabolism = agent.genome.metabolism * dt
        if decision.action == REST:
            metabolism *= self.config.rest_metabolism_multiplier
        if not agent.consume_energy(metabolism):
            self._kill_agent(agent, cause="starvation")
            return

        # 4. ACT: carry out the chosen action (move, eat, reproduce, ...).
        self._apply_decision(agent, decision, dt)
        if not agent.alive or agent.energy <= 0.0:
            self._kill_agent(agent, cause="exhaustion")

    def _apply_decision(self, agent: Agent, decision: Decision, dt: float) -> None:
        action = decision.action
        if action == EAT:
            self._process_eating(agent, decision)
        elif action == FLEE:
            self._move_away(agent, decision.target, dt, speed_multiplier=1.18)
        elif action in {SEEK_FOOD, CHASE_PREY}:
            self._move_towards(agent, decision.target, dt)
        elif action == REPRODUCE:
            self._process_reproduction(agent)
        elif action == WANDER:
            self._wander(agent, dt)
        elif action == REST:
            return
        else:
            self._wander(agent, dt)

    def _process_eating(self, agent: Agent, decision: Decision) -> None:
        target = decision.target
        if agent.species == PREY and target is not None:
            resource = self._resource_from_target(target)
            if resource is not None and agent.distance_to(resource) <= self.config.eat_distance + resource.radius:
                gained = resource.energy_value
                agent.energy = min(self.config.max_agent_energy, agent.energy + gained)
                self.resources.remove(resource)
                self.events.append(
                    ResourceEatenEvent(
                        tick=self.tick_count,
                        agent_id=agent.id,
                        resource_id=resource.id,
                        energy_gained=gained,
                        species=agent.species,
                    )
                )
                return

        if agent.species == PREDATOR and target is not None:
            prey = self._agent_from_target(target)
            if prey is not None and prey.alive and agent.distance_to(prey) <= self.config.capture_distance:
                gained = max(22.0, prey.energy * 0.72)
                agent.energy = min(self.config.max_agent_energy, agent.energy + gained)
                self._kill_agent(prey, cause="predation", predator_id=agent.id, energy_gained=gained)
                return

        self._wander(agent, self.config.dt * 0.35)

    def _process_reproduction(self, agent: Agent) -> None:
        child = agent.reproduce(
            new_id=self._next_agent_id,
            rng=self.rng,
            mutation_rate=self.config.mutation_rate,
            mutation_strength=self.config.mutation_strength,
            energy_fraction=self.config.reproduction_cost_fraction,
            cooldown_ticks=self.config.reproduction_cooldown_ticks,
            jitter_radius=16.0,
            bounds=(self.width, self.height),
            margin=self.config.bounds_margin,
            brain_mutation_rate=self.config.brain_mutation_rate,
            brain_mutation_strength=self.config.brain_mutation_strength,
        )
        if child is None:
            return

        self.agents.append(child)
        self._next_agent_id += 1
        self.metrics.record_birth()
        self.events.append(
            BirthEvent(
                tick=self.tick_count,
                parent_id=agent.id,
                child_id=child.id,
                species=child.species,
                generation=child.generation,
            )
        )

    def _move_towards(
        self,
        agent: Agent,
        target: VisibleEntity | None,
        dt: float,
        *,
        speed_multiplier: float = 1.0,
    ) -> None:
        target_object = self._target_object(target)
        if target_object is None:
            self._wander(agent, dt)
            return
        distance_moved = agent.move_towards(
            target_object,
            dt,
            speed_multiplier=speed_multiplier,
            bounds=(self.width, self.height),
            margin=self.config.bounds_margin,
        )
        self._charge_movement(agent, distance_moved)

    def _move_away(
        self,
        agent: Agent,
        target: VisibleEntity | None,
        dt: float,
        *,
        speed_multiplier: float = 1.0,
    ) -> None:
        target_object = self._target_object(target)
        if target_object is None:
            self._wander(agent, dt)
            return
        distance_moved = agent.move_away_from(
            target_object,
            dt,
            speed_multiplier=speed_multiplier,
            bounds=(self.width, self.height),
            margin=self.config.bounds_margin,
        )
        self._charge_movement(agent, distance_moved)

    def _wander(self, agent: Agent, dt: float) -> None:
        jitter = random_unit_vector(self.rng)
        direction = normalize(
            add(
                multiply(agent.heading, 1.0 - self.config.wander_turn_jitter),
                multiply(jitter, self.config.wander_turn_jitter),
            )
        )
        if direction == (0.0, 0.0):
            direction = jitter
        target = add(agent.position, multiply(direction, max(12.0, agent.genome.speed * 10.0)))
        distance_moved = agent.move_towards(
            target,
            dt,
            speed_multiplier=0.62,
            bounds=(self.width, self.height),
            margin=self.config.bounds_margin,
        )
        self._charge_movement(agent, distance_moved)

    def _charge_movement(self, agent: Agent, distance_moved: float) -> None:
        movement_cost = (
            distance_moved
            * agent.genome.metabolism
            * self.config.movement_cost_multiplier
        )
        if not agent.consume_energy(movement_cost):
            self._kill_agent(agent, cause="exhaustion")

    def _nearest_visible(self, agent: Agent, entities: list[object]) -> VisibleEntity | None:
        if not entities:
            return None
        nearest = min(entities, key=agent.distance_to)
        return VisibleEntity(
            id=getattr(nearest, "id"),
            position=getattr(nearest, "position"),
            distance=agent.distance_to(nearest),
            energy=float(getattr(nearest, "energy", getattr(nearest, "energy_value", 0.0))),
            entity=nearest,
        )

    @staticmethod
    def _target_object(target: VisibleEntity | None) -> object | None:
        if target is None:
            return None
        return target.entity if target.entity is not None else target.position

    def _resource_from_target(self, target: VisibleEntity) -> Resource | None:
        if isinstance(target.entity, Resource) and target.entity in self.resources:
            return target.entity
        return self.resources.get(target.id)

    def _agent_from_target(self, target: VisibleEntity) -> Agent | None:
        if isinstance(target.entity, Agent) and target.entity in self.agents:
            return target.entity
        return self.agents.get(target.id)

    def _age_resources(self, dt: float) -> None:
        for resource in self.resources:
            resource.age += dt
            if resource.regrowth_timer > 0:
                resource.regrowth_timer = max(0, resource.regrowth_timer - 1)

    def _regenerate_resources(self) -> None:
        if len(self.resources) >= self.config.max_resources:
            return
        if self.rng.random() < self.config.resource_regen_probability:
            self.add_resource()

    def _kill_agent(
        self,
        agent: Agent,
        *,
        cause: str,
        predator_id: int | None = None,
        energy_gained: float = 0.0,
    ) -> None:
        if agent.id in self._death_recorded:
            return

        agent.die()
        self._death_recorded.add(agent.id)
        self.metrics.record_death()
        if predator_id is not None:
            self.metrics.record_predation()
            self.events.append(
                PredationEvent(
                    tick=self.tick_count,
                    predator_id=predator_id,
                    prey_id=agent.id,
                    energy_gained=energy_gained,
                    species=agent.species,
                )
            )
        else:
            self.events.append(
                DeathEvent(
                    tick=self.tick_count,
                    agent_id=agent.id,
                    species=agent.species,
                    cause=cause,
                )
            )

    def _remove_dead_agents(self) -> None:
        self.agents[:] = [agent for agent in self.agents if agent.alive]

    def _check_extinctions(self) -> None:
        for species in (PREY, PREDATOR):
            current = self.population(species)
            if self._last_population[species] > 0 and current == 0:
                self.metrics.record_extinction()
                self.events.append(ExtinctionEvent(tick=self.tick_count, species=species))
            self._last_population[species] = current

    def _max_age(self, agent: Agent) -> float:
        return self.config.max_age_prey if agent.species == PREY else self.config.max_age_predator

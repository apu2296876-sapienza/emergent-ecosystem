"""Decision-making policies for autonomous agents.

This module holds the three "brains" an agent can use to pick its next action,
plus the shared machinery around them:

  * ``UtilityDecisionSystem`` -- hand-coded. Scores every action with tuned
    formulas and picks the highest. Classic game AI.
  * ``FSMDecisionSystem``     -- hand-coded. A finite-state machine: checks a
    fixed priority of if/else rules and returns the first that matches.
  * ``NeuralDecisionSystem``  -- learned. Runs each agent's evolved neural
    network (see ``neural.py``) and picks the best-scoring action.

The central design choice is that ALL THREE share the same inputs and the same
set of allowed actions. Every policy receives the same ``Perception`` and, for
the neural policy, the same validity mask (``_valid_actions``) that the rules
enforce. So the three differ ONLY in *which* of the currently-legal actions they
choose -- never in what they can see or what they are allowed to do. That is
what makes the utility-vs-FSM-vs-neural comparison a fair, controlled experiment:
any difference in outcome comes from the decision rule, not from unequal senses
or unequal options.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from src.simulation.agent import Agent
from src.simulation.neural import brain_for_seed, forward
from src.simulation.species import PREDATOR, PREY
from src.utils.constants import (
    ALL_ACTIONS,
    CHASE_PREY,
    EAT,
    FLEE,
    REPRODUCE,
    REST,
    SEEK_FOOD,
    WANDER,
)
from src.utils.random_utils import Vec2, clamp


@dataclass(frozen=True, slots=True)
class VisibleEntity:
    """A perceived entity summarized for utility scoring."""

    id: int
    position: Vec2
    distance: float
    energy: float = 0.0
    entity: object | None = None


@dataclass(frozen=True, slots=True)
class Perception:
    """Local sensory information available to one agent."""

    nearest_resource: VisibleEntity | None = None
    nearest_prey: VisibleEntity | None = None
    nearest_predator: VisibleEntity | None = None
    visible_resources: int = 0
    visible_prey: int = 0
    visible_predators: int = 0
    same_species_nearby: int = 0


@dataclass(frozen=True, slots=True, eq=False)
class Decision:
    """Chosen action, target, winning score, and full utility table."""

    action: str
    target: VisibleEntity | None
    utility: float
    utilities: dict[str, float] = field(default_factory=dict)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.action == other
        if isinstance(other, Decision):
            return (
                self.action == other.action
                and self.target == other.target
                and self.utility == other.utility
            )
        return False

    def __str__(self) -> str:
        return self.action


class FSMState(StrEnum):
    """Finite-state-machine state labels mapped to simulation actions."""

    WANDER = WANDER
    SEEK_FOOD = SEEK_FOOD
    FLEE = FLEE
    CHASE = CHASE_PREY
    EAT = EAT
    REPRODUCE = REPRODUCE
    REST = REST


def target_for_action(
    agent: Agent,
    action: str,
    perception: Perception,
) -> VisibleEntity | None:
    """Map an action name to the perceived entity it operates on.

    Shared by every decision architecture so that utility, FSM, and neural
    policies act on identical affordances; only the *choice* differs.
    """

    if action == FLEE:
        return perception.nearest_predator
    if agent.species == PREY and action in {SEEK_FOOD, EAT}:
        return perception.nearest_resource
    if agent.species == PREDATOR and action in {CHASE_PREY, SEEK_FOOD, EAT}:
        return perception.nearest_prey
    return None


class UtilityDecisionSystem:
    """Score candidate actions and return the highest-utility decision."""

    def __init__(self, eat_distance: float = 8.0, capture_distance: float = 10.0) -> None:
        self.eat_distance = eat_distance
        self.capture_distance = capture_distance

    def choose_action(self, agent: Agent, perception: Perception) -> Decision:
        """Return a structured decision object for the current context."""

        utilities = self.score_actions(agent, perception)
        priority = {
            EAT: 7,
            FLEE: 6,
            REPRODUCE: 5,
            CHASE_PREY: 4,
            SEEK_FOOD: 3,
            REST: 2,
            WANDER: 1,
        }
        action, utility = max(utilities.items(), key=lambda item: (item[1], priority[item[0]]))
        return Decision(
            action=action,
            target=self._target_for_action(agent, action, perception),
            utility=utility,
            utilities=utilities,
        )

    def choose_action_name(self, agent: Agent, perception: Perception) -> str:
        """Convenience wrapper for callers that only need the action name."""

        return self.choose_action(agent, perception).action

    def score_actions(self, agent: Agent, perception: Perception) -> dict[str, float]:
        """Compute a numeric "how good right now" score for every action.

        The numbers below are hand-tuned weights, not learned. Two quantities
        drive most of them: `hunger` (0 = full, 1 = starving) and
        `energy_surplus` (how far above the reproduction threshold the agent is).
        Actions that are impossible in the current context are seeded to -1.0 and
        only raised above zero by the species-specific helpers when they apply,
        so they can never accidentally win.
        """

        # Normalised drivers, both in a fixed range so the scoring stays stable
        # across agents with different reproduction thresholds.
        hunger = clamp(
            1.0 - (agent.energy / max(agent.genome.reproduction_threshold, 1.0)),
            0.0,
            1.0,
        )
        energy_surplus = clamp(
            (agent.energy - agent.genome.reproduction_threshold)
            / max(agent.genome.reproduction_threshold, 1.0),
            0.0,
            1.5,
        )

        # Baseline scores. WANDER/REST are always mildly positive fallbacks;
        # the context-dependent actions start negative and are "switched on"
        # below only when they make sense (a predator is near, food is in range).
        utilities = {
            WANDER: 0.12,
            SEEK_FOOD: 0.05,
            FLEE: -1.0,
            CHASE_PREY: -1.0,
            EAT: -1.0,
            REPRODUCE: -1.0,
            REST: 0.14 + hunger * 0.18,   # resting looks better when hungry/low
        }

        # Reproduction is only worth considering when energy/cooldown allow it;
        # the more surplus energy, the more attractive it becomes.
        if agent.can_reproduce():
            utilities[REPRODUCE] = 0.62 + energy_surplus * 0.30

        if agent.species == PREY:
            self._score_prey(agent, perception, utilities, hunger)
        elif agent.species == PREDATOR:
            self._score_predator(agent, perception, utilities, hunger)
        else:
            raise ValueError(f"Unknown species: {agent.species}")
        return utilities

    def _score_prey(
        self,
        agent: Agent,
        perception: Perception,
        utilities: dict[str, float],
        hunger: float,
    ) -> None:
        if perception.nearest_predator is not None:
            closeness = self._closeness(perception.nearest_predator.distance, agent.genome.vision_radius)
            utilities[FLEE] = 0.35 + closeness * (1.25 + agent.genome.fear_sensitivity)

        if perception.nearest_resource is not None:
            closeness = self._closeness(perception.nearest_resource.distance, agent.genome.vision_radius)
            utilities[SEEK_FOOD] = 0.25 + hunger * 0.78 + closeness * 0.22
            if perception.nearest_resource.distance <= self.eat_distance:
                utilities[EAT] = 0.95 + hunger * 0.35

    def _score_predator(
        self,
        agent: Agent,
        perception: Perception,
        utilities: dict[str, float],
        hunger: float,
    ) -> None:
        if perception.nearest_prey is None:
            utilities[REST] += hunger * 0.24
            return

        closeness = self._closeness(perception.nearest_prey.distance, agent.genome.vision_radius)
        utilities[CHASE_PREY] = (
            0.30
            + hunger * 0.54
            + closeness * 0.42
            + agent.genome.aggression * 0.28
        )
        utilities[SEEK_FOOD] = utilities[CHASE_PREY] * 0.75
        if perception.nearest_prey.distance <= self.capture_distance:
            utilities[EAT] = 1.35 + hunger * 0.30 + agent.genome.aggression * 0.14

    def _target_for_action(
        self,
        agent: Agent,
        action: str,
        perception: Perception,
    ) -> VisibleEntity | None:
        return target_for_action(agent, action, perception)

    @staticmethod
    def _closeness(distance: float, radius: float) -> float:
        if radius <= 0.0:
            return 0.0
        return clamp(1.0 - distance / radius, 0.0, 1.0)


class FSMDecisionSystem:
    """Rule-priority finite-state-machine baseline for comparison."""

    def __init__(self, eat_distance: float = 8.0, capture_distance: float = 10.0) -> None:
        self.eat_distance = eat_distance
        self.capture_distance = capture_distance

    def choose_action(self, agent: Agent, perception: Perception) -> Decision:
        """Return the first matching FSM state as a structured decision."""

        if agent.species == PREY:
            state = self._choose_prey_state(agent, perception)
        elif agent.species == PREDATOR:
            state = self._choose_predator_state(agent, perception)
        else:
            raise ValueError(f"Unknown species: {agent.species}")

        action = state.value
        return Decision(
            action=action,
            target=self._target_for_state(agent, state, perception),
            utility=1.0,
            utilities={state.value: 1.0},
        )

    def choose_action_name(self, agent: Agent, perception: Perception) -> str:
        """Convenience wrapper for callers that only need the action name."""

        return self.choose_action(agent, perception).action

    def _choose_prey_state(self, agent: Agent, perception: Perception) -> FSMState:
        # An FSM has no scores -- behaviour is decided purely by the ORDER of
        # these checks. The first rule that matches wins, so this ordering *is*
        # the prey's priority list: survive first, then eat, then breed, etc.
        if perception.nearest_predator is not None:
            return FSMState.FLEE                       # 1. danger overrides all
        if (
            perception.nearest_resource is not None
            and perception.nearest_resource.distance <= self.eat_distance
        ):
            return FSMState.EAT                        # 2. food in reach -> eat
        if agent.can_reproduce():
            return FSMState.REPRODUCE                  # 3. able -> reproduce
        if perception.nearest_resource is not None and self._is_hungry(agent):
            return FSMState.SEEK_FOOD                  # 4. hungry -> go to food
        if agent.energy < agent.genome.reproduction_threshold * 0.35:
            return FSMState.REST                       # 5. very low -> conserve
        return FSMState.WANDER                         # 6. nothing to do -> roam

    def _choose_predator_state(self, agent: Agent, perception: Perception) -> FSMState:
        if (
            perception.nearest_prey is not None
            and perception.nearest_prey.distance <= self.capture_distance
        ):
            return FSMState.EAT
        if agent.can_reproduce() and (
            perception.nearest_prey is None
            or agent.energy >= agent.genome.reproduction_threshold * 1.45
        ):
            return FSMState.REPRODUCE
        if perception.nearest_prey is not None and agent.energy <= agent.genome.reproduction_threshold * 1.25:
            return FSMState.CHASE
        if perception.nearest_prey is not None and not agent.can_reproduce():
            return FSMState.CHASE
        if agent.energy < agent.genome.reproduction_threshold * 0.45:
            return FSMState.REST
        return FSMState.WANDER

    def _target_for_state(
        self,
        agent: Agent,
        state: FSMState,
        perception: Perception,
    ) -> VisibleEntity | None:
        if state == FSMState.FLEE:
            return perception.nearest_predator
        if agent.species == PREY and state in {FSMState.SEEK_FOOD, FSMState.EAT}:
            return perception.nearest_resource
        if agent.species == PREDATOR and state in {FSMState.CHASE, FSMState.EAT}:
            return perception.nearest_prey
        return None

    @staticmethod
    def _is_hungry(agent: Agent) -> bool:
        return agent.energy < agent.genome.reproduction_threshold * 0.85


MASKED_SCORE = -9.99
"""Sentinel written into the utilities table for actions masked as invalid."""


class NeuralDecisionSystem:
    """Evolved neural-network policy (the learned architecture).

    Each agent carries a small MLP (see ``src.simulation.neural``) whose
    weights are inherited with mutation. This system converts the agent's
    ``Perception`` into 12 normalised inputs, runs the network, masks actions
    that are not currently available (exactly the affordances the utility and
    FSM policies respect), and picks the highest-scoring remaining action.
    """

    def __init__(self, eat_distance: float = 8.0, capture_distance: float = 10.0) -> None:
        self.eat_distance = eat_distance
        self.capture_distance = capture_distance

    def choose_action(self, agent: Agent, perception: Perception) -> Decision:
        """Return the network's preferred valid action as a Decision."""

        if agent.brain is None:
            # Agents are normally given a brain at spawn/birth by the world;
            # this deterministic fallback keeps directly-constructed agents
            # (tests, notebooks) working in neural mode.
            agent.brain = brain_for_seed(agent.id)

        scores = forward(agent.brain, self._inputs(agent, perception))
        valid = self._valid_actions(agent, perception)

        best_action = None
        best_score = float("-inf")
        utilities: dict[str, float] = {}
        for index, action in enumerate(ALL_ACTIONS):
            if action in valid:
                utilities[action] = scores[index]
                if scores[index] > best_score:
                    best_score = scores[index]
                    best_action = action
            else:
                utilities[action] = MASKED_SCORE

        assert best_action is not None  # WANDER/REST are always valid
        return Decision(
            action=best_action,
            target=target_for_action(agent, best_action, perception),
            utility=best_score,
            utilities=utilities,
        )

    def choose_action_name(self, agent: Agent, perception: Perception) -> str:
        """Convenience wrapper for callers that only need the action name."""

        return self.choose_action(agent, perception).action

    def _inputs(self, agent: Agent, perception: Perception) -> list[float]:
        """Build the 12 normalised network inputs from local perception."""

        threshold = max(agent.genome.reproduction_threshold, 1.0)
        hunger = clamp(1.0 - agent.energy / threshold, 0.0, 1.0)
        surplus = clamp((agent.energy - threshold) / threshold, 0.0, 1.5) / 1.5
        vision = agent.genome.vision_radius

        def closeness(entity: VisibleEntity | None) -> float:
            if entity is None or vision <= 0.0:
                return 0.0
            return clamp(1.0 - entity.distance / vision, 0.0, 1.0)

        return [
            1.0,                                                   # bias
            hunger,                                                # 1 = starving
            surplus,                                               # energy above threshold
            1.0 if agent.can_reproduce() else 0.0,                 # reproduction ready
            1.0 if agent.species == PREY else -1.0,                # species flag
            closeness(perception.nearest_resource),                # food proximity
            1.0 if perception.nearest_resource is not None else 0.0,
            closeness(perception.nearest_prey),                    # prey proximity
            1.0 if perception.nearest_prey is not None else 0.0,
            closeness(perception.nearest_predator),                # predator proximity
            1.0 if perception.nearest_predator is not None else 0.0,
            min(perception.same_species_nearby, 8) / 8.0,          # crowding
        ]

    def _valid_actions(self, agent: Agent, perception: Perception) -> set[str]:
        """Actions currently available -- identical affordances to the
        hand-coded policies, so architectures differ only in *choice*."""

        valid = {WANDER, REST}
        if agent.can_reproduce():
            valid.add(REPRODUCE)

        if agent.species == PREY:
            if perception.nearest_predator is not None:
                valid.add(FLEE)
            if perception.nearest_resource is not None:
                valid.add(SEEK_FOOD)
                if perception.nearest_resource.distance <= self.eat_distance:
                    valid.add(EAT)
        elif agent.species == PREDATOR:
            if perception.nearest_prey is not None:
                valid.add(CHASE_PREY)
                valid.add(SEEK_FOOD)
                if perception.nearest_prey.distance <= self.capture_distance:
                    valid.add(EAT)
        return valid


def create_decision_system(
    mode: str,
    *,
    eat_distance: float = 8.0,
    capture_distance: float = 10.0,
) -> UtilityDecisionSystem | FSMDecisionSystem | NeuralDecisionSystem:
    """Construct the requested decision architecture."""

    if mode == "utility":
        return UtilityDecisionSystem(eat_distance=eat_distance, capture_distance=capture_distance)
    if mode == "fsm":
        return FSMDecisionSystem(eat_distance=eat_distance, capture_distance=capture_distance)
    if mode == "neural":
        return NeuralDecisionSystem(eat_distance=eat_distance, capture_distance=capture_distance)
    raise ValueError(
        f"Unknown decision mode: {mode!r}. Expected 'utility', 'fsm', or 'neural'."
    )

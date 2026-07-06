"""Evolvable neural controller ("brain") for agents.

This module implements the learned policy of the project: a tiny multi-layer
perceptron whose weight vector is treated as part of an agent's genome. There
is no gradient descent anywhere -- the network is optimised by the ecosystem
itself. Agents that survive long enough to reproduce pass a mutated copy of
their weights to their offspring, so selection pressure performs the search
in weight space (neuroevolution; cf. Sims 1994; Stanley & Miikkulainen 2002;
Floreano et al. 2008).

Architecture (fixed so every brain in a run has the same shape):

    12 inputs -> 8 tanh hidden units -> 7 linear outputs (one per action)

The 12 inputs are normalised summaries of the agent's local Perception (see
``NeuralDecisionSystem`` in ``decision.py``). The 7 outputs are scores over
the shared action set in ``ALL_ACTIONS`` order; invalid actions are masked by
the decision system before the argmax, so the network chooses among exactly
the same affordances available to the hand-coded utility and FSM policies.

The module is deliberately dependency-free (no numpy, no torch): a forward
pass is ~170 multiply-adds, which pure Python handles easily at ecosystem
scale, and a flat ``tuple[float, ...]`` genome keeps inheritance, mutation,
and serialisation trivial and hashable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

from src.utils.constants import ALL_ACTIONS
from src.utils.random_utils import clamp

# Network shape. Fixed for the whole run so that every agent's brain has the
# same number of weights and the arithmetic below (and inheritance) lines up.
N_INPUTS = 12               # size of the perception vector fed in
N_HIDDEN = 8                # neurons in the single hidden layer
N_OUTPUTS = len(ALL_ACTIONS)  # 7 actions -> one output score per action

# All the network's numbers are stored as ONE flat list, in this exact order:
#   [ W1 (H*I) | b1 (H) | W2 (O*H) | b2 (O) ]
# where W1/W2 are stored row-major (unit 0's weights, then unit 1's, ...).
# For this project that works out to a small, fixed count:
#   W1 = 8*12 = 96,  b1 = 8,  W2 = 7*8 = 56,  b2 = 7   ->  167 numbers total.
# Knowing this layout is the key to reading `forward()` below.
N_PARAMS = N_HIDDEN * N_INPUTS + N_HIDDEN + N_OUTPUTS * N_HIDDEN + N_OUTPUTS

# Generation-zero weights are drawn from a Gaussian with this spread. Small
# values keep the first, un-evolved networks mild rather than erratic.
WEIGHT_INIT_SCALE = 0.60
# Every weight is hard-clamped to [-4, 4]. tanh(4) is already ~0.999 (saturated),
# so larger magnitudes buy nothing and only risk a few weights blowing up over
# many generations of mutation. The clamp keeps the search bounded and stable.
WEIGHT_BOUND = 4.0


@dataclass(frozen=True, slots=True)
class Brain:
    """Immutable flat weight vector for the agent MLP."""

    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.weights) != N_PARAMS:
            raise ValueError(
                f"Brain expects {N_PARAMS} weights, got {len(self.weights)}"
            )

    def as_tuple(self) -> tuple[float, ...]:
        """Return the raw weight vector (for tests, logging, analysis)."""

        return self.weights

    def norm(self) -> float:
        """L2 norm of the weight vector (a cheap 'how far evolved' signal)."""

        return math.sqrt(sum(w * w for w in self.weights))


def random_brain(rng: random.Random | None = None) -> Brain:
    """Create a randomly initialised brain (generation-zero controller)."""

    rng = rng or random.Random()
    return Brain(
        tuple(
            clamp(rng.gauss(0.0, WEIGHT_INIT_SCALE), -WEIGHT_BOUND, WEIGHT_BOUND)
            for _ in range(N_PARAMS)
        )
    )


def mutate_brain(
    brain: Brain,
    *,
    mutation_rate: float = 0.25,
    mutation_strength: float = 0.30,
    rng: random.Random | None = None,
) -> Brain:
    """Return a mutated copy: each weight is perturbed with probability
    ``mutation_rate`` by Gaussian noise of std-dev ``mutation_strength``.

    This mirrors ``genetics.mutate_genome`` so the neural genome evolves under
    exactly the same kind of variation operator as the physiological traits.
    """

    rng = rng or random.Random()
    mutated = []
    for weight in brain.weights:
        # Each weight independently has a `mutation_rate` chance of being nudged
        # by a small Gaussian step. With the defaults (~0.25) that means roughly
        # a quarter of the 167 weights change on each birth, each by a little.
        if rng.random() < mutation_rate:
            weight += rng.gauss(0.0, mutation_strength)
        mutated.append(clamp(weight, -WEIGHT_BOUND, WEIGHT_BOUND))
    return Brain(tuple(mutated))


def inherit_brain(
    parent_a: Brain,
    parent_b: Brain | None = None,
    *,
    mutation_rate: float = 0.25,
    mutation_strength: float = 0.30,
    rng: random.Random | None = None,
) -> Brain:
    """Inherit a brain from one or two parents, then mutate it.

    With two parents the child starts from the element-wise average of the
    parents' weights (same convention as ``genetics.inherit_genome``);
    reproduction in this simulation is asexual, so the single-parent path is
    the one normally exercised.
    """

    if parent_b is None:
        base = parent_a.weights                       # asexual: copy the parent
    else:
        # sexual (unused by default): child starts halfway between two parents
        base = tuple(
            (wa + wb) / 2.0 for wa, wb in zip(parent_a.weights, parent_b.weights)
        )
    # Then apply mutation. Copy + mutation is the whole variation mechanism:
    # good controllers survive, reproduce, and pass on slightly-varied weights,
    # so better networks spread through the population over generations. This is
    # the "learning" -- selection, not gradient descent.
    return mutate_brain(
        Brain(base),
        mutation_rate=mutation_rate,
        mutation_strength=mutation_strength,
        rng=rng,
    )


def brain_for_seed(seed: int) -> Brain:
    """Deterministic brain for a given integer seed (test/fallback helper)."""

    return random_brain(random.Random(seed))


def forward(brain: Brain, inputs: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    """Run the MLP forward pass and return one raw score per action.

    ``inputs`` must have length ``N_INPUTS``; hidden units use tanh, the
    output layer is linear (the decision system masks and argmaxes, so no
    softmax is needed).
    """

    if len(inputs) != N_INPUTS:
        raise ValueError(f"forward expects {N_INPUTS} inputs, got {len(inputs)}")

    w = brain.weights
    # `offset` is a moving cursor into the flat weight list. It walks through the
    # four blocks in order: 0 -> 96 (W1) -> 104 (b1) -> 160 (W2) -> 167 (b2).
    # This is a hand-written version of the usual matrix maths:
    #     hidden = tanh(W1 @ inputs + b1);   outputs = W2 @ hidden + b2
    # done with plain loops so the module needs no numpy/torch.
    offset = 0

    # --- Layer 1: input (12) -> hidden (8), weighted sums only for now --------
    hidden = []
    for h in range(N_HIDDEN):            # for each hidden neuron
        total = 0.0
        row = offset + h * N_INPUTS      # this neuron's 12 weights start here
        for i in range(N_INPUTS):
            total += w[row + i] * inputs[i]   # dot product of weights and inputs
        hidden.append(total)
    offset += N_HIDDEN * N_INPUTS        # step past W1 (offset is now 96)

    # Add each hidden neuron's bias, then squash with tanh. The tanh is the
    # nonlinearity: without it two linear layers would collapse into one and the
    # network could not represent "flee only if a predator is close AND I'm weak".
    for h in range(N_HIDDEN):
        hidden[h] = math.tanh(hidden[h] + w[offset + h])
    offset += N_HIDDEN                   # step past b1 (offset is now 104)

    # --- Layer 2: hidden (8) -> output (7), one score per action -------------
    outputs = []
    for o in range(N_OUTPUTS):           # for each action
        total = 0.0
        row = offset + o * N_HIDDEN      # this action's 8 weights start here
        for h in range(N_HIDDEN):
            total += w[row + h] * hidden[h]
        outputs.append(total)
    offset += N_OUTPUTS * N_HIDDEN       # step past W2 (offset is now 160)

    # Add each action's output bias and return. There is deliberately NO softmax
    # or activation here: the caller only takes the argmax over valid actions,
    # and argmax cares about order, not absolute values or probabilities.
    return tuple(outputs[o] + w[offset + o] for o in range(N_OUTPUTS))

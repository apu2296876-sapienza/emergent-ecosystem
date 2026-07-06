# Emergent Ecosystems: Evolved Neural NPCs vs. Hand-Coded Game AI

A 2D artificial-life simulation where autonomous prey and predator NPCs
live in a continuous world with renewable resources — and where each agent's
decision policy can be an **evolved neural network**, a hand-coded utility
selector, or a finite-state machine, compared like-for-like.

Course topic areas: **artificial life** and **gaming and AI**.

## The research question

> Can an NPC's decision policy be *learned by evolution* and still match
> hand-authored game AI?

Every agent carries a tiny multi-layer perceptron (MLP) whose **weights live in
its genome**. On reproduction the child inherits the parent's weights with
Gaussian mutation — there is **no gradient descent**. Selection pressure
(survive long enough to reproduce) optimises the network in weight space. This
is classic **neuroevolution**, and it is what makes the project a deep-learning /
ALife study rather than pure scripted game AI.

```
x ∈ R^12  (normalised local perception)
h = tanh(W1 x + b1)      # 8 hidden units
s = W2 h + b2            # 7 action scores → argmax over VALID actions
θ = vec(W1, b1, W2, b2)  # the neural genome, inherited + mutated
```

All three policies share identical perception, the same seven actions
(`wander, seek_food, flee, chase_prey, eat, reproduce, rest`), the same validity
masking, and the same physiology — so **only the policy differs**.

## Features

- Real-time Pygame visualization (blue prey triangles, red predator diamonds, green resources).
- Three decision architectures: **evolved neural network**, utility-based AI, and FSM baseline.
- Neuroevolution: per-agent MLP weights inherited with mutation; no backprop, no GPU, pure Python.
- Headless, deterministic simulation core for reproducible experiments.
- Local perception, energy/age/metabolism, predation, and asexual reproduction.
- Six physiological genetic traits inherited with bounded mutation, alongside the neural genome.
- CSV metrics logging and Matplotlib single-run + comparison plots.
- Pytest suite covering genetics, decisions, world dynamics, and the neural controller.

## Installation

The project is tested with Python 3.11 on Windows. Using Python 3.14 can cause Pygame installation issues during wheel build.

```bash
cd emergent-ecosystem
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Python 3.11, Pygame, Matplotlib, and Pytest are required. No external game engine, **no GPU**.

## Running the simulation

```bash
python main.py                   # defaults to the utility policy
python main.py --mode neural     # evolved neural controllers
python main.py --mode fsm        # finite-state-machine baseline
```

Press **`D`** in-game to cycle the decision mode live (utility → fsm → neural).

### Keyboard controls

| Key | Action |
| --- | --- |
| `D` | Cycle decision mode (utility / fsm / neural) live |
| `P` | Add 10 prey near the mouse position |
| `O` | Add 3 predators near the mouse position |
| `F` | Add 20 resources near the mouse position |
| Left click empty space | Add one food resource |
| Left click agent | Select that agent |
| `Shift` + left click | Add one prey at the cursor |
| `Ctrl` + left click | Add one predator at the cursor |
| `R` | Reset simulation |
| `SPACE` | Pause/resume |
| `M` | Toggle metrics/debug overlay, vision radii, target lines, graph, details |
| `S` | Save metrics to `output/metrics.csv` |
| `ESC` | Quit |

## Running experiments

Run all three scenarios across all three policies:

```bash
python -m src.analysis.experiment_runner
```

Options:

```bash
python -m src.analysis.experiment_runner --ticks 500                      # quick smoke test
python -m src.analysis.experiment_runner --decision-modes neural utility  # subset of policies
python -m src.analysis.experiment_runner --experiments baseline --runs 5  # 5 seeds (mean ± std)
```

Scenarios: `baseline` (abundant), `resource_scarcity` (harsh), `predator_pressure`
(many predators). Metrics are written to `output/experiments/{scenario}_{mode}.csv`
with a summary at `output/experiments/summary.csv`. A full 3-scenario × 3-policy
sweep of 3,000 ticks runs on a laptop CPU in a few minutes.

## Generating plots

```bash
python -m src.analysis.plot_metrics output/experiments/baseline_neural.csv   # single-run overview
python -m src.analysis.plot_metrics output/experiments/*.csv                 # comparison plots
```

Plots are written to `output/plots/`. When utility/fsm/neural CSVs for a scenario
are all present, extra `comparison_ai_*` plots contrast the three architectures
directly.

## Sample findings (single seed, 3,000 ticks)

| Scenario | utility prey | fsm prey | **neural prey** |
| --- | --- | --- | --- |
| baseline | 89 | 106 | **86** |
| resource_scarcity | 29 | 25 | **0** |
| predator_pressure | 123 | 100 | **97** |

Evolved controllers are competitive when resources are abundant, but fragile
under scarcity (evolution from random weights has little slack before the
population crashes). Predators collapse in every policy, isolating world-balance
effects from policy effects. See the report for the full write-up.

## Running tests

```bash
pytest
```

Covers genome bounds/mutation/inheritance, utility & FSM selection, world
dynamics and deterministic seeding, and the neural controller (forward-pass
shape, action masking, weight inheritance, neural-mode integration).

## Repository structure

```text
emergent-ecosystem/
  main.py                       # interactive pygame app (--mode, --seed, D to cycle)
  config.py                     # SimulationConfig + scenario presets
  requirements.txt
  src/
    simulation/
      neural.py                 # << evolvable MLP: weights-as-genome, mutation, forward pass
      decision.py               # utility / FSM / NeuralDecisionSystem + shared masking
      agent.py                  # agent state, genome + brain, reproduction
      genetics.py world.py metrics.py events.py resource.py species.py
    rendering/  pygame_renderer.py  ui.py
    analysis/   experiment_runner.py  plot_metrics.py
    utils/      constants.py  random_utils.py
  tests/
    test_neural.py              # neural controller + evolution tests
    test_decision.py  test_genetics.py  test_world.py
```

Generated `output/` CSVs and PNGs are gitignored and recreated by the scripts.


## Known limitations

- Single seed by default (use `--runs` to aggregate; stronger evidence needs mean ± std).
- Fixed neural topology; reproduction is asexual; no spatial memory or planning.
- No terrain, obstacles, seasons, or weather; resources are point entities.
- Predator energetics need rebalancing (predators collapse in all policies).
- Emergence is shown qualitatively; formal statistical testing is future work.

## Future work

- Average over seeds and report confidence intervals.
- Evolve network topology (NEAT-style) rather than fixed weights only.
- Add novelty / intrinsic-motivation pressure to help learning under scarcity.
- Terrain, shelter, sexual reproduction, lineage tracking, richer predator behaviour.

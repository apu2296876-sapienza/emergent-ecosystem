"""Run repeatable headless ecosystem experiments."""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from config import DEFAULT_CONFIG, SimulationConfig
from src.simulation.species import PREDATOR, PREY
from src.simulation.world import World

DEFAULT_TICKS = 3000
DEFAULT_OUTPUT_DIR = Path("output") / "experiments"
DECISION_MODES = ("utility", "fsm", "neural")


def experiment_configs() -> dict[str, SimulationConfig]:
    """Return the three required experimental conditions."""

    return {
        "baseline": replace(
            DEFAULT_CONFIG,
            seed=101,
            initial_prey=46,
            initial_predators=7,
            initial_resources=125,
            max_resources=240,
            resource_regen_probability=0.22,
        ),
        "resource_scarcity": replace(
            DEFAULT_CONFIG,
            seed=202,
            initial_prey=46,
            initial_predators=7,
            initial_resources=38,
            max_resources=90,
            resource_regen_probability=0.055,
        ),
        "predator_pressure": replace(
            DEFAULT_CONFIG,
            seed=303,
            initial_prey=46,
            initial_predators=18,
            initial_resources=125,
            max_resources=240,
            resource_regen_probability=0.22,
        ),
    }


def run_experiment(
    name: str,
    config: SimulationConfig,
    *,
    decision_mode: str = "utility",
    ticks: int = DEFAULT_TICKS,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    seed: int | None = None,
    run_label: str | None = None,
) -> dict[str, int | str]:
    """Run one headless simulation and save its metrics CSV."""

    config = replace(config, decision_mode=decision_mode)
    if seed is not None:
        config = replace(config, seed=seed)
    world = World(config)
    world.populate_initial()
    for _ in range(ticks):
        world.update(config.dt)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_label or f"{name}_{decision_mode}"
    metrics_path = output_dir / f"{run_name}.csv"
    world.metrics.save_csv(metrics_path)

    return {
        "experiment": name,
        "decision_mode": decision_mode,
        "seed": config.seed if config.seed is not None else -1,
        "ticks": ticks,
        "final_prey": world.population(PREY),
        "final_predators": world.population(PREDATOR),
        "final_resources": len(world.resources),
        "births_total": world.metrics.births,
        "deaths_total": world.metrics.deaths,
        "predation_total": world.metrics.predation_events,
        "extinction_events": world.metrics.extinction_events,
        "metrics_csv": metrics_path.as_posix(),
    }


def write_summary(rows: list[dict[str, int | str]], output_dir: Path) -> Path:
    """Save a compact experiment summary CSV."""

    output_path = output_dir / "summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run headless ecosystem experiments.")
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_TICKS,
        help=f"Number of ticks per experiment (default: {DEFAULT_TICKS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for experiment CSV files.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=tuple(experiment_configs().keys()),
        help="Optional subset of experiments to run.",
    )
    parser.add_argument(
        "--decision-modes",
        nargs="+",
        choices=DECISION_MODES,
        default=list(DECISION_MODES),
        help="Decision architecture(s) to run for each experiment.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Independent seeds per (experiment, mode) pair; run k uses "
        "seed + 1000*k and files gain a _runK suffix when runs > 1.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = experiment_configs()
    selected = args.experiments or list(configs.keys())
    rows = []
    for name in selected:
        for decision_mode in args.decision_modes:
            for run_index in range(max(1, args.runs)):
                base_seed = configs[name].seed or 0
                label = f"{name}_{decision_mode}"
                if args.runs > 1:
                    label = f"{label}_run{run_index}"
                rows.append(
                    run_experiment(
                        name,
                        configs[name],
                        decision_mode=decision_mode,
                        ticks=args.ticks,
                        output_dir=args.output_dir,
                        seed=base_seed + 1000 * run_index,
                        run_label=label,
                    )
                )
    summary = write_summary(rows, args.output_dir)

    print(f"Saved experiment summary to {summary}")
    for row in rows:
        print(
            f"{row['experiment']}[{row['decision_mode']}]: prey={row['final_prey']}, "
            f"predators={row['final_predators']}, "
            f"resources={row['final_resources']}, "
            f"births={row['births_total']}, "
            f"deaths={row['deaths_total']}, "
            f"predation={row['predation_total']}"
        )


if __name__ == "__main__":
    main()

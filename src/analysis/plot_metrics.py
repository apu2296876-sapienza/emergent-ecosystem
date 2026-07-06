"""Plot metrics CSV files produced by headless ecosystem experiments."""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("output") / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_OUTPUT_DIR = Path("output") / "plots"
DECISION_MODE_SUFFIXES = ("utility", "fsm", "neural")
REQUIRED_METRIC_COLUMNS = {
    "tick",
    "prey_population",
    "predator_population",
    "resource_count",
    "avg_prey_speed",
    "avg_predator_speed",
    "avg_prey_vision",
    "avg_predator_vision",
    "births_total",
    "deaths_total",
    "predation_total",
    "prey_extinct",
    "predators_extinct",
}
ALIASES = {
    "resource_count": ("resource_count", "resources_count"),
    "avg_prey_speed": ("avg_prey_speed", "average_prey_speed"),
    "avg_predator_speed": ("avg_predator_speed", "average_predator_speed"),
    "avg_prey_vision": ("avg_prey_vision", "average_prey_vision_radius"),
    "avg_predator_vision": ("avg_predator_vision", "average_predator_vision_radius"),
    "births_total": ("births_total", "births"),
    "deaths_total": ("deaths_total", "deaths"),
    "predation_total": ("predation_total", "predation_events"),
    "prey_extinct": ("prey_extinct",),
    "predators_extinct": ("predators_extinct",),
}


def load_metrics(path: Path) -> list[dict[str, float]]:
    """Read a metrics CSV into numeric rows."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand literal files and shell-style globs in a deterministic order."""

    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(Path(match) for match in matches)
        else:
            paths.append(Path(pattern))

    existing = [
        path
        for path in paths
        if path.exists() and path.is_file() and _looks_like_metrics_csv(path)
    ]
    if not existing:
        raise FileNotFoundError(f"No metrics CSV files matched: {', '.join(patterns)}")
    return existing


def plot_single_experiment(csv_path: Path, output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Generate the required overview plots for one experiment CSV."""

    rows = load_metrics(csv_path)
    if not rows:
        raise ValueError(f"No metrics rows found in {csv_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    ticks = [row["tick"] for row in rows]
    experiment = csv_path.stem

    fig, axes = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    ax_pop, ax_resources, ax_speed, ax_vision, ax_events, ax_blank = axes.flatten()

    ax_pop.plot(ticks, _series(rows, "prey_population"), label="Prey", color="#4ca6ff")
    ax_pop.plot(ticks, _series(rows, "predator_population"), label="Predators", color="#eb5c52")
    ax_pop.set_title("Populations over time")
    ax_pop.set_xlabel("Tick")
    ax_pop.set_ylabel("Agents")
    ax_pop.legend()

    ax_resources.plot(ticks, _series(rows, "resource_count"), label="Resources", color="#5ecc68")
    ax_resources.set_title("Resources over time")
    ax_resources.set_xlabel("Tick")
    ax_resources.set_ylabel("Resources")

    ax_speed.plot(ticks, _series(rows, "avg_prey_speed"), label="Prey speed", color="#4ca6ff")
    ax_speed.plot(ticks, _series(rows, "avg_predator_speed"), label="Predator speed", color="#eb5c52")
    ax_speed.set_title("Average speed over time")
    ax_speed.set_xlabel("Tick")
    ax_speed.legend()

    ax_vision.plot(ticks, _series(rows, "avg_prey_vision"), label="Prey vision", color="#4ca6ff")
    ax_vision.plot(ticks, _series(rows, "avg_predator_vision"), label="Predator vision", color="#eb5c52")
    ax_vision.set_title("Average vision radius over time")
    ax_vision.set_xlabel("Tick")
    ax_vision.legend()

    ax_events.plot(ticks, _series(rows, "births_total"), label="Births", color="#b9d06b")
    ax_events.plot(ticks, _series(rows, "deaths_total"), label="Deaths", color="#aaaaaa")
    ax_events.plot(ticks, _series(rows, "predation_total"), label="Predation", color="#ffae7c")
    ax_events.set_title("Cumulative births, deaths, and predation")
    ax_events.set_xlabel("Tick")
    ax_events.legend()

    ax_blank.axis("off")
    ax_blank.text(
        0.0,
        0.92,
        f"Experiment: {experiment}\nFinal prey: {rows[-1]['prey_population']:.0f}\n"
        f"Final predators: {rows[-1]['predator_population']:.0f}\n"
        f"Prey extinct: {_value(rows[-1], 'prey_extinct'):.0f}\n"
        f"Predators extinct: {_value(rows[-1], 'predators_extinct'):.0f}",
        va="top",
        fontsize=11,
    )

    output = output_dir / f"{experiment}_overview.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return [output]


def plot_comparison(csv_paths: list[Path], output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Generate comparison plots across multiple experiment CSV files."""

    output_dir.mkdir(parents=True, exist_ok=True)
    datasets = {path.stem: load_metrics(path) for path in csv_paths}
    datasets = {name: rows for name, rows in datasets.items() if rows}
    if not datasets:
        raise ValueError("No metrics rows found in comparison CSV files.")

    outputs = [
        _comparison_plot(datasets, "prey_population", "Prey population comparison", output_dir / "comparison_prey_population.png"),
        _comparison_plot(datasets, "predator_population", "Predator population comparison", output_dir / "comparison_predator_population.png"),
        _comparison_plot(datasets, "predation_total", "Predation events comparison", output_dir / "comparison_predation_events.png"),
        _comparison_plot(datasets, "prey_extinct", "Prey extinction comparison", output_dir / "comparison_prey_extinction.png"),
        _comparison_plot(datasets, "predators_extinct", "Predator extinction comparison", output_dir / "comparison_predator_extinction.png"),
    ]
    outputs.extend(_ai_architecture_plots(datasets, output_dir))
    return outputs


def plot_metrics(csv_paths: list[str], output_dir: Path = DEFAULT_OUTPUT_DIR) -> list[Path]:
    """Plot one or more CSV files and return saved image paths."""

    paths = expand_paths(csv_paths)
    if len(paths) == 1:
        return plot_single_experiment(paths[0], output_dir)
    return plot_comparison(paths, output_dir)


def _series(rows: list[dict[str, float]], key: str) -> list[float]:
    return [_value(row, key) for row in rows]


def _value(row: dict[str, float], key: str) -> float:
    for candidate in ALIASES.get(key, (key,)):
        if candidate in row:
            return row[candidate]
    if key in {"prey_extinct", "predators_extinct"}:
        return 0.0
    raise KeyError(key)


def _looks_like_metrics_csv(path: Path) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = set(next(reader, []))
    except OSError:
        return False
    return REQUIRED_METRIC_COLUMNS.issubset(header)


def _comparison_plot(
    datasets: dict[str, list[dict[str, float]]],
    key: str,
    title: str,
    output_path: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name, rows in datasets.items():
        ax.plot(_series(rows, "tick"), _series(rows, key), label=name)
    ax.set_title(title)
    ax.set_xlabel("Tick")
    ax.legend()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def _ai_architecture_plots(
    datasets: dict[str, list[dict[str, float]]],
    output_dir: Path,
) -> list[Path]:
    ai_datasets = {
        name: rows
        for name, rows in datasets.items()
        if _split_mode_suffix(name)[1] in DECISION_MODE_SUFFIXES
    }
    if len(ai_datasets) < 2:
        return []
    return [
        _comparison_plot(
            ai_datasets,
            "prey_population",
            "AI architecture comparison: prey population",
            output_dir / "comparison_ai_prey_population.png",
        ),
        _comparison_plot(
            ai_datasets,
            "predator_population",
            "AI architecture comparison: predator population",
            output_dir / "comparison_ai_predator_population.png",
        ),
        _comparison_plot(
            ai_datasets,
            "predation_total",
            "AI architecture comparison: predation events",
            output_dir / "comparison_ai_predation_events.png",
        ),
        _comparison_plot(
            ai_datasets,
            "prey_extinct",
            "AI architecture comparison: prey extinction",
            output_dir / "comparison_ai_prey_extinction.png",
        ),
        _comparison_plot(
            ai_datasets,
            "predators_extinct",
            "AI architecture comparison: predator extinction",
            output_dir / "comparison_ai_predator_extinction.png",
        ),
    ]


def _split_mode_suffix(name: str) -> tuple[str, str | None]:
    base = re.sub(r"_run\d+$", "", name)
    for suffix in DECISION_MODE_SUFFIXES:
        marker = f"_{suffix}"
        if base.endswith(marker):
            return base[: -len(marker)], suffix
    return base, None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ecosystem metrics CSV files.")
    parser.add_argument("csv_paths", nargs="+", help="CSV path(s) or glob(s), e.g. output/experiments/*.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for PNG plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = plot_metrics(args.csv_paths, args.output_dir)
    for output in outputs:
        print(f"Saved plot to {output}")


if __name__ == "__main__":
    main()

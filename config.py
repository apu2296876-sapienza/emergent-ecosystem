"""Runtime configuration for the emergent ecosystem simulation."""

from __future__ import annotations

from dataclasses import dataclass, replace


Color = tuple[int, int, int]
ColorA = tuple[int, int, int, int]


@dataclass(slots=True)
class SimulationConfig:
    """Tunable parameters shared by real-time play and experiments."""

    screen_width: int = 1100
    screen_height: int = 760
    world_width: int = 1100
    world_height: int = 760
    fps: int = 60
    seed: int | None = 7
    decision_mode: str = "utility"

    width: int | None = None
    height: int | None = None

    initial_prey: int = 46
    initial_predators: int = 7
    initial_resources: int = 125
    max_resources: int = 240

    dt: float = 1.0
    updates_per_frame: int = 1
    metrics_interval: int = 1

    resource_energy: float = 38.0
    resource_radius: float = 4.5
    resource_regen_probability: float = 0.22
    resource_patch_size: int = 12
    resource_cluster_spread: float = 30.0

    starting_energy_prey: float = 88.0
    starting_energy_predator: float = 124.0
    max_agent_energy: float = 190.0
    max_age_prey: float = 2100.0
    max_age_predator: float = 2400.0

    eat_distance: float = 8.0
    capture_distance: float = 10.0
    movement_cost_multiplier: float = 0.035
    rest_metabolism_multiplier: float = 0.38
    bounds_margin: float = 8.0

    reproduction_cooldown_ticks: int = 78
    reproduction_cost_fraction: float = 0.44
    mutation_rate: float = 0.16
    mutation_strength: float = 0.085
    brain_mutation_rate: float = 0.25
    brain_mutation_strength: float = 0.30

    wander_turn_jitter: float = 0.42

    add_prey_count: int = 10
    add_predator_count: int = 3
    add_resource_count: int = 20
    intervention_spread: float = 34.0

    show_metrics_overlay: bool = True
    show_vision_radius: bool = True
    show_population_graph: bool = True
    vision_draw_limit: int = 140
    target_line_draw_limit: int = 80
    selection_radius: float = 12.0
    metrics_save_path: str = "output/metrics.csv"

    background_color: Color = (19, 23, 23)
    grid_color: Color = (30, 37, 36)
    resource_color: Color = (88, 210, 106)
    resource_outline_color: Color = (34, 108, 52)
    prey_color: Color = (70, 165, 255)
    prey_outline_color: Color = (22, 84, 142)
    predator_color: Color = (235, 83, 76)
    predator_outline_color: Color = (130, 34, 36)
    flee_indicator_color: Color = (246, 224, 116)
    chase_indicator_color: Color = (255, 174, 112)
    reproduce_indicator_color: Color = (198, 240, 130)
    vision_prey_color: ColorA = (70, 165, 255, 28)
    vision_predator_color: ColorA = (235, 83, 76, 30)
    panel_color: ColorA = (12, 16, 17, 214)
    panel_border_color: Color = (58, 70, 70)
    text_color: Color = (228, 235, 231)
    muted_text_color: Color = (162, 176, 173)
    status_color: Color = (174, 226, 156)
    paused_color: Color = (245, 232, 146)
    selection_color: Color = (255, 245, 160)
    target_line_color: Color = (245, 225, 120)

    def __post_init__(self) -> None:
        """Keep legacy `width`/`height` aliases synced with world size."""

        if self.width is not None:
            self.world_width = self.width
        if self.height is not None:
            self.world_height = self.height
        self.width = self.world_width
        self.height = self.world_height


DEFAULT_CONFIG = SimulationConfig()

EXPERIMENT_PRESETS: dict[str, SimulationConfig] = {
    "baseline": replace(DEFAULT_CONFIG, seed=11),
    "resource_scarcity": replace(
        DEFAULT_CONFIG,
        seed=22,
        initial_resources=35,
        max_resources=75,
        resource_regen_probability=0.06,
    ),
    "predator_pressure": replace(
        DEFAULT_CONFIG,
        seed=33,
        initial_prey=52,
        initial_predators=17,
        initial_resources=135,
        resource_regen_probability=0.18,
    ),
}

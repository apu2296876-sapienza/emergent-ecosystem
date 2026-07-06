"""Real-time Pygame entry point for the emergent ecosystem prototype."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pygame

from config import DEFAULT_CONFIG, SimulationConfig
from src.rendering.pygame_renderer import PygameRenderer
from src.rendering.ui import format_save_message
from src.simulation.decision import create_decision_system
from src.simulation.world import World

DECISION_MODES = ("utility", "fsm", "neural")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the interactive ecosystem.")
    parser.add_argument(
        "--mode",
        choices=DECISION_MODES,
        default=DEFAULT_CONFIG.decision_mode,
        help="Decision architecture for all agents (default: %(default)s). "
        "Press D in-game to cycle modes live.",
    )
    parser.add_argument("--seed", type=int, default=None, help="World seed override.")
    return parser.parse_args()


def create_world(config: SimulationConfig) -> World:
    """Create and populate a deterministic world from config values."""

    world = World(config)
    world.populate_initial()
    return world


def main() -> None:
    """Run the simulation, event loop, rendering, and metric saving."""

    args = parse_args()
    config = replace(DEFAULT_CONFIG, decision_mode=args.mode)
    if args.seed is not None:
        config = replace(config, seed=args.seed)
    pygame.init()
    pygame.display.set_caption(f"Emergent Ecosystems [{config.decision_mode}]")
    screen = pygame.display.set_mode((config.screen_width, config.screen_height))
    clock = pygame.time.Clock()

    world = create_world(config)
    renderer = PygameRenderer(world, screen=screen)

    paused = False
    running = True
    try:
        while running:
            fps = clock.get_fps()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        renderer.set_status("Paused" if paused else "Running")
                    elif event.key == pygame.K_r:
                        world = create_world(config)
                        renderer.world = world
                        renderer.selected_agent_id = None
                        renderer.set_status("Simulation reset")
                        paused = False
                    elif event.key == pygame.K_d:
                        current = DECISION_MODES.index(world.config.decision_mode)
                        mode = DECISION_MODES[(current + 1) % len(DECISION_MODES)]
                        world.config.decision_mode = mode
                        world.decision_system = create_decision_system(
                            mode,
                            eat_distance=world.config.eat_distance,
                            capture_distance=world.config.capture_distance,
                        )
                        pygame.display.set_caption(f"Emergent Ecosystems [{mode}]")
                        renderer.set_status(f"Decision mode: {mode}")
                    elif event.key == pygame.K_m:
                        renderer.show_metrics = not renderer.show_metrics
                    elif event.key == pygame.K_s:
                        path = world.metrics.save_csv(Path(config.metrics_save_path))
                        renderer.set_status(format_save_message(path))
                    else:
                        renderer.handle_keydown(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    renderer.handle_mouse_button(event)

            if not paused:
                for _ in range(config.updates_per_frame):
                    world.update(config.dt)

            renderer.render(paused=paused, fps=fps)
            pygame.display.flip()
            clock.tick(config.fps)
    finally:
        pygame.quit()


if __name__ == "__main__":
    main()

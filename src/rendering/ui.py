"""Pygame UI overlay for live simulation metrics and controls."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from src.simulation.species import PREDATOR, PREY

if TYPE_CHECKING:
    from src.simulation.agent import Agent

CONTROL_ROWS = [
    "D change mood",
    "P add 10 prey",
    "O add 3 predators",
    "F add 20 resources",
    "LMB select/add food",
    "Shift+LMB add prey",
    "Ctrl+LMB add predator",
    "SPACE pause/resume",
    "M metrics/debug",
    "S save metrics",
    "R reset",
    "ESC quit",
]


def draw_overlay(
    surface: pygame.Surface,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    world: object,
    *,
    fps: float,
    paused: bool,
    show_metrics: bool,
    status_message: str,
    selected_agent: "Agent | None" = None,
) -> None:
    """Draw live counts, events, FPS, help, and status text."""

    if not show_metrics:
        label = "Paused" if paused else "Running"
        color = world.config.paused_color if paused else world.config.status_color
        _draw_text(surface, font, label, (16, 14), color)
        if status_message:
            _draw_text(surface, small_font, status_message, (16, 42), world.config.status_color)
        return

    panel_width = 306
    panel_height = 438
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill(world.config.panel_color)
    pygame.draw.rect(panel, world.config.panel_border_color, panel.get_rect(), 1)
    surface.blit(panel, (12, 12))

    state = "Paused" if paused else "Running"
    state_color = world.config.paused_color if paused else world.config.status_color
    _draw_text(surface, font, state, (24, 22), state_color)
    _draw_text(surface, small_font, f"FPS {fps:05.1f}", (204, 28), world.config.muted_text_color)

    rows = [
        f"Tick: {world.tick_count}",
        f"Prey: {world.population(PREY)}",
        f"Predators: {world.population(PREDATOR)}",
        f"Resources: {len(world.resources)}",
        f"Births: {world.metrics.births}",
        f"Deaths: {world.metrics.deaths}",
        f"Predation: {world.metrics.predation_events}",
        f"Extinctions: {world.metrics.extinction_events}",
    ]

    y = 58
    for row in rows:
        _draw_text(surface, small_font, row, (24, y), world.config.text_color)
        y += 19

    y += 6
    _draw_text(surface, small_font, "Controls", (24, y), world.config.muted_text_color)
    y += 20
    for row in CONTROL_ROWS:
        _draw_text(surface, small_font, row, (24, y), world.config.text_color)
        y += 18

    if status_message:
        _draw_text(surface, small_font, status_message, (24, panel_height - 22), world.config.status_color)

    if selected_agent is not None:
        _draw_selected_agent_panel(surface, small_font, world, selected_agent)


def format_save_message(path: Path) -> str:
    """Create a compact status string after a metrics save."""

    return f"Saved {path.as_posix()}"


def _draw_selected_agent_panel(
    surface: pygame.Surface,
    small_font: pygame.font.Font,
    world: object,
    agent: "Agent",
) -> None:
    panel_width = 306
    panel_height = 250
    x = 12
    y = surface.get_height() - panel_height - 12
    panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
    panel.fill(world.config.panel_color)
    pygame.draw.rect(panel, world.config.selection_color, panel.get_rect(), 1)
    surface.blit(panel, (x, y))

    rows = [
        "Selected agent",
        f"Species: {agent.species}",
        f"Energy: {agent.energy:.1f}",
        f"Age: {agent.age:.0f}",
        f"Generation: {agent.generation}",
        f"Action: {agent.current_action}",
        f"Speed: {agent.genome.speed:.2f}",
        f"Vision: {agent.genome.vision_radius:.1f}",
        f"Metabolism: {agent.genome.metabolism:.3f}",
        f"Repro threshold: {agent.genome.reproduction_threshold:.1f}",
        f"Fear: {agent.genome.fear_sensitivity:.2f}",
        f"Aggression: {agent.genome.aggression:.2f}",
    ]
    text_y = y + 10
    for index, row in enumerate(rows):
        color = world.config.selection_color if index == 0 else world.config.text_color
        _draw_text(surface, small_font, row, (x + 12, text_y), color)
        text_y += 19


def _draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    rendered = font.render(text, True, color)
    surface.blit(rendered, position)

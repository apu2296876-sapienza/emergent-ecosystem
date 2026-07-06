"""Pygame rendering and interaction helpers for the ecosystem."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pygame

if TYPE_CHECKING:
    from config import SimulationConfig

from src.rendering.ui import draw_overlay, format_save_message
from src.simulation.agent import Agent
from src.simulation.decision import VisibleEntity
from src.simulation.resource import Resource
from src.simulation.species import PREDATOR, PREY
from src.simulation.world import World
from src.utils.constants import CHASE_PREY, EAT, FLEE, REPRODUCE, SEEK_FOOD
from src.utils.random_utils import Vec2, add, jittered_position, multiply, normalize, random_position


class PygameRenderer:
    """Draw the world and apply player/environment interventions."""

    def __init__(self, world: World, *, screen: pygame.Surface | None = None) -> None:
        self.world = world
        self.screen = screen or pygame.display.set_mode(
            (self.config.screen_width, self.config.screen_height)
        )
        self.font = pygame.font.SysFont("consolas", 22)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.show_metrics = self.config.show_metrics_overlay
        self.status_message = ""
        self.status_timeout = 0
        self.selected_agent_id: int | None = None

    @property
    def config(self) -> "SimulationConfig":
        """Return the config for the current world."""

        return self.world.config

    def run(self) -> None:
        """Compatibility loop; `main.py` owns the primary loop."""

        clock = pygame.time.Clock()
        paused = False
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_s:
                        path = self.world.metrics.save_csv(Path(self.config.metrics_save_path))
                        self.set_status(format_save_message(path))
                    elif event.key == pygame.K_r:
                        self.world.reset()
                        self.selected_agent_id = None
                        self.set_status("Simulation reset")
                    elif event.key == pygame.K_m:
                        self.show_metrics = not self.show_metrics
                    else:
                        self.handle_keydown(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_button(event)

            if not paused:
                for _ in range(self.config.updates_per_frame):
                    self.world.update(self.config.dt)
            self.render(paused=paused, fps=clock.get_fps())
            pygame.display.flip()
            clock.tick(self.config.fps)
        pygame.quit()

    def handle_keydown(self, key: int) -> None:
        """Handle intervention keys that modify the world."""

        if key == pygame.K_p:
            self.spawn_prey_burst()
        elif key == pygame.K_o:
            self.spawn_predator_burst()
        elif key == pygame.K_f:
            self.spawn_resource_burst()

    def handle_mouse_button(self, event: pygame.event.Event) -> None:
        """Handle world clicks for selection and quick interventions."""

        if event.button != 1:
            return

        world_position = self._from_screen(event.pos)
        modifiers = pygame.key.get_mods()
        if modifiers & pygame.KMOD_SHIFT:
            self.world.add_prey(position=world_position)
            self.set_status("Added prey")
            return
        if modifiers & pygame.KMOD_CTRL:
            self.world.add_predator(position=world_position)
            self.set_status("Added predator")
            return

        clicked_agent = self._agent_at_screen_position(event.pos)
        if clicked_agent is not None:
            self.selected_agent_id = clicked_agent.id
            self.set_status(f"Selected {clicked_agent.species} #{clicked_agent.id}")
            return

        if len(self.world.resources) < self.config.max_resources:
            self.world.add_resource(position=world_position)
            self.set_status("Added food")

    def spawn_prey_burst(self) -> None:
        center = self._intervention_position()
        for _ in range(self.config.add_prey_count):
            self.world.add_prey(position=self._near(center))
        self.set_status(f"Added {self.config.add_prey_count} prey")

    def spawn_predator_burst(self) -> None:
        center = self._intervention_position()
        for _ in range(self.config.add_predator_count):
            self.world.add_predator(position=self._near(center))
        self.set_status(f"Added {self.config.add_predator_count} predators")

    def spawn_resource_burst(self) -> None:
        center = self._intervention_position()
        spawned = 0
        for _ in range(self.config.add_resource_count):
            if len(self.world.resources) >= self.config.max_resources:
                break
            self.world.add_resource(position=self._near(center))
            spawned += 1
        self.set_status(f"Added {spawned} resources")

    def set_status(self, message: str) -> None:
        self.status_message = message
        self.status_timeout = int(self.config.fps * 2.2)

    def render(self, *, paused: bool, fps: float) -> None:
        """Render the current world state to the Pygame surface."""

        self.screen.fill(self.config.background_color)
        self._draw_grid()
        self._draw_resources()
        if self.show_metrics and self.config.show_vision_radius:
            self._draw_vision_radii()
            self._draw_target_lines()
        self._draw_agents()
        self._draw_selected_agent()
        if self.show_metrics and self.config.show_population_graph:
            self._draw_population_graph()
        draw_overlay(
            self.screen,
            self.font,
            self.small_font,
            self.world,
            fps=fps,
            paused=paused,
            show_metrics=self.show_metrics,
            status_message=self.status_message,
            selected_agent=self.selected_agent,
        )
        self._tick_status()

    def _intervention_position(self) -> Vec2:
        mouse = pygame.mouse.get_pos()
        if self.screen.get_rect().collidepoint(mouse):
            return (float(mouse[0]), float(mouse[1]))
        return random_position(
            self.config.world_width,
            self.config.world_height,
            self.world.rng,
            self.config.bounds_margin,
        )

    def _near(self, center: Vec2) -> Vec2:
        return jittered_position(
            center,
            self.config.intervention_spread,
            self.config.world_width,
            self.config.world_height,
            self.world.rng,
            self.config.bounds_margin,
        )

    def _tick_status(self) -> None:
        self.status_timeout = max(0, self.status_timeout - 1)
        if self.status_timeout == 0:
            self.status_message = ""

    def _draw_grid(self) -> None:
        spacing = 50
        for x in range(0, self.config.screen_width, spacing):
            pygame.draw.line(self.screen, self.config.grid_color, (x, 0), (x, self.config.screen_height))
        for y in range(0, self.config.screen_height, spacing):
            pygame.draw.line(self.screen, self.config.grid_color, (0, y), (self.config.screen_width, y))

    def _draw_resources(self) -> None:
        for resource in self.world.resources:
            self._draw_resource(resource)

    def _draw_resource(self, resource: Resource) -> None:
        position = self._to_screen(resource.position)
        radius = max(2, int(resource.radius))
        pygame.draw.circle(self.screen, self.config.resource_outline_color, position, radius + 1)
        pygame.draw.circle(self.screen, self.config.resource_color, position, radius)

    def _draw_agents(self) -> None:
        for agent in self.world.agents:
            if agent.species == PREY:
                self._draw_prey(agent)
            elif agent.species == PREDATOR:
                self._draw_predator(agent)

    def _draw_prey(self, agent: Agent) -> None:
        points = self._triangle_points(agent, size=max(7.0, 5.5 + agent.genome.speed * 1.3))
        outline = self._triangle_points(agent, size=max(8.5, 7.0 + agent.genome.speed * 1.3))
        pygame.draw.polygon(self.screen, self.config.prey_outline_color, outline)
        pygame.draw.polygon(self.screen, self.config.prey_color, points)
        self._draw_action_indicator(agent, self._to_screen(agent.position), 11)

    def _draw_predator(self, agent: Agent) -> None:
        forward = normalize(agent.heading)
        if forward == (0.0, 0.0):
            forward = (1.0, 0.0)
        side = (-forward[1], forward[0])
        size = max(8.0, 6.0 + agent.genome.aggression * 3.4)
        outline = self._diamond_points(agent.position, forward, side, size + 1.8)
        points = self._diamond_points(agent.position, forward, side, size)
        pygame.draw.polygon(self.screen, self.config.predator_outline_color, outline)
        pygame.draw.polygon(self.screen, self.config.predator_color, points)
        self._draw_action_indicator(agent, self._to_screen(agent.position), int(size + 4))

    def _triangle_points(self, agent: Agent, size: float) -> list[tuple[int, int]]:
        forward = normalize(agent.heading)
        if forward == (0.0, 0.0):
            forward = (1.0, 0.0)
        side = (-forward[1], forward[0])
        nose = add(agent.position, multiply(forward, size))
        left = add(agent.position, add(multiply(forward, -size * 0.62), multiply(side, size * 0.62)))
        right = add(agent.position, add(multiply(forward, -size * 0.62), multiply(side, -size * 0.62)))
        return [self._to_screen(nose), self._to_screen(left), self._to_screen(right)]

    def _diamond_points(
        self,
        position: Vec2,
        forward: Vec2,
        side: Vec2,
        size: float,
    ) -> list[tuple[int, int]]:
        return [
            self._to_screen(add(position, multiply(forward, size))),
            self._to_screen(add(position, multiply(side, size * 0.78))),
            self._to_screen(add(position, multiply(forward, -size))),
            self._to_screen(add(position, multiply(side, -size * 0.78))),
        ]

    def _draw_action_indicator(self, agent: Agent, position: tuple[int, int], radius: int) -> None:
        if agent.current_action == FLEE:
            pygame.draw.circle(self.screen, self.config.flee_indicator_color, position, radius, 1)
        elif agent.current_action in {CHASE_PREY, EAT}:
            pygame.draw.circle(self.screen, self.config.chase_indicator_color, position, radius, 1)
        elif agent.current_action == REPRODUCE:
            pygame.draw.circle(self.screen, self.config.reproduce_indicator_color, position, radius, 1)
        elif agent.current_action == SEEK_FOOD:
            pygame.draw.circle(self.screen, self.config.resource_color, position, max(3, radius - 2), 1)

    def _draw_vision_radii(self) -> None:
        for agent in list(self.world.agents)[: self.config.vision_draw_limit]:
            color = self.config.vision_prey_color if agent.species == PREY else self.config.vision_predator_color
            self._draw_alpha_circle(color, agent.position, agent.genome.vision_radius)

    def _draw_alpha_circle(self, color: tuple[int, int, int, int], position: Vec2, radius: float) -> None:
        if radius <= 0:
            return
        pygame.draw.circle(self.screen, color[:3], self._to_screen(position), int(radius), 1)

    def _draw_target_lines(self) -> None:
        agents = list(self.world.agents)[: self.config.target_line_draw_limit]
        selected = self.selected_agent
        if selected is not None and selected not in agents:
            agents.append(selected)
        for agent in agents:
            target = self._current_target(agent)
            if target is None:
                continue
            start = self._to_screen(agent.position)
            end = self._to_screen(target.position)
            pygame.draw.line(self.screen, self.config.target_line_color, start, end, 1)

    def _draw_selected_agent(self) -> None:
        selected = self.selected_agent
        if selected is None:
            self.selected_agent_id = None
            return
        position = self._to_screen(selected.position)
        pygame.draw.circle(self.screen, self.config.selection_color, position, 17, 2)
        self._draw_action_label(selected, position)

    def _draw_action_label(self, agent: Agent, position: tuple[int, int]) -> None:
        label = self.small_font.render(agent.current_action, True, self.config.selection_color)
        x = position[0] - label.get_width() // 2
        y = position[1] - 30
        background = pygame.Rect(x - 3, y - 2, label.get_width() + 6, label.get_height() + 4)
        pygame.draw.rect(self.screen, self.config.panel_color[:3], background)
        pygame.draw.rect(self.screen, self.config.selection_color, background, 1)
        self.screen.blit(label, (x, y))

    def _draw_population_graph(self) -> None:
        records = self.world.metrics.records[-120:]
        if len(records) < 2:
            return
        width = 260
        height = 86
        x0 = self.config.screen_width - width - 14
        y0 = 14
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill(self.config.panel_color)
        pygame.draw.rect(panel, self.config.panel_border_color, panel.get_rect(), 1)
        self.screen.blit(panel, (x0, y0))

        max_value = max(
            1,
            max(
                max(row["prey_population"], row["predator_population"], row["resource_count"])
                for row in records
            ),
        )
        self._draw_series(records, "resource_count", self.config.resource_color, x0, y0, width, height, max_value)
        self._draw_series(records, "prey_population", self.config.prey_color, x0, y0, width, height, max_value)
        self._draw_series(records, "predator_population", self.config.predator_color, x0, y0, width, height, max_value)
        self._label_graph(x0, y0)

    def _draw_series(
        self,
        records: list[dict[str, float | int]],
        key: str,
        color: tuple[int, int, int],
        x0: int,
        y0: int,
        width: int,
        height: int,
        max_value: float,
    ) -> None:
        points: list[tuple[int, int]] = []
        for index, row in enumerate(records):
            x = x0 + 8 + int(index * (width - 16) / max(1, len(records) - 1))
            y = y0 + height - 8 - int(float(row[key]) * (height - 22) / max_value)
            points.append((x, y))
        if len(points) > 1:
            pygame.draw.lines(self.screen, color, False, points, 2)

    def _label_graph(self, x0: int, y0: int) -> None:
        label = self.small_font.render("Population", True, self.config.muted_text_color)
        self.screen.blit(label, (x0 + 8, y0 + 5))

    def _to_screen(self, position: Vec2) -> tuple[int, int]:
        scale_x = self.config.screen_width / max(1, self.config.world_width)
        scale_y = self.config.screen_height / max(1, self.config.world_height)
        return (int(position[0] * scale_x), int(position[1] * scale_y))

    def _from_screen(self, position: tuple[int, int]) -> Vec2:
        scale_x = self.config.world_width / max(1, self.config.screen_width)
        scale_y = self.config.world_height / max(1, self.config.screen_height)
        return (float(position[0] * scale_x), float(position[1] * scale_y))

    @property
    def selected_agent(self) -> Agent | None:
        if self.selected_agent_id is None:
            return None
        return self.world.agents.get(self.selected_agent_id)

    def _agent_at_screen_position(self, position: tuple[int, int]) -> Agent | None:
        world_position = self._from_screen(position)
        nearest: Agent | None = None
        nearest_distance = self.config.selection_radius
        for agent in self.world.agents:
            distance = agent.distance_to(world_position)
            if distance <= nearest_distance:
                nearest = agent
                nearest_distance = distance
        return nearest

    def _current_target(self, agent: Agent) -> VisibleEntity | None:
        if not agent.alive:
            return None
        perception = self.world.perceive(agent)
        decision = self.world.decision_system.choose_action(agent, perception)
        return decision.target

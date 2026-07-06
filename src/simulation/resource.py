"""Plant/resource entities consumed by prey."""

from __future__ import annotations

from dataclasses import dataclass

from src.utils.random_utils import Vec2


@dataclass(slots=True, init=False)
class Resource:
    """An edible plant patch in continuous 2D space."""

    id: int
    x: float
    y: float
    energy_value: float
    radius: float
    regrowth_timer: int
    age: float

    def __init__(
        self,
        id: int = 0,
        x: float | Vec2 = 0.0,
        y: float | None = None,
        energy_value: float | None = None,
        radius: float = 4.5,
        regrowth_timer: int = 0,
        age: float = 0.0,
        *,
        position: Vec2 | None = None,
        energy: float | None = None,
    ) -> None:
        if position is not None:
            x, y = position
        elif isinstance(x, tuple):
            x, y = x
        elif y is None:
            y = 0.0

        self.id = id
        self.x = float(x)
        self.y = float(y)
        self.energy_value = float(energy_value if energy_value is not None else energy if energy is not None else 25.0)
        self.radius = float(radius)
        self.regrowth_timer = int(regrowth_timer)
        self.age = float(age)

    @property
    def position(self) -> Vec2:
        """Tuple position alias for renderers and vector helpers."""

        return (self.x, self.y)

    @position.setter
    def position(self, value: Vec2) -> None:
        self.x = float(value[0])
        self.y = float(value[1])

    @property
    def energy(self) -> float:
        """Backward-compatible alias for `energy_value`."""

        return self.energy_value

    @energy.setter
    def energy(self, value: float) -> None:
        self.energy_value = float(value)

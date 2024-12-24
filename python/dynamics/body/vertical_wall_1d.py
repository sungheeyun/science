"""
fixed vertical wall
"""

from typing import Sequence, Any

from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.artist import Artist

from dynamics.body.fixed_body_base import FixedBodyBase


class VerticalWall1D(FixedBodyBase):
    _WALL_HEIGHT: float = 1.0
    _WALL_WIDTH: float = 0.1
    _WALL_COLOR: str = "red"

    def __init__(self, loc_x: float | int, **kwargs) -> None:
        super().__init__(0, (loc_x, 0), None)
        plt_kwargs: dict[str, Any] = dict(fill=True, color=self._WALL_COLOR)
        plt_kwargs.update(**kwargs)

        vertices = (
            (self.loc[0] - self._WALL_WIDTH / 2.0, -self._WALL_HEIGHT / 2.0),
            (self.loc[0] - self._WALL_WIDTH / 2.0, self._WALL_HEIGHT / 2.0),
            (self.loc[0] + self._WALL_WIDTH / 2.0, self._WALL_HEIGHT / 2.0),
            (self.loc[0] + self._WALL_WIDTH / 2.0, -self._WALL_HEIGHT / 2.0),
        )

        self._polygon: Polygon = Polygon(vertices, **plt_kwargs)

    def update(self, t_1: float, t_2: float, forces: Any) -> None:
        pass

    @property
    def kinetic_energy(self) -> float:
        return 0.0

    # visualization

    def add_objs(self, ax: Axes) -> None:
        ax.add_patch(self._polygon)

    def update_obj(self) -> None:
        pass

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._polygon]

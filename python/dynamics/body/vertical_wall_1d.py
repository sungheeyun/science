"""
fixed vertical wall
"""

from typing import Sequence, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.artist import Artist

from dynamics.body.body_base import BodyBase


class VerticalWall1D(BodyBase):
    _WALL_HEIGHT: float = 1.0
    _WALL_WIDTH: float = 0.1
    _WALL_COLOR: str = "red"

    def __init__(self, loc_x: float, plt_kwargs: dict[str, Any] | None = None) -> None:
        super().__init__(np.inf, (loc_x, 0.0), None)
        _plt_kwargs: dict[str, Any] = dict(fill=True, color=self._WALL_COLOR)
        if plt_kwargs is not None:
            _plt_kwargs.update(**plt_kwargs)

        vertices = (
            (self.loc[0] - self._WALL_WIDTH / 2.0, -self._WALL_HEIGHT / 2.0),
            (self.loc[0] - self._WALL_WIDTH / 2.0, self._WALL_HEIGHT / 2.0),
            (self.loc[0] + self._WALL_WIDTH / 2.0, self._WALL_HEIGHT / 2.0),
            (self.loc[0] + self._WALL_WIDTH / 2.0, -self._WALL_HEIGHT / 2.0),
        )

        self._polygon: Polygon = Polygon(vertices, **_plt_kwargs)

    def update(self, t_1: float, t_2: float, forces: Any) -> None:
        pass

    def add_objs(self, ax: Axes) -> None:
        ax.add_patch(self._polygon)

    def update_obj(self) -> None:
        pass

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._polygon]

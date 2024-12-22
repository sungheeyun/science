"""
rigid ball
"""

from typing import Any, Sequence
import math

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from dynamics.body.body_base import BodyBase


class RigidBall(BodyBase):
    _UNIT_MASS_BALL_RADIUS: float = 0.1
    _BALL_COLOR: str = "blue"

    def __init__(
        self,
        mass: float = 1.0,
        init_loc: np.ndarray | list[float | int] | tuple[float | int, ...] | None = None,
        init_v: np.ndarray | list[float | int] | tuple[float | int, ...] | None = None,
        **kwargs
    ) -> None:
        super().__init__(mass, init_loc, init_v)
        circ_kwargs: dict[str, Any] = dict(
            radius=self._UNIT_MASS_BALL_RADIUS * math.pow(self.mass, 1.0 / 3.0),
            color=self._BALL_COLOR,
            fill=True,
        )
        circ_kwargs.update(**kwargs)

        self._obj = Circle((self.loc[0], self.loc[1]), **circ_kwargs)

    def update_obj(self) -> None:
        self._obj.center = (self.loc[0], self.loc[1])

    def add_objs(self, ax: Axes) -> None:
        ax.add_patch(self._obj)

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._obj]

    @property
    def updated_objs(self) -> Sequence[Artist]:
        return self.objs

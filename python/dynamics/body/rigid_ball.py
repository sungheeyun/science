"""
rigid ball
"""

from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.patches import Circle

from dynamics.body.body_base import BodyBase


class RigidBall(BodyBase):
    _BALL_RADIUS: float = 0.1
    _BALL_COLOR: str = "blue"

    def __init__(
        self,
        mass: float = 1.0,
        init_loc: np.ndarray | list[float] | tuple[float, ...] | None = None,
        init_v: np.ndarray | list[float] | tuple[float, ...] | None = None,
        **kwargs
    ) -> None:
        super().__init__(mass, init_loc, init_v)
        circ_kwargs: dict[str, Any] = dict(
            radius=self._BALL_RADIUS, color=self._BALL_COLOR, fill=True
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

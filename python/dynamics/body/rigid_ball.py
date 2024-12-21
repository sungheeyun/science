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
    def __init__(
        self,
        mass: float = 1.0,
        init_loc: np.ndarray | list[float] | tuple[float, ...] | None = None,
        init_v: np.ndarray | list[float] | tuple[float, ...] | None = None,
        **kwargs
    ) -> None:
        super().__init__(mass, init_loc, init_v)
        circ_kwargs: dict[str, Any] = dict(radius=0.1, color="blue", fill=True)
        circ_kwargs.update(**kwargs)

        self._obj = Circle((self._cur_loc[0], self._cur_loc[1]), **circ_kwargs)

    def update_obj(self) -> None:
        self._obj.center = (self._cur_loc[0], self._cur_loc[1])

    def add_obj(self, ax: Axes) -> None:
        ax.add_patch(self._obj)

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._obj]

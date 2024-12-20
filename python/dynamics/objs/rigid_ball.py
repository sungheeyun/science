"""
rigid ball
"""

from typing import Any

import numpy as np
from matplotlib.patches import Circle, Patch

from dynamics.objs.body_base import BodyBase


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

        self._circle: Circle = Circle((self._cur_loc[0], self._cur_loc[1]), **circ_kwargs)

    @property
    def obj(self) -> Patch:
        return self._circle

    def _update_obj(self) -> None:
        self._circle.center = (self._cur_loc[0], self._cur_loc[1])

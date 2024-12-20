"""
frictional force
"""

from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist

from dynamics.objs.body_base import BodyBase
from dynamics.force.one_obj_force_base import OneObjForceBase
from matplotlib.lines import Line2D


class HorizontalFrictionalOneObjForce(OneObjForceBase):
    def __init__(
        self,
        coef_friction: float,
        boundary: float,
        obj_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert coef_friction >= 0.0, coef_friction

        self._coef_friction: float = coef_friction
        self._boundary: float = boundary

        self._num_stripes: int = 50
        self._x_stretch: float = 5.0
        self._height: float = 0.1
        x_1d_p: np.ndarray = np.linspace(
            self._boundary - self._x_stretch, self._boundary, self._num_stripes
        )
        self._line2d_list: list[Line2D] = [
            Line2D(
                xdata=x_1d_p[idx : idx + 2],  # noqa: E203
                ydata=[-self._height, 0.0],
                linewidth=1.5,
                color="black",
                linestyle="-",
                alpha=0.5,
            )
            for idx in range(x_1d_p.size - 1)
        ]

    def _one_obj_force(self, time: float, obj: BodyBase) -> np.ndarray:
        return np.array(
            [0.0 if obj.loc[0] >= self._boundary else (-self._coef_friction * obj.vel[0]), 0.0]
        )

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return np.zeros_like(x_1d)

    @property
    def objs(self) -> Sequence[Artist]:
        return self._line2d_list

"""
frictional force
"""

from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from dynamics.body.body_base import BodyBase
from dynamics.force.force_base import ForceBase
from matplotlib.lines import Line2D


class HorizontalFrictionalForce1D(ForceBase):
    _FRICTIONAL_FORCE_HEIGHT: float = 0.1

    def __init__(
        self,
        coef_friction: float,
        boundary: float,
        obj_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert coef_friction >= 0.0, coef_friction
        plt_kwargs: dict[str, Any] = dict(
            linewidth=1.5,
            color="black",
            linestyle="-",
            alpha=0.5,
        )
        if obj_kwargs is not None:
            plt_kwargs.update(**obj_kwargs)

        self._coef_friction: float = coef_friction
        self._boundary: float = boundary

        x_1d_p: np.ndarray = np.linspace(
            self._boundary - self._FRICTIONAL_FORCE_STRETCH,
            self._boundary,
            int(10 * self._FRICTIONAL_FORCE_STRETCH),
        )
        self._line2d_list: list[Line2D] = (
            list()
            if self._coef_friction == 0.0
            else [
                Line2D(
                    xdata=x_1d_p[idx : idx + 2],  # noqa:E203
                    ydata=[-self._FRICTIONAL_FORCE_HEIGHT, 0.0],
                    **plt_kwargs  # noqa: E203
                )
                for idx in range(x_1d_p.size - 1)
            ]
        )

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        return np.array(
            [0.0 if body.loc[0] >= self._boundary else (-self._coef_friction * body.vel[0]), 0.0]
        )

    def body_potential_energy(self, body: BodyBase) -> float:
        return 0.0

    @property
    def potential_energy(self) -> float:
        return 0.0

    # visualization

    def add_objs(self, ax: Axes) -> None:
        for line2d in self._line2d_list:
            ax.add_artist(line2d)

    @property
    def objs(self) -> Sequence[Artist]:
        return self._line2d_list

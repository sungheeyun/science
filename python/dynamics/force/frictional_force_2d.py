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


class FrictionalForce2D(ForceBase):
    _FRICTIONAL_FORCE_HEIGHT: float = 0.1

    def __init__(
        self,
        coef_friction: float,
        upper_right_pnt: np.ndarray | tuple[float, ...] | list[float],
        obj_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert coef_friction >= 0.0, coef_friction
        plt_kwargs: dict[str, Any] = dict(
            linewidth=1.5,
            color="black",
            linestyle="-",
            alpha=0.1,
        )
        if obj_kwargs is not None:
            plt_kwargs.update(**obj_kwargs)

        self._coef_friction: float = coef_friction
        self._upper_right_pnt: np.ndarray = np.array(upper_right_pnt, float)

        xy_1d_p: np.ndarray = np.linspace(
            0.0, self._FRICTIONAL_FORCE_STRETCH, int(10 * self._FRICTIONAL_FORCE_STRETCH)
        )
        self._line2d_list: list[Line2D] = (
            list()
            if self._coef_friction == 0.0
            else [
                Line2D(
                    xdata=[
                        self._upper_right_pnt[0] - self._FRICTIONAL_FORCE_STRETCH,
                        self._upper_right_pnt[0] - xy,
                    ],
                    ydata=[
                        self._upper_right_pnt[1] - xy_1d_p[xy_1d_p.size - idx - 1],
                        self._upper_right_pnt[1],
                    ],
                    **plt_kwargs
                )
                for idx, xy in enumerate(xy_1d_p)
            ]
            + [
                Line2D(
                    xdata=[self._upper_right_pnt[0] - xy, self._upper_right_pnt[0]],
                    ydata=[
                        self._upper_right_pnt[1] - self._FRICTIONAL_FORCE_STRETCH,
                        self._upper_right_pnt[1] - xy_1d_p[xy_1d_p.size - idx - 1],
                    ],
                    **plt_kwargs
                )
                for idx, xy in enumerate(xy_1d_p[:-1])
            ]
        )

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        return (
            -self._coef_friction * body.vel
            if np.all(body.loc < self._upper_right_pnt)
            else np.zeros_like(body.loc)
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

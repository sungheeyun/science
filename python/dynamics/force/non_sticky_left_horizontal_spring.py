"""
non-sticky left horizontal spring
"""

from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from dynamics.force.force_base import ForceBase
from dynamics.body.body_base import BodyBase


class NonStickyLeftHorizontalSpring(ForceBase):
    def __init__(
        self,
        spring_constant: float,
        equilibrium_point: float,
        obj_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert spring_constant > 0.0, spring_constant
        self._spring_constant: float = spring_constant
        self._equilibrium_point: float = equilibrium_point
        self._obj_kwargs: dict[str, Any] = self._DEFAULT_SPRING_OBJ_KWARGS.copy()
        self._cur_x: float = self._equilibrium_point
        if obj_kwargs is not None:
            self._obj_kwargs.update(**obj_kwargs)

        self._num_coils: int = int(self._NUM_COILS_PER_UNIT_LEN * self._SPRING_X_STRETCH)
        self._t_1d_p: np.ndarray = np.linspace(
            0.0, self._num_coils * 2.0 * np.pi, self._NUM_PLT_POINTS
        )
        self._ydata: np.ndarray = self._SPRING_WIDTH * np.sin(self._t_1d_p)

        self._line2d: Line2D = self._create_obj()

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        self._cur_x = body.loc[0]
        force_x: float = (
            0.0
            if self._cur_x >= self._equilibrium_point
            else self._spring_constant * (self._equilibrium_point - self._cur_x)
        )
        return np.array([force_x, 0.0])

    def body_potential_energy(self, body: BodyBase) -> float:
        return 0.0

    @property
    def potential_energy(self) -> float:
        return (
            0.5 * self._spring_constant * (self._cur_x - self._equilibrium_point) ** 2.0
            if self._cur_x < self._equilibrium_point
            else 0.0
        )

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return (
            0.5
            * self._spring_constant
            * (x_1d < self._equilibrium_point)
            * np.power(x_1d - self._equilibrium_point, 2.0)
        )

    # visualization

    def _create_obj(self) -> Line2D:
        return Line2D(
            xdata=self._equilibrium_point
            - np.linspace(self._SPRING_X_STRETCH, 0.0, self._t_1d_p.size),
            ydata=self._ydata,
            **self._obj_kwargs,
        )

    def add_objs(self, ax: Axes) -> None:
        ax.add_artist(self._line2d)

    def update_objs(self) -> None:
        self._line2d.set_xdata(
            np.linspace(
                self._equilibrium_point - self._SPRING_X_STRETCH,
                self._cur_x if self._cur_x <= self._equilibrium_point else self._equilibrium_point,
                self._t_1d_p.size,
            )
        )

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._line2d]

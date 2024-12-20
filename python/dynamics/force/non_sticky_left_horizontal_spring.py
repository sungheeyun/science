"""
non-sticky left horizontal spring
"""

from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

from dynamics.force.one_body_force_base import OneBodyForceBase
from dynamics.body.body_base import BodyBase


class NonStickyLeftHorizontalSpring(OneBodyForceBase):
    def __init__(
        self,
        spring_constant: float,
        equilibrium_point: float,
        obj_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert spring_constant > 0.0, spring_constant
        self._spring_constant: float = spring_constant
        self._equilibrium_point: float = equilibrium_point
        self._obj_kwargs: dict[str, Any] = dict(linestyle="-", color="blue", linewidth=1.5)
        if obj_kwargs is not None:
            self._obj_kwargs.update(**obj_kwargs)

        self._num_coils: int = 25
        self._num_pnt_p: int = 1000
        self._amplitude: float = 0.1
        self._x_stretch: float = 5.0
        self._t_1d_p: np.ndarray = np.linspace(0.0, self._num_coils * 2.0 * np.pi, self._num_pnt_p)
        self._ydata: np.ndarray = self._amplitude * np.sin(self._t_1d_p)

        self._line2d: Line2D = self._create_obj()

    def _create_obj(self) -> Line2D:
        return Line2D(
            xdata=self._equilibrium_point - np.linspace(self._x_stretch, 0.0, self._t_1d_p.size),
            ydata=self._ydata,
            **self._obj_kwargs,
        )

    def _one_obj_force(self, time: float, obj: BodyBase) -> np.ndarray:
        force_x: float = (
            0.0
            if obj.loc[0] >= self._equilibrium_point
            else self._spring_constant * (self._equilibrium_point - obj.loc[0])
        )
        return np.array([force_x, 0.0])

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return (
            0.5
            * self._spring_constant
            * (x_1d < self._equilibrium_point)
            * np.power(x_1d - self._equilibrium_point, 2.0)
        )

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._line2d]

    def update_obj(self, time: float, loc: np.ndarray) -> None:
        self._line2d.set_xdata(
            np.linspace(
                self._equilibrium_point - self._x_stretch,
                loc[0] if loc[0] <= self._equilibrium_point else self._equilibrium_point,
                self._t_1d_p.size,
            )
        )

"""
non-sticky left horizontal spring
"""

import math
from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.axes import Axes

from dynamics.body.bodies import Bodies
from dynamics.body.fixed_body_base import FixedBodyBase
from dynamics.force.spring_base import SpringBase
from dynamics.body.body_base import BodyBase


class NonStickyLeftHorizontalSpring(SpringBase):
    _SPRING_X_STRETCH: float = 5.0

    def __init__(
        self, spring_constant: float | int, equilibrium_point: float | int, **kwargs
    ) -> None:
        super().__init__(spring_constant)
        self._equilibrium_point: float = float(equilibrium_point)
        self._cur_x: float = self._equilibrium_point

        # visualization

        self._obj_kwargs: dict[str, Any] = dict(
            linestyle="-",
            color="blue",
            alpha=0.5,
            linewidth=self._SPRING_UNIT_CONSTANT_LINE_WIDTH
            * math.pow(self.spring_constant, 1.0 / 3.0),
        )
        self._obj_kwargs.update(**kwargs)

        self._num_coils: int = max(
            int(self._SPRING_NUM_COILS_PER_UNIT_LEN * self._SPRING_X_STRETCH),
            self._SPRING_MIN_NUM_COILS,
        )
        self._t_1d_p: np.ndarray = np.linspace(
            0.0, self._num_coils * 2.0 * np.pi, self._NUM_PLT_POINTS_PER_COIL * self._num_coils
        )
        self._ydata: np.ndarray = 0.5 * self._SPRING_WIDTH * np.sin(self._t_1d_p)

        self._line2d: Line2D = self._create_obj()

    # getters
    @property
    def equilibrium_point(self) -> float:
        return self._equilibrium_point

    # simulation

    def register_force(self, bodies: Bodies) -> None:
        for body in bodies.bodies:
            body.register_force(self)

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        self._cur_x = body.loc[0]
        force_x: float = (
            0.0
            if self._cur_x >= self._equilibrium_point
            else self.spring_constant * (self._equilibrium_point - self._cur_x)
        )
        return np.array([force_x, 0.0])

    # potential energy

    def body_potential_energy(self, body: BodyBase) -> float:
        return 0.0

    @property
    def potential_energy(self) -> float:
        return (
            0.5 * self.spring_constant * (self._cur_x - self._equilibrium_point) ** 2.0
            if self._cur_x < self._equilibrium_point
            else 0.0
        )

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return (
            0.5
            * self.spring_constant
            * (x_1d < self._equilibrium_point)
            * np.power(x_1d - self._equilibrium_point, 2.0)
        )

    # potential energy

    def min_energy_matrices(self, bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
        num_coordinates: int = bodies.num_coordinates
        a_2d: np.ndarray = np.zeros((num_coordinates, num_coordinates))
        b_1d: np.ndarray = np.zeros(num_coordinates)

        for body in bodies.bodies:
            if isinstance(body, FixedBodyBase):
                continue

            for idx in bodies.coordinate_indices(body):
                a_2d[idx, idx] = self.spring_constant
                b_1d[idx] = self.spring_constant * self.equilibrium_point

        return a_2d, b_1d

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

    @property
    def updated_objs(self) -> Sequence[Artist]:
        return self.objs

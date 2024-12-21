"""
non-sticky left horizontal spring
"""

import math
from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes
from numpy import linalg as la
from matplotlib.artist import Artist
from matplotlib.lines import Line2D

from dynamics.force.force_base import ForceBase
from dynamics.body.body_base import BodyBase
from dynamics.body.vertical_wall_1d import VerticalWall1D


class Spring(ForceBase):

    def __init__(
        self,
        spring_constant: float,
        natural_length: float,
        body_1: BodyBase,
        body_2: BodyBase,
        obj_kwargs: dict[str, Any] | None = None,
    ) -> None:
        assert spring_constant > 0.0, spring_constant
        assert natural_length > 0.0, natural_length

        self._spring_constant: float = spring_constant
        self._natural_length: float = natural_length
        self._body_1: BodyBase = body_1
        self._body_2: BodyBase = body_2

        self._obj_kwargs: dict[str, Any] = dict(
            linestyle="-",
            color="blue",
            linewidth=self._UNIT_SPRINT_CONSTANT_LINE_WIDTH
            * math.pow(self._spring_constant, 1.0 / 3.0),
        )
        if obj_kwargs is not None:
            self._obj_kwargs.update(**obj_kwargs)

        self._num_coils: int = int(self._NUM_COILS_PER_UNIT_LEN * self._natural_length)
        self._t_1d_p: np.ndarray = np.linspace(
            0.0, self._num_coils * 2.0 * np.pi, self._NUM_PLT_POINTS
        )
        self._ydata: np.ndarray = self._SPRING_WIDTH * np.sin(self._t_1d_p)

        self._line2d: Line2D = self._create_obj()

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        if body is self._body_1:
            return -self._second_body_force(time)
        elif body is self._body_2:
            return self._second_body_force(time)
        else:
            return np.zeros(2)

    def _second_body_force(self, time: float) -> np.ndarray:
        vec_2_1: np.ndarray = self._body_2.loc - self._body_1.loc
        return (
            -self._spring_constant * (la.norm(vec_2_1) - self._natural_length) / la.norm(vec_2_1)
        ) * vec_2_1

    def body_potential_energy(self, body: BodyBase) -> float:
        return 0.0

    @property
    def potential_energy(self) -> float:
        return (
            0.5
            * self._spring_constant
            * float(la.norm(self._body_1.loc - self._body_2.loc) - self._natural_length) ** 2.0
        )

    def x_potential_energy(self, body: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        if not isinstance(self._body_1, VerticalWall1D) and not isinstance(
            self._body_2, VerticalWall1D
        ):
            return np.zeros_like(x_1d)

        center_pnt: float = 0.0
        if isinstance(self._body_1, VerticalWall1D):
            center_pnt = float(
                self._body_1.loc[0]
                + self._natural_length * (1.0 if body.loc[1] > self._body_1.loc[0] else -1.0)
            )
        else:
            center_pnt = float(
                self._body_2.loc[0]
                + self._natural_length * (1.0 if body.loc[1] > self._body_2.loc[0] else -1.0)
            )

        return 0.5 * self._spring_constant * np.power(x_1d - center_pnt, 2.0)

    # visualization

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._line2d]

    def add_objs(self, ax: Axes) -> None:
        ax.add_artist(self._line2d)

    def _create_obj(self) -> Line2D:
        coordinate_2d: np.ndarray = self._coordinate_2d
        # print(coordinate_2d.shape)
        # print(coordinate_2d[0].shape)
        # print(coordinate_2d[1].shape)
        return Line2D(xdata=coordinate_2d[0], ydata=coordinate_2d[1], **self._obj_kwargs)

    def update_objs(self) -> None:
        self._line2d.set_data(self._coordinate_2d)

    @property
    def _coordinate_2d(self) -> np.ndarray:
        theta: float = math.atan2(
            self._body_2.loc[1] - self._body_1.loc[1], self._body_2.loc[0] - self._body_1.loc[0]
        )

        return (
            np.dot(
                np.vstack(
                    (
                        np.linspace(
                            0.0, la.norm(self._body_2.loc - self._body_1.loc), self._t_1d_p.size
                        ),
                        self._ydata,
                    )
                ).T,
                np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]),
            )
            + self._body_1.loc
        ).T

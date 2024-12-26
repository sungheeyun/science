"""
non-sticky left horizontal spring
"""

import math
from typing import Any, Sequence

import numpy as np
from matplotlib.axes import Axes
from numpy.linalg import norm
from matplotlib.artist import Artist
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from dynamics.bodies.bodies import Bodies
from dynamics.bodies.fixed_body_base import FixedBodyBase
from dynamics.bodies.point_mass import PointMass
from dynamics.forces.spring_base import SpringBase
from dynamics.bodies.body_base import BodyBase
from dynamics.bodies.vertical_wall_1d import VerticalWall1D


class Spring(SpringBase):
    def __init__(
        self,
        spring_constant: float | int,
        natural_length: float | int,
        body_1: BodyBase,
        body_2: BodyBase,
        **kwargs
    ) -> None:
        super().__init__(spring_constant)

        assert natural_length >= 0.0, natural_length
        self._natural_length: float = float(natural_length)
        self._body_1: BodyBase = body_1
        self._body_2: BodyBase = body_2

        self._body_1.register_force(self)
        self._body_2.register_force(self)

        # visualization

        self._spring_kwargs: dict[str, Any] = dict(
            linestyle="-",
            color="blue",
            alpha=0.5,
            linewidth=self._SPRING_UNIT_CONSTANT_LINE_WIDTH
            * math.pow(self.spring_constant, 1.0 / 3.0),
        )
        self._spring_kwargs.update(**kwargs)
        self._poly_kwargs: dict[str, Any] = self._spring_kwargs.copy()
        self._poly_kwargs.update(alpha=0.3)

        self._num_coils: int = max(
            int(self._SPRING_NUM_COILS_PER_UNIT_LEN * self._natural_length),
            self._SPRING_MIN_NUM_COILS,
        )
        self._t_1d_p: np.ndarray = np.linspace(
            0.0, self._num_coils * 2.0 * np.pi, self._NUM_PLT_POINTS_PER_COIL * self._num_coils
        )
        self._ydata: np.ndarray = 0.5 * self._SPRING_WIDTH * np.sin(self._t_1d_p)

        self._obj_list: list[Artist] = self._create_objs()

    # getters
    @property
    def natural_length(self) -> float:
        return self._natural_length

    # dynamics simulation

    def register_force(self, bodies: Bodies) -> None:
        pass

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        if body is self._body_1:
            return -self._second_body_force(time)
        elif body is self._body_2:
            return self._second_body_force(time)
        else:
            return np.zeros(2)

    def _second_body_force(self, time: float) -> np.ndarray:
        vec_2_1: np.ndarray = self._body_2.loc - self._body_1.loc
        assert norm(vec_2_1) > 0.0, vec_2_1
        return (
            -self.spring_constant * (norm(vec_2_1) - self._natural_length) / norm(vec_2_1)
        ) * vec_2_1

    # potential energy

    def min_energy_matrices(self, bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
        _a_2d_1, _b_1d_1 = self._min_energy_matrices(self._body_1, self._body_2, bodies)
        _a_2d_2, _b_1d_2 = self._min_energy_matrices(self._body_2, self._body_1, bodies)

        return _a_2d_1 + _a_2d_2, _b_1d_1 + _b_1d_2

    def _min_energy_matrices(
        self, body_1: BodyBase, body_2: BodyBase, bodies: Bodies
    ) -> tuple[np.ndarray, np.ndarray]:
        num_coordinates: int = bodies.num_coordinates
        a_2d: np.ndarray = np.zeros((num_coordinates, num_coordinates))
        b_1d: np.ndarray = np.zeros(num_coordinates)

        if isinstance(body_1, PointMass):
            indices_1: tuple[int, ...] = bodies.coordinate_indices(body_1)
            if isinstance(body_2, PointMass):
                indices_2: tuple[int, ...] = bodies.coordinate_indices(body_2)
                for _idx, idx_1 in enumerate(indices_1):
                    idx_2: int = indices_2[_idx]
                    a_2d[idx_1, idx_1] = self.spring_constant
                    a_2d[idx_1, idx_2] = -self.spring_constant
            else:
                assert isinstance(body_2, FixedBodyBase), (body_1.__class__, body_2.__class__)
                for _idx, idx_1 in enumerate(indices_1):
                    a_2d[idx_1, idx_1] = self.spring_constant
                    b_1d[idx_1] = self.spring_constant * body_2.loc[_idx]
        else:
            assert isinstance(body_1, FixedBodyBase) and isinstance(body_2, PointMass), (
                body_1.__class__,
                body_2.__class__,
            )

        return a_2d, b_1d

    def body_potential_energy(self, body: BodyBase) -> float:
        return 0.0

    @property
    def potential_energy(self) -> float:
        return (
            0.5
            * self.spring_constant
            * float(norm(self._body_1.loc - self._body_2.loc) - self._natural_length) ** 2.0
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

        return 0.5 * self.spring_constant * np.power(x_1d - center_pnt, 2.0)

    # visualization

    @property
    def objs(self) -> Sequence[Artist]:
        return self._obj_list

    @property
    def updated_objs(self) -> Sequence[Artist]:
        return self._obj_list

    def add_objs(self, ax: Axes) -> None:
        for line2d in self.objs:
            ax.add_artist(line2d)

    def _create_objs(self) -> list[Artist]:
        coordinate_2d: np.ndarray = self._coordinate_2d
        spring_coordinate_2d: np.ndarray = coordinate_2d[:, :-4]
        return [
            Line2D(
                xdata=spring_coordinate_2d[0], ydata=spring_coordinate_2d[1], **self._spring_kwargs
            ),
            Polygon(coordinate_2d[:, -4:].T, **self._poly_kwargs),
        ]

    def update_objs(self) -> None:
        coordinate_2d: np.ndarray = self._coordinate_2d
        self.objs[0].set_data(coordinate_2d[:, :-4])  # type:ignore
        self.objs[1].set_xy(coordinate_2d[:, -4:].T)  # type:ignore

    @property
    def _coordinate_2d(self) -> np.ndarray:
        theta: float = math.atan2(
            self._body_2.loc[1] - self._body_1.loc[1], self._body_2.loc[0] - self._body_1.loc[0]
        )
        spring_length: float = float(norm(self._body_2.loc - self._body_1.loc))

        return (
            0.5
            * (
                np.dot(
                    np.hstack(
                        (
                            np.vstack(
                                (
                                    spring_length * np.linspace(-1.0, 1.0, self._t_1d_p.size),
                                    2.0 * self._ydata,
                                )
                            ),
                            [
                                self.natural_length * np.array([-1, -1, 1, 1], float),
                                self._SPRING_WIDTH * np.array([-1, 1, 1, -1], float),
                            ],
                        )
                    ).T,
                    np.array(
                        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
                    ),
                )
                + (self._body_1.loc + self._body_2.loc)
            )
        ).T

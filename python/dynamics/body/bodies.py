"""
bodies
"""

from functools import reduce
from typing import Any, Sequence

import numpy as np
from numpy import linalg as la
from matplotlib.axes import Axes
from matplotlib.artist import Artist

from dynamics.body.body_base import BodyBase


class Bodies:
    _SIM_TIME_STEP: float = 1e-4
    _SIM_TIME_STEP_CONST_VEL: float = 1e-4

    @classmethod
    def set_time_step_lengths(cls, sim_time_step: float, sim_time_step_const_vel: float) -> None:
        assert sim_time_step > 0.0, sim_time_step
        assert sim_time_step_const_vel > 0.0, sim_time_step_const_vel

        cls._SIM_TIME_STEP = sim_time_step
        cls._SIM_TIME_STEP_CONST_VEL = sim_time_step_const_vel

    def __init__(self, *args) -> None:
        self._bodies: list[BodyBase] = list(args)
        self._cur_time: float = 0.0

    @property
    def bodies(self) -> list[BodyBase]:
        return self._bodies

    def update(self, next_time: float, forces: Any) -> None:
        assert next_time >= self._cur_time, (next_time, self._cur_time)
        if next_time == self._cur_time:
            return

        max_vel: float = float(
            max([la.norm(body._cur_vel) for body in self.bodies])  # type:ignore
        )

        t_step: float = min(
            self._SIM_TIME_STEP, self._SIM_TIME_STEP_CONST_VEL / (max_vel if max_vel > 0.0 else 1.0)
        )
        self._update_bodies(next_time, t_step, forces)
        self._cur_time = next_time

        self.update_objs()

    def _update_bodies(self, next_time: float, t_step: float, forces: Any) -> None:
        t_stamps: np.ndarray = np.hstack((np.arange(self._cur_time, next_time, t_step), next_time))
        for idx, t_1 in enumerate(t_stamps[:-1]):
            for body in self.bodies:
                body.update(t_1, t_stamps[idx + 1], forces)

    @property
    def kinetic_energy(self) -> float:
        return sum([body.kinetic_energy for body in self.bodies])

    def potential_energy(self, forces: Any) -> float:
        return sum([body.body_potential_energy(forces) for body in self.bodies])

    @property
    def dissipated_energy(self) -> float:
        return sum([body.dissipated_energy for body in self.bodies])

    # visualization

    def add_objs(self, ax: Axes) -> None:
        for body in self.bodies:
            body.add_objs(ax)

    def update_objs(self) -> None:
        for body in self.bodies:
            body.update_obj()

    @property
    def objs(self) -> Sequence[Artist]:
        return reduce(list.__add__, [list(body.objs) for body in self.bodies])  # type:ignore

"""
bodies
"""

from functools import reduce
from typing import Sequence

import numpy as np
from numpy import linalg as la
from matplotlib.axes import Axes
from matplotlib.artist import Artist

from dynamics.body.body_base import BodyBase
from dynamics.force.forces import Forces


class Bodies:
    _SIM_TIME_STEP: float = 1e-4
    _SIM_TIME_STEP_CONST_VEL: float = 1e-4

    def __init__(self, *args) -> None:
        self._bodies: list[BodyBase] = list(args)
        self._cur_time: float = 0.0

    def update(self, next_time: float, forces: Forces) -> None:
        assert next_time >= self._cur_time, (next_time, self._cur_time)
        if next_time == self._cur_time:
            return

        max_vel: float = float(
            max([la.norm(body._cur_vel) for body in self._bodies])  # type:ignore
        )

        t_step: float = min(
            self._SIM_TIME_STEP, self._SIM_TIME_STEP_CONST_VEL / (max_vel if max_vel > 0.0 else 1.0)
        )
        self._update_bodies(next_time, t_step, forces)
        self._cur_time = next_time

        self.update_objs()

    def _update_bodies(self, next_time: float, t_step: float, forces: Forces) -> None:
        t_stamps: np.ndarray = np.hstack((np.arange(self._cur_time, next_time, t_step), next_time))
        for idx, t_1 in enumerate(t_stamps[:-1]):
            for body in self._bodies:
                body.update(t_1, t_stamps[idx + 1], forces)

    def add_objs(self, ax: Axes) -> None:
        for body in self._bodies:
            body.add_objs(ax)

    def update_objs(self) -> None:
        for body in self._bodies:
            body.update_obj()

    @property
    def objs(self) -> Sequence[Artist]:
        return reduce(list.__add__, [list(body.objs) for body in self._bodies])  # type:ignore

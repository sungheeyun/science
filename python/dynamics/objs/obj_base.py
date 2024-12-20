"""
base class for all objects in dynamics package
"""

from abc import ABC, abstractmethod

import numpy as np
import numpy.linalg as la
from matplotlib.patches import Patch

from dynamics.force.forces import Forces


class ObjBase(ABC):
    def __init__(
        self,
        mass: float,
        init_loc: np.ndarray | list[float] | tuple[float, ...] | None,
        init_vel: np.ndarray | list[float] | tuple[float, ...] | None,
    ) -> None:
        self.mass: float = mass
        self._cur_loc: np.ndarray = (
            np.array([0, 0], float) if init_loc is None else np.array(init_loc, float)
        )
        self._cur_vel: np.ndarray = (
            np.array([0, 0], float) if init_vel is None else np.array(init_vel, float)
        )
        self._cur_time: float = 0.0

    def update(self, next_time: float, forces: Forces) -> None:
        assert next_time >= self._cur_time, (next_time, self._cur_time)
        if next_time == self._cur_time:
            return

        t_step: float = min(1e-4, 1e-4 / la.norm(self._cur_vel))  # type: ignore
        self._update_loc_and_vel(forces, self._cur_time, next_time, t_step)
        self._cur_time = next_time

        self._update_obj()
        forces.update_obj(next_time, self._cur_loc)

    @property
    @abstractmethod
    def obj(self) -> Patch:
        pass

    @abstractmethod
    def _update_obj(self) -> None:
        pass

    @property
    def loc(self) -> np.ndarray:
        return self._cur_loc

    @property
    def vel(self) -> np.ndarray:
        return self._cur_vel

    def _update_loc_and_vel(
        self, forces: Forces, t_start: float, t_end: float, t_step: float
    ) -> None:
        t_stamps: np.ndarray = np.hstack((np.arange(t_start, t_end, t_step), t_end))

        for idx, t_1 in enumerate(t_stamps[:-1]):
            t_2: float = t_stamps[idx + 1]

            next_loc: np.ndarray = (t_2 - t_1) * self._cur_vel + self._cur_loc
            self._cur_vel += (
                (t_2 - t_1)
                * forces.force((t_1 + t_2) / 2.0, (self._cur_loc + next_loc) / 2.0, self._cur_vel)
                / self.mass
            )

            self._cur_loc = next_loc

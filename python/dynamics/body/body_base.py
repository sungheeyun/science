"""
base class for all objects in dynamics package
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes


class BodyBase(ABC):
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

    def update(self, t_1: float, t_2: float, forces: Any) -> None:
        next_loc: np.ndarray = (t_2 - t_1) * self._cur_vel + self._cur_loc
        self._cur_vel += (t_2 - t_1) * forces.force((t_1 + t_2) / 2.0, self) / self.mass
        self._cur_loc = next_loc

    @abstractmethod
    def add_objs(self, ax: Axes) -> None:
        pass

    @abstractmethod
    def update_obj(self) -> None:
        pass

    @property
    def loc(self) -> np.ndarray:
        return self._cur_loc

    @property
    def vel(self) -> np.ndarray:
        return self._cur_vel

    def _update_loc_and_vel(self, forces: Any, t_start: float, t_end: float, t_step: float) -> None:
        t_stamps: np.ndarray = np.hstack((np.arange(t_start, t_end, t_step), t_end))

        for idx, t_1 in enumerate(t_stamps[:-1]):
            t_2: float = t_stamps[idx + 1]

            next_loc: np.ndarray = (t_2 - t_1) * self._cur_vel + self._cur_loc
            self._cur_vel += (
                (t_2 - t_1) * forces.one_body_force((t_1 + t_2) / 2.0, self) / self.mass
            )

            self._cur_loc = next_loc

    @property
    @abstractmethod
    def objs(self) -> Sequence[Artist]:
        pass

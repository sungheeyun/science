"""
base class for all objects in dynamics package
"""

from abc import ABC, abstractmethod
from typing import Any, Sequence

import numpy as np
from numpy.linalg import norm
from matplotlib.artist import Artist
from matplotlib.axes import Axes


class BodyBase(ABC):
    def __init__(
        self,
        mass: float,
        init_loc: np.ndarray | list[float] | tuple[float, ...] | None,
        init_vel: np.ndarray | list[float] | tuple[float, ...] | None,
    ) -> None:
        self._mass: float = mass
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

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def loc(self) -> np.ndarray:
        return self._cur_loc

    @property
    def vel(self) -> np.ndarray:
        return self._cur_vel

    @property
    def loc_text(self):
        return "(" + ", ".join(f"{loc:.2f}" for loc in self._cur_loc) + ")"

    @property
    def vel_text(self):
        return "(" + ", ".join(f"{vel:.2f}" for vel in self._cur_vel) + ")"

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * float(norm(self.vel)) ** 2.0

    def body_potential_energy(self, forces: Any) -> float:
        return sum([force.body_potential_energy(self) for force in forces._forces])

    # visualization

    @property
    @abstractmethod
    def objs(self) -> Sequence[Artist]:
        pass

    @abstractmethod
    def add_objs(self, ax: Axes) -> None:
        pass

    @abstractmethod
    def update_obj(self) -> None:
        pass

"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib.artist import Artist


class ForceBase(ABC):
    def force(self, time: float, loc: np.ndarray, vel: np.ndarray) -> np.ndarray:
        return self._force(time, loc, vel)

    @abstractmethod
    def _force(self, time: float, loc: np.ndarray, vel: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        pass

    @property
    def objs(self) -> list[Artist]:
        return list()

    def update_obj(self, time: float, loc: np.ndarray) -> None:
        pass

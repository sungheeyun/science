"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib.artist import Artist


class ForceBase(ABC):
    def force(self, time: float, loc: np.ndarray) -> np.ndarray:
        return self._force(time, loc)

    @abstractmethod
    def _force(self, time: float, loc: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        pass

    @property
    def obj(self) -> Artist | None:
        return None

    def update_obj(self, time: float, loc: np.ndarray) -> None:
        pass

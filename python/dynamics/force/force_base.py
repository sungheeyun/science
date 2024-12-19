"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod

import numpy as np


class ForceBase(ABC):
    @abstractmethod
    def force(self, time: float, loc: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        pass

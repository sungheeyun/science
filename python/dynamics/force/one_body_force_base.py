"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist

from dynamics.body.body_base import BodyBase


class OneBodyForceBase(ABC):
    def one_obj_force(self, time: float, obj: BodyBase) -> np.ndarray:
        return self._one_obj_force(time, obj)

    @abstractmethod
    def _one_obj_force(self, time: float, obj: BodyBase) -> np.ndarray:
        pass

    @abstractmethod
    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        pass

    @property
    def objs(self) -> Sequence[Artist]:
        return list()

    def update_obj(self, time: float, loc: np.ndarray) -> None:
        pass

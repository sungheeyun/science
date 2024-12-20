"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist

from dynamics.objs.obj_base import ObjBase


class ForceBase(ABC):
    def force(self, time: float, obj: ObjBase) -> np.ndarray:
        return self._force(time, obj)

    @abstractmethod
    def _force(self, time: float, obj: ObjBase) -> np.ndarray:
        pass

    @abstractmethod
    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        pass

    @property
    def objs(self) -> Sequence[Artist]:
        return list()

    def update_obj(self, time: float, loc: np.ndarray) -> None:
        pass

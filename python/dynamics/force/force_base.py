"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod
from typing import Sequence, Any

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from dynamics.body.body_base import BodyBase


class ForceBase(ABC):
    _DEFAULT_SPRING_OBJ_KWARGS: dict[str, Any] = dict(linestyle="-", color="blue", linewidth=1.5)

    @property
    def objs(self) -> Sequence[Artist]:
        return list()

    def update_obj(self) -> None:
        pass

    def x_potential_energy(self, body: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return np.zeros_like(x_1d)

    @abstractmethod
    def force(self, time: float, boyd: BodyBase) -> np.ndarray:
        pass

    def add_obj(self, ax: Axes) -> None:
        pass

"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from dynamics.body.body_base import BodyBase


class ForceBase(ABC):
    _UNIT_SPRINT_CONSTANT_LINE_WIDTH: float = 1.0
    _FRICTIONAL_FORCE_STRETCH: float = 50.0

    _NUM_COILS_PER_UNIT_LEN: int = 10
    _SPRING_X_STRETCH: float = 5.0
    _NUM_PLT_POINTS: int = 1000
    _SPRING_WIDTH: float = 0.1

    @property
    def is_frictional_force(self) -> bool:
        return False

    @abstractmethod
    def force(self, time: float, boyd: BodyBase) -> np.ndarray:
        pass

    def x_potential_energy(self, body: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return np.zeros_like(x_1d)

    @abstractmethod
    def body_potential_energy(self, body: BodyBase) -> float:
        pass

    @property
    @abstractmethod
    def potential_energy(self) -> float:
        pass

    # visualization

    @property
    def objs(self) -> Sequence[Artist]:
        return list()

    def update_objs(self) -> None:
        pass

    def add_objs(self, ax: Axes) -> None:
        pass

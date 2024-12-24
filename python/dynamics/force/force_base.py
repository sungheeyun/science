"""
base class for sources of force, e.g., spring, gravity, electric or magnetic fields, etc.

"""

from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from dynamics.body.body_base import BodyBase
from dynamics.body.bodies import Bodies


class ForceBase(ABC):

    _NUM_PLT_POINTS_PER_COIL: int = 10

    @property
    def is_frictional_force(self) -> bool:
        return False

    @abstractmethod
    def register_force(self, bodies: Bodies) -> None:
        pass

    @abstractmethod
    def force(self, time: float, boyd: BodyBase) -> np.ndarray:
        pass

    # potential energy

    @abstractmethod
    def min_energy_matrices(self, bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

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

    @property
    def updated_objs(self) -> Sequence[Artist]:
        return list()

    def update_objs(self) -> None:
        pass

    def add_objs(self, ax: Axes) -> None:
        pass

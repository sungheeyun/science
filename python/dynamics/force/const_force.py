"""
constant force
"""

import numpy as np

from dynamics.force.force_base import ForceBase


class ConstForce(ForceBase):
    def __init__(self, force: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._force: np.ndarray = np.array(force, float)

    def force(self, time: float, loc: np.ndarray) -> np.ndarray:
        return self._force

    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        return -self._force[0] * x_1d

"""
collection of force sources
"""

import numpy as np

from dynamics.force.force_base import ForceBase


class Forces(ForceBase):
    def __init__(self, *forces: ForceBase):
        self._forces: tuple[ForceBase, ...] = forces

    def force(self, time: float, loc: np.ndarray) -> np.ndarray:
        return np.vstack([force.force(time, loc) for force in self._forces]).sum(axis=0)

    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        return np.vstack([force.x_potential_energy(x_1d) for force in self._forces]).sum(axis=0)

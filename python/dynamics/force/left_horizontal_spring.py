"""
spring
"""

import numpy as np

from dynamics.force.force_base import ForceBase


class LeftHorizontalSpring(ForceBase):
    def __init__(self, spring_constant: float, equilibrium_point: float) -> None:
        assert spring_constant > 0.0, spring_constant
        self._spring_constant: float = spring_constant
        self._equilibrium_point: float = equilibrium_point

    def force(self, time: float, loc: np.ndarray) -> np.ndarray:
        force_x: float = (
            0.0
            if loc[0] >= self._equilibrium_point
            else self._spring_constant * (self._equilibrium_point - loc[0])
        )
        return np.array([force_x, 0.0])

    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        return (
            0.5
            * self._spring_constant
            * (x_1d < self._equilibrium_point)
            * np.power(x_1d - self._equilibrium_point, 2.0)
        )

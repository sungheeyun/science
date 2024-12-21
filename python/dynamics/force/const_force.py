"""
constant force
"""

import numpy as np

from dynamics.force.force_base import ForceBase
from dynamics.body.body_base import BodyBase


class ConstForce(ForceBase):
    def __init__(self, force: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._force_vec: np.ndarray = np.array(force, float)

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        return self._force_vec

    def body_potential_energy(self, body: BodyBase) -> float:
        return -float(np.dot(self._force_vec, body.loc))

    def x_potential_energy(self, body: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return -self._force_vec[0] * x_1d

"""
constant force
"""

import numpy as np

from dynamics.bodies.bodies import Bodies
from dynamics.forces.force_base import ForceBase
from dynamics.bodies.body_base import BodyBase


class ConstForce(ForceBase):
    def __init__(self, force: np.ndarray | list[float | int] | tuple[float | int, ...]) -> None:
        self._force_vec: np.ndarray = np.array(force, float)

    def register_force(self, bodies: Bodies) -> None:
        for body in bodies.bodies:
            body.register_force(self)

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        return self._force_vec

    def body_potential_energy(self, body: BodyBase) -> float:
        return -float(np.dot(self._force_vec, body.loc))

    @property
    def potential_energy(self) -> float:
        return 0.0

    def x_potential_energy(self, body: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return -self._force_vec[0] * x_1d

"""
gravity like force, i.e., force exerting in a determined direction whose magnitude is proportional
to the mass of an object
"""

import numpy as np

from dynamics.body.bodies import Bodies
from dynamics.body.body_base import BodyBase
from dynamics.force.force_base import ForceBase


class GravityLike(ForceBase):
    def __init__(self, acceleration: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._acceleration: np.ndarray = np.array(acceleration, float)

    def attach_force(self, bodies: Bodies) -> None:
        for body in bodies.bodies:
            body.attach_force(self)

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        return body.mass * self._acceleration

    def body_potential_energy(self, body: BodyBase) -> float:
        return -body.mass * float(np.dot(self._acceleration, body.loc))

    @property
    def potential_energy(self) -> float:
        return 0.0

    def x_potential_energy(self, body: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return -body.mass * self._acceleration[0] * x_1d

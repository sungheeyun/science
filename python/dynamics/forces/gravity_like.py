"""
gravity like force, i.e., force exerting in a determined direction whose magnitude is proportional
to the mass of an object
"""

import numpy as np

from dynamics.bodies.bodies import Bodies
from dynamics.bodies.body_base import BodyBase
from dynamics.bodies.fixed_body_base import FixedBodyBase
from dynamics.forces.force_base import ForceBase


class GravityLike(ForceBase):
    def __init__(
        self, acceleration: np.ndarray | list[float | int] | tuple[float | int, ...]
    ) -> None:
        self._acceleration: np.ndarray = np.array(acceleration, float)

    @property
    def acceleration(self) -> np.ndarray:
        return self._acceleration

    def register_force(self, bodies: Bodies) -> None:
        for body in bodies.bodies:
            body.register_force(self)

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        return body.mass * self._acceleration

    # potential energy

    def body_potential_energy(self, body: BodyBase) -> float:
        return -body.mass * float(np.dot(self._acceleration, body.loc))

    @property
    def potential_energy(self) -> float:
        return 0.0

    def x_potential_energy(self, body: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return -body.mass * self._acceleration[0] * x_1d

    # potential energy solving

    def min_energy_matrices(self, bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
        num_coordinates: int = bodies.num_coordinates
        b_1d: np.ndarray = np.zeros(num_coordinates)
        for body in bodies.bodies:
            if isinstance(body, FixedBodyBase):
                continue
            for _idx, idx in enumerate(bodies.coordinate_indices(body)):
                b_1d[idx] = self.acceleration[_idx] * body.mass

        return np.zeros((num_coordinates, num_coordinates)), b_1d

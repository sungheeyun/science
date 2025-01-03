"""
base class for frictional force classes
"""

import numpy as np

from dynamics.bodies.bodies import Bodies
from dynamics.forces.force_base import ForceBase


class FrictionalForceBase(ForceBase):
    _FRICTIONAL_FORCE_STRETCH: float = 10.0

    @property
    def is_frictional_force(self) -> bool:
        return True

    def register_force(self, bodies: Bodies) -> None:
        for body in bodies.bodies:
            body.register_force(self)

    # potential energy solving

    def min_energy_matrices(self, bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((bodies.num_coordinates, bodies.num_coordinates)), np.zeros(
            bodies.num_coordinates
        )

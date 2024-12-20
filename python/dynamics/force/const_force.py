"""
constant force
"""

import numpy as np

from dynamics.force.one_body_force_base import OneBodyForceBase
from dynamics.body.body_base import BodyBase


class ConstOneBodyForce(OneBodyForceBase):
    def __init__(self, force: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._force_vec: np.ndarray = np.array(force, float)

    def _one_obj_force(self, time: float, obj: BodyBase) -> np.ndarray:
        return self._force_vec

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return -self._force_vec[0] * x_1d

"""
gravity like force, i.e., force exerting in a determined direction whose magnitude is proportional
to the mass of an object
"""

import numpy as np

from dynamics.objs.body_base import BodyBase
from dynamics.force.one_obj_force_base import OneObjForceBase


class GravityLike(OneObjForceBase):
    def __init__(self, acceleration: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._acceleration: np.ndarray = np.array(acceleration, float)

    def _one_obj_force(self, time: float, obj: BodyBase) -> np.ndarray:
        return obj.mass * self._acceleration

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return -obj.mass * self._acceleration[0] * x_1d

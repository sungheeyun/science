"""
gravity like force, i.e., force exerting in a determined direction whose magnitude is proportional
to the mass of an object
"""

import numpy as np

from dynamics.objs.obj_base import ObjBase
from dynamics.force.force_base import ForceBase


class ConstForce(ForceBase):
    def __init__(self, acceleration: np.ndarray) -> None:
        self._acceleration: np.ndarray = np.array(acceleration, float)

    def _force(self, time: float, obj: ObjBase) -> np.ndarray:
        return obj.mass * self._acceleration

    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

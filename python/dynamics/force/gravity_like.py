"""
gravity like force, i.e., force exerting in a determined direction whose magnitude is proportional
to the mass of an object
"""

import numpy as np

from dynamics.objs.obj_base import ObjBase
from dynamics.force.force_base import ForceBase


class GravityLike(ForceBase):
    def __init__(self, acceleration: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._acceleration: np.ndarray = np.array(acceleration, float)

    def _force(self, time: float, obj: ObjBase) -> np.ndarray:
        return obj.mass * self._acceleration

    def x_potential_energy(self, obj: ObjBase, x_1d: np.ndarray) -> np.ndarray:
        return -obj.mass * self._acceleration[0] * x_1d

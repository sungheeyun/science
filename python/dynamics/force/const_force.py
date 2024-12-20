"""
constant force
"""

import numpy as np

from dynamics.force.force_base import ForceBase
from dynamics.objs.obj_base import ObjBase


class ConstForce(ForceBase):
    def __init__(self, force: np.ndarray | list[float] | tuple[float, ...]) -> None:
        self._force_vec: np.ndarray = np.array(force, float)

    def _force(self, time: float, obj: ObjBase) -> np.ndarray:
        return self._force_vec

    def x_potential_energy(self, obj: ObjBase, x_1d: np.ndarray) -> np.ndarray:
        return -self._force_vec[0] * x_1d

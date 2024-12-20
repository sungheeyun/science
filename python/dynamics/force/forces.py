"""
collection of force sources
"""

from functools import reduce
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist

from dynamics.objs.body_base import BodyBase
from dynamics.force.one_obj_force_base import OneObjForceBase


class Forces(OneObjForceBase):
    def __init__(self, *forces: OneObjForceBase):
        self._forces: tuple[OneObjForceBase, ...] = forces

    def _one_obj_force(self, time: float, obj: BodyBase) -> np.ndarray:
        return np.vstack([force.one_obj_force(time, obj) for force in self._forces]).sum(axis=0)

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return np.vstack([force.x_potential_energy(obj, x_1d) for force in self._forces]).sum(
            axis=0
        )

    @property
    def objs(self) -> Sequence[Artist]:
        return reduce(
            list.__add__,  # type:ignore
            [list(force.objs) for force in self._forces if force.objs is not None],
        )

    def update_obj(self, time: float, loc: np.ndarray) -> None:
        [force.update_obj(time, loc) for force in self._forces]  # type:ignore

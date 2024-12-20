"""
collection of force sources
"""

from functools import reduce
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist

from dynamics.objs.obj_base import ObjBase
from dynamics.force.force_base import ForceBase


class Forces(ForceBase):
    def __init__(self, *forces: ForceBase):
        self._forces: tuple[ForceBase, ...] = forces

    def _force(self, time: float, obj: ObjBase) -> np.ndarray:
        return np.vstack([force.force(time, obj) for force in self._forces]).sum(axis=0)

    def x_potential_energy(self, x_1d: np.ndarray) -> np.ndarray:
        return np.vstack([force.x_potential_energy(x_1d) for force in self._forces]).sum(axis=0)

    @property
    def objs(self) -> Sequence[Artist]:
        return reduce(
            list.__add__,  # type:ignore
            [list(force.objs) for force in self._forces if force.objs is not None],
        )

    def update_obj(self, time: float, loc: np.ndarray) -> None:
        [force.update_obj(time, loc) for force in self._forces]  # type:ignore

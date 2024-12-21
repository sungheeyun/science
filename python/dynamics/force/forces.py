"""
collection of force sources
"""

from functools import reduce
from typing import Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from dynamics.body.body_base import BodyBase
from dynamics.force.force_base import ForceBase


class Forces:
    def __init__(self, *forces: ForceBase):
        self._forces: tuple[ForceBase, ...] = forces

    def add_objs(self, ax: Axes):
        for force in self._forces:
            force.add_objs(ax)

    @property
    def forces(self) -> tuple[ForceBase, ...]:
        return self._forces

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        return np.vstack([force.force(time, body) for force in self._forces]).sum(axis=0)

    @property
    def potential_energy(self) -> float:
        return sum([force.potential_energy for force in self._forces])

    # visualization

    @property
    def objs(self) -> Sequence[Artist]:
        return reduce(list.__add__, [list(force.objs) for force in self._forces])  # type:ignore

    def update_objs(self) -> None:
        for force in self._forces:
            force.update_objs()

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return np.vstack([force.x_potential_energy(obj, x_1d) for force in self._forces]).sum(
            axis=0
        )

"""
collection of force sources
"""

from functools import reduce
from typing import Sequence

import numpy as np
from numpy.linalg import solve
from matplotlib.artist import Artist
from matplotlib.axes import Axes

from dynamics.body.body_base import BodyBase
from dynamics.body.bodies import Bodies
from dynamics.force.force_base import ForceBase


class Forces:
    def __init__(self, *forces: ForceBase):
        self._forces: tuple[ForceBase, ...] = forces

    def add_objs(self, ax: Axes):
        for force in self._forces:
            force.add_objs(ax)

    def register_forces(self, bodies: Bodies) -> None:
        for force in self._forces:
            force.register_force(bodies)

    @property
    def forces(self) -> tuple[ForceBase, ...]:
        return self._forces

    def force(self, time: float, body: BodyBase) -> tuple[np.ndarray, np.ndarray]:
        frictional_force: np.ndarray = np.vstack(
            [force.force(time, body) for force in self._forces if force.is_frictional_force]
        ).sum(axis=0)

        non_frictional_force: np.ndarray = np.vstack(
            [force.force(time, body) for force in self._forces if not force.is_frictional_force]
        ).sum(axis=0)

        return non_frictional_force + frictional_force, frictional_force

    # energy

    @property
    def potential_energy(self) -> float:
        return sum([force.potential_energy for force in self._forces])

    def approx_min_energy(self, bodies: Bodies) -> None:
        """
        move bodies to (approximate) min energy locations
        """
        a_2d, b_1d = self._min_energy_matrices(bodies)
        equilibrium_loc: np.ndarray = solve(a_2d, b_1d)
        bodies.set_body_locs(equilibrium_loc)

    def _min_energy_matrices(self, bodies: Bodies) -> tuple[np.ndarray, np.ndarray]:
        a_list, b_list = zip(*[force.min_energy_matrices(bodies) for force in self.forces])
        return np.array(a_list, float).sum(axis=0), np.array(b_list, float).sum(axis=0)

    # visualization

    @property
    def objs(self) -> Sequence[Artist]:
        return reduce(list.__add__, [list(force.objs) for force in self._forces])  # type:ignore

    @property
    def updated_objs(self) -> Sequence[Artist]:
        return reduce(
            list.__add__, [list(force.updated_objs) for force in self._forces]  # type:ignore
        )

    def update_objs(self) -> None:
        for force in self._forces:
            force.update_objs()

    def x_potential_energy(self, obj: BodyBase, x_1d: np.ndarray) -> np.ndarray:
        return np.vstack([force.x_potential_energy(obj, x_1d) for force in self._forces]).sum(
            axis=0
        )

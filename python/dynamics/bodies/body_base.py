"""
base class for all objects in dynamics package
"""

from typing import Any

import numpy as np
from numpy.linalg import norm

from dynamics.obj_base import ObjBase


class BodyBase(ObjBase):
    def __init__(
        self,
        mass: float | int,
        init_loc: np.ndarray | list[float | int] | tuple[float | int, ...] | None,
        init_vel: np.ndarray | list[float | int] | tuple[float | int, ...] | None,
    ) -> None:
        self._mass: float = float(mass)
        self._cur_loc: np.ndarray = (
            np.array([0, 0], float) if init_loc is None else np.array(init_loc, float)
        )
        self._cur_vel: np.ndarray = (
            np.array([0, 0], float) if init_vel is None else np.array(init_vel, float)
        )

        self._forces: list[Any] = list()
        self._dissipated_energy: float = 0.0

    def force(self, time: float) -> tuple[np.ndarray, np.ndarray]:
        frictional_force_list: list[np.ndarray] = [
            force.force(time, self) for force in self._forces if force.is_frictional_force
        ]
        frictional_force: np.ndarray = (
            np.vstack(frictional_force_list).sum(axis=0)
            if frictional_force_list
            else np.zeros_like(self.loc)
        )

        non_frictional_force_list: list[np.ndarray] = [
            force.force(time, self) for force in self._forces if not force.is_frictional_force
        ]

        non_frictional_force: np.ndarray = (
            np.vstack(non_frictional_force_list).sum(axis=0)
            if non_frictional_force_list
            else np.zeros(self.loc)
        )

        return non_frictional_force + frictional_force, frictional_force

    def register_force(self, force: Any) -> None:
        self._forces.append(force)

    def update(self, t_1: float, t_2: float, forces: Any) -> None:
        ori_loc: np.ndarray = self.loc.copy()
        ori_vel: np.ndarray = self.vel.copy()

        force_1, frictional_force_1 = self.force((t_1 + t_2) / 2.0)
        self._cur_loc += (t_2 - t_1) * self.vel
        force_2, frictional_force_2 = self.force((t_1 + t_2) / 2.0)

        force: np.ndarray = (force_1 + force_2) / 2.0
        frictional_force: np.ndarray = (frictional_force_1 + frictional_force_2) / 2.0
        self._cur_vel += (t_2 - t_1) * force / self.mass

        avg_vel: np.ndarray = (ori_vel + self.vel) / 2.0
        self._cur_loc = ori_loc + (t_2 - t_1) * avg_vel

        self._dissipated_energy += -np.dot(frictional_force, self.vel) * (t_2 - t_1)

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def loc(self) -> np.ndarray:
        return self._cur_loc

    @loc.setter
    def loc(self, value: np.ndarray) -> None:
        assert self.loc.shape == value.shape, (self.loc.shape, value.shape)
        self._cur_loc = value

    @property
    def vel(self) -> np.ndarray:
        return self._cur_vel

    @vel.setter
    def vel(self, value: np.ndarray) -> None:
        assert self.vel.shape == value.shape, (self.vel.shape, value.shape)
        self._cur_vel = value

    @property
    def loc_text(self):
        return "(" + ", ".join(f"{loc:.2f}" for loc in self.loc) + ")"

    @property
    def vel_text(self):
        return "(" + ", ".join(f"{vel:.2f}" for vel in self.vel) + ")"

    @property
    def kinetic_energy(self) -> float:
        return 0.5 * self.mass * norm(self.vel).item() ** 2.0

    def body_potential_energy(self, forces: Any) -> float:
        return sum([force.body_potential_energy(self) for force in forces._forces])

    @property
    def dissipated_energy(self) -> float:
        return self._dissipated_energy

    @property
    def momentum(self) -> np.ndarray:
        return self.mass * self.vel

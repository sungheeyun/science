"""
electric-force-like force
"""

import numpy as np
from numpy.linalg import norm

from dynamics.bodies.body_base import BodyBase
from dynamics.forces.force_base import ForceBase


class ElectricForceLike(ForceBase):
    def __init__(
        self,
        coefficient: float | int,
        exponent: float | int,
        threshold: float | int,
        body_1: BodyBase,
        body_2: BodyBase,
    ) -> None:
        assert exponent >= 1.0, exponent
        assert threshold > 0.0, threshold
        self._coefficient: float = float(coefficient)
        self._exponent: float = float(exponent)
        self._threshold: float = float(threshold)
        self._threshold_coefficient: float = self._threshold
        self._body_1: BodyBase = body_1
        self._body_2: BodyBase = body_2

        self._body_1.register_force(self)
        self._body_2.register_force(self)

        self._threshold_force: float = self._threshold_coefficient * np.power(
            self._threshold, -self._exponent
        )
        self._threshold_potential_energy: float = (
            (self._coefficient / (self._exponent - 1.0))
            * np.power(self._threshold, -(self._exponent - 1))
            if self._exponent > 1.0
            else -self._coefficient * np.log(self._threshold)
        )

    # dynamics simulation

    def force(self, time: float, body: BodyBase) -> np.ndarray:
        if body is self._body_1:
            return -self._second_body_force(time)
        elif body is self._body_2:
            return self._second_body_force(time)
        else:
            return np.zeros(2)

    def _second_body_force(self, time: float) -> np.ndarray:
        vec_2_1: np.ndarray = self._body_2.loc - self._body_1.loc
        dist: float = norm(vec_2_1).item()
        assert dist > 0.0, (self._body_1.loc, self._body_2.loc, vec_2_1, dist)
        return (
            (self._coefficient / np.power(dist, self._exponent + 1.0)) * vec_2_1
            if dist > self._threshold
            else (self._threshold_force / self._threshold) * vec_2_1
        )

    # potential energy

    @property
    def potential_energy(self) -> tuple[float, float]:
        dist: float = norm(self._body_1.loc - self._body_2.loc).item()
        return (
            (
                (self._coefficient / (self._exponent - 1.0))
                * np.power(dist, -(self._exponent - 1.0))
                if self._exponent > 1.0
                else -self._coefficient * np.log(dist)
            )
            if dist > self._threshold
            else (
                self._threshold_potential_energy
                - 0.5 * (self._threshold_force / self._threshold) * np.power(dist, 2.0)
                + 0.5 * (self._threshold_force / self._threshold) * np.power(self._threshold, 2.0)
            )
        ), 0.0

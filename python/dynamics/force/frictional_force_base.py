"""
base class for frictional force classes
"""

from abc import ABC

from dynamics.body.bodies import Bodies
from dynamics.force.force_base import ForceBase


class FrictionalForceBase(ForceBase, ABC):
    @property
    def is_frictional_force(self) -> bool:
        return True

    def attach_force(self, bodies: Bodies) -> None:
        for body in bodies.bodies:
            body.attach_force(self)

"""
base class for frictional force classes
"""

from abc import ABC

from dynamics.force.force_base import ForceBase


class FrictionalForceBase(ForceBase, ABC):
    @property
    def is_frictional_force(self) -> bool:
        return True

"""
base class for accessories
"""

from abc import abstractmethod

from dynamics.obj_base import ObjBase


class AccessoryBase(ObjBase):
    @abstractmethod
    def update(self, time: float) -> None:
        pass

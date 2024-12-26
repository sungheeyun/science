"""
base class for all objects, i.e., bodies and accessories
"""

from abc import ABC, abstractmethod
from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.artist import Artist


class ObjBase(ABC):

    # visualization

    @abstractmethod
    def add_objs(self, ax: Axes) -> None:
        """
        add self object to Axes
        """
        pass

    def update_objs(self) -> None:
        pass

    @property
    @abstractmethod
    def objs(self) -> Sequence[Artist]:
        """
        return objects used for self's visualization
        """
        pass

    @property
    def updated_objs(self) -> Sequence[Artist]:
        """
        return only those objects updated for every frame among all those used for self's
        visualization
        """
        return list()

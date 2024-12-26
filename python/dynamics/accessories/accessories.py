"""
collection of accessories
"""

from matplotlib.axes import Axes

from dynamics.accessories.accessory_base import AccessoryBase


class Accessories:
    def __init__(self, *args) -> None:
        self._accessories: list[AccessoryBase] = list(args)

    def update(self, time: float) -> None:
        for accessory in self._accessories:
            accessory.update(time)

        self.update_objs()

    # visualization

    def add_objs(self, ax: Axes) -> None:
        for accessory in self._accessories:
            accessory.add_objs(ax)

    def update_objs(self) -> None:
        for accessory in self._accessories:
            accessory.update_objs()

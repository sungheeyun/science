"""
fan - devised for testing alias with Nyquist criteria
"""

from typing import Any, Sequence

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from dynamics.accessories.accessory_base import AccessoryBase


class Fan(AccessoryBase):
    def __init__(
        self,
        loc: np.ndarray | list[float | int] | tuple[float | int, ...],
        radius: float | int,
        freq: float | int,
        num_blades: int = 3,
        **kwargs
    ) -> None:
        assert radius > 0.0, freq
        assert freq > 0.0, freq
        assert num_blades > 2, num_blades
        self._loc: np.ndarray = np.array(loc, float)
        self._radius: float = float(radius)
        self._freq: float = float(freq)
        self._num_blades: int = num_blades

        self._time: float = 0.0

        circ_kwargs: dict[str, Any] = dict(color="black", fill=False)
        circ_kwargs.update(kwargs)

        # visualization
        assert len(self._loc) == 2, (self._loc, len(self._loc))
        self._circle: Circle = Circle((self._loc[0], self._loc[1]), self._radius, **circ_kwargs)

        blade_coordinates: np.ndarray = self._blade_coordinates
        self._blade_list: list[Line2D] = [
            Line2D(
                xdata=blade_coordinates[:, idx, 0],
                ydata=blade_coordinates[:, idx, 1],
                color=circ_kwargs["color"],
            )
            for idx in range(num_blades)
        ]
        self._objs: list[Artist] = self._blade_list + [self._circle]

    # simulation
    def update(self, time: float) -> None:
        self._time = time

    # visualization
    def add_objs(self, ax: Axes) -> None:
        ax.add_patch(self._circle)
        for blade in self._blade_list:
            ax.add_artist(blade)

    @property
    def objs(self) -> Sequence[Artist]:
        return self._objs

    def update_objs(self) -> None:
        blade_coordinates: np.ndarray = self._blade_coordinates
        for idx, blade in enumerate(self._blade_list):
            blade.set_data(blade_coordinates[:, idx].T)

    @property
    def _blade_coordinates(self) -> np.ndarray:
        theta_1d: np.ndarray = (2 * np.pi) * (
            np.linspace(0, 1, self._num_blades, endpoint=False) + self._time * self._freq
        )
        end_1d: np.ndarray = (
            self._radius * np.vstack((np.cos(theta_1d), np.sin(theta_1d))).T + self._loc
        )
        center_1d: np.ndarray = np.zeros_like(end_1d) + self._loc
        return np.array([center_1d, end_1d])

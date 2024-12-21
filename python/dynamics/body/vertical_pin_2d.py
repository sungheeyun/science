"""
fixed vertical wall
"""

from typing import Sequence, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.artist import Artist

from dynamics.body.body_base import BodyBase


class VerticalPin2D(BodyBase):
    _PIN_RADIUS: float = 0.05
    _PIN_COLOR: str = "red"

    def __init__(
        self,
        loc: np.ndarray | list[float] | tuple[float, ...],
        plt_kwargs: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(np.inf, loc, None)
        circ_kwargs: dict[str, Any] = dict(
            radius=self._PIN_RADIUS, color=self._PIN_COLOR, fill=True
        )
        if plt_kwargs is not None:
            circ_kwargs.update(**plt_kwargs)

        self._circle: Circle = Circle(tuple(self.loc), **circ_kwargs)

    def update(self, t_1: float, t_2: float, forces: Any) -> None:
        pass

    @property
    def kinetic_energy(self) -> float:
        return 0.0

    # visualization

    def add_objs(self, ax: Axes) -> None:
        ax.add_patch(self._circle)

    def update_obj(self) -> None:
        pass

    @property
    def objs(self) -> Sequence[Artist]:
        return [self._circle]

"""
fixed vertical wall
"""

from typing import Sequence, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.artist import Artist

from dynamics.bodies.fixed_body_base import FixedBodyBase


class VerticalPin2D(FixedBodyBase):
    _PIN_RADIUS: float = 0.05
    _PIN_COLOR: str = "red"

    def __init__(
        self, loc: np.ndarray | list[float | int] | tuple[float | int, ...], **kwargs
    ) -> None:
        super().__init__(0.0, loc, None)
        circ_kwargs: dict[str, Any] = dict(
            radius=self._PIN_RADIUS, color=self._PIN_COLOR, fill=True
        )
        circ_kwargs.update(**kwargs)

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

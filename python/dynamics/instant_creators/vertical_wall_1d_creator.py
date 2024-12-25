"""
1d vertical wall creator instantiating VerticalWall1D from user-entered input data

"""

from dynamics.bodies.vertical_wall_1d import VerticalWall1D
from dynamics.instant_creators.constants import Constants


class VerticalWall1DCreator:
    @staticmethod
    def create(
        data: dict[str, str | float | int], constants: Constants
    ) -> tuple[str, VerticalWall1D]:
        _data: dict[str, str | float | int] = data.copy()
        _id: str = _data.pop("id")  # type:ignore
        position: float | int = _data.pop("position")  # type:ignore
        return _id, VerticalWall1D(position, **_data)

"""
2d vertical pin creator instantiating VerticalPin2D from user-entered input data

"""

from dynamics.bodies.vertical_pin_2d import VerticalPin2D


class VerticalPin2DCreator:
    @staticmethod
    def create(data: dict[str, str | list[float | int]]) -> tuple[str, VerticalPin2D]:
        _data: dict[str, str | list[float | int]] = data.copy()
        _id: str = _data.pop("id")  # type:ignore
        position: list[float | int] = _data.pop("position")  # type:ignore
        return _id, VerticalPin2D(position, **_data)

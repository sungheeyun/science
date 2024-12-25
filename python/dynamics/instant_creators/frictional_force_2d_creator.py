"""
2d vertical pin creator instantiating VerticalPin2D from user-entered input data

"""

from dynamics.forces.frictional_force_2d import FrictionalForce2D
from dynamics.instant_creators.constants import Constants


class FrictionalForce2DCreator:
    @staticmethod
    def create(
        data: dict[str, str | float | int | list[float | int]],
        constants: Constants,
    ) -> FrictionalForce2D:
        _data: dict[str, str | float | int | list[float | int]] = data.copy()
        _data.pop("id", None)
        coefficient_of_friction: float | int = constants.value(
            _data.pop("coefficient_of_friction")  # type:ignore
        )
        upper_right_point: list[float | int] = _data.pop("upper_right_point")  # type:ignore
        return FrictionalForce2D(coefficient_of_friction, upper_right_point, **_data)

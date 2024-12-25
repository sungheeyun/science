"""
1d horizontal frictional force creator
instantiating HorizontalFrictionalForce1D from user-entered input data

"""

from dynamics.forces.horizontal_frictional_force_1d import HorizontalFrictionalForce1D
from dynamics.instant_creators.constants import Constants


class HorizontalFrictionalForce1DCreator:
    @staticmethod
    def create(
        data: dict[str, str | float | int],
        constants: Constants,
    ) -> HorizontalFrictionalForce1D:
        _data: dict[str, str | float | int] = data.copy()
        _data.pop("id", None)
        coefficient_of_friction: float | int = constants.value(
            _data.pop("coefficient_of_friction")  # type:ignore
        )
        boundary: float | int = constants.value(_data.pop("boundary"))  # type:ignore
        return HorizontalFrictionalForce1D(coefficient_of_friction, boundary, **_data)

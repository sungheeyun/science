"""
Fan creator instantiating Fan from user-entered input data

"""

from dynamics.accessories.fan import Fan
from dynamics.instant_creators.constants import Constants


class FanCreator:
    @staticmethod
    def create(
        data: dict[str, str | int | float | list[float | int]],
        constants: Constants,
    ) -> Fan:
        _data: dict[str, str | int | float | list[float | int]] = data.copy()

        _data.pop("id", None)
        center: list[float | int] = constants.value(_data.pop("center"))  # type:ignore
        radius: float | int = constants.value(_data.pop("radius"))  # type:ignore
        freq: float | int = constants.value(_data.pop("freq"))  # type:ignore
        num_blades: int = constants.value(_data.pop("num_blades"))  # type:ignore

        return Fan(center, radius, freq, num_blades, **_data)

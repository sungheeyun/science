"""
rigid ball creator instantiating RigidBall from user-entered input data

"""

from dynamics.bodies.point_mass import PointMass
from dynamics.instant_creators.constants import Constants


class PointMassCreator:
    @staticmethod
    def create(
        data: dict[str, str | int | float | list[float | int]], constants: Constants
    ) -> tuple[str, PointMass]:
        _data: dict[str, str | int | float | list[float | int]] = data.copy()
        _id: str = _data.pop("id")  # type:ignore
        mass: float | int = constants.value(_data.pop("mass"))  # type:ignore
        position: list[float | int] | None = _data.pop("position", None)  # type:ignore
        velocity: list[float | int] | None = _data.pop("velocity", None)  # type:ignore
        return _id, PointMass(mass, position, velocity, **_data)

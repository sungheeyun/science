"""
rigid ball creator instantiating RigidBall from user-entered input data

"""

from dynamics.body.rigid_ball import RigidBall
from dynamics.instant_creators.constants import Constants


class RigidBallCreator:
    @staticmethod
    def create(
        data: dict[str, str | int | float | list[float | int]], constants: Constants
    ) -> tuple[str, RigidBall]:
        _data: dict[str, str | int | float | list[float | int]] = data.copy()
        _id: str = _data.pop("id")  # type:ignore
        mass: float | int = constants.value(_data.pop("mass"))  # type:ignore
        position: list[float | int] | None = _data.pop("position", None)  # type:ignore
        velocity: list[float | int] | None = _data.pop("velocity", None)  # type:ignore
        return _id, RigidBall(mass, position, velocity, **_data)

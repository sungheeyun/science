"""
Spring creator instantiating Spring from user-entered input data

"""

from dynamics.bodies.body_base import BodyBase
from dynamics.forces.spring import Spring
from dynamics.instant_creators.constants import Constants


class SpringCreator:
    @staticmethod
    def create(
        data: dict[str, str | int | float],
        id_body_map: dict[str, BodyBase],
        constants: Constants,
    ) -> Spring:
        _data: dict[str, str | int | float] = data.copy()

        _data.pop("id", None)
        spring_constant: float | int = constants.value(_data.pop("spring_constant"))  # type:ignore
        natural_length: float | int = constants.value(_data.pop("natural_length"))  # type:ignore
        body_id_1, body_id_2 = _data.pop("bodies")  # type:ignore
        body_1: BodyBase = id_body_map[body_id_1]  # type:ignore
        body_2: BodyBase = id_body_map[body_id_2]  # type:ignore

        return Spring(spring_constant, natural_length, body_1, body_2, **_data)

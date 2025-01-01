"""
electric-force-like force creator instantiating ElectricForceLike from user-entered input data

"""

from dynamics.forces.electric_force_like import ElectricForceLike
from dynamics.bodies.body_base import BodyBase
from dynamics.instant_creators.constants import Constants


class ElectricForceLikeCreator:
    @staticmethod
    def create(
        data: dict[str, str | int | float],
        id_body_map: dict[str, BodyBase],
        constants: Constants,
    ) -> ElectricForceLike:
        _data: dict[str, str | float | int] = data.copy()

        _data.pop("id", None)

        coefficient: float | int = constants.value(_data.pop("coefficient"))
        exponent: float | int = constants.value(_data.pop("exponent"))
        threshold: float | int = constants.value(_data.pop("threshold"))

        body_id_1: str
        body_id_2: str
        body_id_1, body_id_2 = _data.pop("bodies")  # type: ignore
        body_1: BodyBase = id_body_map[body_id_1]
        body_2: BodyBase = id_body_map[body_id_2]

        assert len(_data) == 0, _data

        return ElectricForceLike(coefficient, exponent, threshold, body_1, body_2)

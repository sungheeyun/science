"""
2d vertical pin creator instantiating VerticalPin2D from user-entered input data

"""

from dynamics.forces.gravity_like import GravityLike


class GravityLikeCreator:
    @staticmethod
    def create(
        data: dict[str, list[float | int]],
    ) -> GravityLike:
        _data: dict[str, list[float | int]] = data.copy()
        _data.pop("id", None)
        acceleration: list[float | int] = _data.pop("acceleration")  # type:ignore
        return GravityLike(acceleration)

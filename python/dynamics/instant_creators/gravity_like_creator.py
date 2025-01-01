"""
gravity-like force creator instantiating GravityLike from user-entered input data

"""

from dynamics.forces.gravity_like import GravityLike


class GravityLikeCreator:
    @staticmethod
    def create(data: dict[str, str | list[float | int]]) -> GravityLike:
        _data: dict[str, str | list[float | int]] = data.copy()
        _data.pop("id", None)
        acceleration: list[float | int] = _data.pop("acceleration")  # type:ignore

        assert len(_data) == 0, _data

        return GravityLike(acceleration)

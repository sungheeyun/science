"""
non-sticky left horizontal spring creator
instantiating NonStickyLeftHorizontalSpring from user-entered input data
"""

from dynamics.forces.non_sticky_left_horizontal_spring import NonStickyLeftHorizontalSpring
from dynamics.instant_creators.constants import Constants


class NonStickyLeftHorizontalSpringCreator:
    @staticmethod
    def create(
        data: dict[str, str | float | int], constants: Constants
    ) -> NonStickyLeftHorizontalSpring:
        _data: dict[str, str | float | int] = data.copy()

        _data.pop("id", None)
        spring_constant: float | int = constants.value(_data.pop("spring_constant"))  # type:ignore
        equilibrium_point: float | int = constants.value(_data.pop("equilibrium_point"))

        return NonStickyLeftHorizontalSpring(spring_constant, equilibrium_point, **_data)

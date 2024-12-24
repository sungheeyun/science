"""
base class for all types of springs
"""

from dynamics.force.force_base import ForceBase


class SpringBase(ForceBase):
    _SPRING_UNIT_CONSTANT_LINE_WIDTH: float = 1.0
    _SPRING_NUM_COILS_PER_UNIT_LEN: int = 15
    _SPRING_MIN_NUM_COILS: int = 10
    _SPRING_WIDTH: float = 0.2

    def __init__(self, spring_constant: float | int) -> None:
        assert spring_constant > 0.0, spring_constant
        self._spring_constant: float = float(spring_constant)

    # getters

    @property
    def spring_constant(self) -> float:
        return self._spring_constant

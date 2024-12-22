"""
base class for all types of springs
"""

from dynamics.force.force_base import ForceBase


class SpringBase(ForceBase):
    def __init__(self, spring_constant: float | int) -> None:
        assert spring_constant > 0.0, spring_constant
        self._spring_constant: float = float(spring_constant)

    # getters

    @property
    def spring_constant(self) -> float:
        return self._spring_constant

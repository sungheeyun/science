"""
store and retrieve constant values
"""

from logging import Logger, getLogger

logger: Logger = getLogger()


class Constants:
    def __init__(self) -> None:
        self._name_value_map: dict[str, float | int] = dict()

    def assign(self, name: str, value: str) -> None:
        if name in self._name_value_map:
            logger.warning(f"The value of a constant named '{name}' is overwritten.")

        self._name_value_map[name] = eval(value)

    def value(self, name_or_value: str | float | int) -> float | int:
        if isinstance(name_or_value, str):
            try:
                return self._name_value_map[name_or_value]
            except KeyError:
                return eval(name_or_value)
        return name_or_value

import math

from ghoshell_moss_contrib.prototypes.ros2_robot.abcd import JointValueParser

__all__ = ["DegreeToRadiansParser", "default_parsers"]


class DegreeToRadiansParser(JointValueParser):
    """
    角度对弧度的转换器.
    """

    @classmethod
    def name(cls) -> str:
        return "degree_to_radians"

    def from_value_to_position(self, value: float) -> float:
        return round(math.radians(value), 5)

    def from_position_to_value(self, position: float) -> float:
        return round(math.degrees(position), 5)


default_parsers = {
    DegreeToRadiansParser.name(): DegreeToRadiansParser(),
}

import logging
from abc import ABC, abstractmethod
from typing import Optional

import yaml
from ghoshell_common.contracts import LoggerItf, Storage
from ghoshell_common.helpers import yaml_pretty_dump
from pydantic import ValidationError

from ghoshell_moss_contrib.prototypes.ros2_robot.abcd import JointValueParser, MOSSRobotManager
from ghoshell_moss_contrib.prototypes.ros2_robot.joint_parsers import default_parsers
from ghoshell_moss_contrib.prototypes.ros2_robot.models import (
    RobotInfo,
)


class MemoryRobotManager(MOSSRobotManager):
    def __init__(self, robot: RobotInfo, value_parsers: Optional[dict[str, JointValueParser]] = None):
        self._robot = robot
        self._value_parsers = value_parsers or default_parsers

    def robot(self) -> RobotInfo:
        return self._robot

    def joint_value_parsers(self) -> dict[str, JointValueParser]:
        return self._value_parsers

    def save_robot(self, robot: RobotInfo) -> None:
        self._robot = robot


class StorageRobotManager(MOSSRobotManager, ABC):
    def __init__(
        self,
        filename: str,
        storage: Storage,
        *,
        parsers: Optional[dict[str, JointValueParser]] = None,
        default_robot: RobotInfo | None = None,
        logger: LoggerItf | None = None,
    ):
        self._storage = storage
        self._filename = filename
        self._logger = logger or logging.getLogger(__name__)
        self._default_robot = default_robot
        self._parsers = parsers or default_parsers

    def robot(self) -> RobotInfo:
        if self._storage.exists(self._filename):
            content = self._storage.get(self._filename)
            robot = self._unmarshal_robot(content)
            if robot is not None:
                return robot
        if not self._default_robot:
            raise FileNotFoundError(f"robot file {self._filename} not found")
        return self._default_robot

    @abstractmethod
    def _unmarshal_robot(self, content: bytes) -> Optional[RobotInfo]:
        pass

    @abstractmethod
    def _marshal_robot(self, robot: RobotInfo) -> bytes:
        pass

    def joint_value_parsers(self) -> dict[str, JointValueParser]:
        return self._parsers

    def save_robot(self, robot: RobotInfo) -> None:
        content = self._marshal_robot(robot)
        self._storage.put(self._filename, content)


class YamlStorageRobotManager(StorageRobotManager):
    def _unmarshal_robot(self, content: bytes) -> Optional[RobotInfo]:
        try:
            data = yaml.safe_load(content)
            return RobotInfo(**data)
        except ValidationError:
            return None

    def _marshal_robot(self, robot: RobotInfo) -> bytes:
        data = robot.model_dump(exclude_none=True)
        return yaml_pretty_dump(data).encode()

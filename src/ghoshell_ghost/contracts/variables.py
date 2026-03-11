from typing import Dict, Any, Optional, Tuple, Iterable
from types import ModuleType
from pydantic import BaseModel, Field
from ghoshell_common.entity import EntityMeta, to_entity_meta, from_entity_meta, is_entity_type

__all__ = [
    'Variables',
]


class Variables(BaseModel):
    """
    上下文相关的变量讯息. 可以存储标量, 可序列化变量, 还有 pydantic 可序列化变量.
    """

    properties: Dict[str, EntityMeta] = Field(
        default_factory=dict,
    )

    def set_prop(self, name: str, value: Any):
        self.properties[name] = to_entity_meta(value)

    def get_prop(self, name: str, module: Optional[ModuleType] = None) -> Any:
        if name not in self.properties:
            return None
        value = self.properties[name]
        return from_entity_meta(value, module)

    @staticmethod
    def allow_prop(value: Any) -> bool:
        if isinstance(value, BaseModel):
            return True
        elif isinstance(value, bool):
            return True
        elif isinstance(value, str):
            return True
        elif isinstance(value, int):
            return True
        elif isinstance(value, float):
            return True
        elif isinstance(value, list):
            return True
        elif isinstance(value, dict):
            return True
        elif is_entity_type(value):
            return True
        return False

    def iter_props(self, module: Optional[ModuleType] = None) -> Iterable[Tuple[str, Any]]:
        for name in self.properties:
            value = self.properties[name]
            yield name, from_entity_meta(value, module)

    def join(self, variables: "Variables") -> "Variables":
        """
        合并两个 variables, 以右侧的为优先.
        """
        copied = self.model_copy(deep=True)
        if copied.module is None:
            copied.module = variables.module
        if copied.code is None:
            copied.code = variables.code
        if copied.execute_code is None:
            copied.execute_code = variables.execute_code
            copied.executed = variables.executed

        for key, val in variables.properties.items():
            copied.properties[key] = val
        return copied

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Dict, ClassVar, Any, Type
from ghoshell_common.helpers import generate_import_path


class State(BaseModel):
    version: str = Field(default="", description="state version, optimis lock")
    name: str = Field(description="The name of the state object.")
    description: str = Field(default="", description="The description of the state object.")
    schema: Dict[str, Any] = Field(description="the json schema of the state")
    default: Dict[str, Any] = Field(description="the default value of the state")


class StateModel(BaseModel, ABC):
    """
    通过强类型的方式对 State 进行建模.
    """
    state_desc: ClassVar[str] = ""
    state_name: ClassVar[str] = ""

    @classmethod
    def to_state(cls) -> State:
        name = cls.state_name or generate_import_path(cls)
        description = cls.state_desc or cls.__doc__ or ""
        default = cls().model_dump()
        schema = cls.model_json_schema()
        return State(name=name, description=description, schema=schema, default=default)


class StateStore(ABC):

    @abstractmethod
    def register(self, state: State) -> None:
        pass

    @abstractmethod
    def register_model(self, model: StateModel) -> None:
        pass

    @abstractmethod
    async def get(self, state_name: str) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def save_model(self, model: StateModel) -> None:
        pass

    @abstractmethod
    async def get_model(self, model: Type[StateModel]) -> StateModel:
        pass

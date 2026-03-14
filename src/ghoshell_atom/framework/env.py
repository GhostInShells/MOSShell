from typing_extensions import Self
from pydantic import BaseModel, Field


class AtomEnviron(BaseModel):
    """
    Atom 的环境变量建模设计.
    通过强类型的方式取值.
    """

    @classmethod
    def from_env(cls):
        """
        从环境变量中直接获取关键数据
        """
        from os import environ
        return cls(**environ)

from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

__all__ = ["Expressions"]


class ExpressionItem(BaseModel):
    chars: str = Field(description="expression 所使用的符号")
    description: str = Field(description="expression 对应的描述.")
    ctml: str = Field(description="expression 所对应的 ctml")


class ExpressionData(BaseModel):
    items: list[ExpressionItem] = Field(default_factory=list, description="所有已经创建的符号.")


class Expressions(ABC):
    """
    将多轨实现变成极少 token 的单轨实现的设计.
    它能注册几个表情符号, 将表情符号和 CTML 建立对应关系.
    并且提供 special tokens, 让 Interpreter 解析时自动将对应的 token 展开为完整的 CTML
    """

    @abstractmethod
    async def define_expression(self, chars: str, description: str, ctml__) -> None:
        """
        定义一个 expression 符号.

        :param chars: expression 所使用的符号. 如果和已有的重合, 会覆盖掉已有的.
        :param description: 对这个 expression 的描述. 要非常简单, 最好一个单词.
        :param ctml__: 基于 ctml 语法定义的行为逻辑.
        """
        pass

    @abstractmethod
    def data(self) -> ExpressionData:
        """
        返回完整的数据结构.
        """
        pass

    @abstractmethod
    async def read_expression(self, chars: str) -> str:
        """
        :param chars: expression 所使用的符号.
        :return: 返回 expression 的 CTML
        """
        pass

    @abstractmethod
    async def instruction(self) -> str:
        """
        说明对应关系.
        """
        pass

    @abstractmethod
    async def remove_expression(self, chars: str) -> str:
        """
        移除 expression.
        """
        pass

    @abstractmethod
    def special_tokens(self) -> dict[str, str]:
        """
        返回 expression chars 对应的 ctml
        """
        pass

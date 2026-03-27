from typing import TypedDict, Required, Optional, Any
from typing_extensions import Self
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field

__all__ = ['Content', 'ContentModel']


class Content(TypedDict, total=False):
    """
    这是你链路中流动的‘弱类型容器’
    它一定有 type，且兼容 Anthropic 的 Dict 结构
    """
    type: Required[str]
    # 使用 Any 或平铺字段，保持灵活性
    # 这样在做 content_model.from_content(dict) 时不会报错
    text: Optional[str]
    source: Optional[dict[str, Any]]

    # anthropic 不支持的数据结构.
    raw: Optional[dict[str, Any]]


class ContentModel(BaseModel, ABC):
    """
    强类型的数据结构同时兼容 Anthropic.
    """
    source: Optional[dict[str, Any]] = Field(
        default=None,
        description="the content source type"
    )

    @classmethod
    @abstractmethod
    def content_type(cls) -> str:
        """
        content type of the model
        """
        pass

    def to_content(self) -> Content:
        content = Content(type=self.content_type())
        raw = self.model_dump(exclude_none=True, exclude={'type', 'source'})
        if raw:
            content['raw'] = raw
        if self.source is not None:
            content['source'] = self.source
        return content

    @classmethod
    def from_content(cls, content: Content) -> Self | None:
        if cls.content_type() != content['type']:
            return None
        source = content.get('source')
        raw = content.get('raw') or {}
        raw['source'] = source
        return cls.model_construct(**raw)

    def to_anthropic(self) -> dict[str, Any]:
        """
        真正输出给 Anthropic API 的形态。
        将 raw 中的字段提升到顶层，以符合 Anthropic 的 Content Block Schema。
        """
        # 1. 拿到基础内容
        content = self.to_content()

        # 2. 准备输出字典
        out: dict[str, Any] = dict(type=content["type"])

        # 3. 提取 source (Image 等多模态需要)
        source = content.get("source")
        if source is not None:
            out["source"] = content["source"]

        # 4. 提取 text (Text 类型需要)
        text = content.get("text")
        if text is not None:
            out["text"] = content["text"]
        return out

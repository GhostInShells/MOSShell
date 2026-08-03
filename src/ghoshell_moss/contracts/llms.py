"""LLM provider contract — model configuration, client protocols, and provider resolution."""

from typing import Literal, Iterable
from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from .configs import ConfigType
from ghoshell_moss.message import Message, Content
from ghoshell_common.helpers import import_from_path
from ghoshell_container import IoCContainer

__all__ = [
    "ClientProtocol",
    "ServiceConfig",
    "ModelConfig",
    "Provider",
    "ResolvedModel",
    "LLMConfig",
    "register_converter",
    "clear_converters",
    "MessageContentConverter",
]

# gemini 这么没牌面吗?
ClientProtocol = Literal["anthropic", "openai"]
ModelTag = str
ModelName = str
DefaultModelTag = Literal['small_fast_model', 'flash', 'pro']


class MessageContentConverter(ABC):

    @abstractmethod
    def convert(
            self,
            container: IoCContainer,
            content: Content,
    ) -> Iterable[Content]:
        pass


_converters: dict[str, MessageContentConverter | None] = {}


def register_converter(import_path: str, converter: MessageContentConverter | None) -> None:
    """注册 content converter 实例，用于测试或运行时注入。

    import_path 需与 ModelConfig.converters 中配置的路径一致。
    """
    _converters[import_path] = converter


def clear_converters() -> None:
    """清空所有已注册的 converter 缓存。"""
    _converters.clear()


class ModelConfig(BaseModel):
    """
    单个模型的配置。
    """

    model: ModelName = Field(
        default="$ANTHROPIC_MODEL",
        description="模型名，如 claude-opus-4-7",
    )
    description: str = Field(
        default="",
        description="human-readable description — used by list displays and Ghost model-selection channels",
    )
    tags: dict[ModelTag, ModelName] = Field(
        default_factory=dict,
    )
    context_window: int = Field(
        default=200000,
        description="上下文窗口大小 (tokens)",
    )
    max_output_tokens: int = Field(
        default=4096,
        description="最大输出 tokens",
    )
    content_types: list[str] = Field(
        default=["text"],
        description="支持的 Content.type 列表。'*' 表示接受所有类型，不做过滤。",
    )
    converters: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "content 转换器的 import path，格式 'pkg.module:attr'。"
            "发 prompt 前，不支持的 content type 依次走：converter 适配 → 文本退化。"
            "转换器返回 Content 表示使用转换结果；"
            "未配置或无结果时降级为文本表示 (Message.content_as_string)。"
        ),
    )

    def unwrap_tag(self, model_tag: str) -> 'ModelConfig':
        if not model_tag or model_tag not in self.tags:
            return self
        model_name = self.tags[model_tag]
        tags = dict(self.tags)
        tags['default'] = model_name
        return self.model_copy(
            update={'model': model_name, 'tags': tags},
        )

    def accepts(self, content: Content) -> bool:
        """检查当前模型是否支持某个 content type。'*' 表示通配所有。"""
        if "*" in self.content_types:
            return True
        return content['type'] in self.content_types

    def convert(self, container: IoCContainer, message: Message) -> Message:
        """转义 message：原生支持 → converter 适配 → 文本退化。"""
        new_message = Message(meta=message.meta)
        for content in message.contents:
            if self.accepts(content):
                new_message.contents.append(content)
            else:
                converted = list(self.convert_content(container, content))
                if converted:
                    for c in converted:
                        new_message.with_content(c)
                else:
                    # 降级为文本表示，确保模型至少能"读到"内容
                    degraded = Message.content_as_string(content)
                    if degraded:
                        new_message.with_content(
                            Content(type="text", text=degraded)
                        )
        return new_message

    def convert_content(self, container: IoCContainer, content: Content) -> Iterable[Content]:
        """转义 content. """
        global _converters
        content_type = content['type']
        if content_type not in self.converters:
            yield from []
            return
        converter_import_path = self.converters.get(content_type)
        if converter_import_path in _converters:
            cached_converter = _converters[converter_import_path]
            if cached_converter is None:
                yield from []
                return
            else:
                yield from cached_converter.convert(container, content)
                return
        try:
            converter = import_from_path(converter_import_path)
            if isinstance(converter, MessageContentConverter):
                yield from converter.convert(container, content)
                return
        except Exception:
            pass
        # 缓存 converter import path, 避免重复导入.
        _converters[converter_import_path] = None
        yield from []
        return


class ServiceConfig(BaseModel):
    """
    单个 LLM 服务的连接配置.
    仅提供配置项.
    """

    name: str = Field(description="服务名，如 deepseek / anthropic / openai")
    base_url: str = Field(
        default="$ANTHROPIC_BASE_URL",
        description="API base URL"
    )
    api_key: str = Field(
        default="$ANTHROPIC_API_KEY",
        description="API key，以 $ 开头从环境变量读取",
    )
    protocol: ClientProtocol = Field(
        default="anthropic",
        description="API 协议类型",
    )


class Provider(BaseModel):
    """模型供应商。一个服务 + 它的模型阵容。"""
    service: ServiceConfig = Field(
        description="所属的服务",
    )
    default: ModelConfig = Field(
        description="default model",
    )
    models: dict[str, ModelConfig] = Field(
        default_factory=dict,
        description="其它的模型.",
    )

    def get_model(self, name: str, tag: ModelTag | None = None) -> ModelConfig:
        model = self.models.get(name, self.default)
        return model.unwrap_tag(tag)


class ResolvedModel(BaseModel):
    """模型查找结果：模型 + 到达它的服务。"""

    service: ServiceConfig
    model: ModelConfig

    @property
    def client_protocol(self) -> ClientProtocol:
        return self.service.protocol


class LLMConfig(ConfigType):
    """LLM 配置中心。存储在 workspace configs/ 目录下。"""
    default: Provider = Field(
        default_factory=lambda: Provider(
            service=ServiceConfig(
                name='anthropic',
                base_url='$ANTHROPIC_BASE_URL',
                api_key='$ANTHROPIC_API_KEY',
                protocol='anthropic',
            ),
            default=ModelConfig(
                model="$ANTHROPIC_MODEL",
                description="Default Anthropic model — general-purpose, multimodal (text + image)",
                tags={
                    'small_fast_model': "$ANTHROPIC_SMALL_FAST_MODEL",
                },
            )
        ),
    )
    providers: dict[str, Provider] = Field(
        default_factory=lambda: dict(
            deepseek=Provider(
                service=ServiceConfig(
                    name='deepseek',
                    base_url='$DEEPSEEK_ANTHROPIC_BASE_URL',
                    api_key='$DEEPSEEK_API_KEY',
                    protocol='anthropic',
                ),
                default=ModelConfig(
                    model="$DEEPSEEK_MODEL",
                    description="DeepSeek via Anthropic-compatible protocol — cost-efficient reasoning",
                    tags={
                        'small_fast_model': "$DEEPSEEK_SMALL_FAST_MODEL",
                    },
                )
            ),
            deepseek_openai=Provider(
                service=ServiceConfig(
                    name='deepseek_openai',
                    base_url='$DEEPSEEK_OPENAI_BASE_URL',
                    api_key='$DEEPSEEK_API_KEY',
                    protocol='openai',
                ),
                default=ModelConfig(
                    model="$DEEPSEEK_MODEL",
                    description="DeepSeek via OpenAI-compatible protocol — for OpenAI client testing",
                    tags={
                        'small_fast_model': "$DEEPSEEK_SMALL_FAST_MODEL",
                    },
                )
            ),
        )
    )

    @classmethod
    def conf_name(cls) -> str:
        return 'llms'

    def get_model(
            self,
            provider: str = "",
            model: str = "",
            tag: ModelTag | None = None,
            *,
            no_fallback: bool = False,
    ) -> ResolvedModel:
        """获取模型配置。

        零参数：返回默认 provider 的默认模型。
        provider：指定 provider 的默认模型（同时匹配 default 和 providers）。
        tag：对结果 unwrap 标签（如 small_fast_model → 实际模型名）。
        model：在所有 provider 中按模型名精确搜索。

        Ghost 运作时自己选模型：get_model(provider="deepseek", tag="pro")。
        """
        if not provider and not model:
            return self._get_default()

        if provider:
            p = self._resolve_provider(provider)
            if p is None:
                if no_fallback:
                    raise KeyError(f"Provider {provider!r} not found")
                return self._get_default()
            return ResolvedModel(
                service=p.service,
                model=p.get_model(model, tag),
            )

        # 未指定 provider，按 model 名搜索所有 provider
        if model:
            for p in self._all_providers():
                if model in p.models:
                    return ResolvedModel(
                        service=p.service,
                        model=p.get_model(model, tag),
                    )
            if no_fallback:
                raise KeyError(f"Model {model!r} not found in any provider")
            return self._get_default()

        return self._get_default()

    def list_models(self, provider: str = "") -> list[ResolvedModel]:
        """列出可用模型，可按 provider 过滤。"""
        result: list[ResolvedModel] = []
        if provider:
            p = self._resolve_provider(provider)
            providers_to_search = [p] if p else []
        else:
            providers_to_search = list(self._all_providers())

        for p in providers_to_search:
            result.append(ResolvedModel(service=p.service, model=p.default))
            for model_config in p.models.values():
                result.append(ResolvedModel(service=p.service, model=model_config))
        return result

    def get_service(self, name: str) -> ServiceConfig:
        """按名称获取服务配置。"""
        p = self._resolve_provider(name)
        if p is None:
            raise KeyError(f"Service {name!r} not found")
        return p.service

    @property
    def services(self) -> list[ServiceConfig]:
        """所有已注册的服务（去重）。"""
        seen: set[str] = set()
        result: list[ServiceConfig] = []
        for p in self._all_providers():
            if p.service.name not in seen:
                seen.add(p.service.name)
                result.append(p.service)
        return result

    def _resolve_provider(self, name: str) -> Provider | None:
        """按名称查找 provider，同时检查 default 和 providers。"""
        if self.default.service.name == name:
            return self.default
        return self.providers.get(name)

    def _all_providers(self):
        """所有 provider 的迭代器：default 优先。"""
        yield self.default
        yield from self.providers.values()

    def _get_default(self) -> ResolvedModel:
        return ResolvedModel(
            service=self.default.service,
            model=self.default.default,
        )

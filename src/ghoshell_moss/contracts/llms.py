import importlib
from typing import Callable, ClassVar, Literal
from pydantic import BaseModel, Field
from .configs import ConfigType

__all__ = [
    "ApiType",
    "ModelType",
    "LLMServiceConfig",
    "LLMModelConfig",
    "LLMModelAndService",
    "LLMConfig",
    "load_converter",
]

ApiType = Literal["anthropic", "openai"]
ModelType = Literal["default", "pro", "flash"]


class LLMServiceConfig(BaseModel):
    """单个 LLM 服务的连接配置。"""

    name: str = Field(description="服务名，如 deepseek / anthropic / openai")
    base_url: str = Field(description="API base URL")
    api_key: str = Field(
        default="$ANTHROPIC_API_KEY",
        description="API key，以 $ 开头从环境变量读取",
    )
    api_type: ApiType = Field(
        default="anthropic",
        description="API 协议类型",
    )


class LLMModelConfig(BaseModel):
    """单个模型的配置。"""

    MODEL_TYPE_DEFAULT: ClassVar[ModelType] = "default"
    MODEL_TYPE_PRO: ClassVar[ModelType] = "pro"
    MODEL_TYPE_FLASH: ClassVar[ModelType] = "flash"

    model: str = Field(description="模型名，如 claude-opus-4-7")
    service: str = Field(description="所属服务名，引用 LLMServiceConfig.name")
    model_type: ModelType | str = Field(
        default="default",
        description="模型类型标签，用于降级匹配。预定义: default/pro/flash，可扩展",
    )
    context_window: int = Field(
        default=200000,
        description="上下文窗口大小 (tokens)",
    )
    max_output_tokens: int = Field(
        default=4096,
        description="最大输出 tokens",
    )
    protocols: list[str] = Field(
        default=["text"],
        description="支持的 Content.type 列表，如 text / image",
    )
    converter: str | None = Field(
        default=None,
        description=(
            "content 转换器的 import path，格式 'pkg.module:callable'。"
            "callable 签名: (Content) -> Content | None。"
            "发 prompt 前，protocols 不支持的 content type 走此转换器；"
            "转换器返回 None 表示丢弃该 content，返回 Content 表示使用转换结果。"
            "None（默认）表示不支持的 content 直接丢弃。"
        ),
    )

    def accepts(self, content_type: str) -> bool:
        """检查当前模型是否支持某个 content type。"""
        return content_type in self.protocols


class LLMModelAndService(BaseModel):
    """get_model() 的 joined 返回值。"""

    model: LLMModelConfig
    service: LLMServiceConfig


class LLMConfig(ConfigType):
    """LLM 配置中心。存储在 workspace configs/ 目录下。"""

    services: list[LLMServiceConfig] = Field(
        default_factory=list,
        description="可用的 LLM 服务列表",
    )
    models: list[LLMModelConfig] = Field(
        default_factory=list,
        description="可用的模型列表",
    )
    default_model: str = Field(
        default="",
        description="系统默认模型名，get_model() 无匹配时的最终降级目标",
    )

    @classmethod
    def conf_name(cls) -> str:
        return "llm"

    def get_model(
        self,
        service: str = "",
        model_type: ModelType | str = "",
        *,
        no_fallback: bool = False,
    ) -> LLMModelAndService:
        """按 service 和 model_type 查找模型。

        无筛选条件时直接返回 default_model；有筛选条件时精确匹配优先，
        无匹配降级到 default_model。no_fallback=True 时无匹配抛 KeyError。
        """
        if not service and not model_type:
            return self._get_default()

        candidates = self.models
        if service:
            candidates = [m for m in candidates if m.service == service]
        if model_type:
            candidates = [m for m in candidates if m.model_type == model_type]

        if candidates:
            return self._join(candidates[0])

        if no_fallback:
            raise KeyError(
                f"No model matched service={service!r} model_type={model_type!r}"
            )

        return self._get_default()

    def list_models(self, service: str = "") -> list[LLMModelConfig]:
        """列出可用模型，可按 service 过滤。"""
        if service:
            return [m for m in self.models if m.service == service]
        return self.models

    def list_services(self) -> list[LLMServiceConfig]:
        """列出已注册的服务。"""
        return self.services

    def get_service(self, name: str) -> LLMServiceConfig:
        """按名称获取服务配置。"""
        for s in self.services:
            if s.name == name:
                return s
        raise KeyError(f"Service {name!r} not found")

    def _join(self, model: LLMModelConfig) -> LLMModelAndService:
        return LLMModelAndService(
            model=model,
            service=self.get_service(model.service),
        )

    def _get_default(self) -> LLMModelAndService:
        if not self.default_model:
            raise ValueError("No default_model configured")
        for m in self.models:
            if m.model == self.default_model:
                return self._join(m)
        raise KeyError(
            f"Default model {self.default_model!r} not found in models"
        )


def load_converter(import_path: str) -> Callable:
    """从 import path 加载 content 转换器。

    import_path 格式: 'pkg.module:callable_name'
    返回的 callable 签名为 (Content) -> Content | None：
    None 表示丢弃该 content，Content 表示转换成功。

    :raises ImportError: 模块不存在
    :raises AttributeError: callable 名称不存在
    :raises ValueError: import_path 格式不正确
    :raises TypeError: 指定路径不是 callable
    """
    if ":" not in import_path:
        raise ValueError(
            f"Invalid converter import path {import_path!r}, "
            f"expected format 'pkg.module:callable_name'"
        )
    module_path, _, attr_name = import_path.partition(":")
    module = importlib.import_module(module_path)
    converter = getattr(module, attr_name)
    if not callable(converter):
        raise TypeError(
            f"Converter {import_path!r} is not callable: {type(converter)}"
        )
    return converter

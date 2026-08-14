"""LLM provider contract — model configuration, client protocols, and provider resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Iterable, Type, Callable, Generic, TypeVar, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pydantic import BaseModel, Field, AwareDatetime
from .configs import ConfigType
from ghoshell_moss.anchor import Anchor
from ghoshell_common.helpers import import_from_path
from ghoshell_container import IoCContainer
from pathlib import Path

if TYPE_CHECKING:
    from ghoshell_moss.message import Content, Message

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
    "Effort",
    "TokenCount",
]

# gemini 这么没牌面吗?
ClientProtocol = Literal["anthropic", "openai"]
ModelTag = str
ModelName = str
DefaultModelTag = Literal['small_fast_model', 'flash', 'pro']
# thinking effort 刻度 — no..max。引擎按协议映射到 pydantic-ai
# (anthropic_effort: low..max / openai_reasoning_effort: none..xhigh)。
Effort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]


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
        from ghoshell_moss.message import Content, Message

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


class ModelRef(BaseModel):
    """模型引用的无密钥同构 — 从 ResolvedModel 挑出不泄密字段投影.

    ``ResolvedModel`` 携带 api_key / base_url, 不可直接落盘或跨进程传递。
    ``ModelRef`` 排除这两类, 保留其余字段与 ResolvedModel 同构, 可完整描述
    一个模型引用, 作为通用载体:
    - ``moss llms list`` 展示模型时的结构化标识
    - benchmark 结果溯源 (``BenchmarkRun.model``)

    运行时经 ``LLMConfig.get_model()`` 反查为 ``ResolvedModel`` 再调用,
    密钥只在内存中复活。
    """

    service: str = Field(
        description="service.name — 对应 LLMConfig provider 的 service.name",
    )
    protocol: str = Field(
        description="service.protocol — anthropic / openai",
    )
    model: str = Field(
        description="model.model — 已 resolve 的真实模型名",
    )
    description: str = Field(
        default="",
        description="model.description — 人类可读描述",
    )
    tags: dict[str, str] = Field(
        default_factory=dict,
        description="model.tags — 模型标签映射 (tag -> 实际模型名)",
    )
    context_window: int = Field(
        default=200000,
        description="model.context_window — 上下文窗口 (tokens)",
    )
    max_output_tokens: int = Field(
        default=4096,
        description="model.max_output_tokens — 最大输出 tokens",
    )
    content_types: list[str] = Field(
        default_factory=list,
        description="model.content_types — 原生支持的 content type 列表",
    )

    @classmethod
    def from_resolved(cls, resolved: ResolvedModel) -> 'ModelRef':
        """从解析结果构造 — 丢弃 service 的 api_key / base_url, 其余同构."""
        service = resolved.service
        model = resolved.model
        return cls(
            service=service.name,
            protocol=service.protocol,
            model=model.model,
            description=model.description,
            tags=dict(model.tags),
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            content_types=list(model.content_types),
        )

    def resolve(self, conf: LLMConfig, *, no_fallback: bool = False) -> ResolvedModel:
        """从配置中心反查为可调用的 ResolvedModel (密钥复活, 仅内存)."""
        return conf.get_model(
            provider=self.service,
            model=self.model,
            no_fallback=no_fallback,
        )


RESULT_MODEL = TypeVar('RESULT_MODEL', bound=BaseModel)


class LLMFuncResultRecord(BaseModel):
    """Model func 单次调用的弱数据结果 — 持久化与 benchmark 汇总的最小形态.

    由 ``LLMFuncResult.to_record()`` 产出。result 是结构化输出的 dict 表示
    (已展平), 入库 / 写 jsonl 不依赖具体 result_type。反向 (record → 强类型)
    意义不大 — 结构化只在内存中存活。
    """

    result: dict[str, Any] | None = Field(
        default=None,
        description="结构化输出的 dict 表示; 调用未指定 result_type 或解析失败时为 None",
    )
    content: str = Field(
        default="",
        description="模型原始文本输出; 结构化模式下模型可能仅在 tool call 中返回, 此字段可为空",
    )
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="token 开销 (标准 Usage 的 dict 表示)",
    )
    cast: float = Field(
        default=0.0,
        description="单次调用耗时 (秒)",
    )
    retries: int = Field(
        default=0,
        description="本轮调用内部的模型重试次数",
    )


class LLMFuncResult(BaseModel, Generic[RESULT_MODEL]):
    """Model func 单次调用的强类型结果.

    ``result`` 是 ``result_type`` 的实例 (RESULT_MODEL); 调用未指定结果类型时为
    None, 此时 ``content`` 承载全部输出。需要持久化时 ``to_record()`` 转为弱数据。
    """

    result: RESULT_MODEL | None = Field(
        default=None,
        description="结构化输出 (result_type 实例); 无 result_type 时为 None",
    )
    content: str = Field(
        default="",
        description="模型原始文本输出",
    )
    usage: dict[str, Any] = Field(
        default_factory=dict,
        description="token 开销",
    )
    cast: float = Field(
        default=0.0,
        description="单次调用耗时 (秒)",
    )
    retries: int = Field(
        default=0,
        description="本轮调用内部的模型重试次数",
    )
    anchor: Anchor | None = Field(
        default=None,
        description=(
            "本次调用的认知锚 (Anchor) — call(export_anchor=...) 给定文件名时产出。"
            "锚是独立持久化路径 (.anchor.yml), 不随 to_record() 进入弱数据 record。"
        ),
    )

    def to_record(self) -> LLMFuncResultRecord:
        """转为弱数据 record, 用于持久化。result 展平为 dict。锚不携带。"""
        result_dict = self.result.model_dump() if self.result is not None else None
        return LLMFuncResultRecord(
            result=result_dict,
            content=self.content,
            usage=self.usage,
            cast=self.cast,
            retries=self.retries,
        )


class BenchmarkMeta(BaseModel):
    """benchmark 元信息 — bench.md 的 YAML frontmatter 部分, 模型无关.

    声明 benchmark 在验证什么、产物结构是什么、公共 instruction 与用例文件。
    模型不在此处 — 由运行时的 ``BenchmarkRun`` 绑定, 因此同一 benchmark 可换模型重跑对比。
    """

    title: str = Field(description="benchmark 标题")
    description: str = Field(
        default="",
        description="benchmark 说明 (设计动机 / 验证目标)",
    )
    version: str = Field(
        default="v1.0.0",
        description="benchmark 版本, 用于结果溯源",
    )
    created: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="创建时间",
    )
    updated: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="更新时间",
    )
    # TODO(命名): 此字段是 benchmark 级公共 instruction, 与 case 级 instruction
    # 重名易混淆, 语义上更接近"benchmark 的说明/背景"。名称待定。
    instruction: str = Field(
        default="",
        description="benchmark 级缺省 instruction; case 未指定 instruction 时使用",
    )
    result_type: str = Field(
        description="产物结构 — module:attr 指向 BaseModel 类, 逐 case 的结构化输出类型",
    )
    cases_file: str | None = Field(
        default=None,
        description="case.jsonl 路径, 相对运行 cwd 解析",
    )

    @classmethod
    def read_from_markdown(cls, file: Path) -> 'BenchmarkMeta':
        """从 bench.md 读取: YAML frontmatter 解析为 meta, markdown 正文作为 description."""
        raise NotImplementedError('todo')

    def dump_to_markdown(self, file: Path) -> None:
        """写回 bench.md: 去掉默认值与 None, 序列化 frontmatter + 正文."""
        raise NotImplementedError('todo')


class BenchmarkCase(BaseModel):
    """单个 benchmark 用例.

    ``prompt`` / ``instruction`` 可以是字符串本身, 或是相对运行 cwd 的文件路径
    (cwd 下同目录文件约定)。``expected`` 是打分参考。
    """

    label: str = Field(description="用例唯一标识, 结果中以此关联")
    description: str = Field(
        default="",
        description="用例说明",
    )
    times: int = Field(
        default=1,
        description="本用例重复执行次数",
    )
    instruction: str = Field(
        default="",
        description="本用例 instruction (字符串或相对 cwd 的文件路径); 空则回退到 meta.instruction",
    )
    prompt: str = Field(
        description="发给模型的 prompt (字符串或相对 cwd 的文件路径)",
    )
    expected: str = Field(
        default="",
        description="期望输出, 打分标准参考",
    )
    thinking: str | None = Field(
        default=None,
        description=(
            "本用例的 thinking block (字符串或相对 cwd 的文件路径) — 内观 hint。"
            "空则回退到 run 级 thinking。策略变量: 评分 hint 可 per-case 混排"
            "(放 instruction / thinking / 省略)。"
        ),
    )
    effort: Effort | None = Field(
        default=None,
        description="本用例 thinking effort (none..max); 空则回退到 run 级 effort",
    )


class BenchmarkRun(BaseModel):
    """一次 benchmark 运行的声明 — 某模型跑某 meta.

    绑定 ``ModelRef`` (无密钥), 结果可安全持久化并溯源;
    同一 meta 换模型重跑即产生多个 run, 用于对比。
    """

    label: str = Field(description="本次运行标识")
    description: str = Field(
        default="",
        description="运行说明",
    )
    meta: BenchmarkMeta = Field(
        description="benchmark 定义 (不含模型)",
    )
    model: ModelRef = Field(
        description="本次运行绑定的模型 (无密钥引用)",
    )


class BenchmarkRecord(BaseModel):
    """benchmark 完整结果产物 — 一次运行 + 逐 case 结果.

    持久化为 jsonl: 首行 run (含 meta 与 model), 后续每行一个弱数据结果。
    """

    run: BenchmarkRun = Field(
        description="运行声明 (meta + 模型)",
    )
    results: list[LLMFuncResultRecord] = Field(
        default_factory=list,
        description="逐 case 结果, 弱数据形态",
    )

    @classmethod
    def read_from_jsonl(cls, file: Path) -> 'BenchmarkRecord':
        """从结果 jsonl 读回 BenchmarkRecord."""
        raise NotImplementedError('todo')


@dataclass(frozen=True)
class TokenCount:
    """Token 计数结果 — 结构化返回, 含编码与服务信息.

    ``count`` = ``len(encode(text))`` (tiktoken 无 count_tokens 捷径)。
    ``estimate`` 标记非 openai 协议的估算 — tiktoken 是 OpenAI 的分词器。
    ``tokens`` 默认 None, 显式要求 (``include_tokens=True``) 才物化。
    """

    count: int
    service: str = ""
    model: str = ""
    encoding: str = ""
    estimate: bool = False
    tokens: tuple[int, ...] | None = None


class LLMFuncs(ABC):
    """model func 引擎契约 — 模型调用的最小协议.

    输入字符串 (instruction + prompt), 输出结构化 BaseModel (可选 — 不指定则 content 承载原文), 单轮无状态。
    引擎无关: pydantic-ai 是首个实现, 底层 API / 未来消息引擎可替换。
    """

    @abstractmethod
    async def call(
            self,
            *,
            instruction: str,
            prompt: str,
            result_type: Type[RESULT_MODEL] | None = None,
            model: ResolvedModel,
            effort: Effort | None = None,
            export_anchor: str | Path | None = None,
            anchor_description: str = "",
            input_anchor: Anchor | None = None,
            thinking: str | None = None,
    ) -> LLMFuncResult[RESULT_MODEL]:
        """单轮模型调用: instruction + prompt -> 结构化 result_type 结果.

        ``prompt`` — 纯字符串, moss-free。moss 协议 (Message / @ 文件) 的
        prompt 走 ``MossLLMFuncs`` (``call_prompt`` / ``call_messages``)。
        ``result_type`` — 结构化输出类型 (BaseModel 子类)。None = 纯文本输出,
        ``result`` 为 None, 原文由 ``content`` 承载。
        ``model`` 由调用方解析 (``LLMConfig.get_model()``), 引擎不负责选模型。
        ``effort`` — thinking effort 刻度 (none..max), 不进 config, 引擎按协议
        映射到 pydantic-ai 的 effort 字段 (anthropic_effort / openai_reasoning_effort)。
        ``export_anchor`` — 锚的目标文件名 (无 ``.anchor.yml`` 后缀, 可含路径如
        ``.anchors/my-call``)。None = 不产锚; ``""`` = 自动生成带 uid 的名字
        (``call-<uid[:8]>``); 其它 = 稳定地址 (重跑覆盖, 版本由 git 治理)。
        锚经 ``LLMFuncResult.anchor`` 携带出来。``anchor_description`` 是锚的
        一句说明 (meta.description)。
        ``input_anchor`` — 消费的锚 (Anchor 对象, 抽象层只约束锚本身)。从锚还原
        上次调用的 turn 链 ([request/response], 含 thinking) 作为 message_history
        拼在本次调用之前做内观; 产出锚的 turns 自动延续被消费的链条。仅支持
        CallAnchor payload — 由强类型校验 (``CallAnchor.from_anchor``) 判定,
        不匹配抛 NotImplementedError。文件 → Anchor 的读取由调用方经
        ``Anchor.from_file`` 完成 (数据结构对协议自解释), 引擎不接触路径。
        None = 冷启动。
        ``thinking`` — 人工插入的 thinking block (内观 A/B 实验工具)。构造
        ``ModelResponse(parts=[ThinkingPart])`` 拼在 message_history 末尾 (若
        有 input_anchor, 在 anchor turns 之后), 让模型把这段思考当作自己的
        既有立场 (内观), 而非需要回复的用户输入 (外观 — 那只是塞进 prompt)。
        thinking 本身不进锚的语义字段, 它以 ThinkingPart 出现在 turns 里。
        """

    @abstractmethod
    def count_tokens(
            self,
            text: str,
            *,
            model: ResolvedModel | None = None,
            include_tokens: bool = False,
    ) -> TokenCount:
        """统计字符串的 token 数 — 同步纯函数.

        性能: CPU-bound BPE 分词, O(n), ``include_tokens=True`` 会物化 token
        id 列表 (长文本有内存开销)。协程调用者必须卸载到线程池
        (asyncio.to_thread / anyio.to_thread / run_in_executor)。

        ``model`` 选择分词器; None 用引擎默认 (o200k_base)。
        非 openai 协议的计数是估算 (tiktoken 是 OpenAI 分词器) —
        ``TokenCount.estimate`` 携带该标志, 由调用方决定如何标注。
        """

    @abstractmethod
    async def run_benchmark(
            self,
            meta: BenchmarkMeta,
            model: ResolvedModel,
            *,
            cwd: Path | None = None,
            output_file: Path | None = None,
            effort: Effort | None = None,
            thinking: str | None = None,
    ) -> BenchmarkRecord:
        """运行一个 benchmark: 用 ``model`` 逐条跑 ``meta.cases_file`` 的用例, 汇总.

        ``model`` 由调用方解析 (``LLMConfig.get_model()``), 引擎不负责选模型。
        ``cwd`` 默认当前进程工作目录 — case 的 prompt/instruction 文件路径相对它解析。
        ``output_file`` 给定则结果写为 jsonl。
        ``effort`` / ``thinking`` — 透传给每个 case 的调用 (策略变量: 评分 hint
        可放 instruction / thinking / 省略, 供 A/B 对比)。
        """


class MossLLMFuncs(LLMFuncs):
    """LLMFuncs + moss prompt protocol — moss 耦合从这里开始.

    在 moss-free 的 ``call(prompt: str)`` 之上加两个 moss 接口:
    - ``call_prompt(text)`` — prompt 文本经 @ 文件协议
      (``message_from_prompt``) 生成 Message 块, 委派给 ``call_messages``。
    - ``call_messages(prompt)`` — 直接收 moss Message 块, 引擎转换为其
      模型 parts 后调用。抽象, 引擎实现。
    """

    async def call_prompt(
            self,
            *,
            text: str,
            instruction: str,
            result_type: Type[RESULT_MODEL] | None = None,
            model: ResolvedModel,
            base_dir: str | Path | None = None,
            expose_file_meta: bool = False,
            effort: Effort | None = None,
            export_anchor: str | Path | None = None,
            anchor_description: str = "",
            input_anchor: Anchor | None = None,
            thinking: str | None = None,
    ) -> LLMFuncResult[RESULT_MODEL]:
        """Prompt 文本 → @ 文件协议生成 Message 块 → call_messages.

        ``text`` 是 Prompt 源 (支持 @ 文件引用), 不是裸字符串。
        ``base_dir`` — 相对 @ref 的解析基准 (默认 cwd)。
        ``expose_file_meta`` — 文件 meta 暴露 flag (可丢弃/可使用层)。
        其余参数同 ``call``。
        """
        from ghoshell_moss.message import message_from_prompt

        blocks = message_from_prompt(
            text, base_dir=base_dir, expose_file_meta=expose_file_meta,
        )
        return await self.call_messages(
            instruction=instruction,
            prompt=blocks,
            result_type=result_type,
            model=model,
            effort=effort,
            export_anchor=export_anchor,
            anchor_description=anchor_description,
            input_anchor=input_anchor,
            thinking=thinking,
        )

    @abstractmethod
    async def call_messages(
            self,
            *,
            instruction: str,
            prompt: list[Message],
            result_type: Type[RESULT_MODEL] | None = None,
            model: ResolvedModel,
            effort: Effort | None = None,
            export_anchor: str | Path | None = None,
            anchor_description: str = "",
            input_anchor: Anchor | None = None,
            thinking: str | None = None,
    ) -> LLMFuncResult[RESULT_MODEL]:
        """直接收 moss Message 块 (list[Message]) → 引擎转换为模型 parts 后调用.

        ``prompt`` 是 @ 生成 (``message_from_prompt``) 或手建的 Message 块。
        其余参数同 ``call``。
        """

"""Pydantic-ai 引擎实现 — LLMFuncs 的默认引擎.

依赖 ``[ghost]`` extra (pydantic-ai >= 1.90.0). import 是惰性的, 模块级 import
不会触发 pydantic-ai 加载; 实际调用时 (任何 public 方法) 才会。
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type

from pydantic import ValidationError

from ghoshell_moss.anchor import Anchor
from ghoshell_moss.agents.pydantic_ai_utils import deserialize_messages, serialize_messages
from ghoshell_moss.contracts.llms import (
    RESULT_MODEL,
    BenchmarkCase,
    BenchmarkMeta,
    BenchmarkRecord,
    BenchmarkRun,
    CallSettings,
    Effort,
    LLMConfig,
    LLMFuncResult,
    LLMFuncResultRecord,
    MossLLMFuncs,
    ModelRef,
    ResolvedModel,
    TokenCount,
)
from ghoshell_moss.contracts.logger import LoggerItf, get_moss_logger
from ghoshell_moss.llms.pydantic_ai_adapter.call_anchor import CallAnchor
from ghoshell_moss.message import Message
from ghoshell_container import Container, IoCContainer

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.run import AgentRunResult

__all__ = ["PydanticAIFuncs"]


class PydanticAIFuncs(MossLLMFuncs):
    """LLMFuncs 的 pydantic-ai 引擎实现.

    构造函数绑定 ``config`` (LLMConfig) + ``logger``; 调用时只传配置路径
    (provider/model/tag), 引擎内部解析 ResolvedModel 再 build_agent。
    三个入口共享私有 ``_call_impl`` (build_agent + history + anchor + run):
    ``call(prompt: str)`` 是 moss-free 字符串入口, ``call_messages(prompt:
    list[Message])`` 是 moss 块入口 (经转换协议), ``call_prompt(text: str)``
    继承 ``MossLLMFuncs`` 默认 (@ 文件协议 → call_messages)。run_benchmark()
    在 call 之上加 case 循环与结果持久化。
    """

    def __init__(
            self,
            logger: LoggerItf | None = None,
            config: LLMConfig | None = None,
            container: IoCContainer | None = None,
    ) -> None:
        self._logger = logger or get_moss_logger()
        self._config = (config or LLMConfig()).resolve()
        # convert() 走 converter 适配时需要 IoC 依赖; 无 container 时用空容器
        # (default 主路径是 accept-or-degrade, 不需要依赖)。
        self._container = container or Container()

    def _resolve(
            self,
            provider: str = "",
            model: str = "",
            tag: str | None = None,
    ) -> ResolvedModel:
        """配置路径 → ResolvedModel, 校验关键 env var 已就绪."""
        resolved = self._config.get_model(provider=provider, model=model, tag=tag)
        for name, field in (
            ("api_key", resolved.service.api_key),
            ("base_url", resolved.service.base_url),
        ):
            if isinstance(field, str) and field.startswith("$"):
                raise ValueError(
                    f"env var {field[1:]} is not set (service "
                    f"{resolved.service.name!r} {name})"
                )
        return resolved

    async def call(
        self,
        *,
        instruction: str,
        prompt: str,
        result_type: Type[RESULT_MODEL] | None = None,
        provider: str = "",
        model: str = "",
        tag: str | None = None,
        settings: CallSettings | None = None,
        effort: Effort | None = None,
        export_anchor: str | Path | None = None,
        anchor_description: str = "",
        input_anchor: Anchor | None = None,
        thinking: str | None = None,
    ) -> LLMFuncResult[RESULT_MODEL]:
        """moss-free 字符串入口 — prompt 原样传给 agent.run."""
        return await self._call_impl(
            instruction=instruction,
            user_prompt=prompt,
            result_type=result_type,
            resolved=self._resolve(provider=provider, model=model, tag=tag),
            settings=settings,
            effort=effort,
            export_anchor=export_anchor,
            anchor_description=anchor_description,
            input_anchor=input_anchor,
            thinking=thinking,
        )

    async def call_messages(
        self,
        *,
        instruction: str,
        prompt: list[Message],
        result_type: Type[RESULT_MODEL] | None = None,
        provider: str = "",
        model: str = "",
        tag: str | None = None,
        settings: CallSettings | None = None,
        effort: Effort | None = None,
        export_anchor: str | Path | None = None,
        anchor_description: str = "",
        input_anchor: Anchor | None = None,
        thinking: str | None = None,
    ) -> LLMFuncResult[RESULT_MODEL]:
        """moss Message 块入口 — 经消息转换协议映射为 pydantic-ai parts.

        ``with_meta=True`` 渲染 Message 携带的 meta 层 (如 @ 文件协议
        expose_file_meta 设的 tag="file" + path/type/size); 无 tag 的纯文本
        块不受影响。content_types 过滤: 按目标模型的 ``ModelConfig.content_types``
        逐 message 跑 ``convert()`` — 原生支持的类型保真, 不支持的降级为文本
        占位 (或经 converters 适配), 防止把图片裸发给纯文本模型 (如 deepseek-v4-pro)。
        """
        from ghoshell_moss.llms.pydantic_ai_adapter.conversion import messages_to_parts

        resolved = self._resolve(provider=provider, model=model, tag=tag)
        filtered = [
            resolved.model.convert(self._container, message)
            for message in prompt
        ]
        return await self._call_impl(
            instruction=instruction,
            user_prompt=messages_to_parts(filtered, with_meta=True),
            result_type=result_type,
            resolved=resolved,
            settings=settings,
            effort=effort,
            export_anchor=export_anchor,
            anchor_description=anchor_description,
            input_anchor=input_anchor,
            thinking=thinking,
        )

    async def _call_impl(
        self,
        *,
        instruction: str,
        user_prompt: str | list[Any],
        result_type: Type[RESULT_MODEL] | None,
        resolved: ResolvedModel,
        settings: CallSettings | None,
        effort: Effort | None,
        export_anchor: str | Path | None,
        anchor_description: str,
        input_anchor: Anchor | None,
        thinking: str | None,
    ) -> LLMFuncResult[RESULT_MODEL]:
        """build_agent + history + anchor + agent.run — 两个入口共享的单点.

        ``user_prompt`` 是已就绪的 pydantic-ai prompt (str 或 UserContent parts),
        由 call / call_messages 各自完成转换后传入。
        ``resolved`` 必须是已 resolve 的 ResolvedModel (api_key 已解密, 仅内存).
        ``settings`` / ``effort`` 透传给 build_agent, 由它按协议映射.
        ``export_anchor`` — 锚的目标文件名 (无 .anchor.yml 后缀, 可含路径).
        None = 不产锚; ``""`` = 自动生成带 uid 的名字; 其它 = 稳定地址.
        调用前后各落一次锚: 调用前写请求帧 (调用失败也保留请求锚), 成功后
        覆写为完整帧 (instruction + turns — 标准序列化的 request/response,
        含 thinking), 锚经 ``LLMFuncResult.anchor`` 携带出来。
        ``input_anchor`` — 消费的锚 (Anchor 对象). 从锚还原 turn 链作为
        message_history 拼在本次调用之前做内观 (仅支持 CallAnchor payload,
        其它类型抛 NotImplementedError — 由强类型校验判定); 产出锚的 turns
        自动延续被消费的链条.
        ``thinking`` — 人工插入的 thinking block (内观 A/B 实验工具), 构造
        ``ModelResponse(parts=[ThinkingPart])`` 拼在 message_history 末尾,
        让模型把它当作自己的既有立场而非需回复的用户输入. 以 ThinkingPart
        进入 turns, 不进锚的语义字段.
        """
        from ghoshell_moss.llms.pydantic_ai_adapter.client import build_agent

        anchor_dir, base_name = _resolve_anchor_target(export_anchor)
        agent = build_agent(
            resolved,
            settings=settings,
            effort=effort,
        )
        history = _build_history(input_anchor, thinking)

        anchor = None
        if anchor_dir is not None:
            anchor = _build_call_anchor(
                instruction, result_type, resolved, effort,
                name=base_name, description=anchor_description,
            )
            anchor.dump_to_dir(anchor_dir, anchor.meta.name)

        self._logger.debug(
            "llms call: service=%s model=%s effort=%s settings=%s",
            resolved.service.name, resolved.model.model, effort, settings,
        )
        start = time.perf_counter()
        try:
            result = await agent.run(
                user_prompt,
                output_type=result_type,
                instructions=instruction or None,
                message_history=history,
            )
        except Exception:
            self._logger.exception(
                "llms call failed: service=%s model=%s",
                resolved.service.name, resolved.model.model,
            )
            raise
        elapsed = time.perf_counter() - start
        self._logger.debug("llms call done: elapsed=%.2fs", elapsed)
        output = result.output
        typed = output if result_type is not None and isinstance(output, result_type) else None

        llm_kwargs = dict(
            result=typed,
            content=_extract_text(result),
            usage=_dataclass_asdict(result.usage) if result.usage else {},
            cast=elapsed,
            retries=0,
        )
        llm_result = (
            LLMFuncResult[result_type](**llm_kwargs)
            if result_type is not None
            else LLMFuncResult(**llm_kwargs)
        )
        if anchor is not None:
            frame = CallAnchor(
                instruction=instruction,
                model=ModelRef.from_resolved(resolved),
                result_type=_type_path(result_type),
                effort=effort,
                turns=serialize_messages(result.all_messages()),
                result=typed.model_dump() if typed is not None else None,
            )
            anchor.payload = frame.model_dump(exclude_none=True, exclude={"meta"})
            anchor.dump_to_dir(anchor_dir, anchor.meta.name)
            llm_result.anchor = anchor
        return llm_result

    async def run_benchmark(
        self,
        meta: BenchmarkMeta,
        *,
        provider: str = "",
        model: str = "",
        tag: str | None = None,
        cwd: Path | None = None,
        output_file: Path | None = None,
        effort: Effort | None = None,
        thinking: str | None = None,
    ) -> BenchmarkRecord:
        """逐 case 跑 benchmark, 汇总为 BenchmarkRecord.

        ``meta.result_type`` (module:attr) 解析为 BaseModel 类型后用于所有 case;
        case 的 prompt / instruction 可能是相对 cwd 的文件路径, 自动解析.
        ``effort`` / ``thinking`` 逐 case 透传给 call (策略 A/B: hint 放
        instruction / thinking / 省略).
        """
        from ghoshell_common.helpers import import_from_path

        cwd = cwd or Path.cwd()
        if not meta.result_type:
            raise ValueError("BenchmarkMeta.result_type is required")
        result_type = import_from_path(meta.result_type)
        cases = _load_cases(meta, cwd)
        resolved = self._resolve(provider=provider, model=model, tag=tag)
        run = BenchmarkRun(
            label=meta.title,
            meta=meta,
            model=ModelRef.from_resolved(resolved),
        )
        results: list[LLMFuncResultRecord] = []
        for case in cases:
            for _ in range(case.times):
                inst = _resolve_instruction(case, meta, cwd)
                prompt_str = _resolve_file_value(case.prompt, cwd)
                case_thinking = (
                    _resolve_file_value(case.thinking, cwd)
                    if case.thinking is not None else thinking
                )
                case_effort = case.effort if case.effort is not None else effort
                res = await self.call(
                    instruction=inst,
                    prompt=prompt_str,
                    result_type=result_type,
                    provider=provider,
                    model=model,
                    tag=tag,
                    effort=case_effort,
                    thinking=case_thinking,
                )
                results.append(res.to_record())

        record = BenchmarkRecord(run=run, results=results)
        if output_file:
            _dump_record(record, output_file)
        return record

    def count_tokens(
        self,
        text: str,
        *,
        provider: str = "",
        model: str = "",
        tag: str | None = None,
        include_tokens: bool = False,
    ) -> TokenCount:
        """tiktoken 计数 — openai 协议精确, 非 openai 协议为估算.

        路径全空或非 openai 协议时回退 o200k_base 并标 estimate。
        tiktoken 惰性 import (依赖 ghost extra 的 pydantic-ai-slim[openai])。
        """
        import tiktoken

        if not (provider or model or tag):
            service, model_name, estimate = "", "", True
            enc = tiktoken.get_encoding("o200k_base")
        else:
            resolved = self._resolve(provider=provider, model=model, tag=tag)
            service, model_name = resolved.service.name, resolved.model.model
            estimate = resolved.client_protocol != "openai"
            try:
                enc = tiktoken.encoding_for_model(model_name)
            except KeyError:
                enc = tiktoken.get_encoding("o200k_base")

        ids = enc.encode(text)
        return TokenCount(
            count=len(ids),
            service=service,
            model=model_name,
            encoding=enc.name,
            estimate=estimate,
            tokens=tuple(ids) if include_tokens else None,
        )


# ── helpers ────────────────────────────────────────────────────────────


def _type_path(cls: Type | None) -> str:
    """module:attr 路径 — 指向输出 schema, 可经 import_from_path 还原. None → 空."""
    if cls is None:
        return ""
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve_anchor_target(export_anchor: str | Path | None) -> tuple[Path | None, str]:
    """把 export_anchor 解析为 (目标目录, 文件名 stem).

    None → (None, "") 不产锚; ``""`` → (cwd, "") 自动生成 uid 名字;
    其它 → 文件名/路径 (无后缀), 目录取所在路径的父目录.
    """
    if export_anchor is None:
        return None, ""
    if export_anchor == "":
        return Path("."), ""
    target = Path(export_anchor)
    return target.parent, target.stem


def _build_call_anchor(
        instruction: str,
        result_type: Type[RESULT_MODEL] | None,
        model: ResolvedModel,
        effort: Effort | None,
        *,
        name: str,
        description: str,
) -> Anchor:
    """组装请求帧 CallAnchor (无 turns) → 转弱 Anchor. name 空则按 uid 自动命名."""
    request = CallAnchor(
        instruction=instruction,
        model=ModelRef.from_resolved(model),
        result_type=_type_path(result_type),
        effort=effort,
    )
    anchor = request.to_anchor(name=name, description=description)
    if not anchor.meta.name:
        anchor.meta.name = f"call-{anchor.meta.uid[:8]}"
    return anchor


def _load_history(anchor: Anchor | None) -> list[ModelMessage] | None:
    """从输入锚还原 turn 链作为 message_history — 消费侧 (内观回灌).

    ``anchor`` 抽象层只约束 Anchor; 引擎只消费 CallAnchor payload。判断交给
    强类型 — ``CallAnchor.from_anchor`` 的结构化校验 (而非手工比较 ref 字符串):
    载荷不匹配 CallAnchor 即 NotImplementedError — LLMFunc 支持的锚类型有限,
    不支持的显式拒绝。空 turns (失败保留的请求帧) 返回 None — 无响应可内观,
    冷启动。
    """
    if anchor is None:
        return None
    try:
        rebuilt = CallAnchor.from_anchor(anchor)
    except ValidationError as e:
        raise NotImplementedError(
            f"unsupported input anchor (ref={anchor.meta.ref!r}) — "
            f"payload does not validate as CallAnchor: {e}"
        ) from e
    if not rebuilt.turns:
        return None
    return deserialize_messages(rebuilt.turns)


def _build_history(
        input_anchor: Anchor | None,
        thinking: str | None,
) -> list[ModelMessage] | None:
    """组装 message_history — 消费锚的 turns + 人工 thinking block.

    顺序: ``[anchor turns...] + [ModelResponse(ThinkingPart)]``, 然后
    ``agent.run`` 追加新 prompt 的 request。thinking 作为孤立 ModelResponse
    的 ThinkingPart 注入 — 模型把它当作自己的既有立场 (内观), 而非需要
    回复的用户输入 (外观)。无锚且无 thinking → None (冷启动)。
    """
    history = _load_history(input_anchor)
    if thinking is None:
        return history
    from pydantic_ai.messages import ModelResponse, ThinkingPart

    thinking_turn: list[ModelMessage] = [
        ModelResponse(parts=[ThinkingPart(content=thinking)])
    ]
    if history is None:
        return thinking_turn
    return [*history, *thinking_turn]


def _extract_text(result: AgentRunResult[Any]) -> str:
    """从 pydantic-ai 结果提取原始文本 (TextPart 拼接).

    结构化模式下模型可能仅在 tool call 返回内容, 此值可能为空。
    """
    from pydantic_ai.messages import TextPart

    parts: list[str] = []
    for msg in result.all_messages():
        for part in msg.parts:
            if isinstance(part, TextPart):
                parts.append(part.content)
    return "".join(parts)


def _dataclass_asdict(obj: Any) -> dict[str, Any]:
    """dataclass 转 dict (RunUsage 等). 非 dataclass 返回空 dict。"""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    return {}


def _load_cases(meta: BenchmarkMeta, cwd: Path) -> list[BenchmarkCase]:
    """从 ``meta.cases_file`` (相对 cwd) 读取 jsonl, 每行一个 BenchmarkCase。"""
    if not meta.cases_file:
        return []
    path = cwd / meta.cases_file
    if not path.is_file():
        raise FileNotFoundError(f"cases file not found: {path}")
    cases: list[BenchmarkCase] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(BenchmarkCase(**json.loads(line)))
    return cases


def _resolve_file_value(value: str, cwd: Path) -> str:
    """若 value 相对 cwd 存在且为文件, 读文件返回; 否则原样返回字符串."""
    if not value:
        return value
    candidate = (cwd / value).resolve()
    try:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    except OSError:
        pass
    return value


def _resolve_instruction(case: BenchmarkCase, meta: BenchmarkMeta, cwd: Path) -> str:
    """case instruction → meta instruction 三级回退, 两者都可能指向文件."""
    value = case.instruction or meta.instruction
    return _resolve_file_value(value, cwd)


def _dump_record(record: BenchmarkRecord, file: Path) -> None:
    """jsonl 持久化: 首行 run (meta + model_ref), 后续每行一个 record."""
    lines: list[str] = [record.run.model_dump_json(exclude_none=True)]
    for r in record.results:
        lines.append(r.model_dump_json(exclude_none=True))
    file.write_text("\n".join(lines) + "\n", encoding="utf-8")

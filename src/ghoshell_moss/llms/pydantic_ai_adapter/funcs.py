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
from ghoshell_moss.contracts.llms import (
    RESULT_MODEL,
    BenchmarkCase,
    BenchmarkMeta,
    BenchmarkRecord,
    BenchmarkRun,
    Effort,
    LLMFuncResult,
    LLMFuncResultRecord,
    LLMFuncs,
    ModelRef,
    ResolvedModel,
    TokenCount,
)
from ghoshell_moss.llms.pydantic_ai_adapter.call_anchor import CallAnchor

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.run import AgentRunResult

__all__ = ["PydanticAIFuncs"]


class PydanticAIFuncs(LLMFuncs):
    """LLMFuncs 的 pydantic-ai 引擎实现.

    依赖 ``build_agent(resolved)`` 构造 pydantic-ai Agent, 再通过 ``run`` (async)
    执行单轮调用。call() 填充 LLMFuncResult (result/content/usage/cast/retries);
    run_benchmark() 在此基础上加 case 循环与结果持久化。
    """

    async def call(
        self,
        *,
        instruction: str,
        prompt: str,
        result_type: Type[RESULT_MODEL],
        model: ResolvedModel,
        effort: Effort | None = None,
        export_anchor: str | Path | None = None,
        anchor_description: str = "",
        input_anchor: Anchor | None = None,
        thinking: str | None = None,
    ) -> LLMFuncResult[RESULT_MODEL]:
        """单轮模型调用 — build_agent + agent.run(output_type, instructions).

        ``model`` 必须是已 resolve 的 ResolvedModel (api_key 已解密, 仅内存).
        ``effort`` 透传给 build_agent, 由它按协议映射到 effort 字段.
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
        agent = build_agent(model, effort=effort)
        history = _build_history(input_anchor, thinking)

        anchor = None
        if anchor_dir is not None:
            anchor = _build_call_anchor(
                instruction, result_type, model, effort,
                name=base_name, description=anchor_description,
            )
            anchor.dump_to_dir(anchor_dir, anchor.meta.name)

        start = time.perf_counter()
        result = await agent.run(
            prompt,
            output_type=result_type,
            instructions=instruction or None,
            message_history=history,
        )
        elapsed = time.perf_counter() - start
        output = result.output
        typed = output if isinstance(output, result_type) else None

        llm_result = LLMFuncResult[result_type](
            result=typed,
            content=_extract_text(result),
            usage=_dataclass_asdict(result.usage) if result.usage else {},
            cast=elapsed,
            retries=0,
        )
        if anchor is not None:
            frame = CallAnchor(
                instruction=instruction,
                model=ModelRef.from_resolved(model),
                result_type=_type_path(result_type),
                effort=effort,
                turns=_serialize_messages(result.all_messages()),
                result=typed.model_dump() if typed is not None else None,
            )
            anchor.payload = frame.model_dump(exclude_none=True, exclude={"meta"})
            anchor.dump_to_dir(anchor_dir, anchor.meta.name)
            llm_result.anchor = anchor
        return llm_result

    async def run_benchmark(
        self,
        meta: BenchmarkMeta,
        model: ResolvedModel,
        *,
        cwd: Path | None = None,
        output_file: Path | None = None,
    ) -> BenchmarkRecord:
        """逐 case 跑 benchmark, 汇总为 BenchmarkRecord.

        ``meta.result_type`` (module:attr) 解析为 BaseModel 类型后用于所有 case;
        case 的 prompt / instruction 可能是相对 cwd 的文件路径, 自动解析.
        """
        from ghoshell_common.helpers import import_from_path

        cwd = cwd or Path.cwd()
        if not meta.result_type:
            raise ValueError("BenchmarkMeta.result_type is required")
        result_type = import_from_path(meta.result_type)
        cases = _load_cases(meta, cwd)
        run = BenchmarkRun(
            label=meta.title,
            meta=meta,
            model=ModelRef.from_resolved(model),
        )
        results: list[LLMFuncResultRecord] = []
        for case in cases:
            for _ in range(case.times):
                inst = _resolve_instruction(case, meta, cwd)
                prompt_str = _resolve_file_value(case.prompt, cwd)
                res = await self.call(
                    instruction=inst,
                    prompt=prompt_str,
                    result_type=result_type,
                    model=model,
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
        model: ResolvedModel | None = None,
        include_tokens: bool = False,
    ) -> TokenCount:
        """tiktoken 计数 — openai 协议精确, 非 openai 协议为估算.

        ``model`` 为 None 或非 openai 协议时回退 o200k_base 并标 estimate。
        tiktoken 惰性 import (依赖 ghost extra 的 pydantic-ai-slim[openai])。
        """
        import tiktoken

        if model is None:
            service, model_name, estimate = "", "", True
            enc = tiktoken.get_encoding("o200k_base")
        else:
            service, model_name = model.service.name, model.model.model
            estimate = model.client_protocol != "openai"
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


def _type_path(cls: Type) -> str:
    """module:attr 路径 — 指向输出 schema, 可经 import_from_path 还原."""
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
        result_type: Type[RESULT_MODEL],
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


def _serialize_messages(messages: list[ModelMessage]) -> list[dict[str, Any]]:
    """pydantic-ai 标准序列化 message history — 保住 thinking/text/tool 所有 part."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")


def _deserialize_messages(turns: list[dict[str, Any]]) -> list[ModelMessage]:
    """pydantic-ai 标准序列化反向 — dict 列表还原为 ModelMessage (消费锚)."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter

    return ModelMessagesTypeAdapter.validate_python(turns)


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
    return _deserialize_messages(rebuilt.turns)


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

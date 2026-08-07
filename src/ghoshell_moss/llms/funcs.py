"""Pydantic-ai 引擎实现 — LLMFuncs 的默认引擎.

依赖 ``[ghost]`` extra (pydantic-ai >= 1.90.0). import 是惰性的, 模块级 import
不会触发 pydantic-ai 加载; 实际调用时 (任何 public 方法) 才会。
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Type

from ghoshell_moss.contracts.llms import (
    RESULT_MODEL,
    BenchmarkCase,
    BenchmarkMeta,
    BenchmarkRecord,
    BenchmarkRun,
    LLMFuncResult,
    LLMFuncResultRecord,
    LLMFuncs,
    ModelRef,
    ResolvedModel,
)

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
    ) -> LLMFuncResult[RESULT_MODEL]:
        """单轮模型调用 — build_agent + agent.run(output_type, instructions).

        ``model`` 必须是已 resolve 的 ResolvedModel (api_key 已解密, 仅内存).
        """
        from ghoshell_moss.llms.client import build_agent

        agent = build_agent(model)
        start = time.perf_counter()
        result = await agent.run(
            prompt,
            output_type=result_type,
            instructions=instruction or None,
        )
        elapsed = time.perf_counter() - start
        output = result.output
        return LLMFuncResult[result_type](
            result=output if isinstance(output, result_type) else None,
            content=_extract_text(result),
            usage=_dataclass_asdict(result.usage) if result.usage else {},
            cast=elapsed,
            retries=0,
        )

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


# ── helpers ────────────────────────────────────────────────────────────


def _extract_text(result) -> str:
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


def _dataclass_asdict(obj) -> dict:
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

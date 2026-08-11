"""Tests for PydanticAIFuncs — pydantic-ai engine implementation.

Uses mock Agent (no real API calls). Validates structured output,
content extraction, to_record conversion, and benchmark loop.
"""

import json
import tempfile
from pathlib import Path
from typing import Type
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest
from pydantic import BaseModel

from ghoshell_moss.contracts.llms import (
    BenchmarkCase,
    BenchmarkMeta,
    BenchmarkRecord,
    LLMFuncResult,
    LLMFuncResultRecord,
    ModelRef,
    ResolvedModel,
    ServiceConfig,
    ModelConfig,
)
from ghoshell_moss.llms.pydantic_ai_adapter.funcs import (
    PydanticAIFuncs,
    _extract_text,
    _resolve_file_value,
    _dataclass_asdict,
)


class Score(BaseModel):
    value: int


class Tag(BaseModel):
    label: str
    confidence: float


# ── fixtures ──────────────────────────────────────────────────────────


def _make_resolved() -> ResolvedModel:
    return ResolvedModel(
        service=ServiceConfig(
            name="test", base_url="https://test.local/v1",
            api_key="sk-test", protocol="openai",
        ),
        model=ModelConfig(model="test-model"),
    )


def _make_mock_agent(output: BaseModel | None = None, text_parts: list[str] | None = None):
    """Return a mock Agent with run() -> AgentRunResult.

    run() returns a mock with .output, .all_messages(), and .usage.
    """
    from pydantic_ai.messages import TextPart, ModelResponse, ModelRequest
    import dataclasses

    agent = MagicMock()
    result = MagicMock()
    result.output = output
    result.usage = None

    # build all_messages() with text parts
    messages = []
    if text_parts:
        msg = MagicMock(spec=ModelResponse)
        msg.parts = []
        for t in text_parts:
            part = MagicMock(spec=TextPart)
            part.content = t
            msg.parts.append(part)
        messages.append(msg)
    result.all_messages.return_value = messages
    agent.run = AsyncMock(return_value=result)
    return agent


# ── unit: _extract_text ──────────────────────────────────────────────


class TestExtractText:
    def test_empty(self):
        result = MagicMock()
        result.all_messages.return_value = []
        assert _extract_text(result) == ""

    def test_single_text_part(self):
        from pydantic_ai.messages import TextPart, ModelResponse

        part = MagicMock(spec=TextPart)
        part.content = "hello world"
        msg = MagicMock(spec=ModelResponse)
        msg.parts = [part]
        result = MagicMock()
        result.all_messages.return_value = [msg]
        assert _extract_text(result) == "hello world"

    def test_multiple_parts(self):
        from pydantic_ai.messages import TextPart, ModelResponse

        p1 = MagicMock(spec=TextPart)
        p1.content = "hello"
        p2 = MagicMock(spec=TextPart)
        p2.content = " world"
        msg = MagicMock(spec=ModelResponse)
        msg.parts = [p1, p2]
        result = MagicMock()
        result.all_messages.return_value = [msg]
        assert _extract_text(result) == "hello world"


# ── unit: _resolve_file_value ────────────────────────────────────────


class TestResolveFileValue:
    def test_empty(self, tmp_path):
        assert _resolve_file_value("", tmp_path) == ""

    def test_exists(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("file content")
        assert _resolve_file_value(str(f), tmp_path) == "file content"

    def test_not_exists(self, tmp_path):
        assert _resolve_file_value("plain string", tmp_path) == "plain string"


# ── unit: _dataclass_asdict ──────────────────────────────────────────


class TestDataclassAsDict:
    def test_dataclass(self):
        import dataclasses

        @dataclasses.dataclass
        class Foo:
            x: int = 1
            y: str = "a"

        assert _dataclass_asdict(Foo()) == {"x": 1, "y": "a"}

    def test_non_dataclass(self):
        assert _dataclass_asdict(42) == {}
        assert _dataclass_asdict("str") == {}


# ── integration: PydanticAIFuncs.call() ──────────────────────────────


@pytest.mark.asyncio
async def test_call_structured_output():
    """Structured call returns typed LLMFuncResult."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=8), text_parts=["thinking..."])

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        result = await funcs.call(
            instruction="you are helpful",
            prompt="rate this",
            result_type=Score,
            model=_make_resolved(),
        )

    assert isinstance(result, LLMFuncResult)
    assert result.result == Score(value=8)
    assert result.content == "thinking..."
    assert result.cast > 0
    assert result.retries == 0
    agent.run.assert_awaited_once()


@pytest.mark.asyncio
async def test_call_null_result():
    """When output is not the expected type, result stays None."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output="plain string", text_parts=["hello"])

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        result = await funcs.call(
            instruction="",
            prompt="hi",
            result_type=Score,
            model=_make_resolved(),
        )

    assert result.result is None
    assert result.content == "hello"


@pytest.mark.asyncio
async def test_call_to_record():
    """LLMFuncResult.to_record() converts to weak-data record."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Tag(label="greeting", confidence=0.95))

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        result = await funcs.call(
            instruction="",
            prompt="classify",
            result_type=Tag,
            model=_make_resolved(),
        )

    record = result.to_record()
    assert isinstance(record, LLMFuncResultRecord)
    assert record.result == {"label": "greeting", "confidence": 0.95}


# ── integration: PydanticAIFuncs.run_benchmark() ─────────────────────


@pytest.mark.asyncio
async def test_run_benchmark_basic(tmp_path: Path):
    """Benchmark loop: load cases, call each, produce record."""
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps({"label": "c1", "prompt": "hello", "expected": "hi", "times": 2})
        + "\n"
        + json.dumps({"label": "c2", "prompt": "bye", "times": 1})
        + "\n"
    )
    meta = BenchmarkMeta(
        title="test-bench",
        description="desc",
        result_type="ghoshell_moss.contracts.llms:ModelRef",
        cases_file=cases_file.name,
    )
    resolved = _make_resolved()

    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Tag(label="ok", confidence=0.5))

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        record = await funcs.run_benchmark(
            meta, resolved, cwd=tmp_path,
        )

    assert isinstance(record, BenchmarkRecord)
    assert record.run.label == "test-bench"
    assert record.run.model.service == "test"
    assert record.run.model.model == "test-model"
    # 2 cases: c1 * 2 + c2 * 1 = 3 result records
    assert len(record.results) == 3
    assert all(isinstance(r, LLMFuncResultRecord) for r in record.results)
    assert sorted(r.model_dump_json() for r in record.results)  # smoke


@pytest.mark.asyncio
async def test_run_benchmark_output_file(tmp_path: Path):
    """When output_file is set, jsonl is written."""
    cases_file = tmp_path / "cases.jsonl"
    cases_file.write_text(
        json.dumps({"label": "c1", "prompt": "hello"}) + "\n"
    )
    meta = BenchmarkMeta(
        title="bench", result_type="ghoshell_moss.contracts.llms:ModelRef",
        cases_file=cases_file.name,
    )
    output = tmp_path / "results.jsonl"
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=1))

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        await funcs.run_benchmark(meta, _make_resolved(), cwd=tmp_path, output_file=output)

    assert output.is_file()
    lines = output.read_text().strip().splitlines()
    assert len(lines) == 2  # run + 1 result
    assert "bench" in lines[0]


def test_run_benchmark_missing_result_type():
    """result_type is required — pydantic rejects at construction."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match="result_type"):
        BenchmarkMeta(title="x", cases_file="c.jsonl")


# ── unit: count_tokens (tiktoken, no mocks — pure local computation) ──


class TestCountTokens:
    def _resolved(self, protocol: str, model: str) -> ResolvedModel:
        return ResolvedModel(
            service=ServiceConfig(name="s", base_url="http://x", api_key="k", protocol=protocol),
            model=ModelConfig(model=model),
        )

    def test_openai_gpt4o_uses_o200k(self):
        f = PydanticAIFuncs()
        r = f.count_tokens("hello world", model=self._resolved("openai", "gpt-4o"))
        assert r.encoding == "o200k_base"
        assert r.estimate is False
        assert r.count == 2
        assert r.service == "s"
        assert r.model == "gpt-4o"

    def test_openai_gpt4_uses_cl100k(self):
        f = PydanticAIFuncs()
        r = f.count_tokens("hello world", model=self._resolved("openai", "gpt-4"))
        assert r.encoding == "cl100k_base"
        assert r.estimate is False

    def test_non_openai_is_estimate_with_fallback(self):
        f = PydanticAIFuncs()
        r = f.count_tokens("hello world", model=self._resolved("anthropic", "claude-sonnet-4-6"))
        assert r.estimate is True
        assert r.encoding == "o200k_base"  # tiktoken 不认识 → 回退

    def test_no_model_is_estimate_and_blank(self):
        f = PydanticAIFuncs()
        r = f.count_tokens("hello world")
        assert r.estimate is True
        assert r.service == ""
        assert r.model == ""

    def test_include_tokens_materializes_ids(self):
        f = PydanticAIFuncs()
        r = f.count_tokens("hi there", model=self._resolved("openai", "gpt-4o"), include_tokens=True)
        assert r.tokens is not None
        assert len(r.tokens) == r.count == 2


# ── ModelRef safety ──────────────────────────────────────────────────


def test_modelref_no_secret_leak():
    resolved = ResolvedModel(
        service=ServiceConfig(
            name="deepseek", base_url="https://secret.internal/v1",
            api_key="sk-super-secret", protocol="anthropic",
        ),
        model=ModelConfig(model="deepseek-chat"),
    )
    ref = ModelRef.from_resolved(resolved)
    data = ref.model_dump_json()
    assert "sk-super-secret" not in data
    assert "secret.internal" not in data
    assert ref.service == "deepseek"
    assert ref.model == "deepseek-chat"

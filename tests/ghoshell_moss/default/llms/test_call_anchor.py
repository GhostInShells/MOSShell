"""Tests for LLMFuncs.call anchor production — the call's key frame.

Uses mock Agent (no real API calls). Validates: anchor file written with the
two-section yaml, meta + payload fields, ref points to the payload
definition, file -> CallAnchor round-trip reconstructs the call, and the
request frame survives a failed call. export_anchor is a target filename
(no suffix); '' auto-generates a uid-based name.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel

from ghoshell_moss.anchor import Anchor, AnchorMeta
from ghoshell_moss.contracts.llms import (
    ModelConfig,
    ResolvedModel,
    ServiceConfig,
)
from ghoshell_moss.llms.call_anchor import CallAnchor
from ghoshell_moss.llms.funcs import PydanticAIFuncs, _type_path


class Score(BaseModel):
    value: int


def _make_resolved() -> ResolvedModel:
    return ResolvedModel(
        service=ServiceConfig(
            name="test", base_url="https://test.local/v1",
            api_key="sk-test", protocol="openai",
        ),
        model=ModelConfig(model="test-model"),
    )


def _make_mock_agent(output, *, text_parts=None, exc=None):
    """Mock Agent: run() -> AgentRunResult, or raises ``exc``."""
    from pydantic_ai.messages import ModelResponse, TextPart

    agent = MagicMock()
    if exc is not None:
        agent.run = AsyncMock(side_effect=exc)
        return agent
    result = MagicMock()
    result.output = output
    result.usage = None
    messages = []
    if text_parts:
        msg = MagicMock(spec=ModelResponse)
        msg.parts = [MagicMock(spec=TextPart, content=t) for t in text_parts]
        messages.append(msg)
    result.all_messages.return_value = messages
    agent.run = AsyncMock(return_value=result)
    return agent


def _read_anchor(path: Path) -> tuple[dict, dict]:
    """Split the two-section yaml: (meta, payload)."""
    meta_text, payload_text = path.read_text(encoding="utf-8").split("---", 1)
    return yaml.safe_load(meta_text), yaml.safe_load(payload_text)


def _anchor_files(dir: Path) -> list[Path]:
    return sorted(dir.glob("*.anchor.yml"))


# ── produce ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_produces_named_anchor(tmp_path: Path):
    """export_anchor=filename writes a two-section anchor; result carries it."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=8), text_parts=["thinking..."])

    with patch("ghoshell_moss.llms.client.build_agent", return_value=agent):
        result = await funcs.call(
            instruction="you are helpful",
            prompt="rate this",
            result_type=Score,
            model=_make_resolved(),
            export_anchor=tmp_path / "my-call",
        )

    files = _anchor_files(tmp_path)
    assert files == [tmp_path / "my-call.anchor.yml"]
    assert result.anchor is not None
    assert result.anchor.meta.uid
    assert result.anchor.meta.name == "my-call"

    meta, payload = _read_anchor(files[0])
    assert meta["uid"] == result.anchor.meta.uid
    assert meta["name"] == "my-call"
    assert meta["ref"] == CallAnchor.ref()
    assert meta["created"]
    assert payload["instruction"] == "you are helpful"
    assert payload["prompt"] == "rate this"
    assert payload["model"]["service"] == "test"
    assert payload["model"]["model"] == "test-model"
    assert payload["result_type"] == _type_path(Score)
    assert payload["result"] == {"value": 8}
    assert payload["content"] == "thinking..."


@pytest.mark.asyncio
async def test_call_auto_uid_name(tmp_path: Path, monkeypatch):
    """export_anchor='' -> auto call-<uid[:8]> name in cwd."""
    monkeypatch.chdir(tmp_path)
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=1), text_parts=["ok"])

    with patch("ghoshell_moss.llms.client.build_agent", return_value=agent):
        result = await funcs.call(
            instruction="",
            prompt="hi",
            result_type=Score,
            model=_make_resolved(),
            export_anchor="",
        )

    files = _anchor_files(tmp_path)
    assert len(files) == 1
    assert result.anchor is not None
    assert files[0].name == f"{result.anchor.meta.name}.anchor.yml"
    assert result.anchor.meta.name.startswith("call-")
    assert result.anchor.meta.name.endswith(result.anchor.meta.uid[:8])


@pytest.mark.asyncio
async def test_no_export_no_anchor(tmp_path: Path):
    """Without export_anchor, no anchor is produced or written."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=3))

    with patch("ghoshell_moss.llms.client.build_agent", return_value=agent):
        result = await funcs.call(
            instruction="",
            prompt="hi",
            result_type=Score,
            model=_make_resolved(),
        )

    assert result.anchor is None
    assert _anchor_files(tmp_path) == []


# ── reconstruct ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_anchor_reconstructs_call(tmp_path: Path):
    """From the anchor file -> CallAnchor: the call is reconstructible.

    A model curls meta.ref to learn the payload shape, then from_anchor
    rebuilds the typed request — the protocol's single key proposition.
    """
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=8), text_parts=["thinking..."])

    with patch("ghoshell_moss.llms.client.build_agent", return_value=agent):
        result = await funcs.call(
            instruction="you are helpful",
            prompt="rate this",
            result_type=Score,
            model=_make_resolved(),
            export_anchor=tmp_path / "my-call",
        )

    meta, payload = _read_anchor(_anchor_files(tmp_path)[0])
    rebuilt = CallAnchor.from_anchor(
        Anchor(meta=AnchorMeta(**meta), payload=payload)
    )
    assert rebuilt.instruction == "you are helpful"
    assert rebuilt.prompt == "rate this"
    assert rebuilt.model.service == "test"
    assert rebuilt.model.model == "test-model"
    assert rebuilt.result_type == _type_path(Score)
    assert rebuilt.result == {"value": 8}
    assert rebuilt.content == "thinking..."


# ── failure ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_frame_survives_failed_call(tmp_path: Path):
    """Request anchor is dumped before the call — a failed call keeps it."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=None, exc=RuntimeError("boom"))

    with patch("ghoshell_moss.llms.client.build_agent", return_value=agent):
        with pytest.raises(RuntimeError, match="boom"):
            await funcs.call(
                instruction="you are helpful",
                prompt="rate this",
                result_type=Score,
                model=_make_resolved(),
                export_anchor=tmp_path / "my-call",
            )

    files = _anchor_files(tmp_path)
    assert files == [tmp_path / "my-call.anchor.yml"]
    meta, payload = _read_anchor(files[0])
    assert meta["ref"] == CallAnchor.ref()
    assert payload["instruction"] == "you are helpful"
    assert payload["prompt"] == "rate this"
    assert payload["model"]["service"] == "test"
    assert "result" not in payload
    assert "content" not in payload

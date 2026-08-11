"""Tests for LLMFuncs.call anchor production — the call's key frame.

Uses mock Agent (no real API calls). Validates: anchor file written with the
two-section yaml, meta + payload fields, ref points to the payload
definition, the turns capture the full request/response (incl thinking) in
pydantic-ai standard serialization, file -> CallAnchor round-trip
reconstructs the call, and the request frame survives a failed call.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from pydantic import BaseModel
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    UserPromptPart,
)

from ghoshell_moss.anchor import Anchor, AnchorMeta
from ghoshell_moss.contracts.llms import (
    ModelConfig,
    ModelRef,
    ResolvedModel,
    ServiceConfig,
)
from ghoshell_moss.llms.pydantic_ai_adapter.call_anchor import CallAnchor
from ghoshell_moss.llms.pydantic_ai_adapter.funcs import PydanticAIFuncs, _type_path


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


def _make_mock_agent(
        output: BaseModel | None,
        *,
        instruction: str = "",
        prompt: str = "",
        thinking: str | None = None,
        text_parts: list[str] | None = None,
        exc: Exception | None = None,
) -> MagicMock:
    """Mock Agent: run() -> AgentRunResult with a real request/response pair.

    all_messages() returns [ModelRequest(system+user), ModelResponse(thinking?,
    text...)] so the anchor's turns carry the full turn in standard
    serialization — exactly what the real engine produces.
    """
    agent = MagicMock()
    if exc is not None:
        agent.run = AsyncMock(side_effect=exc)
        return agent
    result = MagicMock()
    result.output = output
    result.usage = None
    request = ModelRequest(parts=[
        SystemPromptPart(content=instruction),
        UserPromptPart(content=prompt),
    ])
    resp_parts: list[ThinkingPart | TextPart] = []
    if thinking:
        resp_parts.append(ThinkingPart(content=thinking))
    for t in (text_parts or []):
        resp_parts.append(TextPart(content=t))
    result.all_messages.return_value = [request, ModelResponse(parts=resp_parts)]
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
    agent = _make_mock_agent(
        output=Score(value=8),
        instruction="you are helpful",
        prompt="rate this",
        thinking="i reasoned carefully",
        text_parts=["thinking..."],
    )

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
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
    assert payload["model"]["service"] == "test"
    assert payload["model"]["model"] == "test-model"
    assert payload["result_type"] == _type_path(Score)
    assert payload["result"] == {"value": 8}

    # turns = full request/response in standard serialization, thinking preserved
    turns = payload["turns"]
    assert len(turns) == 2
    assert [p["part_kind"] for p in turns[0]["parts"]] == ["system-prompt", "user-prompt"]
    assert turns[0]["parts"][0]["content"] == "you are helpful"
    assert turns[0]["parts"][1]["content"] == "rate this"
    assert [p["part_kind"] for p in turns[1]["parts"]] == ["thinking", "text"]
    assert turns[1]["parts"][0]["content"] == "i reasoned carefully"
    assert turns[1]["parts"][1]["content"] == "thinking..."


@pytest.mark.asyncio
async def test_call_auto_uid_name(tmp_path: Path, monkeypatch):
    """export_anchor='' -> auto call-<uid[:8]> name in cwd."""
    monkeypatch.chdir(tmp_path)
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=1), text_parts=["ok"])

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
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

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
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
    turns round-trips the full request/response incl thinking.
    """
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(
        output=Score(value=8),
        instruction="you are helpful",
        prompt="rate this",
        thinking="i reasoned carefully",
        text_parts=["thinking..."],
    )

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
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
    assert rebuilt.model.service == "test"
    assert rebuilt.model.model == "test-model"
    assert rebuilt.result_type == _type_path(Score)
    assert rebuilt.result == {"value": 8}
    assert len(rebuilt.turns) == 2
    assert [p["part_kind"] for p in rebuilt.turns[1]["parts"]] == ["thinking", "text"]
    assert rebuilt.turns[1]["parts"][0]["content"] == "i reasoned carefully"


# ── consume ────────────────────────────────────────────────────────────


def test_anchor_from_file_round_trip(tmp_path: Path):
    """dump_to_dir then from_file restores meta + payload (SPEC §3/§4)."""
    anchor = Anchor(
        meta=AnchorMeta(name="trip", description="d", ref="https://example.com/x"),
        payload={"a": 1, "b": ["x", "y"]},
    )
    path = anchor.dump_to_dir(tmp_path, "trip")
    rebuilt = Anchor.from_file(path)
    assert rebuilt.meta.uid == anchor.meta.uid
    assert rebuilt.meta.name == "trip"
    assert rebuilt.meta.ref == "https://example.com/x"
    assert rebuilt.payload == {"a": 1, "b": ["x", "y"]}


@pytest.mark.asyncio
async def test_call_consumes_anchor(tmp_path: Path):
    """input_anchor (Anchor object) re-injects turns as message_history.

    The history passed to agent.run is the deserialized turn chain — request
    and response incl thinking preserved (内观). The anchor object is the
    constraint the interface takes; no paths at this layer.
    """
    funcs = PydanticAIFuncs()
    producer = _make_mock_agent(
        output=Score(value=8),
        instruction="base instruction",
        prompt="first call",
        thinking="i reasoned carefully",
        text_parts=["ok"],
    )
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=producer):
        produced = await funcs.call(
            instruction="base instruction",
            prompt="first call",
            result_type=Score,
            model=_make_resolved(),
            export_anchor=tmp_path / "base",
        )
    assert produced.anchor is not None

    consumer = _make_mock_agent(output=Score(value=9), text_parts=["done"])
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=consumer):
        await funcs.call(
            instruction="continue",
            prompt="follow up",
            result_type=Score,
            model=_make_resolved(),
            input_anchor=produced.anchor,
        )

    history = consumer.run.await_args.kwargs["message_history"]
    assert history is not None
    assert len(history) == 2
    assert [p.part_kind for p in history[0].parts] == ["system-prompt", "user-prompt"]
    assert [p.part_kind for p in history[1].parts] == ["thinking", "text"]
    assert history[1].parts[0].content == "i reasoned carefully"


@pytest.mark.asyncio
async def test_call_consumes_anchor_from_file(tmp_path: Path):
    """Path → Anchor.from_file (data-structure self-explaining) → consume."""
    funcs = PydanticAIFuncs()
    producer = _make_mock_agent(
        output=Score(value=8),
        instruction="base instruction",
        prompt="first call",
        thinking="i reasoned carefully",
        text_parts=["ok"],
    )
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=producer):
        produced = await funcs.call(
            instruction="base instruction",
            prompt="first call",
            result_type=Score,
            model=_make_resolved(),
            export_anchor=tmp_path / "base",
        )
    path = _anchor_files(tmp_path)[0]

    consumer = _make_mock_agent(output=Score(value=9), text_parts=["done"])
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=consumer):
        await funcs.call(
            instruction="continue",
            prompt="follow up",
            result_type=Score,
            model=_make_resolved(),
            input_anchor=Anchor.from_file(path),
        )

    history = consumer.run.await_args.kwargs["message_history"]
    assert history is not None
    assert len(history) == 2
    assert history[1].parts[0].content == "i reasoned carefully"


@pytest.mark.asyncio
async def test_call_unsupported_anchor_ref(tmp_path: Path):
    """A non-CallAnchor payload is rejected with NotImplementedError.

    LLMFunc supports a narrow set of anchors; unsupported ones fail loudly.
    The strong type (from_anchor structural validation) is the judge — no
    manual ref-string comparison.
    """
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=1))
    foreign = Anchor(
        meta=AnchorMeta(name="foreign", ref="https://example.com/other-payload"),
        payload={"foo": "bar"},
    )
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        with pytest.raises(NotImplementedError, match="unsupported input anchor"):
            await funcs.call(
                instruction="",
                prompt="hi",
                result_type=Score,
                model=_make_resolved(),
                input_anchor=foreign,
            )
    agent.run.assert_not_called()


@pytest.mark.asyncio
async def test_call_request_frame_anchor_cold_start(tmp_path: Path):
    """An anchor with empty turns (failed-call request frame) → cold start."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=1))
    request_frame = CallAnchor(
        instruction="x",
        model=ModelRef(service="test", protocol="openai", model="test-model"),
        result_type=_type_path(Score),
        turns=[],
    ).to_anchor(name="req")
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        await funcs.call(
            instruction="",
            prompt="hi",
            result_type=Score,
            model=_make_resolved(),
            input_anchor=request_frame,
        )
    assert agent.run.await_args.kwargs["message_history"] is None


# ── thinking (内观) ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_call_thinking_injects_introspection(tmp_path: Path):
    """thinking -> a lone ModelResponse(ThinkingPart) is the message_history.

    The block is injected as the model's OWN prior reasoning (内观) — not a
    user prompt that invites a reply (外观).
    """
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=Score(value=5), text_parts=["ok"])
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        await funcs.call(
            instruction="",
            prompt="question",
            result_type=Score,
            model=_make_resolved(),
            thinking="my established position",
        )
    history = agent.run.await_args.kwargs["message_history"]
    assert len(history) == 1
    assert isinstance(history[0], ModelResponse)
    assert [p.part_kind for p in history[0].parts] == ["thinking"]
    assert history[0].parts[0].content == "my established position"


@pytest.mark.asyncio
async def test_call_thinking_chains_into_anchor(tmp_path: Path):
    """The injected thinking block lands in the produced anchor's turns."""
    funcs = PydanticAIFuncs()
    result = MagicMock()
    result.output = Score(value=7)
    result.usage = None
    result.all_messages.return_value = [
        ModelResponse(parts=[ThinkingPart(content="my position")]),
        ModelRequest(parts=[
            SystemPromptPart(content="sys"),
            UserPromptPart(content="question"),
        ]),
        ModelResponse(parts=[TextPart(content="answer")]),
    ]
    agent = MagicMock()
    agent.run = AsyncMock(return_value=result)
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
        await funcs.call(
            instruction="sys",
            prompt="question",
            result_type=Score,
            model=_make_resolved(),
            export_anchor=tmp_path / "t",
            thinking="my position",
        )

    meta, payload = _read_anchor(_anchor_files(tmp_path)[0])
    turns = payload["turns"]
    assert len(turns) == 3
    assert turns[0]["kind"] == "response"
    assert [p["part_kind"] for p in turns[0]["parts"]] == ["thinking"]
    assert turns[0]["parts"][0]["content"] == "my position"


@pytest.mark.asyncio
async def test_call_thinking_after_anchor_history(tmp_path: Path):
    """With input_anchor, the thinking block follows the anchor turns."""
    funcs = PydanticAIFuncs()
    producer = _make_mock_agent(
        output=Score(value=8),
        instruction="base instruction",
        prompt="first call",
        thinking="prior reasoning",
        text_parts=["ok"],
    )
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=producer):
        produced = await funcs.call(
            instruction="base instruction",
            prompt="first call",
            result_type=Score,
            model=_make_resolved(),
            export_anchor=tmp_path / "base",
        )
    assert produced.anchor is not None

    consumer = _make_mock_agent(output=Score(value=9), text_parts=["done"])
    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=consumer):
        await funcs.call(
            instruction="continue",
            prompt="follow up",
            result_type=Score,
            model=_make_resolved(),
            input_anchor=produced.anchor,
            thinking="fresh position",
        )
    history = consumer.run.await_args.kwargs["message_history"]
    assert len(history) == 3
    assert [p.part_kind for p in history[0].parts] == ["system-prompt", "user-prompt"]
    assert [p.part_kind for p in history[1].parts] == ["thinking", "text"]
    assert [p.part_kind for p in history[2].parts] == ["thinking"]
    assert history[2].parts[0].content == "fresh position"


# ── failure ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_frame_survives_failed_call(tmp_path: Path):
    """Request anchor is dumped before the call — a failed call keeps it."""
    funcs = PydanticAIFuncs()
    agent = _make_mock_agent(output=None, exc=RuntimeError("boom"))

    with patch("ghoshell_moss.llms.pydantic_ai_adapter.client.build_agent", return_value=agent):
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
    assert payload["model"]["service"] == "test"
    assert payload["result_type"] == _type_path(Score)
    assert payload["turns"] == []  # request frame: no history yet
    assert "result" not in payload

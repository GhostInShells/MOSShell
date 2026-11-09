"""agentPresets 读面行为证据 — 信封 args + 返回值建模.

覆盖:
- ``agentPresets/list`` 是无参动词: 信封 args 必须是 ``{}``, 不是 ``{"request": None}``.
- ``agentPresets/read`` 的唯一参数是 preset id: args 必须是 ``{"agentPreset": <id>}``.
- roster 的坏声明: ``broken`` 有值时仍然留在 roster 里, 与「消失」分开.
- dsh 会省略可选字段 (name / description / broken); 缺席不等于坏掉.

钉的是**发给 dsh 的信封形状** —— 那是与引擎的契约, 不是内部实现细节;
``_invoke`` 只是把它截下来看的地方.
"""

import logging

import pytest

from ghoshell_dsh.v0_2_0.client import DshClient
from ghoshell_dsh.v0_2_0.types.presets import (
    AgentPresetDocument,
    AgentPresetRoster,
    AgentPresetRow,
)


class _RecordingClient(DshClient):
    """把 ``_invoke`` 换成脚本应答的 client: 不碰 http, 只验信封与模型解析."""

    def __init__(self, answers: dict[str, dict]) -> None:
        super().__init__("http://dsh.invalid", logging.getLogger("test.presets"))
        self._answers = answers
        self.calls: list[tuple[str, dict]] = []

    async def _invoke(self, method, args, timeout=None):  # type: ignore[override]
        self.calls.append((method, args))
        return self._answers[method]


@pytest.mark.asyncio
async def test_list_is_a_no_arg_verb():
    client = _RecordingClient(
        {"agentPresets/list": {"presets": [{"id": "web", "isDefault": True}]}}
    )
    try:
        roster = await client.agent_presets()
    finally:
        await client.close()

    assert client.calls == [("agentPresets/list", {})]
    assert isinstance(roster, AgentPresetRoster)
    assert [row.id for row in roster.presets] == ["web"]
    assert roster.presets[0].isDefault is True


@pytest.mark.asyncio
async def test_read_sends_the_preset_id_as_the_only_arg():
    client = _RecordingClient(
        {
            "agentPresets/read": {
                "agentPreset": "web",
                "content": "- plugin: a\n- plugin: b\n",
                "name": "Web",
            }
        }
    )
    try:
        document = await client.agent_preset("web")
    finally:
        await client.close()

    assert client.calls == [("agentPresets/read", {"agentPreset": "web"})]
    assert isinstance(document, AgentPresetDocument)
    assert document.agentPreset == "web"
    assert document.content == "- plugin: a\n- plugin: b\n"
    assert document.description is None


def test_a_broken_preset_stays_in_the_roster():
    """坏声明照样在 roster 里 —— 「消失」和「坏掉」是两件不同的事."""
    roster = AgentPresetRoster.model_validate(
        {
            "presets": [
                {"id": "web", "isDefault": True, "name": "Web", "description": "for the web"},
                {"id": "half", "broken": "missing plugin: x"},
            ]
        }
    )
    assert [row.id for row in roster.presets] == ["web", "half"]
    assert roster.presets[1].broken == "missing plugin: x"
    assert roster.presets[1].name is None
    assert roster.presets[1].isDefault is False


def test_row_defaults_match_what_dsh_may_omit():
    row = AgentPresetRow.model_validate({"id": "silent"})
    assert row.name is None
    assert row.description is None
    assert row.broken is None
    assert row.isDefault is False

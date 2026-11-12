"""session/list 与 session/rename — roster 的读和写.

覆盖:
- list_sessions 走 ``_request`` 包络, 解析 items 里的 projections.values.title.
- rename_session 走 ``request`` 包络 (两条 typert 路由的参数键不同), 一个 by-id 的改名.
"""

import asyncio

from ghoshell_dsh.v0_2_0.launcher import DshConnection, DshConnectionConfig


def _conn() -> DshConnection:
    return DshConnection(DshConnectionConfig())


def test_list_sessions_parses_items_and_uses_the_typert_envelope():
    conn = _conn()
    calls: list[tuple[str, object, str]] = []

    async def typert(method, request=None, *, args_key="_request", **kwargs):
        calls.append((method, request, args_key))
        return {
            "items": [
                {
                    "sessionId": "s1",
                    "updatedAt": 1794473812769,
                    "running": True,
                    "blank": False,
                    "cwd": "/proj",
                    "projections": {"values": {"title": "Ψ · 26-11-12 16:56:52"}},
                },
                {"sessionId": "s2", "updatedAt": 1, "running": False, "blank": True},
            ]
        }

    conn.client.typert = typert  # type: ignore[method-assign]
    items = asyncio.run(conn.list_sessions())

    assert [item.sessionId for item in items] == ["s1", "s2"]
    assert items[0].cwd == "/proj"
    assert items[0].projections is not None
    assert items[0].projections.values["title"] == "Ψ · 26-11-12 16:56:52"
    assert items[1].projections is None
    assert calls == [("session/list", {}, "_request")]


def test_rename_session_sends_the_request_under_its_own_key():
    conn = _conn()
    calls: list[tuple[str, object, str]] = []

    async def typert(method, request=None, *, args_key="_request", **kwargs):
        calls.append((method, request, args_key))
        return {}

    conn.client.typert = typert  # type: ignore[method-assign]
    asyncio.run(conn.rename_session("s1", "Ω · 26-11-12 16:56–18:07"))

    assert calls == [
        ("session/rename", {"sessionId": "s1", "title": "Ω · 26-11-12 16:56–18:07"}, "request")
    ]

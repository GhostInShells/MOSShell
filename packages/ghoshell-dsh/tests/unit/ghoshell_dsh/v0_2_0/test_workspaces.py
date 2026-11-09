"""workspace 镜像与变更等待 — follow 帧维护列表, mutation 等自己的状态落地.

覆盖:
- baseline 重建镜像; upsert 增量更新; remove 摘除.
- wait_workspace: 已在镜像里立即返回; 帧后到时被唤醒; 超时返回 None.
- create_workspace: 回执即权威 (无帧也能拿到), 帧到了则以帧为准.
"""

import asyncio

from ghoshell_dsh.v0_2_0.launcher import DshConnection, DshConnectionConfig, WorkspaceView


def _row(workspace_id: str, path: str, title: str = "") -> dict:
    return {
        "workspaceId": workspace_id,
        "path": path,
        "title": title or path.rsplit("/", 1)[-1],
        "sessionIds": [],
        "createdAt": "2026-11-10T00:00:00Z",
        "updatedAt": "2026-11-10T00:00:00Z",
    }


def _conn() -> DshConnection:
    return DshConnection(DshConnectionConfig())


def test_baseline_rebuilds_the_mirror():
    conn = _conn()
    conn._dispatch_workspace_frame(
        {"type": "baseline", "value": {"items": [_row("w1", "/tmp/one"), _row("w2", "/tmp/two")]}}
    )
    assert [item.workspaceId for item in conn._workspaces.values()] == ["w1", "w2"]
    assert conn._workspace_baseline_event.is_set()


def test_upsert_and_remove_keep_the_mirror_live():
    conn = _conn()
    conn._dispatch_workspace_frame({"type": "baseline", "value": {"items": [_row("w1", "/tmp/one")]}})
    conn._dispatch_workspace_frame({"type": "upsert", "workspace": _row("w2", "/tmp/two")})
    assert set(conn._workspaces) == {"w1", "w2"}
    conn._dispatch_workspace_frame({"type": "upsert", "workspace": _row("w1", "/tmp/one", "renamed")})
    assert conn._workspaces["w1"].title == "renamed"
    conn._dispatch_workspace_frame({"type": "remove", "workspaceId": "w1"})
    assert set(conn._workspaces) == {"w2"}


def test_wait_returns_at_once_when_already_present():
    conn = _conn()
    conn._workspaces["w1"] = WorkspaceView.model_validate(_row("w1", "/tmp/one"))
    assert asyncio.run(conn.wait_workspace("w1", timeout=0.01)) is not None


def test_wait_wakes_up_when_the_frame_arrives():
    conn = _conn()

    async def scenario():
        task = asyncio.create_task(conn.wait_workspace("w9", timeout=1.0))
        await asyncio.sleep(0.01)
        conn._dispatch_workspace_frame({"type": "upsert", "workspace": _row("w9", "/tmp/nine")})
        return await task

    assert asyncio.run(scenario()).workspaceId == "w9"


def test_wait_times_out_without_a_frame():
    conn = _conn()
    assert asyncio.run(conn.wait_workspace("absent", timeout=0.05)) is None


def test_create_workspace_uses_the_receipt_when_no_frame_comes():
    conn = _conn()
    calls: list[tuple[str, dict]] = []

    async def remote(method, request, **kwargs):
        calls.append((method, request))
        return {"workspace": _row("w3", "/tmp/three"), "created": True}

    conn.client.remote = remote  # type: ignore[method-assign]
    view, created = asyncio.run(conn.create_workspace("/tmp/three", wait=0.05))
    assert (view.workspaceId, created) == ("w3", True)
    assert "w3" in conn._workspaces
    assert calls[0][0] == "workspace/create"

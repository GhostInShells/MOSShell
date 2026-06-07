"""scope meta + session metadata 全链路集成验证。

用临时 workspace 启动 Host，验证 scope meta 创建/删除 + sessions.jsonl 追加 + meta.yaml 写。
不依赖项目 .moss_ws，避免污染开发环境。
"""
import asyncio
import os
import tempfile
from pathlib import Path

from ghoshell_moss.core.blueprint.environment import Environment, ScopeMeta
from ghoshell_moss.core.blueprint.session import SessionRecord, SessionMetadata
from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.host import Host


async def _verify():
    # 1. 创建临时 workspace
    tmp = Path(tempfile.mkdtemp(prefix="moss_test_"))
    workspace = tmp / ".moss_ws"
    Environment.init_workspace(workspace)
    print(f"[1] workspace created: {workspace}")

    scope = "test_metadata_lifecycle"
    env = Environment(workspace, session_scope=scope)
    host = Host(env=env)

    # 2. 启动 host (async context)
    runtime_storage = LocalStorage(workspace / "runtime")
    scope_meta_file = workspace / "runtime" / "scopes" / f"scope-{scope}.yml"
    session = None

    matrix = host.matrix()
    async with matrix as mtx:
        session = mtx.session
        sid = session.session_id
        print(f"[2] host started — session_id={sid}")

        # 3. scope meta 文件存在
        assert scope_meta_file.exists(), "scope meta file not created"
        scope_meta = env.read_scope_meta()
        assert scope_meta is not None
        assert scope_meta.session_id == sid
        assert scope_meta.host_pid == os.getpid()
        print(f"[3] scope meta ok — host_pid={scope_meta.host_pid} session_id={sid}")

        # 4. session metadata (meta.yaml) 已写入
        meta_loaded = session.meta
        assert meta_loaded is not None, "meta.yaml should be readable via Session.meta"
        assert meta_loaded.session_id == sid
        assert meta_loaded.session_scope == scope
        assert meta_loaded.host_pid == os.getpid()
        print(f"[4] session.meta ok — mode={meta_loaded.mode_name} ghost={meta_loaded.ghost_name}")

        # 5. sessions.jsonl 有一条记录
        records = list(session.scope_storage.read_models("sessions", SessionRecord))
        assert len(records) == 1, f"expected 1 record, got {len(records)}"
        assert records[0].session_id == sid
        print(f"[5] sessions.jsonl ok — 1 record: {records[0].session_id}")

        first_sid = sid

    # 6. 退出后 scope meta 已删除
    assert not scope_meta_file.exists(), "scope meta should be deleted on clean exit"
    print("[6] scope meta deleted on exit")

    # 7. 再次启动 — 新 session_id + JSONL 追加
    env2 = Environment(workspace, session_scope=scope)
    host2 = Host(env=env2)
    matrix2 = host2.matrix()
    async with matrix2 as mtx2:
        sid2 = mtx2.session.session_id
        assert sid2 != first_sid, "new session_id should differ"
        print(f"[7] second host started — new session_id={sid2}")

        records2 = list(mtx2.session.scope_storage.read_models("sessions", SessionRecord))
        assert len(records2) == 2, f"expected 2 records, got {len(records2)}"
        assert records2[0].session_id == first_sid
        assert records2[1].session_id == sid2
        print(f"[8] sessions.jsonl ok — 2 records: {first_sid[:8]}... {sid2[:8]}...")

    print(f"\nOK — all checks passed")


if __name__ == "__main__":
    asyncio.run(_verify())

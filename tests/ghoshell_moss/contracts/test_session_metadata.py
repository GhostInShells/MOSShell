"""Tests for ScopeMeta, SessionRecord, SessionMetadata storage flows."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import BaseModel

from ghoshell_moss.contracts.workspace import LocalStorage
from ghoshell_moss.core.blueprint.environment import Environment, ScopeMeta
from ghoshell_moss.core.blueprint.session import SessionMetadata, SessionRecord


# -- ScopeMeta model -------------------------------------------------------


def test_scope_meta_model():
    meta = ScopeMeta(
        session_scope="default",
        session_id="abc123",
        mode="default",
        host_pid=99999,
    )
    d = meta.model_dump()
    assert d["session_scope"] == "default"
    assert d["session_id"] == "abc123"
    assert d["mode"] == "default"
    assert d["host_pid"] == 99999


def test_scope_meta_rejects_missing_fields():
    with pytest.raises(Exception):
        ScopeMeta(session_scope="s")  # missing session_id, mode, host_pid


# -- SessionRecord JSONL roundtrip ------------------------------------------


def test_session_record_jsonl_roundtrip(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    r1 = SessionRecord(session_id="a", created_at="2026-06-07T00:00:00+00:00")
    r2 = SessionRecord(session_id="b", created_at="2026-06-07T01:00:00+00:00")

    storage.append_model("sessions", r1)
    storage.append_model("sessions", r2)

    items = list(storage.read_models("sessions", SessionRecord))
    assert len(items) == 2
    assert items[0].session_id == "a"
    assert items[1].session_id == "b"


def test_session_record_read_empty(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    items = list(storage.read_models("sessions", SessionRecord))
    assert items == []


# -- SessionMetadata YAML roundtrip -----------------------------------------


def test_session_metadata_yaml_roundtrip(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    meta = SessionMetadata(
        session_id="abc",
        session_scope="default",
        mode_name="default",
        ghost_name="mock",
        host_cell_address="host/default",
        host_pid=12345,
        created_at="2026-06-07T00:00:00+00:00",
        title="test session",
    )
    storage.write_yaml("meta", meta)

    loaded = storage.read_yaml("meta", SessionMetadata)
    assert loaded is not None
    assert loaded.session_id == "abc"
    assert loaded.mode_name == "default"
    assert loaded.ghost_name == "mock"
    assert loaded.host_pid == 12345
    assert loaded.title == "test session"
    assert loaded.description == ""


def test_session_metadata_defaults(tmp_path: Path):
    storage = LocalStorage(tmp_path)
    meta = SessionMetadata(
        session_id="x",
        session_scope="s",
        mode_name="m",
        ghost_name="None",
        host_cell_address="host/m",
        host_pid=1,
        created_at="now",
    )
    storage.write_yaml("meta", meta)
    loaded = storage.read_yaml("meta", SessionMetadata)
    assert loaded is not None
    assert loaded.title == ""
    assert loaded.updated_at == ""


# -- Environment scope meta methods -----------------------------------------


def test_write_and_read_scope_meta(tmp_path: Path):
    """写入 scope meta 到约定路径，读回验证。"""
    workspace = tmp_path / ".moss_ws"
    workspace.mkdir()
    # write runtime/scopes dir via env method
    env = Environment(workspace, session_scope="test_scope")
    env.write_scope_meta()
    assert env.scope_meta_path.exists()
    assert 1 == env.scope_meta_path.read_text().count(f"host_pid: {os.getpid()}")

    meta = env.read_scope_meta()
    assert meta is not None
    assert meta.session_scope == "test_scope"
    assert meta.session_id == env.session_id
    assert meta.host_pid == os.getpid()
    assert meta.mode == env.moss_mode_name


def test_read_scope_meta_missing(tmp_path: Path):
    """文件不存在 → None。"""
    workspace = tmp_path / ".moss_ws"
    workspace.mkdir()
    env = Environment(workspace, session_scope="no_file")
    assert env.read_scope_meta() is None


def test_read_scope_meta_zombie_pid(tmp_path: Path):
    """文件存在但 PID 已死 → None。"""
    workspace = tmp_path / ".moss_ws"
    workspace.mkdir()
    env = Environment(workspace, session_scope="zombie")
    env.write_scope_meta()
    assert env.read_scope_meta() is not None  # own pid alive

    # 覆写为不可能存活的 PID
    env.scope_meta_path.write_text(
        env.scope_meta_path.read_text().replace(
            f"host_pid: {os.getpid()}",
            "host_pid: 1",
        )
    )
    # PID 1 (init/launchd) 不可能等于当前进程，但 pid_exists 可能返回 True
    # 用 -1 确保 dead
    env.scope_meta_path.write_text(
        env.scope_meta_path.read_text().replace(
            "host_pid: 1",
            "host_pid: -1",
        )
    )
    assert env.read_scope_meta() is None


def test_delete_scope_meta(tmp_path: Path):
    """删除 scope meta。"""
    workspace = tmp_path / ".moss_ws"
    workspace.mkdir()
    env = Environment(workspace, session_scope="del_test")
    env.write_scope_meta()
    assert env.scope_meta_path.exists()
    env.delete_scope_meta()
    assert not env.scope_meta_path.exists()


def test_delete_scope_meta_missing_ok(tmp_path: Path):
    """删除不存在的文件不报错。"""
    workspace = tmp_path / ".moss_ws"
    workspace.mkdir()
    env = Environment(workspace, session_scope="no_file")
    env.delete_scope_meta()  # no error

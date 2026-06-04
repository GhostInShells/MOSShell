"""set_mode() must update _cell_address when derived from template."""

import os
import pytest
from pathlib import Path

from ghoshell_moss.core.blueprint.environment import (
    Environment, DEFAULT_CELL_ADDRESS, ENV_CELL_ADDRESS_KEY, ENV_MOSS_MODE_KEY,
)


@pytest.fixture(autouse=True)
def clean_env():
    """清理可能影响测试的环境变量."""
    old = {k: os.environ.pop(k, None) for k in [ENV_MOSS_MODE_KEY, ENV_CELL_ADDRESS_KEY]}
    yield
    for k, v in old.items():
        if v is not None:
            os.environ[k] = v


def test_set_mode_updates_cell_address_from_template(tmp_path):
    """当 cell_address 从模板派生时，set_mode 应同步更新."""
    env = Environment(workspace_path=tmp_path, mode="initial")
    assert env.cell_address == DEFAULT_CELL_ADDRESS.format(mode="initial")

    env.set_mode("new_mode")
    assert env.moss_mode_name == "new_mode"
    assert env.cell_address == DEFAULT_CELL_ADDRESS.format(mode="new_mode")


def test_set_mode_preserves_explicit_cell_address(tmp_path):
    """当 MOSS_CELL_ADDRESS 显式指定时，set_mode 不应覆盖."""
    os.environ[ENV_CELL_ADDRESS_KEY] = "fractal/remote_node"
    env = Environment(workspace_path=tmp_path, mode="initial")
    assert env.cell_address == "fractal/remote_node"

    env.set_mode("new_mode")
    assert env.cell_address == "fractal/remote_node"


def test_set_mode_no_env_var_means_template_derived(tmp_path):
    """无 MOSS_CELL_ADDRESS 环境变量时，cell_address 来自模板."""
    assert ENV_CELL_ADDRESS_KEY not in os.environ
    env = Environment(workspace_path=tmp_path, mode="test")
    assert env.cell_address == "host/test"

"""
Cell blueprint abstraction unit tests.

Scope:
  - 三域模型的数据契约 (NodeManifest / CellRuntimeInfo / Cell + CellEvent)
  - ExecSpec 字段与 arguments 拆分
  - NodeManifest 与文件系统往返 (write/read + INSTALL 状态推导 + from_script 匝道)
  - Cell 派生 property (address / fullname / is_host / unique_name)
  - build helpers (build_node_from_manifest / build_host_cell)
  - address 三段结构 (§ZZ-10) + normalize / parse_address
  - 历史锚点负向断言 (WW-3 no interpreter, WW-6 no exit, 契约中不该出现的字段)

Out of scope:
  zenoh 集成 (归 test_zenoh_presence / test_zenoh_mesh).
  matrix 层组装 (归 matrix workstream).
  cell lifecycle 副作用 (enter_cell_lifecycle / discover_this_node, 依赖 workspace lock).
"""

import textwrap
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from ghoshell_moss.core.blueprint.cell import (
    Cell,
    CellEvent,
    CellRuntimeInfo,
    DuplicatedError,
    ExecSpec,
    HOST_ROLE,
    NODE_ROLE,
    NodeLauncher,
    NodeManifest,
    NodeScriptCategory,
    ROLES,
    build_host_cell,
    build_cell_from_node,
    make_address,
    normalize,
    parse_address,
)


# ── ExecSpec ─────────────────────────────────────────────────────────


class TestExecSpec:

    def test_defaults(self):
        spec = ExecSpec()
        assert spec.command == 'python'
        assert spec.args == 'main.py'
        assert spec.env == {}

    def test_arguments_shlex_split(self):
        # args 是 str, .arguments property 走 shlex.split — 引号包裹保留空格.
        spec = ExecSpec(command='python', args="main.py --flag 'value with spaces'")
        assert spec.arguments == ['main.py', '--flag', 'value with spaces']

    def test_custom_command(self):
        spec = ExecSpec(command='uv', args='run main.py')
        assert spec.command == 'uv'
        assert spec.arguments == ['run', 'main.py']

    def test_no_interpreter_field(self):
        # WW-3: interpreter 融合 launcher⊗package, 从字段面永久剔除.
        assert 'interpreter' not in ExecSpec.model_fields

    def test_no_from_run_sugar(self):
        # 早期 ExecSpec.from_run/to_run 糖已收敛回字段直填 (frontmatter 层还有 `run:` 糖).
        assert not hasattr(ExecSpec, 'from_run')
        assert not hasattr(ExecSpec, 'to_run')


# ── NodeManifest ──────────────────────────────────────────────────────


class TestNodeManifest:

    def test_write_read_roundtrip(self, tmp_path: Path):
        original = NodeManifest(
            name='g1_loco',
            category='bodies',
            singleton=True,
            description='g1 locomotion',
            exec=ExecSpec(command='python', args='-m g1.locomotion', env={'RATE': '50'}),
            instruction='body doc',
        )
        original.write_file(tmp_path)
        loaded = NodeManifest.read_from_file(tmp_path / NodeManifest.MANIFEST_FILENAME)
        assert loaded.name == original.name
        assert loaded.category == original.category
        assert loaded.singleton == original.singleton
        assert loaded.description == original.description
        assert loaded.instruction == original.instruction
        assert loaded.exec.command == original.exec.command
        assert loaded.exec.args == original.exec.args
        assert loaded.exec.env == original.exec.env

    def test_installed_recovers_from_marker(self, tmp_path: Path):
        (tmp_path / NodeManifest.MANIFEST_FILENAME).write_text(
            textwrap.dedent(
                """\
                ---
                name: needs_install
                ---
                requires deps
                """,
            ),
        )
        # 无 INSTALL_FILENAME → installed 默认视为 True (无额外依赖).
        m1 = NodeManifest.read_from_file(tmp_path / NodeManifest.MANIFEST_FILENAME)
        assert m1.installed is True

        (tmp_path / NodeManifest.INSTALL_FILENAME).write_text('pip install foo')
        # 有 INSTALL_FILENAME 但无 INSTALLED_FILE → installed=False (需完成安装).
        m2 = NodeManifest.read_from_file(tmp_path / NodeManifest.MANIFEST_FILENAME)
        assert m2.installed is False

        (tmp_path / NodeManifest.INSTALLED_FILE).touch()
        m3 = NodeManifest.read_from_file(tmp_path / NodeManifest.MANIFEST_FILENAME)
        assert m3.installed is True

    def test_singleton_default_true(self):
        m = NodeManifest(name='n')
        assert m.singleton is True

    def test_installed_not_persisted_to_frontmatter(self, tmp_path: Path):
        # installed 由文件系统推导, 不落 frontmatter — 避免 declared vs actual 双源.
        m = NodeManifest(name='x', installed=False)
        m.write_file(tmp_path)
        content = (tmp_path / NodeManifest.MANIFEST_FILENAME).read_text()
        assert 'installed' not in content

    def test_read_from_directory_missing_returns_none(self, tmp_path: Path):
        assert NodeManifest.read_from_directory(tmp_path) is None

    def test_read_from_directory_hits(self, tmp_path: Path):
        (tmp_path / NodeManifest.MANIFEST_FILENAME).write_text(
            "---\nname: x\n---\nbody\n"
        )
        m = NodeManifest.read_from_directory(tmp_path)
        assert m is not None
        assert m.name == 'x'
        assert m.instruction == 'body'

    def test_cwd_derives_from_file(self, tmp_path: Path):
        m = NodeManifest(name='x', file=str((tmp_path / 'sub' / NodeManifest.MANIFEST_FILENAME)))
        assert m.cwd == (tmp_path / 'sub').resolve()


class TestFromScript:

    def test_find_upward_hits_ancestor(self, tmp_path: Path):
        (tmp_path / NodeManifest.MANIFEST_FILENAME).write_text(
            "---\nname: ancestor_cell\n---\n"
        )
        deep = tmp_path / 'a' / 'b' / 'c'
        deep.mkdir(parents=True)
        script = deep / 'entry.py'
        script.write_text('print(1)')
        m = NodeManifest.from_script(script)
        assert m.name == 'ancestor_cell'
        # from_script 会用脚本 exec_spec 覆写继承来的 exec.
        assert str(script) in m.exec.args

    def test_downgrade_when_no_ancestor(self, tmp_path: Path):
        # 找不到 ancestor NODE.md → 临时身份, 不拒绝运行.
        script = tmp_path / 'lone.py'
        script.write_text('print(1)')
        m = NodeManifest.from_script(script)
        assert m.name == script.stem
        assert m.category == NodeScriptCategory
        # 匝道降级: command = sys.executable 绝对路径.
        import sys
        assert m.exec.command == sys.executable
        assert str(script) in m.exec.args

    def test_custom_exec_spec_overrides(self, tmp_path: Path):
        script = tmp_path / 'lone.py'
        script.write_text('')
        override = ExecSpec(command='uv', args='run lone.py')
        m = NodeManifest.from_script(script, exec_spec=override)
        assert m.exec.command == 'uv'
        assert m.exec.args == 'run lone.py'


# ── CellRuntimeInfo ──────────────────────────────────────────────────


def _make_cell(role: str = NODE_ROLE, name: str = 'x', **overrides) -> Cell:
    defaults = dict(
        role=role, name=name, home='/tmp/x',
    )
    defaults.update(overrides)
    return Cell(**defaults)


class TestCellRuntimeInfo:

    def test_minimum_fields(self):
        cell = _make_cell()
        info = CellRuntimeInfo(address=cell.address, cell=cell)
        assert info.pid == 0
        assert info.pgid == 0
        assert info.start_time > 0  # default_factory = time.time

    def test_no_exit_fields(self):
        # WW-6: ledger 不加 exit 记录 (单写者原则 — spawn 现场唯一写入时机).
        cell = _make_cell()
        info = CellRuntimeInfo(address=cell.address, cell=cell)
        for banned in ('exit_code', 'exited_at', 'exit_status'):
            assert not hasattr(info, banned), (
                f"CellRuntimeInfo leaked {banned!r} — exit 信息归 CellEvent + Signal, "
                "不入 ledger (WW-6)."
            )

    def test_locker_name_uses_fullname(self):
        # locker_name 用 fullname (category_name), 不含 uid — 治理域内 fullname 唯一即锁唯一.
        cell = _make_cell(name='cam', category='sensors')
        info = CellRuntimeInfo(address=cell.address, cell=cell)
        assert info.locker_name() == normalize(cell.fullname)
        assert cell.uid not in info.locker_name()

    def test_write_read_runtime_dir_roundtrip(self, tmp_path: Path):
        cell = _make_cell()
        info = CellRuntimeInfo(address=cell.address, pid=12345, pgid=12345, cell=cell)
        info.write_to_runtime_dir(tmp_path)
        loaded = CellRuntimeInfo.read_from_runtime_dir(tmp_path, cell.address)
        assert loaded is not None
        assert loaded.address == info.address
        assert loaded.pid == info.pid
        assert loaded.cell.name == cell.name

    def test_read_deletes_invalid(self, tmp_path: Path):
        cell = _make_cell()
        path = CellRuntimeInfo.filepath(tmp_path, cell.address)
        path.write_text('not valid json')
        assert CellRuntimeInfo.read_from_runtime_dir(tmp_path, cell.address) is None
        assert not path.exists(), "invalid runtime file should be deleted on read"

    def test_iter_runtime_info_skips_invalid(self, tmp_path: Path):
        cell = _make_cell()
        info = CellRuntimeInfo(address=cell.address, cell=cell)
        info.write_to_runtime_dir(tmp_path)
        (tmp_path / 'junk.json').write_text('{ not json')
        found = list(CellRuntimeInfo.iter_runtime_info(tmp_path))
        assert len(found) == 1
        assert found[0].address == cell.address


# ── Cell ─────────────────────────────────────────────────────────────


class TestCell:

    def test_address_is_property_three_segment(self):
        cell = Cell(role=NODE_ROLE, name='cam', uid='ABC12345', home='/tmp')
        # §ZZ-10 三段结构.
        assert cell.address == f'{NODE_ROLE}/cam/ABC12345'

    def test_fullname_without_category(self):
        cell = _make_cell(name='cam')
        assert cell.fullname == 'cam'

    def test_fullname_with_category(self):
        cell = _make_cell(name='cam', category='sensors')
        assert cell.fullname == 'sensors_cam'

    def test_is_host_reflects_role(self):
        assert _make_cell(role=HOST_ROLE, name='m').is_host is True
        assert _make_cell(role=NODE_ROLE, name='m').is_host is False

    def test_unique_name_combines_fullname_and_uid_prefix(self):
        cell = Cell(role=NODE_ROLE, name='cam', uid='ABC12345XYZ', home='/tmp')
        # unique_name = fullname + uid[:8]
        assert cell.unique_name == f'cam_{"ABC12345"}'

    def test_providing_default_empty(self):
        assert _make_cell().providing == []

    def test_providing_rejects_unknown_literal(self):
        # 膜类型是 matrix 版本演进事件 — cell 作者不能自扩.
        with pytest.raises(ValidationError):
            _make_cell(providing=['resource'])

    def test_role_rejects_unknown(self):
        with pytest.raises(ValidationError):
            Cell(role='bridge', name='x', home='/tmp')

    def test_is_local_matches_env_project_id(self):
        env = Mock()
        env.project_id = 'proj-a'
        assert _make_cell(project_id='proj-a').is_local(env) is True
        assert _make_cell(project_id='proj-b').is_local(env) is False

    def test_no_state_field(self):
        # 历史锚点: 曾一度加过 CellState.state / failure 字段, 现已退回 Cell 数据模型
        # 只描述"存在与提供什么", 不描述生命周期状态 — 状态由 CellEvent 承载 (推拉).
        for banned in ('state', 'failure'):
            assert banned not in Cell.model_fields, (
                f"Cell leaked {banned!r} — 生命周期状态归 CellEvent, 不入 Cell 数据模型."
            )

    def test_no_address_field(self):
        # address 是 property (make_address(role, name, uid)), 不是可直接设的字段.
        assert 'address' not in Cell.model_fields

    def test_update_touches_updated_timestamp(self):
        cell = _make_cell()
        before = cell.updated
        # 强制不同时间戳.
        cell.updated = cell.updated.replace(year=2020)
        cell.update()
        assert cell.updated != before or cell.updated.year != 2020

    def test_serialization_roundtrip(self):
        cell = Cell(
            role=NODE_ROLE, name='cam', uid='ABC12345', home='/tmp',
            category='sensors', project_id='proj-a', providing=['channel'],
        )
        js = cell.model_dump_json()
        loaded = Cell.model_validate_json(js)
        assert loaded == cell
        assert loaded.address == cell.address


# ── Build helpers ────────────────────────────────────────────────────


def _mock_env(tmp_path: Path, **overrides):
    env = Mock()
    env.project_id = overrides.get('project_id', 'proj-x')
    env.project_name = overrides.get('project_name', 'projx')
    env.cell_runtimes_dir = tmp_path / 'runtime'
    env.cell_runtimes_dir.mkdir(exist_ok=True)
    env.workspace_path = tmp_path
    env.this_cell_address = overrides.get('this_cell_address', '')
    env.dump_cell_env = Mock(return_value={})
    if 'moss_meta' in overrides:
        env.moss_meta = overrides['moss_meta']
    else:
        meta = Mock()
        meta.name = 'moss_test'
        env.moss_meta = meta
    return env


class TestBuildHelpers:

    def test_build_node_home_from_manifest_file(self, tmp_path: Path):
        env = _mock_env(tmp_path)
        manifest_dir = tmp_path / 'apps' / 'cam'
        manifest_dir.mkdir(parents=True)
        manifest = NodeManifest(
            name='cam',
            file=str(manifest_dir / NodeManifest.MANIFEST_FILENAME),
        )
        cell = build_cell_from_node(env, manifest)
        assert cell.role == NODE_ROLE
        assert cell.name == 'cam'
        assert cell.home == str(manifest_dir.resolve())
        assert cell.project_id == 'proj-x'
        assert cell.project_name == 'projx'

    def test_build_node_home_fallback_when_file_missing(self, tmp_path: Path):
        env = _mock_env(tmp_path)
        manifest = NodeManifest(name='cam', category='sensors')  # file=''
        cell = build_cell_from_node(env, manifest)
        # 无 file → 临时 workspace 在 env.cell_runtimes_dir/{fullname}
        expected = (env.cell_runtimes_dir / cell.fullname).resolve()
        assert cell.home == str(expected)

    def test_build_node_uid_per_call(self, tmp_path: Path):
        env = _mock_env(tmp_path)
        manifest = NodeManifest(name='cam')
        c1 = build_cell_from_node(env, manifest)
        c2 = build_cell_from_node(env, manifest)
        assert c1.uid != c2.uid, "每次 spawn 独立 uid, 保证 address 全局唯一"

    def test_build_node_alias_overrides_name(self, tmp_path: Path):
        env = _mock_env(tmp_path)
        manifest = NodeManifest(name='cam')
        cell = build_cell_from_node(env, manifest, name='cam_alias')
        assert cell.name == 'cam_alias'

    def test_build_host_uses_moss_meta(self, tmp_path: Path):
        env = _mock_env(tmp_path)
        env.project_id = 'proj-42'
        cell = build_host_cell(env)
        assert cell.role == HOST_ROLE
        assert cell.name == env.moss_meta.name
        assert cell.uid == 'proj-42'
        assert cell.singleton is True
        assert cell.home == str(tmp_path.absolute())
        # host address = host / {moss_name} / {project_id}
        assert cell.address == f'{HOST_ROLE}/{env.moss_meta.name}/proj-42'


# ── NodeLauncher ─────────────────────────────────────────────────────


class TestNodeLauncher:
    """NodeLauncher 组装是 project.NodeManager.get_node_launcher 的下层,
    这里覆盖 python 关键字解析 + dump_cell_env 传参约定."""

    def test_python_command_resolves_to_sys_executable(self, tmp_path: Path):
        import sys
        env = _mock_env(tmp_path)
        manifest = NodeManifest(name='x', exec=ExecSpec(command='python', args='main.py'))
        launcher = NodeLauncher.from_manifest(env, manifest)
        assert launcher.run[0] == sys.executable
        assert launcher.run[1:] == ['main.py']

    def test_custom_command_pass_through(self, tmp_path: Path):
        env = _mock_env(tmp_path)
        manifest = NodeManifest(name='x', exec=ExecSpec(command='uv', args='run main.py'))
        launcher = NodeLauncher.from_manifest(env, manifest)
        assert launcher.run == ['uv', 'run', 'main.py']

    def test_dump_cell_env_receives_address_pair(self, tmp_path: Path):
        env = _mock_env(tmp_path, this_cell_address='host/moss_test/proj-x')
        manifest = NodeManifest(name='x')
        launcher = NodeLauncher.from_manifest(env, manifest)
        env.dump_cell_env.assert_called_once()
        kwargs = env.dump_cell_env.call_args.kwargs
        assert kwargs['cell_address'] == launcher.runtime.address
        assert kwargs['parent_cell_address'] == 'host/moss_test/proj-x'


# ── CellEvent ────────────────────────────────────────────────────────


class TestCellEvent:

    def test_refetch_default_true(self):
        e = CellEvent(address='node/x/uid1')
        assert e.refetch is True

    def test_refetch_false_explicit(self):
        e = CellEvent(address='node/x/uid1', refetch=False, content='pure log')
        assert e.refetch is False

    def test_no_kind_field(self):
        # 2026-07-11 明示: 不做 kind 枚举, 用 content 自由文本.
        assert 'kind' not in CellEvent.model_fields

    def test_no_terminal_field(self):
        # cell 下线由 liveness 消失承载, 不在 event 上做 terminal 标记.
        assert 'terminal' not in CellEvent.model_fields


# ── address 三段结构 (§ZZ-10) ─────────────────────────────────────────


class TestAddress:

    def test_make_address_valid(self):
        assert make_address(NODE_ROLE, 'cam', 'uid8') == f'{NODE_ROLE}/cam/uid8'
        assert make_address(HOST_ROLE, 'moss', 'proj-1') == f'{HOST_ROLE}/moss/proj-1'

    def test_make_address_rejects_unknown_role(self):
        with pytest.raises(ValueError):
            make_address('bridge', 'x', 'uid')

    def test_make_address_rejects_empty_segment(self):
        with pytest.raises(ValueError):
            make_address(NODE_ROLE, '', 'uid')
        with pytest.raises(ValueError):
            make_address(NODE_ROLE, 'x', '')

    def test_make_address_rejects_slash_in_segment(self):
        with pytest.raises(ValueError):
            make_address(NODE_ROLE, 'a/b', 'uid')

    def test_parse_address_roundtrip(self):
        addr = make_address(NODE_ROLE, 'cam', 'uid8')
        role, name, uid = parse_address(addr)
        assert (role, name, uid) == (NODE_ROLE, 'cam', 'uid8')

    def test_parse_address_wrong_segment_count(self):
        with pytest.raises(ValueError):
            parse_address(f'{NODE_ROLE}/cam')
        with pytest.raises(ValueError):
            parse_address(f'{NODE_ROLE}/a/b/c')

    def test_parse_address_rejects_unknown_role(self):
        with pytest.raises(ValueError):
            parse_address('bridge/x/uid')

    def test_roles_frozenset_matches_literals(self):
        assert ROLES == frozenset({HOST_ROLE, NODE_ROLE})


# ── normalize ────────────────────────────────────────────────────────


class TestNormalize:

    def test_replaces_all_separators(self):
        # 覆盖 / \ . - 四种分隔符, 输出可作 filename + python identifier.
        assert normalize('worker/cam.front-1\\sub') == 'worker_cam_front_1_sub'


# ── DuplicatedError ──────────────────────────────────────────────────


class TestDuplicatedError:

    def test_is_runtime_error(self):
        # singleton 执法产物, 走 RuntimeError 而不是 ValueError — 触发时机是运行时冲突.
        assert issubclass(DuplicatedError, RuntimeError)

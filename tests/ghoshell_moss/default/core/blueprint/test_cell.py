"""
Cell blueprint abstraction unit tests.

Scope:
  三域模型 (CellManifest / CellRecord / CellPresence) 的数据契约,
  ExecSpec run: 糖的双向语义, CELL.md 文件读写往返,
  向上认亲 (find_upward / from_script) 与匝道降级,
  CellEvent refetch 默认与 Watcher/Presence ABC 完整性.

Out of scope:
  zenoh 集成 (需两 session, 归 test_zenoh_presence_watcher.py, 后补).
  matrix 层集成 (归 matrix workstream, 与本文件无关).
"""

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from ghoshell_moss.core.blueprint.cell import (
    CellEvent,
    CellManifest,
    CellPresence,
    CellRecord,
    CellState,
    DuplicatedError,
    ExecSpec,
    Presence,
    Watcher,
    make_address,
    normalize,
)


# ── ExecSpec ─────────────────────────────────────────────────────────


class TestExecSpec:

    def test_from_run_string(self):
        spec = ExecSpec.from_run('python main.py --flag')
        assert spec.command == 'python'
        assert spec.args == ['main.py', '--flag']
        assert spec.env == {}

    def test_from_run_list(self):
        spec = ExecSpec.from_run(['uv', 'run', 'main.py'])
        assert spec.command == 'uv'
        assert spec.args == ['run', 'main.py']

    def test_from_run_with_env(self):
        spec = ExecSpec.from_run('python main.py', env={'A': '1'})
        assert spec.env == {'A': '1'}

    def test_from_run_empty_rejected(self):
        with pytest.raises(ValueError):
            ExecSpec.from_run('')
        with pytest.raises(ValueError):
            ExecSpec.from_run([])

    def test_to_run_roundtrip(self):
        original = 'python main.py --flag value with spaces'
        spec = ExecSpec.from_run(original)
        # shlex.join 引号包裹带空格的段, 语义等价.
        rebuilt = ExecSpec.from_run(spec.to_run())
        assert rebuilt.command == spec.command
        assert rebuilt.args == spec.args

    def test_no_interpreter_field(self):
        # WW-3: interpreter 字段死. ExecSpec 只有 command/args/env.
        spec = ExecSpec(command='python')
        assert not hasattr(spec, 'interpreter')


# ── CellManifest ──────────────────────────────────────────────────────


class TestCellManifest:

    def test_write_read_roundtrip(self, tmp_path: Path):
        original = CellManifest(
            name='g1_loco',
            taxonomy='bodies',
            singleton='host',
            description='g1 locomotion',
            exec=ExecSpec.from_run(
                '../../.venv/bin/python -m g1.locomotion',
                env={'RATE': '50'},
            ),
            instruction='body doc',
        )
        original.write_file(tmp_path)
        loaded = CellManifest.read_from_file(tmp_path / 'CELL.md')
        assert loaded == original

    def test_installed_recovers_from_marker(self, tmp_path: Path):
        (tmp_path / 'CELL.md').write_text(
            textwrap.dedent(
                """\
                ---
                name: needs_install
                run: python main.py
                ---
                requires deps
                """,
            ),
        )
        # 无 INSTALL.md → installed=True.
        m1 = CellManifest.read_from_file(tmp_path / 'CELL.md')
        assert m1.installed is True

        (tmp_path / 'INSTALL.md').write_text('run: pip install foo')
        # 有 INSTALL.md 但无 .installed → installed=False.
        m2 = CellManifest.read_from_file(tmp_path / 'CELL.md')
        assert m2.installed is False

        (tmp_path / '.installed').touch()
        m3 = CellManifest.read_from_file(tmp_path / 'CELL.md')
        assert m3.installed is True

    def test_singleton_domain_default_none(self):
        m = CellManifest(name='n', exec=ExecSpec(command='python'))
        assert m.singleton == 'none'

    def test_singleton_literal_strict(self):
        with pytest.raises(ValidationError):
            CellManifest(name='n', singleton='rogue')

    def test_installed_not_in_frontmatter(self, tmp_path: Path):
        m = CellManifest(
            name='x',
            exec=ExecSpec.from_run('python main.py'),
            installed=False,
        )
        m.write_file(tmp_path)
        content = (tmp_path / 'CELL.md').read_text()
        assert 'installed' not in content

    def test_exec_serialized_as_run_sugar(self, tmp_path: Path):
        m = CellManifest(
            name='x',
            exec=ExecSpec(command='python', args=['main.py']),
        )
        m.write_file(tmp_path)
        content = (tmp_path / 'CELL.md').read_text()
        # run: 糖, 不是嵌套 dict.
        assert 'run:' in content
        assert 'exec:' not in content


class TestFromScript:

    def test_find_upward_hits_ancestor(self, tmp_path: Path):
        (tmp_path / 'CELL.md').write_text(
            textwrap.dedent(
                """\
                ---
                name: ancestor_cell
                run: python main.py
                ---
                """,
            ),
        )
        deep = tmp_path / 'a' / 'b' / 'c'
        deep.mkdir(parents=True)
        script = deep / 'entry.py'
        script.write_text('print(1)')
        m = CellManifest.from_script(script)
        assert m.name == 'ancestor_cell'
        assert m.taxonomy == ''  # 从 CELL.md 读, 未声明为空.

    def test_from_script_downgrade_when_no_manifest(self, tmp_path: Path):
        script = tmp_path / 'lone.py'
        script.write_text('print(1)')
        m = CellManifest.from_script(script)
        assert m.taxonomy == 'script'
        assert m.name.startswith('lone_')
        assert m.exec is not None
        assert m.exec.command == 'python'
        assert str(script) in m.exec.args


# ── CellRecord ────────────────────────────────────────────────────────


class TestCellRecord:

    def test_minimum_fields(self):
        r = CellRecord(
            address='w/x/y',
            pid=1234,
            start_time=1000.0,
            cwd='/tmp',
        )
        assert r.alias == ''
        assert r.pgid == 0
        assert r.spawner == ''

    def test_no_exit_fields(self):
        # WW-6: ledger 不加 exit 记录, 单写者原则.
        r = CellRecord(address='a', pid=1, start_time=0.0, cwd='/tmp')
        for banned in ('exit_code', 'exited_at', 'exit_status'):
            assert not hasattr(r, banned), f"CellRecord leaked {banned!r}"


# ── CellPresence ──────────────────────────────────────────────────────


class TestCellPresence:

    def test_default_state_is_ready(self):
        p = CellPresence(address='a/b/c')
        assert p.state is CellState.READY

    def test_no_channel_interface_field(self):
        # 2026-07-12 契约: announce payload 永不携带膜内容.
        p = CellPresence(address='a/b/c')
        assert not hasattr(p, 'channel_interface')

    def test_membrane_default_empty(self):
        p = CellPresence(address='a/b/c')
        assert p.membrane == []

    def test_membrane_accepts_channel(self):
        p = CellPresence(address='a/b/c', membrane=['channel'])
        assert p.membrane == ['channel']

    def test_membrane_rejects_unknown_literal(self):
        # v1 唯一支持 'channel'. 加类型 = matrix 版本演进事件, cell 作者不能自扩.
        with pytest.raises(ValidationError):
            CellPresence(address='a/b/c', membrane=['resource'])

    def test_serialization_roundtrip(self):
        p = CellPresence(
            address='w/x/y',
            alias='cam',
            state=CellState.READY,
            failure='',
            project_id='proj-a',
            is_host=True,
            membrane=['channel'],
        )
        js = p.model_dump_json()
        loaded = CellPresence.model_validate_json(js)
        assert loaded == p


# ── CellEvent ─────────────────────────────────────────────────────────


class TestCellEvent:

    def test_refetch_default_true(self):
        e = CellEvent(address='a/b/c')
        assert e.refetch is True

    def test_refetch_false_explicit(self):
        e = CellEvent(address='a/b/c', refetch=False, content='pure log')
        assert e.refetch is False

    def test_no_terminal_field(self):
        # CellLog.terminal 语义由 liveness DELETE 承载, 不在事件层.
        e = CellEvent(address='a/b/c')
        assert not hasattr(e, 'terminal')

    def test_no_kind_field(self):
        # 不做 kind 枚举 (2026-07-11 人类明示).
        e = CellEvent(address='a/b/c')
        assert not hasattr(e, 'kind')


# ── ABC 完整性 ────────────────────────────────────────────────────────


class TestABCs:

    def test_presence_abstractmethods(self):
        expected = {
            '__aenter__', '__aexit__',
            'this', 'announce', 'revoke', 'provide', 'publish_event',
        }
        assert Presence.__abstractmethods__ == expected

    def test_watcher_abstractmethods(self):
        expected = {
            '__aenter__', '__aexit__',
            'view', 'refresh',
            'on_change', 'on_event',
            'recent_events', 'wait_present',
            'accept', 'release', 'accepted',
        }
        assert Watcher.__abstractmethods__ == expected


# ── 工具函数 ──────────────────────────────────────────────────────────


class TestUtilities:

    def test_normalize_replaces_separators(self):
        assert normalize('worker/cam.front-1') == 'worker_cam_front_1'

    def test_make_address_joins(self):
        assert make_address('worker', 'cam', 'uid-8') == 'worker/cam/uid-8'

    def test_duplicated_error_is_runtime_error(self):
        assert issubclass(DuplicatedError, RuntimeError)

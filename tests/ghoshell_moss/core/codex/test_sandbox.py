import pytest
from ghoshell_moss.core.codex.sandbox import Sandbox, SANDBOX_BUILTINS


class TestSandboxBasics:
    """沙盒基础功能：exec、变量持久化、stdout 捕获。"""

    def test_exec_and_stdout_capture(self):
        s = Sandbox(name="test")
        output = s.exec("print('hello')")
        assert output == "hello\n"
        s.close()

    def test_variable_persistence(self):
        s = Sandbox(name="test")
        s.exec("x = 42")
        assert s.get("x") == 42
        s.exec("x = x + 1")
        assert s.get("x") == 43
        s.close()

    def test_variable_from_previous_call_visible(self):
        s = Sandbox(name="test")
        s.exec("a = 10")
        output = s.exec("print(a)")
        assert "10" in output
        s.close()

    def test_get_raises_for_missing(self):
        s = Sandbox(name="test")
        with pytest.raises(AttributeError, match="has no attribute"):
            s.get("nonexistent")
        s.close()

    def test_set_and_get(self):
        s = Sandbox(name="test")
        s.set("obj", {"key": "value"})
        assert s.get("obj") == {"key": "value"}
        s.close()

    def test_exec_returns_empty_for_no_output(self):
        s = Sandbox(name="test")
        output = s.exec("x = 1")
        assert output == ""
        s.close()

    def test_name_property(self):
        s = Sandbox(name="my_sandbox")
        assert s.name == "my_sandbox"
        s.close()


class TestSandboxBuiltins:
    """builtins 控制。"""

    def test_default_blocks_dangerous(self):
        s = Sandbox(name="test")
        output = s.exec("open('/etc/passwd')")
        assert "NameError" in output
        s.close()

    def test_default_blocks_import(self):
        s = Sandbox(name="test")
        output = s.exec("import os")
        assert "ImportError" in output
        s.close()

    def test_default_blocks_eval(self):
        s = Sandbox(name="test")
        output = s.exec("eval('1+1')")
        assert "NameError" in output
        s.close()

    def test_default_blocks_exec(self):
        s = Sandbox(name="test")
        output = s.exec("exec('x=1')")
        assert "NameError" in output
        s.close()

    def test_default_blocks_compile(self):
        s = Sandbox(name="test")
        output = s.exec("compile('1+1', '', 'eval')")
        assert "NameError" in output
        s.close()

    def test_default_blocks_input(self):
        s = Sandbox(name="test")
        output = s.exec("input()")
        assert "NameError" in output
        s.close()

    def test_default_blocks_breakpoint(self):
        s = Sandbox(name="test")
        output = s.exec("breakpoint()")
        assert "NameError" in output
        s.close()

    def test_none_builtins_allows_all(self):
        s = Sandbox(name="test", builtins=None)
        output = s.exec("import sys; print(sys.version_info[0])")
        assert "NameError" not in output
        assert output.strip().isdigit()
        s.close()

    def test_safe_builtins_available(self):
        s = Sandbox(name="test")
        output = s.exec("print(len('hello'))")
        assert "5" in output
        assert "NameError" not in output
        s.close()

    def test_sandbox_builtins_constant(self):
        assert "__import__" not in SANDBOX_BUILTINS
        assert "open" not in SANDBOX_BUILTINS
        assert "print" in SANDBOX_BUILTINS
        assert "len" in SANDBOX_BUILTINS
        assert "range" in SANDBOX_BUILTINS
        assert "int" in SANDBOX_BUILTINS
        assert "str" in SANDBOX_BUILTINS
        assert "list" in SANDBOX_BUILTINS
        assert "dict" in SANDBOX_BUILTINS
        assert "isinstance" in SANDBOX_BUILTINS
        assert "hasattr" in SANDBOX_BUILTINS
        assert "Exception" in SANDBOX_BUILTINS


class TestSandboxParentChild:
    """父子沙盒共享命名空间。"""

    def test_child_shares_parent_namespace(self):
        parent = Sandbox(name="parent", builtins=None)
        parent.exec("x = 10")
        child = Sandbox(name="child", parent=parent)
        assert child.get("x") == 10
        parent.close()
        child.close()

    def test_child_side_effect_visible_to_parent(self):
        parent = Sandbox(name="parent", builtins=None)
        child = Sandbox(name="child", parent=parent)
        child.exec("y = 20")
        assert parent.get("y") == 20
        parent.close()
        child.close()

    def test_child_close_preserves_parent_namespace(self):
        parent = Sandbox(name="parent", builtins=None)
        child = Sandbox(name="child", parent=parent)
        child.exec("z = 30")
        child.close()
        assert parent.get("z") == 30
        parent.close()

    def test_child_closed_blocks_exec(self):
        parent = Sandbox(name="parent", builtins=None)
        child = Sandbox(name="child", parent=parent)
        child.close()
        with pytest.raises(RuntimeError, match="closed"):
            child.exec("x = 1")
        parent.close()

    def test_child_with_restricted_builtins_parent_full(self):
        parent = Sandbox(name="parent", builtins=None)
        parent.exec("import sys")
        child = Sandbox(name="child", parent=parent, builtins=SANDBOX_BUILTINS)
        # child can't import (restricted) but can access parent's imports
        assert "sys" in [k for k in child._module.__dict__ if not k.startswith("_")]
        output = child.exec("import os")
        assert "ImportError" in output
        parent.close()
        child.close()

    def test_parent_property(self):
        parent = Sandbox(name="parent", builtins=None)
        child = Sandbox(name="child", parent=parent)
        assert child.parent is parent
        assert parent.parent is None
        parent.close()
        child.close()


class TestSandboxLifecycle:
    """生命周期：context manager、on_init、on_destroy。"""

    def test_context_manager(self):
        with Sandbox(name="test") as s:
            s.exec("x = 42")
            assert s.get("x") == 42
        assert s._closed

    def test_on_init_called(self):
        initialized = []

        def init(sb):
            initialized.append(sb.name)

        s = Sandbox(name="test", on_init=init)
        assert initialized == ["test"]
        s.close()

    def test_on_destroy_called_on_close(self):
        destroyed = []

        def destroy(sb):
            destroyed.append(sb.name)

        s = Sandbox(name="test", on_destroy=destroy)
        s.close()
        assert destroyed == ["test"]

    def test_on_destroy_called_once(self):
        destroyed = []

        def destroy(sb):
            destroyed.append(1)

        s = Sandbox(name="test", on_destroy=destroy)
        s.close()
        s.close()  # second close is no-op
        assert len(destroyed) == 1

    def test_exec_after_close_raises(self):
        s = Sandbox(name="test")
        s.close()
        with pytest.raises(RuntimeError, match="closed"):
            s.exec("x = 1")


class TestSandboxExceptionHandling:
    """异常捕获。"""

    def test_exception_traceback_returned(self):
        s = Sandbox(name="test")
        output = s.exec("1/0")
        assert "ZeroDivisionError" in output
        assert "Traceback" in output
        s.close()

    def test_variables_preserved_after_exception(self):
        s = Sandbox(name="test")
        s.exec("x = 42")
        s.exec("1/0")  # exception, but x is still there
        assert s.get("x") == 42
        s.close()

    def test_stdout_before_exception_captured(self):
        s = Sandbox(name="test")
        output = s.exec("print('before')\n1/0")
        assert "before" in output
        assert "ZeroDivisionError" in output
        s.close()

"""
Unit tests for Sandbox — safe ModuleType-based execution environment.
"""

import sqlite3

import pytest

from ghoshell_moss.core.codex.sandbox import SANDBOX_BUILTINS, Sandbox


# -- basic exec ----------------------------------------------------------

def test_exec_returns_result_variable():
    s = Sandbox()
    r = s.exec("__result__ = 42")
    assert r.returns == 42
    assert r.std_output == ''


def test_exec_captures_stdout():
    s = Sandbox()
    r = s.exec("print('hello'); print('world')")
    assert r.std_output == 'hello\nworld\n'


def test_exec_returns_and_prints():
    s = Sandbox()
    r = s.exec("print('side effect'); __result__ = 99")
    assert r.returns == 99
    assert r.std_output == 'side effect\n'


# -- variable persistence (REPL-like) ------------------------------------

def test_variables_persist_across_exec_calls():
    s = Sandbox()
    s.exec("x = 10")
    s.exec("y = x + 5")
    r = s.exec("__result__ = y")
    assert r.returns == 15


def test_functions_and_classes_persist():
    s = Sandbox()
    s.exec("class Dog:\n    def bark(self):\n        return 'woof'")
    r = s.exec("__result__ = Dog().bark()")
    assert r.returns == 'woof'


# -- builtins safety -----------------------------------------------------

@pytest.mark.parametrize("dangerous", [
    "__import__('os')",
    "open('/etc/passwd')",
    "eval('1+1')",
    "exec('x=1')",
    "compile('x=1', '', 'exec')",
    "input()",
    "breakpoint()",
])
def test_dangerous_builtins_blocked(dangerous):
    s = Sandbox()
    r = s.exec(dangerous)
    assert r.exception is not None, f"expected error for: {dangerous}"


def test_safe_builtins_work():
    s = Sandbox()
    r = s.exec("""
x = list(range(5))
y = sum(x)
z = [str(i) for i in x]
__result__ = (y, isinstance(z, list), len(z))
""")
    assert r.returns == (10, True, 5)


def test_custom_builtins():
    custom = {"print": print, "len": len, "int": int}
    s = Sandbox(builtins=custom)
    s.exec("x = len([1, 2, 3])")
    r = s.exec("__result__ = x")
    assert r.returns == 3
    r = s.exec("range(5)")
    assert r.exception is not None


def test_full_builtins_when_none():
    s = Sandbox(builtins=None)
    r = s.exec("import json; __result__ = json.dumps({'a': 1})")
    assert r.returns == '{"a": 1}'


# -- lifecycle hooks -----------------------------------------------------

def test_on_init_hook():
    def init(sb: Sandbox):
        sb.set("answer", 42)

    s = Sandbox(on_init=init)
    r = s.exec("__result__ = answer")
    assert r.returns == 42


def test_on_destroy_hook():
    destroyed = []

    def destroy(sb: Sandbox):
        destroyed.append(sb._name)

    s = Sandbox(name="test_sb", on_destroy=destroy)
    s.close()
    assert destroyed == ["test_sb"]


def test_context_manager():
    s = Sandbox()
    with s:
        s.exec("x = 1")
    with pytest.raises(RuntimeError, match="closed"):
        s.exec("x = 2")


# -- get / set -----------------------------------------------------------

def test_get_set():
    s = Sandbox()
    s.set("pi", 3.14)
    assert s.get("pi") == 3.14
    with pytest.raises(AttributeError):
        s.get("nonexistent")


# -- parent-child namespace sharing --------------------------------------

def test_child_shares_parent_namespace():
    parent = Sandbox(name="parent")
    parent.exec("shared = [1, 2, 3]")

    child = Sandbox(name="child", parent=parent)
    r = child.exec("__result__ = shared")
    assert r.returns == [1, 2, 3]

    child.exec("shared.append(4)")
    r = parent.exec("__result__ = shared")
    assert r.returns == [1, 2, 3, 4]


def test_child_close_does_not_destroy_parent_namespace():
    parent = Sandbox(name="parent")
    parent.exec("x = 100")

    child = Sandbox(name="child", parent=parent)
    child.exec("x = 200")

    child.close()
    r = parent.exec("__result__ = x")
    assert r.returns == 200


def test_parent_close_destroys_children():
    parent = Sandbox(name="parent")
    child = Sandbox(name="child", parent=parent)

    parent.close()
    with pytest.raises(RuntimeError, match="closed"):
        child.exec("x = 1")


def test_cannot_create_child_from_closed_parent():
    parent = Sandbox(name="parent")
    parent.close()
    with pytest.raises(ValueError, match="closed"):
        Sandbox(name="child", parent=parent)


def test_close_is_idempotent():
    s = Sandbox()
    s.close()
    s.close()


# -- closed sandbox ------------------------------------------------------

def test_exec_on_closed_raises():
    s = Sandbox()
    s.close()
    with pytest.raises(RuntimeError, match="closed"):
        s.exec("x = 1")


# -- sqlite3 acceptance (app + sqlite) -----------------------------------

def test_sqlite_injection_acceptance():
    parent = Sandbox(name="db_parent")

    def init_db(sb: Sandbox):
        conn = sqlite3.connect(":memory:")
        sb.set("sqlite3", sqlite3)
        sb.set("conn", conn)

    child = Sandbox(name="db_child", parent=parent, on_init=init_db)

    child.exec("""
sql = sqlite3
conn.execute('CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT)')
conn.execute("INSERT INTO users VALUES (1, 'Alice')")
conn.execute("INSERT INTO users VALUES (2, 'Bob')")
conn.commit()
""")

    r = parent.exec("""
rows = conn.execute('SELECT * FROM users ORDER BY id').fetchall()
__result__ = rows
""")
    assert r.returns == [(1, 'Alice'), (2, 'Bob')]

    conn = parent.get("conn")
    conn.close()
    parent.close()


# -- exception capture & traceback filtering ------------------------------

def test_exception_returned_in_result_not_raised():
    s = Sandbox()
    r = s.exec("1/0")
    assert r.exception is not None
    assert 'ZeroDivisionError' in r.exception
    assert r.traceback is not None
    assert r.returns is None


def test_stdout_preserved_on_error():
    s = Sandbox()
    r = s.exec("print('before'); 1/0; print('after')")
    assert r.std_output == 'before\n'


def test_traceback_excludes_sandbox_internals():
    s = Sandbox(name="test_sb")
    s.exec("def crash():\n    return 1/0")
    r = s.exec("crash()")
    assert r.exception is not None
    assert 'sandbox.py' not in r.traceback
    assert 'test_sb' in r.traceback


def test_traceback_includes_model_code_frames():
    s = Sandbox(name="my_sandbox")
    s.exec("def inner():\n    return 1/0")
    s.exec("def outer():\n    return inner()")
    r = s.exec("outer()")
    assert r.exception is not None
    # Model's code frames should be present
    assert 'my_sandbox' in r.traceback
    assert 'outer' in r.traceback
    assert 'inner' in r.traceback


def test_syntax_error_captured():
    s = Sandbox()
    r = s.exec("x = ")
    assert r.exception is not None
    assert 'SyntaxError' in r.exception


def test_successful_exec_has_no_exception():
    s = Sandbox()
    r = s.exec("x = 1; __result__ = x")
    assert r.exception is None
    assert r.traceback is None
    assert r.returns == 1


def test_sandbox_clears_module_dict_on_root_close():
    s = Sandbox(name="root")
    s.exec("secret = 'api-key-12345'")
    s.close()
    assert s.module.__dict__ == {}

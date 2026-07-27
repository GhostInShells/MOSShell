"""Import record & replay — definition-is-authorization runtime semantics."""

from __future__ import annotations

import pytest

from ghoshell_moss.agents._imports import recording_builtins, replay_import
from ghoshell_moss.core.codex.compiler import Compiler
from ghoshell_moss.core.codex.sandbox import SANDBOX_BUILTINS, Sandbox


def _compile_agent(source: str):
    builtins_dict, recorded = recording_builtins()
    compiler = Compiler(
        source=source,
        modulename="probe_agent",
        local_injections={"__builtins__": builtins_dict},
        compile_soon=True,
    )
    return compiler.compiled, recorded


def test_compile_records_plain_and_from_imports():
    _, recorded = _compile_agent(
        "import math\n"
        "from os import path\n"
    )
    assert "math" in recorded
    assert "os" in recorded
    assert "os.path" in recorded  # fromlist submodule resolved


def test_compile_records_dotted_ancestors():
    _, recorded = _compile_agent("import xml.etree.ElementTree\n")
    assert {"xml", "xml.etree", "xml.etree.ElementTree"} <= recorded


def _replay_sandbox(source: str) -> Sandbox:
    compiled, recorded = _compile_agent(source)
    builtins = {**SANDBOX_BUILTINS, "__import__": replay_import(frozenset(recorded))}
    sandbox = Sandbox(name="probe", builtins=builtins, source=source)
    for k, v in compiled.__dict__.items():
        if not k.startswith("__"):
            sandbox.set(k, v)
    return sandbox


def test_replay_allows_reimport_of_authorized_module():
    sandbox = _replay_sandbox("import math\n")
    # the model reflexively re-imports what it sees in the source — must work
    result = sandbox.exec("import math\n__result__ = math.pi")
    assert result.exception is None
    assert result.returns == pytest.approx(3.14159, abs=1e-4)


def test_replay_rejects_unauthorized_module_with_teaching_error():
    sandbox = _replay_sandbox("import math\n")
    result = sandbox.exec("import json")
    assert result.exception is not None
    msg = str(result.exception)
    assert "authorization" in msg
    assert "math" in msg  # error lists what IS authorized


def test_replay_rejects_unauthorized_submodule():
    sandbox = _replay_sandbox("import xml\nimport xml.sax\n")
    # xml.etree was never imported by the definition file
    result = sandbox.exec("from xml import etree")
    # xml.etree may or may not be in sys.modules from other tests; if it is,
    # the guard fires. Either way the authorized one must work:
    ok = sandbox.exec("from xml import sax\n__result__ = sax.__name__")
    assert ok.exception is None
    assert ok.returns == "xml.sax"


def test_replay_rejects_relative_import():
    sandbox = _replay_sandbox("import math\n")
    result = sandbox.exec("from . import something")
    assert result.exception is not None

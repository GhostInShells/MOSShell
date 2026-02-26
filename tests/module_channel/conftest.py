import sys
from pathlib import Path

import pytest

from ghoshell_moss.channels.module_channel import ModuleChannel


@pytest.fixture(autouse=True)
def _isolate_script_module_counter():
    """Prevent cross-test pollution from the `script_module.py` fixture.

    `tests/module_fixtures/script_module.py` defines a module-level `counter`.
    When that file is loaded via `ModuleChannel` using a script path, the
    implementation may import it as a *package module* (because `tests/` and
    `tests/module_fixtures/` are packages) and the module object is then cached
    in `sys.modules`. Without cleanup, earlier tests can increment `counter` and
    later tests will observe a non-zero start value.
    """

    script_path = (Path(__file__).resolve().parent.parent / "module_fixtures" / "script_module.py").resolve()
    dotted_name, _ = ModuleChannel._derive_module_name_and_sys_path(script_path)
    stable_file_name = ModuleChannel._stable_file_module_name(script_path)

    for name in (dotted_name, stable_file_name):
        existing = sys.modules.get(name)
        if existing is not None and hasattr(existing, "counter"):
            existing.counter = 0

    yield

    sys.modules.pop(dotted_name, None)
    sys.modules.pop(stable_file_name, None)

# ModuleChannel Demo

This example demonstrates how `ModuleChannel` can wrap a Python module into a
runnable MOSS channel.

It covers several common scenarios (fixtures are reused from
`tests/module_fixtures/`):

1. **Prefer sibling `__channel__.py`**: if the target module lives in a directory
   containing `__channel__.py`, `ModuleChannel` loads the predefined `__channel__`
   value from that file.
1. **Auto-wrap a plain module**: when no `__channel__.py` exists, exported
   callables from the module are registered as commands.
1. **Provide an explicit `channel_file`**: manually specify a `.py` file that
   defines `__channel__`.
1. **Wrap a standard library module**: for example, exposing `math` as a channel
   and calling its functions.
1. **Load from a script path (`.py`)**: provide a concrete python file path like
   `tests/module_fixtures/script_module.py` instead of a dotted import path.
   This demo also shows `reload_on_bootstrap` behavior.

## Run

From repo root:

```bash
python examples/module_channel_demo/main.py
```

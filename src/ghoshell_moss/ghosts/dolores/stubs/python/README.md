# DOLORES python tree

- `src/DOLORES/` — capability modules. The ghost's startup appends this `src` to `sys.path`.
- `tests/` — tests for those modules; `conftest.py` makes `DOLORES` importable.

Run the tests from this directory:

    python -m pytest tests -q

The tree is seeded from `src/ghoshell_moss/ghosts/dolores/stubs/python` once per ghost home.

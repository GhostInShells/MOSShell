"""DOLORES — the ghost's own Python capabilities.

Each module under this package is a capability declaration. The first line of a module
docstring is its readable name; `@agent_func` marks functions whose semantics are settled
per call, and `@agent_func`-free modules are ordinary code.

The tree is registered on the ghost's `sys.path` at startup. Keep it importable with no
side effects, and keep the tests next to it in ``python/tests``.
"""

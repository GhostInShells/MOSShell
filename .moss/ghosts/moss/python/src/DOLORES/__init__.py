"""DOLORES — the ghost's own Python capabilities.

Each module under this package is a capability declaration. The first line of a module
docstring is its readable name; `@agent_class` declares a class-level agent surface,
and `@agent_method` marks its blocking ops; modules without them are ordinary code.

The tree is registered on the ghost's `sys.path` at startup. Keep it importable with no
side effects, and keep the tests next to it in ``python/tests``.
"""

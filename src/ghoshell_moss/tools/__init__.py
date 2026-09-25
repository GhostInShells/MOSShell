"""MOSS project-level tools — plain callables that may consume project-level contracts
(e.g. SubprocessFacade) but declare no IoC abstract classes.

Categorized under `tools/`: `tools.git` (read-only git subset, @cli-wrapped)
and `tools.fs` (read / list / glob within the project root). All tools are
read-only; the import surface is the authorization whitelist.
"""

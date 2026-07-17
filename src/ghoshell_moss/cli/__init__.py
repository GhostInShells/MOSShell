"""
ghoshell CLI - Ghost In Shells command line tool
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ghoshell_moss.cli.main import app, main, main_entry

__all__ = ["app", "main", "main_entry"]


def __getattr__(name: str) -> Any:
    """Load the root CLI only when one of its public objects is requested.

    Console entries such as ``moss-run-ghost`` import a sibling module and must
    not pay for, or be broken by, unrelated root-command discovery.
    """
    if name not in __all__:
        raise AttributeError(name)
    from ghoshell_moss.cli.main import app, main, main_entry

    return {"app": app, "main": main, "main_entry": main_entry}[name]

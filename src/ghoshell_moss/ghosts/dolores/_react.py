"""Dolores react — runtime string-template functions over CTML.

A react maps a single-char key to a CTML template with ``%s`` slots. ``moss_define_reacts``
bulk-defines them (merge/overwrite, in-memory only); ``moss_react`` expands one and streams it.
No seed, no notice, no persistence — the model defines what a scenario needs, when it needs it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["React", "ReactStore"]


class React(BaseModel):
    """char → CTML template (positional ``%s`` slots)."""

    char: str = Field(description="single-character key.")
    template: str = Field(default="", description="CTML template; %s slots filled by args.")


class ReactStore:
    """In-memory char → template table. define() merges/overwrites; render() expands."""

    def __init__(self) -> None:
        self._reacts: dict[str, str] = {}

    def define(self, reacts: list[React]) -> list[str]:
        """Bulk define/overwrite. Returns the chars defined. Raises ValueError on a non-single-char key."""
        defined: list[str] = []
        for react in reacts:
            if len(react.char) != 1:
                raise ValueError(f"react key must be a single character, got {react.char!r}")
            self._reacts[react.char] = react.template
            defined.append(react.char)
        return defined

    def render(self, char: str, args: list[str] | None = None) -> str:
        """Expand char's template with positional args. Raises KeyError (unknown char) or ValueError (args mismatch)."""
        template = self._reacts.get(char)
        if template is None:
            raise KeyError(char)
        try:
            return template % tuple(args or [])
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError(f"react {char!r} args {args!r} do not fit template: {error}") from error

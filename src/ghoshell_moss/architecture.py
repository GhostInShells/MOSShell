"""MOSS architecture map — a hand-curated index of the load-bearing modules.

``moss codex architecture`` prints everything between ``# __ARCHITECTURE_MAP_START__``
and ``# __ARCHITECTURE_MAP_END__`` (dedented), so nothing is imported at runtime and
the map never crashes on a missing heavy dependency. Every import lives under
``if TYPE_CHECKING`` for IDE navigation only — it is never executed.

Naming convention for entries:

- ``from <pkg> import <name>`` — when the module's own name is self-explanatory.
- ``import <path> as the_<role>`` — when the leaf name is terse or collides with
  another entry; the alias spells out the module's role (sentence-style names
  like ``the_<x>_of_<y>`` are also fine).

To index a module, add one such line under the matching section. The map is
deliberately curated (详略得当), not exhaustive: sections group modules by
cohesion, and only load-bearing paths are listed.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # __ARCHITECTURE_MAP_START__

    # ============================================================================
    # Core Concepts — what MOSS is
    # ghoshell_moss.core.concepts
    # ============================================================================

    from ghoshell_moss.core.concepts import channel
    from ghoshell_moss.core.concepts import command
    from ghoshell_moss.core.concepts import errors
    from ghoshell_moss.core.concepts import interpreter
    from ghoshell_moss.core.concepts import shell
    from ghoshell_moss.core.concepts import topic

    # ============================================================================
    # Blueprints — how to build with MOSS
    # ghoshell_moss.core.blueprint
    # ============================================================================

    from ghoshell_moss.core.blueprint import channel_builder
    from ghoshell_moss.core.blueprint import environment
    from ghoshell_moss.core.blueprint import ghost
    from ghoshell_moss.core.blueprint import host
    from ghoshell_moss.core.blueprint import matrix
    from ghoshell_moss.core.blueprint import mindflow
    from ghoshell_moss.core.blueprint import session
    from ghoshell_moss.core.blueprint import states_channel

    # ============================================================================
    # Cognitive substrate — memento / ground / message / topic models
    # ============================================================================

    from ghoshell_moss import ground
    from ghoshell_moss import memento
    from ghoshell_moss import message
    from ghoshell_moss.types import topics

    # ============================================================================
    # Contracts — abstract dependencies
    # ghoshell_moss.contracts
    # ============================================================================

    from ghoshell_moss import contracts

    # ============================================================================
    # Implementations — concrete paths
    # ============================================================================

    from ghoshell_moss import bridges
    from ghoshell_moss import cli
    from ghoshell_moss import ghosts
    from ghoshell_moss.core import file_editor
    from ghoshell_moss.core import py_channel
    from ghoshell_moss.core import speech
    from ghoshell_moss.host import tui_entries
    import ghoshell_moss.contracts.voice as the_voice_contract
    import ghoshell_moss.core.topic as the_topic_service
    import ghoshell_moss.host as the_host_implementation
    import ghoshell_moss.host.tui as the_tui_framework
    import ghoshell_moss.host.voice as the_voice_implementation
    import ghoshell_moss.matrix.session as the_session_implementation

    # ============================================================================
    # LLMs — model configuration and engines
    # ghoshell_moss.llms.pydantic_ai_adapter
    # ============================================================================

    import ghoshell_moss.llms.pydantic_ai_adapter.client as the_llms_client
    import ghoshell_moss.llms.pydantic_ai_adapter.funcs as the_llms_function_engine

    # ============================================================================
    # Openbox — prebuilt capabilities
    # ============================================================================

    from ghoshell_moss import channels
    from ghoshell_moss.core.concepts import tools

    # __ARCHITECTURE_MAP_END__


def render() -> str:
    """Return the map: the file body between the map delimiters, dedented.

    No line classification — everything between the two markers is shown with
    leading whitespace stripped.
    """
    import pathlib

    lines = pathlib.Path(__file__).read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    inside = False
    for line in lines:
        stripped = line.strip()
        if stripped == "# __ARCHITECTURE_MAP_START__":
            inside = True
            continue
        if stripped == "# __ARCHITECTURE_MAP_END__":
            break
        if inside:
            out.append(stripped)
    return "\n".join(out).strip() + "\n"

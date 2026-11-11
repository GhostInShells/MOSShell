"""Dolores prototype constants — paths and cross-module protocol names.

Module-level only (no package export): a magic value should be findable here before it is
used in a chain. The ghost home itself is **not** resolved here — components resolve it from
the registered ghost (``home()`` / ``home_meta_dir()``), the only source of truth.
"""

DOLORES_META_DIRECTORY = ".dolores"
"""The ghost home's mechanism directory — prompts, configs, and other meta assets."""

DOLORES_PROMPTS_DIRECTORY = "prompts"
"""Prompt versions under the meta directory: one directory per key, one file per version."""

DOLORES_CONFIGS_DIRECTORY = "configs"
"""Typed runtime configs under the meta directory (YAML, one file per ConfigType)."""

MINDHOOK = "mindhook"
"""The generic container for a message from a parallel-thinking bypass task."""

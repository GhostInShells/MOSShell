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

NIRMANAKAYA_REMINDER = "emanation-reminder"
"""The container a forked emanation receives: which stream it is on, and what capabilities changed."""

ALIVE_EGO_SESSION = "Ψ"
"""The title of the one live ego session — the current breathing span, and only that one.

The title template (``DoloresEgoConfig.session_title``) prefixes every new ego session with it;
the boot rule archives every other Ψ/Ω-titled session, so "exactly one live Ψ" stays true.
"""

ARCHIVED_EGO_SESSION = "Ω"
"""The title prefix of a closed ego session — the span that has ended.

Omega is the letter after psi and the last of the alphabet: Ψ → Ω reads as one life span. It is
also the marker the boot rule greps for, so a crash that left an already-closed session
unarchived is still collected on the next boot.
"""

NIRMANAKAYA = "Emanation"
"""A concurrent self: the same inception and the same existence trajectory, in another context.

The value is the English token the model reads; the doctrinal CJK spelling (應身) is kept as a
gloss in the prompt, where it costs two tokens and carries the reading "the same one, appearing
in several places" rather than "a lesser copy".
"""

NIRMANAKAYA_SESSION = "Δ"
"""The title prefix of a concurrent emanation session — the delta of the same self: a branching
distributary, not a lesser copy. Together with Ψ / Ω / Σ it forms the session-title vocabulary:
Σ supervises, Ψ faces, Δ branches, Ω closes."""

SUPER_EGO_SESSION = "Σ"
"""The title prefix of a superego session.

Sigma is the summation operator: the superego's function is governance, its goal is integration,
and its means is pruning. The letter states the goal, not the means."""

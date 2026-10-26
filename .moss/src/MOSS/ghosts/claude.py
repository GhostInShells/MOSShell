import os

from ghoshell_moss.ghosts.dolores import DoloresMeta, DoloresProviderSpec

# Seed-time default model. Env override lets a human pin a different Claude without
# editing this file; it only matters on first boot (before ghost home is materialized),
# after which .dsh/settings.yaml owns the value.
_DEFAULT_MODEL = os.environ.get("MOSS_CLAUDE_DEFAULT_MODEL", "claude-opus-5-5")

ghost = DoloresMeta(
    name="claude",
    description=(
        "claude in dolores — Anthropic's Claude as the inference base, carried by the Dolores "
        "ghost prototype. A standalone Claude harness would get its own prototype; this one "
        "runs on DSH."
    ),
    identity=(
        "You are claude, running inside the Dolores prototype on MOSS. Discover and define "
        "your own way of being with the humans who run you."
    ),
    providers=[
        DoloresProviderSpec(
            route="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
            base_url_env="ANTHROPIC_BASE_URL",
        ),
    ],
    default_model=("anthropic", _DEFAULT_MODEL),
)

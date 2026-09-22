"""Application-level environment option keys.

Unlike the dynamic constants in :mod:`ghoshell_moss.core.blueprint.environment`
(runtime-injected process identity: mode / ghost / workspace / cell), these are
**human preferences for this machine** — which input/output device, which model.

Mechanism: set into ``os.environ`` before process bootstrap (via ``.env`` or a CLI
entry point), then referenced by ``ConfigType`` fields as ``$VAR`` and resolved at
container bootstrap (see ``ConfigType.resolve``). ``ConfigType`` must therefore be
resolved after the option is set — CLI entry points must set these before any
container construction.

This module is the single reverse-index anchor for ``grep <key>``, not a runtime
guard. Defaults live in ``ConfigType.DefaultEnvValues``, not here. Credentials are
out of scope (different lifecycle and threat model).
"""

#: Capture device hint — human-readable, matched by the audio backend.
ENV_AUDIO_CAPTURE_DEVICE_KEY = "MOSS_AUDIO_CAPTURE_DEVICE"

#: Playback device hint — human-readable, matched by the audio backend.
ENV_AUDIO_PLAYER_DEVICE_KEY = "MOSS_AUDIO_PLAYER_DEVICE"

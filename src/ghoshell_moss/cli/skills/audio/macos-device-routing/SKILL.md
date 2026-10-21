---
name: macos-device-routing
description: Configure MOSS audio device routing on macOS, especially when a Bluetooth headset is the output. Core conflict — a headset opened for capture switches to HFP and kills its A2DP output ("connected but silent"); the fix is to separate capture and player onto different devices.
moss_version: beta2
platform: [macos]
verified: 2026-10-01
---

# macOS Audio Device Routing (MOSS voice)

## When to use this

Configuring MOSS voice on macOS (`--voice speak` / `listen` / `all`), especially when the
output is a **Bluetooth headset**. Typical symptoms:

- `moss audio speak` (CLI) has sound, but speak inside `moss-shell` does not;
- the headset is Bluetooth-"connected", set as system default, volume normal — yet silent;
- with no device configured, audio goes to the display/built-in speakers, never the headset.

## Principle (the root cause — read this first)

A Bluetooth headset on macOS exposes two **mutually exclusive** profiles:

- **A2DP** — media output (stereo, good quality);
- **HFP** — telephony (includes the microphone, mono 16k, poor quality).

**As soon as a process opens the headset's microphone (capture), macOS switches the headset
to HFP and the A2DP media output is preempted.** The result is "headset connected but MOSS is
silent".

On the MOSS side: `moss-shell` opens the whole audio factory (Speech + AEC); AEC needs capture,
so it opens an input device — and if capture points at the *same* Bluetooth headset, the HFP
conflict triggers. The CLI's `moss audio speak` only opens the player, not capture, so it keeps
working — the first signal that this is the issue.

## Diagnosis

1. **List devices**: `moss audio device` (note the headset name under output; config matches by
   substring).
2. **Check whether capture and player point at the same Bluetooth device**. If yes → HFP conflict
   is the likely cause.
3. **Isolate**: run the native system TTS `say "test"` (goes to the system default output).
   - audible → the system/CoreAudio layer is fine; the problem is MOSS's routing choice;
   - silent → the problem is the headset or the OS layer (stalled link, headset out of range);
     fix the headset first.
4. **Confirm where MOSS lands by default**: miniaudio's `device_id=None` (i.e. an empty config)
   uses **the first device in its own enumeration, not the CoreAudio system default** — so it is
   not guaranteed to reach the headset; it may land on the display/speakers.

## Fix

**Separate capture and player onto different devices** so the headset only carries output and
stays on A2DP:

- point capture at a non-Bluetooth device (e.g. the built-in mic);
- point player at the headset (name substring).

Both come from the miniaudio factory's capture/player `device_pattern`, injected via the env
vars `MOSS_AUDIO_CAPTURE_DEVICE` / `MOSS_AUDIO_PLAYER_DEVICE` — write them into the workspace
`.env`, or inject process env with a command-line prefix (more reliable when the shell has not
reached `.env` loading yet).

> For exact field names, defaults, and the matching rule, read the implementation with
> `moss codex get-interface` — do not copy from this document.

## Observed combinations (see frontmatter for the pinned version)

Measured on moss `beta2` / macOS 14 (Darwin 23.6); qualitative reference only:

| capture | player | result |
|---|---|---|
| same Bluetooth headset | same headset | ❌ silent output (HFP preempts A2DP) |
| built-in mic | Bluetooth headset | ✅ listen and speak both work |
| empty | empty | ⚠️ takes miniaudio's first enumerated device, not necessarily the system default/headset |

## Staleness

This skill is pinned to `moss_version` (frontmatter). Device-matching behavior drifts with the
MOSS version, the macOS version, and Bluetooth firmware; use the frontmatter version and date
for stale-checking and pruning.

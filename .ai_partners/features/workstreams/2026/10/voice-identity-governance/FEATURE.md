---
created: 2026-10-26
depends:
- voice-interaction-control
- audio-observability
description: 'Give every sound a provenance: who may hear whom (acoustic coupling
  topology), whose voice a segment carries (identity), and how the chain decides to
  keep, mark, or cancel it — with AEC, cloud speaker verification and local embedding
  surveyed as candidate mechanisms.'
milestone: null
priority: P1
status: draft
status_note: created as pre-research on audio identity topology
title: Voice Identity Governance
updated: '2026-10-26'
---

# Voice Identity Governance

> Use `moss features set-status voice-identity-governance <status> -m "note"` to update state.
> See [TOPOLOGY.md](TOPOLOGY.md) for directory layout and [README.md](README.md) for the full convention.

## Motivation

On D26 evening the human plugged in a wired headset, and the listener began
transcribing the ghost's own TTS back as user speech — twice, verbatim, each
time aborting the ghost's turn. The ghost replied to its own words as if they
were the human's.

The root cause is not "AEC is weak". It is that **no layer in the chain answers
"whose voice is this"**:

- **The acoustic layer** (AEC) only removes a *known reference* through an
  assumed linear path; it is blind when the play/capture device pair is unknown,
  when device DSP is nonlinear, or when the two device clocks drift.
- **The ASR layer** has exactly one input stream and no notion of source; echo
  and human speech are the same kind of thing to it.
- **The interaction layer** knows when the ghost speaks, but only as a boolean
  `is_playing()`; it cannot say "the last 400 ms of what you heard was me",
  which is exactly what echo is.

Meanwhile the audio abstraction itself is single-source by construction: one
`MiniAudioFactory` produces one player + one capture, `_install_aec` requires
both to exist *inside the same factory*, and `AcousticEchoCanceller` has a
single `push_far`. That shape cannot express "listen from A while speaking to
B", let alone a humanoid body speaking in the room while a private line runs to
the human's headset.

The human's framing for this workstream:

> "核心点实际上是抽象拓扑, 治理好抽象拓扑后, 所有问题都好解决."

So this is a topology-first workstream. The mechanisms (AEC, cloud speaker
diarization + voiceprint, local embedding) are surveyed as **candidates that
plug into the topology**, not as the starting point.

## Design Index

- Key design documents: `design/`
- Key discussion records: `discuss/`

## Key Decisions

### D1 — Layer the problem; do not solve it with one component (2026-10-26)

Four tiers, ordered by cost, each covering what the one above cannot:

| Tier | Mechanism | Cost | Covers | Cannot cover |
|---|---|---|---|---|
| 1 | Time gate — "I was speaking" window + tail | ≈0 | echo after the ghost stops | overlap |
| 2 | Text self-match — compare delivered text to what the ghost just said | ≈0 | verbatim echo (both incidents tonight) | ASR-mangled echo |
| 3 | Speaker attribution — SAUC `speaker_id` / voiceprint | +40–80 ms tail | "this is not the human" | overlap; <1 s fragments |
| 4 | AEC — per-sample subtraction | alignment, drift, device pairing | overlap — hearing the human while speaking | anything, if the device pair is unknown |

AEC is therefore an *optimization*, not a foundation. Its only irreplaceable
job is the overlap case: keeping the human audible while the ghost speaks.

### D2 — AEC is an edge in an acoustic coupling graph, not a factory by-product (2026-10-26)

The current `_install_aec` reads `capture.sample_rate` from inside the factory —
an assembler reaching into the assembled. The fix is not "declare the rate in
config" (rejected, see Exploration paths) but moving ownership: AEC has two
ports, `near` (whoever yields microphone samples) and `far` (whoever says "these
samples went out"), and each port may be fed from a different place — in-process
callback, shared-memory ring, socket, RTP. The assembler becomes a small
"coupling graph" declaration: which edge exists, and therefore which AEC
instance exists.

### D3 — Control plane on Matrix, data plane on a side channel (2026-10-26)

Matrix carries the *contract* of audio streams — who publishes, who subscribes,
format, ring/endpoint address, per-stream state and readings. Sample-level audio
does not ride the generic bus.

Rationale: at 16 kHz mono s16 the bandwidth is trivial (32 KB/s, 640 B per 10 ms
frame); what a general bus cannot guarantee is **constant rate**. Serialization
cost is not bytes but boundaries, and the boundary count is what turns into
jitter — which becomes AEC alignment error. Local transport is a shared-memory
ring (zero copy, µs); cross-machine is a standard protocol (RTP/L16, or WebRTC
when jitter buffer and clock recovery are needed).

### D4 — Audio frames carry `(source_id, sample_seq)` (2026-10-26)

Wall-clock timestamps describe *when a message arrived*, not *which position in
the audio timeline it occupies*. Two devices have two crystals. Every frame must
carry its own monotonic sample sequence so the consumer can align, detect gaps
and decide to drop or pad — the governance of drift lives here, not in a buffer
size.

### D5 — Prefer server-side voiceprint registration over a locally grown table (2026-10-26)

The ghost proposed bootstrapping a `speaker_id → identity` table from its own
speaking timeline (clusters that repeatedly fall inside "I was speaking" windows
are "self"). The human pointed out that Volcengine exposes a full voiceprint
lifecycle API, so identity can be *registered* instead of *inferred*:

> "它是有自带声纹注册接口的. 也就是你实际上可以用声纹注册动作来做, 比如你倒数
> 三二一后我说话之类的. 或者给个界面, 界面可以点击通过网页通讯注册声纹. 这样
> 比你治理声纹更好"

Accepted: cluster labels are session-scoped; a voiceprint is a durable identity
across sessions, restarts and devices. The timeline bootstrap survives only as
the fallback when no voiceprint exists.

Two hard constraints discovered with it:

- **The ghost's own voiceprint must be enrolled through the acoustic path**
  (play a known line, record it back through the same speaker→microphone chain),
  never uploaded as clean TTS audio — otherwise the embedding is of a studio
  timbre, not of "me as heard in this room", and matching fails.
- **`AudioUrl` must be reachable by the service.** Registration needs 5–30 s of
  audio at a public URL (TOS or a temporary tunnel). This is the only external
  dependency; it applies at enrolment time only, never at runtime.

GroupId is the natural expression of "one acoustic environment" (a machine + a
headset + a room): change device, change group.

### D6 — This workstream is pre-research (2026-10-26)

Status stays `draft` until the topology is settled. The deliverable first is a
topology document that all three mechanism families can be described in — not an
implementation.

### D7 — One workstream; audio IO follows identity, and is governed as many, not one (2026-10-26)

Audio IO does **not** get a workstream of its own. It has no independent problem
statement: its shape is a consequence of what identity governance needs from it
(frames carrying provenance, an alignable timeline, several inputs coexisting).
Defining IO first would fix the contract before the topology is settled, and then
force the topology around it — the wrong order.

The human's framing:

> "音频交互设计上就是 nodes 体系, 开机自带是为了快速在 beta3 版本能被人体验."
> "因为 mindflow + node 的存在, 你可以管理多个音频 io 而不是系统唯一. 这才是
> m-OS-s 的做法."
> "麦克风阵列本身也是要治理的."
> "我目前觉得 io 的问题是跟随身份识别的, 还不值得作为 feature 定义."

So the host-level `MiniAudioFactory` plus the audio providers are reclassified as
**a default bringup** — a built-in default audio node that merely happens to be
hosted by the host process — not as the architecture. The architecture is: every
audio IO is a node, and the ghost may hold many at once.

**Interchangeability test** — the criterion for "same abstraction":

> Can two audio IOs be swapped without changing the ghost at all?

Laptop mic ↔ headset mic ↔ a browser page ↔ a four-mic array ↔ a mic on a robot
body. If the ghost side needs even one line of change, the two are not the same
abstraction — each is carrying private wiring. `_install_aec`'s "both streams must
come from the same factory" is the archetype of that private wiring.

### D8 — Multiple audio inputs are nuclei, not a special case (2026-10-26)

The multi-IO case needs no new mechanism: **one audio input node = one sensing
nucleus**. The listener already surfaces as `listener_nucleus` (onset packets as
one impulse, deliver packets as another). A second microphone, a browser page or
a robot's ears each become a nucleus whose pulses are impulses in mindflow.

Identity therefore acquires a second job beyond echo rejection: **it is what lets
impulses from different inputs be correlated.** The same person appearing on three
nodes with different delays and gains is not three speakers — and those delays and
gains are exactly the spatial information a single microphone cannot have. That is
the strongest argument for this feature: identity governance is a precondition for
multi-sensory fusion, not a cleanup of audio.

## What this feature is not

- **Not an audio-IO implementation catalog.** Recording, streaming event
  detection, spectrum analysis, voice cloning, on-device ASR — each may become its
  own node, and fragmentation there is welcome (small contracts, one sitting per
  node). This feature produces the *requirements* an IO must satisfy, never the
  list of IOs.
- **Not a bus/transport project.** D3's control-plane/data-plane split is a
  constraint this feature must respect, not a deliverable.
- **Not a commitment to any mechanism.** AEC, voiceprint and local embedding stay
  candidates until the topology decides where each one plugs in.
- **Not a change to the default experience.** More than a dozen ghosts are
  running; the built-in chain stays the default and must not regress.

## Exploration paths

- **"Declare the AEC sample rate in the wiring layer"** — rejected by the human:
  it drags the *near* side into resampling as well, so both directions get
  resampled. A linear resampler inserted into the echo path adds a filter the
  delay estimator must chase. Correct shape: the listen side establishes the
  standard at AEC initialisation (see D2 for where "listen side" moves to).
- **"Keep the cluster table locally and let it grow"** — works, but only as a
  fallback; labels are not identity (see D5).
- **"Put audio on Matrix topics"** — rejected for the data plane; see D3.
- **Offline sanity check of pywebrtc_audio** (D26): feeding synthetic echo
  through our exact framing (near chopped from 50 ms into 10 ms blocks, far
  consumed in order, `stream_delay_ms=0`) yields **28.7–31.5 dB ERLE**. The
  library and the framing are fine; tonight's failure is wiring and environment,
  not algorithm.
- **"Accept the echo and pay in tokens"** — the human's first instinct. Rejected
  as insufficient: tokens are only the bill, the real damage is that the ghost
  treats its own words as instructions.

## Methods

- **ERLE without library support.** `pywebrtc_audio.EchoCanceller` exposes only
  `process` / `reset` / `stream_delay_ms` — no metrics at all (unlike
  `NoiseSuppressor.speech_probability` or `GainController.gain_db`). But ERLE is
  free for us: `_apply_aec` holds both pre- and post-AEC frames, so
  `10*log10(E[in²]/E[out²])` over a 1 s window is one line.
- **Instrument before tuning.** Enumerated readings, none of which exist today:
  AEC mounted (rate, delay hint), far queue length, far underruns (zero padding)
  and overflows (dropped reference), ERLE, and device identity per side.
- **Device identity must be readable.** The A-fix (commit `fb33f8bb`) logs the
  resolved device *name*, but when the pattern is empty the log says "system
  default" — it does not say which device that is. Default input and default
  output are two independent OS settings; nothing guarantees they are the same
  physical unit.
- **Browser capture is a different bargain (not yet a decision, a candidate).**
  `getUserMedia({audio: {echoCancellation, noiseSuppression, autoGainControl}})`
  ships WebRTC's AEC3 in the capture pipeline, and when playback goes through the
  same page's `AudioContext` its far reference is the page's own render stream —
  i.e. the reference problem D2 wrestles with is answered for free. Raw PCM needs
  `AudioWorklet` (128-sample blocks); `MediaRecorder` yields webm/opus, which must
  never feed an AEC reference.
- **Push-to-talk changes the problem class.** Hold-to-speak removes the need for
  VAD, for `end_window_size` tuning, and for echo rejection in the overlapping
  window: the medium becomes half-duplex by construction, and turn-taking
  replaces real-time arbitration. Cost: no hands-free. For a human sitting at a
  keyboard it is a strictly better bargain, which is why the browser node may be
  the cheapest path to a working multi-IO system.

## Implementation Notes

- The human's observation about collaboration, recorded because it shapes the
  plan: "现在的你遇到现有抽象往往不敢动手, 我们需要一个专门的对话去挖细节." —
  the topology deserves its own dedicated conversation rather than being
  improvised inside an implementation session.
- Volcengine SAUC already used by the listener supports `enable_speaker_info` +
  `show_utterances`; `ssd_mode=2` with `voiceprints` is the "short-audio
  voiceprint" product (single main speaker, AI-owner scenarios), +40–80 ms tail
  latency, up to 10 speakers, and — per the vendor's own doc — overlapping
  speech cannot be separated.
- The vendor's own limitation list is the argument for tier 4: "插话/重叠：
  声音重叠的说话人当前无法区分" is exactly the region only AEC covers.

## Open Questions (for the topology conversation)

1. What are the nodes of the coupling graph — *devices* (renderer / capture),
   *streams*, or *logical roles* (the human, the ghost, a body in the room)?
   What is an edge: "A can be heard by B", or "A is coupled to B with a path"?
2. Is identity a property of the **sound** (a segment carries a speaker), of the
   **channel** (this device pair is one identity), or of the **transport** (a
   publisher declares who it speaks for)?
3. Who owns "self" — the producer (knows what it said) or the recogniser (hears
   it)? Tonight both were available and neither was wired.
4. Where does the observation surface attach: per edge (ERLE, queue), per node
   (device identity, rate), or per segment (speaker, confidence)?
5. How does the topology survive a device change, a process split, or a move to
   another machine — what is re-derived, what is re-registered, what must be
   restarted?
6. With N nuclei producing impulses over the same acoustic reality, what is the
   unit of correlation — a segment, a time window, or a source hypothesis? Does
   mindflow need a cross-nucleus join, or does identity live entirely below it?
7. For an IO that runs in a browser (its own AEC3, push-to-talk, page-local
   render reference), is the identity question still ours to answer — or does the
   node report identity as part of its contract, leaving us to govern only how
   nodes combine?
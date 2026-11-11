/**
 * The ghoshell patch: a liveness route, and the anchor verb.
 *
 * This bundle exists so that "install our plugin into a dsh home" can be implemented and
 * verified before the capabilities that ride on it (anchor call, session read). It proves the
 * three things that matter about the mechanism — the bundle resolves by name, the patch layer
 * loads it, and a route answers — with nothing that can go wrong for a second reason.
 *
 * Plain JS on purpose: no build step, so installing is a file copy. `Remote` is a stage-3
 * decorator and this file has no decorator syntax, so the marking is done by hand with a fake
 * context — the same trick the playground probe used, and the reason this file must not be
 * refactored into something "cleaner".
 *
 * ### anchor — the coordinate
 *
 * `anchor` writes down where a session *is*: the log position the next turn would start from,
 * in a form that can leave the process. It reads the session's own objects and adds nothing —
 * `snapshotEvents()` for the position, `requestHeader()` for the config that was answering,
 * `session.header` for the preset and cwd. No llm call, no tool, no workspace: taking a
 * coordinate costs nothing, so there is no `run` to fail halfway.
 *
 * The payload uses the field names of `DshSessionAnchor` on the Python side, so the shape
 * crosses the wire once, unchanged. A null `seq` is not "unknown" — it is "the session's
 * start", which is why an empty log answers null rather than 0.
 */
import { randomUUID } from 'node:crypto'

import { brandString } from '@deepseek-ai/dsh-brand'
import { BlockAssembler, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionLogOffset } from '@deepseek-ai/dsh-session'
import { buildForkSeed } from '@deepseek-ai/dsh-session/fork'
import { setSandboxMode } from '@deepseek-ai/dsh-sandbox-policy'
import { setApprovalPolicy } from '@deepseek-ai/dsh-user-approval'
import { Remote, TypertRemoteService } from '@deepseek-ai/dsh-typert-protocol'

function markRemote(instance, method) {
  const context = {
    kind: 'method',
    name: method,
    static: false,
    private: false,
    addInitializer(initializer) {
      initializer.call(instance)
    },
  }
  Remote(undefined, context)
}

const TURN_START = 'turn/start'
const SESSION_TITLE = 'session/title'

/** The newest position in a log, or null when it has none yet. */
function lastSeq(events) {
  return events.length === 0 ? null : events[events.length - 1].seq
}

/** Which turn contains `seq` — the last `turn/start` at or before it. */
function turnAt(events, seq) {
  let turn = null
  for (const event of events) {
    if (event.seq > seq) break
    if (event.type === TURN_START && typeof event.data?.turn === 'number') turn = event.data.turn
  }
  return turn
}

/**
 * The message bodies a call asks for, in order.
 *
 * A bare string is one text body — the Python side's only piece of sugar. Everything else is the
 * one convention: an array of bodies, each an array of content blocks. Nothing here knows what a
 * "reminder" is; the caller assembles the sequence, and this says it in that order.
 */
function normalizeMessages(raw) {
  if (typeof raw === 'string') return raw === '' ? [] : [raw]
  if (!Array.isArray(raw)) return []
  return raw.filter((body) => {
    if (typeof body === 'string') return body !== ''
    return Array.isArray(body) && body.length > 0
  })
}

/**
 * The seq a call should stop at: `seq` when the caller has one, otherwise the end of the turn.
 *
 * Both are coordinates for the same point, so they are read as one: `seq` is exact and wins when
 * present; `turn` is what a caller who never recorded a seq actually has (a ghost's commit ref
 * carries turns). Neither means "the session's end" — that is a position too, not a guess.
 */
function boundarySeq(events, askedSeq, askedTurn) {
  if (askedSeq !== undefined && askedSeq !== null) return askedSeq
  if (askedTurn !== undefined && askedTurn !== null) return turnBoundarySeq(events, askedTurn)
  return lastSeq(events)
}

/**
 * The last event a turn owns: its `turn/end`, plus the standalone records that trail it before
 * the next `turn/start` (titles, injections) — the same cut the ghost's own seed uses.
 */
function turnBoundarySeq(events, turn) {
  let cut = -1
  for (let i = 0; i < events.length; i++) {
    const event = events[i]
    if (event.type === 'turn/end' && event.data?.turn === turn) cut = i
  }
  if (cut < 0) throw new Error(`chatOffRecord: no turn/end for turn ${turn}`)
  let end = cut
  for (let i = cut + 1; i < events.length; i++) {
    if (events[i].type === TURN_START) break
    end = i
  }
  return events[end].seq
}

/**
 * A service that may be absent in a given deployment.
 *
 * Required services are declared in `inject` — their absence should stop the plugin, loudly.
 * Permission services are the opposite case: they are optional in dsh's composition, and a
 * plugin that refuses to load because a deployment has no approval UI would be worse than one
 * that reports the capability missing when asked.
 */
function optionalService(ctx, key) {
  try {
    return ctx.get(key)
  } catch {
    return undefined
  }
}

/** Milliseconds between two reads of the kernel's own state. Cheap: same process, no I/O. */
const WAIT_POLL_MS = 100

/**
 * The sandbox modes a ghost may switch itself into. `danger-full-access` is absent by design:
 * full access is the human's to grant, never a ghost's — not to itself, not to a clone, not to
 * an agent it spawns.
 */
const REACHABLE_SANDBOX_MODES = ['read-only', 'workspace-write']

/** The id every verb needs, refused in the same words wherever it is missing. */
function requireSessionId(request, verb) {
  const sessionId = typeof request?.sessionId === 'string' ? request.sessionId : ''
  if (sessionId === '') throw new Error(`${verb}: sessionId is required`)
  return sessionId
}

/** The kernel's answer about one session — the one source `waitState` and callers share. */
function stateOf(ctx, sessionId) {
  const agent = ctx.agents.get(sessionId)
  const session = ctx.sessions.get(sessionId)
  const events = session === undefined ? [] : session.snapshotEvents()
  return {
    sessionId,
    attached: agent !== undefined,
    running: agent === undefined ? false : agent.status === 'running',
    hasPending: agent === undefined ? false : agent.inbox.hasPending,
    turn: lastTurn(events),
    seq: lastSeq(events),
    ...(session === undefined ? {} : { policy: policyOf(ctx, session) }),
  }
}

/** The permission pair for one session — the same read `sessionPolicy` exposes, factored out. */
function policyOf(ctx, session) {
  const policy = optionalService(ctx, 'sandboxPolicy')
  const events = session.snapshotEvents()
  return {
    ...(policy === undefined
      ? {}
      : (() => {
          const resolved = policy.resolve({ session })
          return { sandbox: resolved.mode, sandboxRoot: resolved.workspaceRoot }
        })()),
    sandboxOverride: lastOf(events, 'sandbox/mode')?.mode ?? null,
    approval: lastOf(events, 'approval/policy')?.policy ?? null,
  }
}

/**
 * The cut point official fork uses: the last completed turn, extended over trailing standalone
 * events, stopping before anything that belongs to the next one.
 */
function completedBoundary(events) {
  const ends = events.map((event, index) => (event.type === 'turn/end' ? index : -1)).filter((i) => i >= 0)
  if (ends.length === 0) return undefined
  let boundary = ends[ends.length - 1]
  for (const next of events.slice(boundary + 1)) {
    if (next.type === 'turn/start' || next.type === 'agent/inbox/spliced') break
    if (next.type === 'user/message' && next.surfaceOp === 'append') break
    boundary = next.seq
  }
  return boundary
}

/** Agent composition for one preset — the same two steps dsh's own factory takes. */
async function compose(ctx, presetId) {
  const presets = optionalService(ctx, 'agentPresets')
  if (presets === undefined) return { setup: () => {} }
  const resolved = await presets.resolve(presetId)
  return {
    agentPreset: resolved.id,
    setup: async (agentCtx) => {
      await presets.mount(agentCtx, resolved.id)
    },
  }
}

/** The payload of the newest event of one type, or null — the fold for log-only state. */
function lastOf(events, type) {
  let found = null
  for (const event of events) {
    if (event.type === type) found = event.data ?? null
  }
  return found
}

/** The newest turn the log has opened, or null. */
function lastTurn(events) {
  let turn = null
  for (const event of events) {
    if (event.type === TURN_START && typeof event.data?.turn === 'number') turn = event.data.turn
  }
  return turn
}

/** The session's current title, as the log tells it. */
function latestTitle(events) {
  let title = null
  for (const event of events) {
    if (event.type === SESSION_TITLE && typeof event.data?.title === 'string') title = event.data.title
  }
  return title
}

export class GhostShellService extends TypertRemoteService {
  constructor(ctx) {
    super(ctx, 'ghoshell')
    markRemote(this, 'ping')
    markRemote(this, 'anchor')
    markRemote(this, 'chatOffRecord')
    markRemote(this, 'archiveSession')
    markRemote(this, 'sessionState')
    markRemote(this, 'waitState')
    markRemote(this, 'inject')
    markRemote(this, 'llmProviders')
    markRemote(this, 'sessionPolicy')
    markRemote(this, 'setSessionPolicy')
    markRemote(this, 'forkSession')
    markRemote(this, 'attachWorkspace')
  }

  async ping() {
    return { ok: true, from: '@local/ghoshell-dsh', at: new Date().toISOString() }
  }

  /**
   * Where this session is. `request.seq` names an earlier point; absent means now.
   * Everything but the coordinate is "for the reader" — it is read from the log, never
   * asserted, and never used to reconstruct anything.
   */
  async anchor(request) {
    const sessionId = typeof request?.sessionId === 'string' ? request.sessionId : ''
    if (sessionId === '') throw new Error('anchor: sessionId is required')

    const session = this.ctx.sessions.get(sessionId)
    if (session === undefined) throw new Error(`anchor: no live session ${sessionId}`)

    const events = session.snapshotEvents()
    const asked = request?.seq
    let seq
    if (asked === undefined || asked === null) {
      seq = lastSeq(events)
    } else if (!Number.isInteger(asked) || asked < 0 || asked >= events.length) {
      const last = events.length === 0 ? 'none' : `0..${events.length - 1}`
      throw new Error(`anchor: seq ${asked} is outside the log (${last})`)
    } else {
      seq = asked
    }

    const config = session.requestHeader()?.config ?? {}
    const header = session.header ?? {}
    return {
      session_id: sessionId,
      seq,
      turn: seq === null ? null : turnAt(events, seq),
      model: config.model ?? null,
      reasoning_effort: config.reasoningEffort ?? null,
      agent_preset: header.agentPreset ?? null,
      cwd: header.cwd ?? null,
      title: latestTitle(events),
      note: typeof request?.note === 'string' ? request.note : null,
      created: new Date().toISOString(),
    }
  }

  /**
   * What the kernel says about this session *right now*.
   *
   * `agent.status` and `agent.inbox.hasPending` are fields on an object in this process. Reading
   * them is not a distributed read: no transport, no window between snapshot and subscription, no
   * event that could have been missed. That is why this is the authority and the `$events`
   * mirror is not — the mirror is a convenience for display, and it cannot know what it missed.
   */
  async sessionState(request) {
    return stateOf(this.ctx, requireSessionId(request, 'sessionState'))
  }

  /**
   * Block until the session settles, read from the kernel in a tight loop.
   *
   * `condition: "idle"` means "not running **and** nothing queued" — both halves matter, because
   * a session with queued input is about to start and a caller waiting for it to stop would
   * otherwise be told it already had. `"running"` is the other direction.
   *
   * The wait polls. That is the honest shape here: what it polls lives in this process, so the
   * poll costs a property read and the clock is the only one involved. An event-driven wait would
   * need to state what happens when the event does not arrive, and the answer would be "poll".
   */
  async waitState(request) {
    const sessionId = requireSessionId(request, 'waitState')
    const condition = request?.condition === 'running' ? 'running' : 'idle'
    const timeoutMs = typeof request?.timeoutMs === 'number' ? request.timeoutMs : 300000
    const deadline = Date.now() + timeoutMs

    for (;;) {
      const state = stateOf(this.ctx, sessionId)
      const settled = condition === 'running' ? state.running : !state.running && !state.hasPending
      if (settled) return state
      if (Date.now() >= deadline) {
        throw new Error(
          `waitState: ${condition} not reached within ${timeoutMs}ms ` +
            `(running=${state.running}, pending=${state.hasPending})`,
        )
      }
      await new Promise((resolve) => setTimeout(resolve, WAIT_POLL_MS))
    }
  }

  /**
   * Hand one message into the running conversation — `agent.inject`, the official verb.
   *
   * What dsh's `inject` does, and what a caller has to know before using it: the message lands in
   * the inbox's `next-step` slot and is claimed at the **next step boundary** (`preStep`), which
   * is also the next model call. It never takes effect immediately, and it never wakes the
   * driver — on an idle agent the message is held until something else starts a turn. That is
   * the difference between `inject` and `steer`; both land in the same slot, only `steer` wakes.
   *
   * `wait: "claimed"` answers the question "did a step actually take it". The witness is the
   * log, not an event: a claimed message appears as a `user/message` event carrying the same id,
   * inside a turn. If it leaves the inbox without appearing in the log it was cancelled, and
   * that is reported as an error rather than as success.
   */
  async inject(request) {
    const sessionId = requireSessionId(request, 'inject')
    const text = typeof request?.text === 'string' ? request.text : ''
    if (text === '') throw new Error('inject: text must be a non-empty string')

    const agent = this.ctx.agents.get(sessionId)
    if (agent === undefined) throw new Error(`inject: no live agent ${sessionId}`)
    const session = this.ctx.sessions.get(sessionId)
    if (session === undefined) throw new Error(`inject: no live session ${sessionId}`)

    // Hold the official pieces: the message is dsh's, the inbox is dsh's, the claim is dsh's.
    const message = createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
    const woke = agent.status === 'running'
    const fromSeq = lastSeq(session.snapshotEvents())
    agent.inject(message)
    const messageId = message.id

    if (request?.wait !== 'claimed') {
      return { messageId, claimed: false, turn: null, woke }
    }

    const timeoutMs = typeof request?.timeoutMs === 'number' ? request.timeoutMs : 30000
    const deadline = Date.now() + timeoutMs
    for (;;) {
      for (const event of session.snapshotEvents(fromSeq === null ? 0 : fromSeq + 1)) {
        if (event.type === 'user/message' && event.data?.id === messageId) {
          return { messageId, claimed: true, turn: turnAt(session.snapshotEvents(), event.seq), woke }
        }
      }
      const stillQueued = agent.inbox.nextStep.some((queued) => queued.id === messageId)
      if (!stillQueued) {
        throw new Error(`inject: message ${messageId} left the inbox without being claimed (cancelled?)`)
      }
      if (Date.now() >= deadline) {
        throw new Error(`inject: message ${messageId} was not claimed within ${timeoutMs}ms (woke=${woke})`)
      }
      await new Promise((resolve) => setTimeout(resolve, WAIT_POLL_MS))
    }
  }

  /**
   * What this session is allowed to do, and where.
   *
   * Two knobs, not one bit: `sandbox` says where its tools may act, `approval` says whether a
   * tool call needs a human nod. dsh stores both as log events on the session (durable, replayed
   * on restart) and — this is why switching is cheap — neither is a surface event, so changing
   * permission does not enter the transcript and does not disturb the model prefix.
   */
  async sessionPolicy(request) {
    const sessionId = requireSessionId(request, 'sessionPolicy')
    const session = this.ctx.sessions.get(sessionId)
    if (session === undefined) throw new Error(`sessionPolicy: no live session ${sessionId}`)
    const policy = optionalService(this.ctx, 'sandboxPolicy')
    if (policy === undefined) throw new Error('sessionPolicy: this dsh has no sandbox policy service')
    // The overrides are read from the log, not from a service: `approval/policy` is written by a
    // module function and its reader is an *optional* service this deployment may not mount —
    // depending on it would have reported "no approval" for a session that had just switched.
    // The last event of each type IS the override; both are log-only, so this fold is exact.
    return policyOf(this.ctx, session)
  }

  /**
   * Switch this session's permission — and only within what a ghost may switch itself.
   *
   * `danger-full-access` is refused here on purpose. That is not a capability we lack; it is a
   * boundary: full access is something the human grants to a ghost, never something a ghost
   * grants to itself, a clone, or an agent it spawned. The refusal states that, so the boundary
   * survives whoever reads this code next.
   */
  async setSessionPolicy(request) {
    const sessionId = requireSessionId(request, 'setSessionPolicy')
    const session = this.ctx.sessions.get(sessionId)
    if (session === undefined) throw new Error(`setSessionPolicy: no live session ${sessionId}`)

    const sandbox = request?.sandbox
    if (sandbox !== undefined && sandbox !== null) {
      if (!REACHABLE_SANDBOX_MODES.includes(sandbox)) {
        throw new Error(
          `setSessionPolicy: ${JSON.stringify(sandbox)} is not reachable from a ghost ` +
            `(allowed: ${REACHABLE_SANDBOX_MODES.join(', ')}); full access is granted by the human only`,
        )
      }
      setSandboxMode(session, sandbox)
    }

    const approval = request?.approval
    if (approval !== undefined && approval !== null) {
      if (approval !== 'ask' && approval !== 'never') {
        throw new Error(`setSessionPolicy: approval must be "ask" or "never", got ${JSON.stringify(approval)}`)
      }
      setApprovalPolicy(session, approval)
    }

    return this.sessionPolicy({ sessionId })
  }

  /**
   * Fork a session into a **different** working directory — the one thing dsh's own fork refuses.
   *
   * Official `session/fork` builds the seed correctly and then hard-codes the child's cwd to the
   * source's. Everything it uses is public, so this route composes the same pieces with one
   * field changed: `buildForkSeed` for a contiguous, balanced prefix, `agents.create` with the
   * new `meta.cwd`. The child is then a session whose writable boundary is the new directory —
   * because that boundary is the cwd, and nothing else.
   *
   * What it costs is the prefix: the workspace instruction files discovered from the new cwd are
   * different, and anything the system prompt says about the environment changes with them. The
   * cut is the same rule official fork uses — the last completed turn, extended over trailing
   * standalone events, stopping before anything that belongs to the next turn.
   */
  async forkSession(request) {
    const sessionId = requireSessionId(request, 'forkSession')
    const cwd = typeof request?.cwd === 'string' ? request.cwd : ''
    if (cwd === '') throw new Error('forkSession: cwd is required — it is the one thing this differs in')

    const source = this.ctx.sessions.get(sessionId)
    if (source === undefined) throw new Error(`forkSession: no live session ${sessionId}`)

    const events = source.snapshotEvents()
    const boundary = typeof request?.atSeq === 'number' ? request.atSeq : completedBoundary(events)
    if (boundary === undefined || events[boundary]?.seq !== boundary) {
      throw new Error(`forkSession: no completed turn to cut at (atSeq ${String(request?.atSeq)})`)
    }
    const seed = buildForkSeed(events, boundary)

    const preset = typeof request?.agentPreset === 'string' ? request.agentPreset : source.header?.agentPreset
    const composition = await compose(this.ctx, preset)
    const childId = brandString(`session-${randomUUID()}`)
    const config = source.requestHeader()?.config
    await this.ctx.agents.create({
      sessionId: childId,
      seed,
      inheritedEventCount: SessionLogOffset(boundary + 1),
      meta: {
        cwd,
        parentSession: sessionId,
        isSeeded: true,
        ...(composition.agentPreset === undefined ? {} : { agentPreset: composition.agentPreset }),
      },
      ...(config === undefined ? {} : { agentOptions: { provider: config.provider, model: config.model } }),
      setup: composition.setup,
    })

    // Delegation carries permission with it. dsh marks exactly this case on the event
    // (`source: 'delegation'`), and the alternative is worse than it looks: a child that inherits
    // a conversation but not what the conversation was allowed to do is a clone with a different
    // temperament, and the difference shows up as a refused write rather than as a decision.
    // Only overrides are copied — a parent that never switched has nothing to say, and the child
    // falls to the same deployment default it would have anyway.
    const child = this.ctx.sessions.get(childId)
    if (child !== undefined && request?.inheritPolicy !== false) {
      const mode = lastOf(events, 'sandbox/mode')?.mode
      if (mode !== undefined) child.append('sandbox/mode', { mode, source: 'delegation' })
      const approval = lastOf(events, 'approval/policy')?.policy
      if (approval !== undefined) child.append('approval/policy', { policy: approval, source: 'delegation' })
    }
    // Same closing step dsh's own fork takes: put the child in the workspace that owns its
    // directory. Skipped only when asked to, or when this deployment has no registry.
    if (child !== undefined && request?.attach !== false && optionalService(this.ctx, 'workspaceRegistry') !== undefined) {
      const workspace = await optionalService(this.ctx, 'workspaceRegistry').create(cwd)
      await workspace.attachSession(childId)
    }
    return { sessionId: childId, boundary, seededEvents: seed.length }
  }

  /**
   * Register a session under the workspace that owns its directory.
   *
   * A workspace is a *registry of directories*, not a permission boundary: attaching changes
   * which list a session shows up in, not where its tools may act (that is the sandbox root, and
   * that is the session's cwd). dsh does this for its own forks; a session created any other way
   * — ours included — has to say so.
   */
  async attachWorkspace(request) {
    const sessionId = requireSessionId(request, 'attachWorkspace')
    const session = this.ctx.sessions.get(sessionId)
    if (session === undefined) throw new Error(`attachWorkspace: no live session ${sessionId}`)
    const registry = optionalService(this.ctx, 'workspaceRegistry')
    if (registry === undefined) throw new Error('attachWorkspace: this dsh has no workspace registry')

    const path = typeof request?.path === 'string' && request.path !== '' ? request.path : session.header?.cwd
    if (path === undefined || path === '') throw new Error('attachWorkspace: neither path nor session cwd is known')

    const workspace = await registry.create(path)
    await workspace.attachSession(sessionId)
    return { workspaceId: String(workspace.id), path }
  }

  /**
   * Which model routes this dsh can actually reach, live or merely declared.
   *
   * A caller choosing a model without asking first is a caller about to be corrected by an
   * error; this is the cheap question to ask first. `live` separates "the adapter registered
   * this route" from "somebody shipped the knowledge that it exists".
   */
  async llmProviders() {
    const live = this.ctx.llm.listProviders().map((provider) => ({
      id: provider.id,
      name: provider.name,
      live: true,
    }))
    const declared = (this.ctx.llm.listConfigurableProviders?.() ?? []).map((provider) => ({
      id: provider.provider,
      name: provider.displayName,
      live: false,
      declared: provider.declared ?? null,
      error: provider.error ?? null,
    }))
    const seen = new Set(live.map((provider) => provider.id))
    return { providers: [...live, ...declared.filter((provider) => !seen.has(provider.id))] }
  }

  /**
   * One model turn against a session's context at a chosen point, with nothing written back.
   *
   * The prefix is the session's own derived history — the same projection the main path uses to
   * assemble its request, so an off-record call inherits exactly what the conversation saw,
   * including the system prompt (a `system/message` in the history). That is why no `system`
   * option is passed: the adapter would concatenate it *before* the history's system prompt and
   * invalidate the prefix cache.
   *
   * Two requirements, both about not lying:
   *
   * - **A cold session is replayed from its log.** A case outlives its carrier, so asking a step
   *   a question must work when nothing is live: the stored events are sliced at the anchored
   *   position and replayed into a detached session — the same seed the fork path builds, never
   *   published.
   * - **Any position can be anchored, and the coordinate is `seq` first.** A caller that only
   *   knows the turn (the ghost's commit refs carry turns, not seqs) still gets the right cut:
   *   `seq` wins when given, `turn` resolves to that turn's end when it is not.
   *
   * `messages` is an ordered list of bodies: each one is a content-block array, and a bare
   * string is one text body. A reminder and a question are two messages because the caller said
   * two things in sequence — this verb does not know what either word means.
   */
  async chatOffRecord(request) {
    const sessionId = typeof request?.sessionId === 'string' ? request.sessionId : ''
    if (sessionId === '') throw new Error('chatOffRecord: sessionId is required')
    const bodies = normalizeMessages(request?.messages ?? request?.message)
    if (bodies.length === 0) throw new Error('chatOffRecord: messages must contain at least one non-empty body')

    const resolved = await this.resolveSession(sessionId, request?.anchor?.seq, request?.anchor?.turn)
    try {
      return await this.answerOffRecord(resolved.session, request, bodies)
    } finally {
      if (resolved.dispose !== undefined) await resolved.dispose()
    }
  }

  /**
   * Archive a session: take it out of the workspace's active set, keep its log.
   *
   * This is the collection side of a finished case. `stopActivity` asks the composed providers to
   * stop whatever still runs for it; the archive is written first, so a provider failure cannot
   * undo it.
   */
  async archiveSession(request) {
    const sessionId = requireSessionId(request, 'archiveSession')
    const registry = optionalService(this.ctx, 'workspaceRegistry')
    if (registry === undefined) {
      throw new Error('archiveSession: workspaceRegistry is not composed in this deployment')
    }
    await registry.archiveSession(sessionId, { stopActivity: request?.stop === true })
    return { sessionId, archived: true }
  }

  /**
   * The session to ask: the live one when it is at the right point, otherwise a detached replay.
   *
   * A detached replay is built from the *stored* log, so it works after a restart, after the
   * carrier was retired, and at any position — which is exactly what "ask an old step" means.
   * The position is `seq` when the caller has one, otherwise the end of the named `turn`.
   */
  async resolveSession(sessionId, askedSeq, askedTurn) {
    const live = this.ctx.sessions.get(sessionId)
    if (live !== undefined) {
      const events = live.snapshotEvents()
      const boundary = boundarySeq(events, askedSeq, askedTurn)
      if (boundary === lastSeq(events)) {
        return { session: live, dispose: undefined }
      }
      return { session: sliceSession(sessionId, events, live.header, boundary), dispose: undefined }
    }

    // `sessionQuery` is not in this service's inject list (the property guard would throw); the
    // optional-service read returns undefined when a deployment does not compose it.
    const query = optionalService(this.ctx, 'sessionQuery')
    if (query === undefined) {
      throw new Error('chatOffRecord: sessionQuery is not composed; cannot replay a cold session')
    }
    const observed = await query.observeSession(sessionId)
    const events = observed.events
    const boundary = boundarySeq(events, askedSeq, askedTurn)
    const session = sliceSession(sessionId, events, observed.header, boundary)
    const dispose =
      observed[Symbol.asyncDispose]?.bind(observed) ?? observed[Symbol.dispose]?.bind(observed)
    return { session, dispose }
  }

  /** One model call, assembled from the resolved session's own header and history. */
  async answerOffRecord(session, request, bodies) {
    const header = session.requestHeader()
    if (header === undefined) {
      throw new Error('chatOffRecord: session has no request header yet, nothing to inherit')
    }

    const inherited = session.deriveMessages()
    // Tools ride *on top of* the session's own declarations, never in place of them: they are a
    // top-level request field, so a call that changes them starts a new prefix and loses the
    // cache the conversation had built.
    // Extra tools ride *on top of* the session's own declarations, never in place of them.
    // Measured, both ways, on a live session: appending one left `cacheReadTokens` at 9728 and
    // the model answering normally; replacing the list dropped it to 1024 *and* made the model
    // emit its tool-call syntax (DSML) as plain text, because the tools it knows were gone.
    // Removing is therefore the expensive direction — which is also the opposite of what the
    // "tools are top-level, so any change kills the prefix" reasoning predicted.
    // Extra tools are added on top of the session's own declarations. Removal is deliberately not
    // offered: it was measured to cost the cache (9088 -> 1024) and to make the model lose its
    // footing, and a knob whose only effect is that is not a knob worth shipping.
    // `readonly` is the bypass shape: the request is assembled with NOTHING that could write, so
    // the tools menu is empty. That is the mechanical half of "do not call a tool" — a model that
    // is offered a tool answers with one, and an off-record call cannot run it.
    const tools = request?.readonly === true
      ? []
      : [...(header.tools ?? []), ...(Array.isArray(request?.tools) ? request.tools : [])]
    // A patch, not a config: the base is the session's own header, so a call stays inside the
    // conversation it borrowed and only states what is different about this one answer.
    const config = { ...header.config, ...(request?.llm ?? {}) }
    // One user message per body, in the order given: the parts are the caller's, not ours. A
    // reminder and a question are two things said in sequence, and collapsing them into one
    // message would decide a shape the caller already expressed.
    const said = bodies.map((body) => ({
      role: 'user',
      content: typeof body === 'string' ? [{ type: 'text', text: body }] : body,
    }))
    const outcome = await oneShot(this.ctx, {
      ...config,
      messages: [...inherited, ...said],
      toolHistory: session.toolHistory(),
      ...(tools.length === 0 ? {} : { tools }),
      sessionId: session.id,
      ...(typeof request?.maxTokens === 'number' ? { maxTokens: request.maxTokens } : {}),
    })
    const { usage, ...rest } = outcome
    return {
      ...rest,
      ...(usage === null || usage === undefined ? {} : { usage }),
      inherited_messages: inherited.length,
    }
  }
}

/**
 * Replay one event prefix into a detached session — the fork seed, without publishing a child.
 *
 * The seed is what the fork path builds for a child session; here it is handed to a session that
 * is never stored, read, or announced. `inheritedEventCount` is the durable cut, so the replay
 * knows which part it was given.
 */
function sliceSession(sessionId, events, header, boundary) {
  if (boundary === undefined || boundary === null) {
    throw new Error('chatOffRecord: this session has no events to anchor at')
  }
  if (events[boundary]?.seq !== boundary) {
    const last = events.length === 0 ? 'none' : String(events[events.length - 1].seq)
    throw new Error(
      `chatOffRecord: event ${String(boundary)} does not exist in session "${sessionId}" (last seq: ${last})`,
    )
  }
  const seed = buildForkSeed(events, boundary)
  // Not a fork: every seeded event belongs to this session, so the inherited prefix is empty.
  return Session.create(sessionId, seed, header, SessionLogOffset(0))
}

/**
 * One request, assembled by hand — the whole side effect is one round trip.
 *
 * Nothing is appended, nothing is entered, no session is created: this is `ctx.llm.stream`
 * with a message list we built ourselves, which is the only layer that accepts a call that
 * leaves no trace. The cost is reported back because an off-record call is still paid for, and
 * the session's own accounting will never see it.
 */
async function oneShot(ctx, options) {
  const assembler = new BlockAssembler()
  for await (const chunk of ctx.llm.stream(options)) assembler.push(chunk)
  const blocks = assembler.blocks()
  const finish = assembler.finish
  // An empty answer is not an answer: a provider without a usable route or credential can end the
  // stream with nothing in it, and returning "" would make that look like a model that said no.
  if (blocks.length === 0 && assembler.usage === undefined) {
    throw new Error(
      `chatOffRecord: the model returned nothing (finish: ${finish?.kind ?? 'none'}) — ` +
        'check the session route and its credential',
    )
  }
  return {
    text: blocks
      .filter((block) => block.type === 'text')
      .map((block) => block.text)
      .join(''),
    // A response can be a tool call instead of prose. Dropping these here would make the two
    // kinds of answer indistinguishable at the caller — "no text" and "asked for a tool" would
    // look the same, and only one of them means the call is unfinished.
    tool_calls: blocks
      .filter((block) => block.type === 'tool-call')
      .map((block) => ({
        id: block.id ?? '',
        name: block.name ?? '',
        arguments: block.arguments ?? '',
      })),
    usage: assembler.usage ?? null,
    finish_reason: finish?.kind ?? '',
  }
}

export const name = 'ghoshell'

/**
 * The anchor verb reads the live session book; `chatOffRecord` also needs the model.
 */
export const inject = ['sessions', 'llm', 'agents']

/**
 * The service half alone.
 *
 * The page bridge used to ride in this file. It is its own package
 * (`@local/ghoshell-dsh-bridge`) because the two are installed by different owners for
 * different reasons: a ghost home wants the verbs, a page-driving deployment also wants a
 * page it can reach. Welding them meant every consumer carried the other one's cost.
 */
export function apply(ctx) {
  ctx.plugin(GhostShellService)
}

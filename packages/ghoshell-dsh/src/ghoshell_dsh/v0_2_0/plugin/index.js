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
import { BlockAssembler } from '@deepseek-ai/dsh-llm'
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

/** Milliseconds between two reads of the kernel's own state. Cheap: same process, no I/O. */
const WAIT_POLL_MS = 100

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
  }
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
    markRemote(this, 'sessionState')
    markRemote(this, 'waitState')
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
   * One model turn against this session's live context, with nothing written back.
   *
   * The prefix is `deriveMessages()` — the same call the main path uses to assemble its own
   * request, so an off-record call inherits exactly what the conversation sees, including the
   * system prompt (which is a `system/message` in the history). That is also why no `system`
   * option is passed: the adapter would concatenate it *before* the history's system prompt and
   * invalidate the prefix cache.
   *
   * `anchor.seq` is accepted only at the current end for now: assembling an earlier prefix means
   * building an unpublished session from a log slice, which is the next step and not this one.
   * Say so loudly instead of quietly answering from the wrong place.
   */
  async chatOffRecord(request) {
    const sessionId = typeof request?.sessionId === 'string' ? request.sessionId : ''
    if (sessionId === '') throw new Error('chatOffRecord: sessionId is required')
    const message = typeof request?.message === 'string' ? request.message : ''
    if (message === '') throw new Error('chatOffRecord: message must be a non-empty string')

    const session = this.ctx.sessions.get(sessionId)
    if (session === undefined) throw new Error(`chatOffRecord: no live session ${sessionId}`)

    const header = session.requestHeader()
    if (header === undefined) throw new Error('chatOffRecord: session has no request header yet, nothing to inherit')

    const events = session.snapshotEvents()
    const end = lastSeq(events)
    const asked = request?.anchor?.seq
    if (asked !== undefined && asked !== null && asked !== end) {
      throw new Error(`chatOffRecord: only the current end can be anchored yet (asked ${asked}, end ${end})`)
    }

    const inherited = session.deriveMessages()
    const outcome = await oneShot(this.ctx, {
      ...header.config,
      messages: [...inherited, { role: 'user', content: [{ type: 'text', text: message }] }],
      toolHistory: session.toolHistory(),
      ...(header.tools === undefined ? {} : { tools: header.tools }),
      sessionId: session.id,
      ...(typeof request?.maxTokens === 'number' ? { maxTokens: request.maxTokens } : {}),
    })
    return { ...outcome, inherited_messages: inherited.length }
  }
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
  return {
    text: blocks
      .filter((block) => block.type === 'text')
      .map((block) => block.text)
      .join(''),
    usage: assembler.usage ?? null,
    finish_reason: typeof assembler.finish === 'string' ? assembler.finish : '',
  }
}

export const name = 'ghoshell'

/**
 * The anchor verb reads the live session book; `chatOffRecord` also needs the model.
 */
export const inject = ['sessions', 'llm', 'agents']

export function apply(ctx) {
  ctx.plugin(GhostShellService)
}

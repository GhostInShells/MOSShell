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
}

export const name = 'ghoshell'

/** The anchor verb reads the live session book; without it this plugin has nothing to say. */
export const inject = ['sessions']

export function apply(ctx) {
  ctx.plugin(GhostShellService)
}

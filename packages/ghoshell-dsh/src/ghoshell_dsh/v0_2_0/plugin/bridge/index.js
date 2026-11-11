/**
 * The page bridge — host half, as its own dsh bundle.
 *
 * The service bundle (`@local/ghoshell-dsh`) answers RPC verbs about sessions. This one is a
 * different concern: it holds a queue that a dsh *page* polls, so the ghost can ask the page to
 * run code against its own cordis context. Neither half needs the other, so they are two
 * packages: a ghost home that only wants the verbs never loads this one, and never pays its poll.
 *
 * `inject: ['webServer']` is the whole point: cordis applies a plugin only once its dependencies
 * exist, so this half starts when the web server does and simply never starts where there is none
 * (a headless deployment has no page to drive). Probing with `ctx.get` at apply time was the wrong
 * shape — it could silently miss a service that had not been registered yet.
 */
const BRIDGE = '/ghoshell/bridge'

function readJson(req) {
  return new Promise((resolve, reject) => {
    let raw = ''
    req.on('data', (chunk) => { raw += chunk })
    req.on('end', () => {
      try { resolve(raw ? JSON.parse(raw) : {}) } catch (err) { reject(err) }
    })
    req.on('error', reject)
  })
}

function sendJson(res, status, body) {
  res.statusCode = status
  res.setHeader('content-type', 'application/json; charset=utf-8')
  res.setHeader('cache-control', 'no-store')
  res.end(JSON.stringify(body ?? null))
}

/**
 * The page bridge — host half. Three routes and one queue.
 *
 * The ghost posts code; the page polls for it, runs it against its own cordis context, and posts
 * the outcome back. No websocket, no build step, no RPC envelope — it is a transport, so the layer
 * above can stay about *what* to ask a page rather than *how* to reach one.
 *
 * It does not decide who may call it. These are plain routes, so they bypass the api gateway's
 * auth; the boundary lives in whatever exposes `page_exec` to a model (a channel command, a node).
 */
function attachBridge(ctx, webServer) {
  /** @type {Array<{id: string, code: string, timeoutMs: number}>} */
  const queue = []
  /** @type {Map<string, {resolve: (v: any) => void}>} */
  const waiting = new Map()
  let seq = 0
  /** When the page last asked for work — the only liveness signal there is. */
  let lastPullAt = 0

  const pull = webServer.register({
    kind: 'exact',
    path: `${BRIDGE}/pull`,
    async handler(req, res) {
      lastPullAt = Date.now()
      sendJson(res, 200, queue.shift() ?? {})
    },
  })

  const result = webServer.register({
    kind: 'exact',
    path: `${BRIDGE}/result`,
    async handler(req, res) {
      let body
      try { body = await readJson(req) } catch (err) { sendJson(res, 400, { error: 'bad json' }); return }
      const pending = waiting.get(body.id)
      if (!pending) { sendJson(res, 404, { error: 'unknown job' }); return }
      pending.resolve(body)
      sendJson(res, 200, { ok: true })
    },
  })

  const status = webServer.register({
    kind: 'exact',
    path: `${BRIDGE}/status`,
    async handler(req, res) {
      const ago = lastPullAt === 0 ? null : Date.now() - lastPullAt
      sendJson(res, 200, { attached: ago !== null && ago < 5000, lastPullAgoMs: ago, queued: queue.length })
    },
  })

  const exec = webServer.register({
    kind: 'exact',
    path: `${BRIDGE}/exec`,
    async handler(req, res) {
      let body
      try { body = await readJson(req) } catch (err) { sendJson(res, 400, { ok: false, error: 'bad json' }); return }
      if (typeof body.code !== 'string' || body.code.trim() === '') {
        sendJson(res, 400, { ok: false, error: 'code required' })
        return
      }
      // A page that stopped polling is a page we cannot drive; saying so beats queueing into a void.
      if (lastPullAt === 0 || Date.now() - lastPullAt > 5000) {
        sendJson(res, 409, { ok: false, error: 'no page attached (no poll within 5s)' })
        return
      }
      const id = `j${++seq}`
      const timeoutMs = Number(body.timeoutMs) > 0 ? Number(body.timeoutMs) : 5000
      const settled = new Promise((resolve) => waiting.set(id, { resolve }))
      queue.push({ id, code: body.code, timeoutMs })
      const timer = setTimeout(() => {
        const pending = waiting.get(id)
        if (pending) { pending.resolve({ id, ok: false, error: 'host timeout' }); waiting.delete(id) }
      }, timeoutMs + 2000)
      const outcome = await settled
      clearTimeout(timer)
      waiting.delete(id)
      // A page that answers late would otherwise leave the job in the queue and run it twice.
      const stuck = queue.findIndex((job) => job.id === id)
      if (stuck >= 0) queue.splice(stuck, 1)
      sendJson(res, 200, outcome)
    },
  })

  ctx.effect(() => () => {
    pull()
    result()
    status()
    exec()
  })
}

export const name = 'ghoshell-bridge'

export const inject = ['webServer']

export function apply(ctx) {
  attachBridge(ctx, ctx.webServer)
}

/**
 * The browser half of the page bridge.
 *
 * A dsh page is a cordis context. This module attaches to it, polls the host half for a job, runs
 * the job's code against the real `ctx`, and posts the JSON projection back. That is the whole
 * mechanism — and it is why the ghost can drive *dsh's own* layout/focus API instead of painting
 * CSS over the DOM: `ctx.get('layout')` is the live service, not a picture of one.
 *
 * Two facts this file must not hide:
 *
 * - **There is no sandbox.** The code runs in the page's realm: `window`, `document` and `fetch`
 *   are all reachable, and `ctx.get` reaches services without an `inject` declaration. The trust
 *   boundary is the caller, not this file.
 * - **Results must be JSON-projected.** `ctx` methods hand back live objects; only what
 *   `JSON.stringify` can carry crosses back.
 *
 * The loader ABI is the vendored one (`window.__ModuleLoader__` + a CJS-style factory) — a real
 * bundle was read once to learn the shape, and nothing here needs a build step.
 */
window.__ModuleLoader__.load({
  id: '@local/ghoshell-dsh',
  factory: (require) => {
    var module = { exports: {} }
    var exports = module.exports

    const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor
    const BASE = '/ghoshell/bridge'

    function jsonSafe(value) {
      try {
        return JSON.parse(JSON.stringify(value ?? null))
      } catch (err) {
        return { __unserializable: String(value) }
      }
    }

    async function runJob(ctx, job) {
      const started = performance.now()
      try {
        const fn = new AsyncFunction('ctx', job.code)
        const value = await Promise.race([
          fn(ctx),
          new Promise((_, reject) =>
            setTimeout(() => reject(new Error('timeout')), job.timeoutMs || 5000),
          ),
        ])
        return { id: job.id, ok: true, value: jsonSafe(value), ms: Math.round(performance.now() - started) }
      } catch (err) {
        return { id: job.id, ok: false, error: String(err && err.message ? err.message : err) }
      }
    }

    async function apply(ctx) {
      if (window.__ghoshellBridgeStarted) return
      window.__ghoshellBridgeStarted = true
      let busy = false

      async function tick() {
        if (busy) return
        busy = true
        try {
          const res = await fetch(`${BASE}/pull`, { cache: 'no-store' })
          if (res.ok) {
            const job = await res.json()
            if (job && job.id) {
              const outcome = await runJob(ctx, job)
              await fetch(`${BASE}/result`, {
                method: 'POST',
                headers: { 'content-type': 'application/json' },
                body: JSON.stringify(outcome),
              })
            }
          }
        } catch (err) {
          // The host may not be up yet; a failed poll is not an error worth surfacing.
        } finally {
          busy = false
        }
      }

      setInterval(tick, 400)
      tick()
    }

    exports.apply = apply
    exports.inject = ['sidebarRight', 'layout', 'slots', 'sessions', 'shortcuts', 'remote']
    return module.exports
  },
})

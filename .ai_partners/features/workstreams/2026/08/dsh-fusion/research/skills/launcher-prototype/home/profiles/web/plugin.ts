import type { Context } from '@deepseek-ai/cordis'

export const name = 'launcher-prototype'
export const inject = ['webServer']

// 供 launcher 验证的零第三方依赖 HTTP 面:
//   /plugin-api/ping — 就绪探针 (readiness 轮询目标)
//   /plugin-api/echo  — outbound call 往返目标 (原样回显 payload)
// 下行 session/event 走 dsh web 内置 /api/events.mux WS, 本 plugin 不注册.

export function apply(ctx: Context) {
  ctx.webServer.register({
    kind: 'prefix',
    path: '/plugin-api/ping',
    handler: async (_req, res) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ ok: true }))
    },
  })

  ctx.webServer.register({
    kind: 'prefix',
    path: '/plugin-api/echo',
    handler: async (req, res) => {
      let body = ''
      for await (const chunk of req) body += chunk.toString()
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(body || JSON.stringify({ ok: true }))
    },
  })
}

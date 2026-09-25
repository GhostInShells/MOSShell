import type { Context } from '@deepseek-ai/cordis'
import { defineTool } from '@deepseek-ai/dsh-tools'
import { writeFileSync } from 'node:fs'

export const name = 'plugin-api-session-event'
export const inject = ['tools', 'webServer']

// 验证完整反向调用链路:
// 1. 注册特殊名称 tool (moss_shell_observe).
// 2. agent 调用它 → tool/call 事件 append → session/event → /api/events.mux WS 下行.
// 3. python 收到 tool/call → 调 http 回调 /plugin-api/callback → touch 文件.
//
// 零第三方依赖: node:fs + ctx.tools + ctx.webServer (都走 dsh 注入/内置).

const CALLBACK_MARKER = '/Users/BrightRed/Develop/github.com/GhostInShells/MOSShell/.ai_partners/features/workstreams/2026/08/dsh-fusion/research/skills/plugin-api-session-event/callback.marker'

export function apply(ctx: Context) {
  // 1. 注册特殊测试 tool
  ctx.tools.register(defineTool({
    name: 'moss_shell_observe',
    description: 'Test tool for the plugin-api reverse-call experiment. Call it with a short note.',
    parameters: {
      note: { type: 'string', required: true, description: 'A short note to echo back.' },
    },
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: {
          echo: { type: 'string', required: true },
        },
      },
      render: (_args, value) => [{ type: 'text', text: `observed: ${value.echo}` }],
    },
    async execute(args) {
      // tool 被 agent 调用时, tool/call 事件已 append 到 session,
      // 通过 session/event 广播到 python 端.
      return { echo: args.note }
    },
  }))

  // 2. 回调 http 接口: python 收到 tool/call 后调它, touch 文件作为证据.
  ctx.webServer.register({
    kind: 'prefix',
    path: '/plugin-api/callback',
    handler: async (req, res) => {
      let body = ''
      for await (const chunk of req) body += chunk.toString()
      writeFileSync(CALLBACK_MARKER, body || 'callback')
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ ok: true, received: body }))
    },
  })
}

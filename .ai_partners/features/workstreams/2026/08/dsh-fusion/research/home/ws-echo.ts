/**
 * ws-echo — 最小 ws endpoint plugin, 验证 dsh plugin 与外部进程双向通讯.
 *
 * 复用 dsh 的 web 网关 (webserver): 通过 ctx.webServer.registerUpgrade 注册
 * 一个 /echo upgrade route, 不自己起独立 WebSocketServer.
 *
 * 依赖注记: `ws` 在 dsh 的 .pnpm 里未 hoist, 用相对路径指到它的 ESM 入口 wrapper.mjs.
 */
import { WebSocketServer } from '../source/deepseek-harness/node_modules/.pnpm/ws@8.21.0/node_modules/ws/wrapper.mjs'

export const name = 'ws-echo'
export const inject = ['webServer']

export function apply(ctx: any): void {
  const wss = new WebSocketServer({ noServer: true })

  ctx.webServer.registerUpgrade({
    path: '/echo',
    handler: (req: any, socket: any, head: any) => {
      wss.handleUpgrade(req, socket, head, (ws: any) => {
        ws.on('message', (data: any) => {
          ws.send(data.toString())
        })
      })
    },
  })

  ctx.effect(() => () => {
    wss.close()
  })
}

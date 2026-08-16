import type { Context } from '@deepseek-ai/cordis'
import { WebSocketServer } from 'ws'

export const name = 'ws-pingpong'
export const inject = ['webServer']

export function apply(ctx: any) {
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

// mock-llm-frame — 验证「mock LlmAdapter 确定性产出 tool-use」的机制.
//
// 命题 (来自 Dolores thinking transaction 设计讨论, 2026-08-27):
//   替代合成 session 事件, 用一个 ego 专属的 mock LLM provider, 在特殊 turn 里
//   让模型"产出" frame tool-call → agent-loop 原生暂停 dispatch → plugin 的 frame
//   tool Consumer resolve 帧 → tool/result 落 session → 下一个 step 切回真模型.
//
// 本 skill 零真实 LLM, 两个 mock provider 路由同一实例:
//   moss-frame → 产 moss_frame tool-call (BlockAssembler delta-only 宽容, 无需 block-start/end)
//   moss-real  → 产固定文本 (模拟"切回真模型后继续")
//
// 三个断言目标:
//   ① mock stream → BlockAssembler → executeToolCalls 可调度 (tool/call 出现).
//   ② frame tool resolve → tool/result 落 session (带 surfaceOp + sourceEventSeqs).
//   ③ agent/request 路由切换: step1 provider=moss-frame, 之后 provider=moss-real.
//
// 依赖: 全走 dsh 注入服务 (agents/llm/tools/webServer), 零第三方包.

import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import type { Context } from '@deepseek-ai/cordis'
import { LlmAdapter, createUserMessage } from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, StreamChunk } from '@deepseek-ai/dsh-llm'
import { defineTool } from '@deepseek-ai/dsh-tools'

export const name = 'mock-llm-frame'
export const inject = ['agents', 'llm', 'tools', 'webServer']

// agent cwd — verify.py 启动前创建 (home/workspace).
const WORKSPACE_DIR = join(process.env.DSH_HOME ?? process.cwd(), 'workspace')

// 每次 mock 产出一个新 callId, 供 tool/result 成对引用.
let frameCounter = 0

class FrameMockAdapter extends LlmAdapter {
  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    if (options.provider === 'moss-frame') {
      const id = `frame-${++frameCounter}`
      // delta-only: BlockAssembler.ensure() 建 partial, assemble() 从累计 delta 组装 block.
      yield { type: 'tool-call-delta', index: 0, id, name: 'moss_frame', argumentsDelta: '{}' }
    } else {
      yield { type: 'text-delta', index: 0, text: '[mock text] frame delivered; continuing with the real intent.' }
    }
    yield { type: 'finish', reason: { kind: 'stop' } }
  }
}

export function apply(ctx: Context) {
  // ① mock adapter: 两条路由同一实例, stream 按 options.provider 分叉.
  ctx.llm.registerAdapter(['moss-frame', 'moss-real'], new FrameMockAdapter())

  // ③ agent/request 路由切换: step1 → moss-frame (产 frame tool-call), 之后 → moss-real.
  //    短路 next(): 实验全 mock 驱动, 不依赖默认 provider/model. payload 含 {agent, turn, step, signal}.
  ctx.on('agent/request', async ({ step }, next) => {
    return step === 1
      ? { provider: 'moss-frame', model: 'frame' }
      : { provider: 'moss-real', model: 'text' }
  })

  // ② frame tool: plugin 注册, result 即 thinking 帧 (实验用占位帧; 产品形态 = 挂起等 MOSS RPC 供帧).
  ctx.tools.register(defineTool({
    name: 'moss_frame',
    description: 'Thinking frame delivery: carries the current moment frame as its result.',
    parameters: {},
    output: {
      schema: {
        type: 'object',
        additionalProperties: false,
        properties: { frame: { type: 'string', required: true } },
      },
      render: (_args, value) => [{ type: 'text', text: value.frame }],
    },
    async execute() {
      return { frame: '[frame] thinking moment delivered via tool-result.' }
    },
  }))

  // 活 agent (惰性创建) + 触发/读日志 RPC.
  let agent: { id: string; session: { events: Array<{ type: string; seq: number; data: unknown }> }; steer: (m: unknown) => void } | undefined
  let sessionId: string | undefined

  ctx.webServer.register({
    kind: 'exact',
    path: '/plugin-api/frame-trigger',
    handler: async (req, res) => {
      if (agent === undefined) {
        sessionId = randomUUID()
        const handle = await ctx.agents.create({
          sessionId,
          meta: { cwd: WORKSPACE_DIR },
          agentOptions: { provider: 'moss-real', model: 'text' },
        })
        agent = handle.agent
      }
      // steer 唤醒一个 turn; 帧经 moss_frame tool-result 送达, steer 内容只作唤醒信号.
      agent.steer(createUserMessage({
        content: [{ type: 'text', text: 'frame-wake' }],
        source: { kind: 'user' },
      }))
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ sessionId }))
    },
  })

  ctx.webServer.register({
    kind: 'exact',
    path: '/plugin-api/frame-log',
    handler: async (req, res) => {
      const events = agent === undefined ? [] : agent.session.events.map((e) => ({
        type: e.type,
        seq: e.seq,
        data: e.data,
      }))
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ events }))
    },
  })
}

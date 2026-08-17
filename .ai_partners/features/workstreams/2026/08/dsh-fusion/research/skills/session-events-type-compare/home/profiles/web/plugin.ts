import type { Context } from '@deepseek-ai/cordis'

// 类型文档注解 — Node 24 strip-types 会擦除 `import type`, 无运行时依赖.
// 形状对照 dsh 源码:
//   SessionEventMap:  packages/core/session/src/types.ts
//   UserMessage/AssistantMessage/ToolResultMessage: packages/llm/llm/src/message.ts
//   ContentBlock/StreamChunk/TokenUsage/...:        packages/llm/llm/src/types.ts
import type {
  SessionEvent,
  SessionEventType,
} from '@deepseek-ai/dsh-session'

export const name = 'session-events-type-compare'
export const inject = ['webServer']

// 本 skill 的 plugin 只做一件事: 暴露一个 HTTP rpc, 返回 TS 侧构造的
// session event 各类型 mock 实例 (ground truth). Python verify.py 拉取后,
// 喂给 ghoshell_moss.agents.deepseek_harness.session_events 的强类型模型,
// 验证类型转换正确、且与 TS 类型保持同步.
//
// 零第三方依赖: 只用 ctx.webServer (dsh 注入).

// 固定信封字段: seq 单调递增, time 用固定 epoch 毫秒 (便于 round-trip 断言).
const T = 1723800000000

// 逐条构造, 覆盖全部 13 种事件类型 + 嵌套判别联合的多样变体.
// surface 事件 (user/message, assistant/message, tool/result) 携带
// surfaceOp + sourceEventSeqs; 其余事件不含.
const MOCKS: SessionEvent[] = [
  // 1-2. turn 生命周期
  { type: 'turn/start', seq: 1, time: T, data: { turn: 0 } },
  { type: 'turn/end', seq: 2, time: T + 1, data: { turn: 0, reason: { kind: 'completed' } } },
  // 3. turn/end 的 aborted + user cancel cause 变体
  {
    type: 'turn/end', seq: 3, time: T + 2,
    data: { turn: 1, reason: { kind: 'aborted', reason: { kind: 'user' } } },
  },

  // 4-5. step 生命周期
  { type: 'step/start', seq: 4, time: T + 3, data: { turn: 0, step: 0 } },
  { type: 'step/end', seq: 5, time: T + 4, data: { turn: 0, step: 0 } },

  // 6. user/message: 人类 prompt, source=user, text 块
  {
    type: 'user/message', seq: 6, time: T + 5,
    data: {
      id: 'u1', role: 'user',
      content: [{ type: 'text', text: 'hello' }],
      source: { kind: 'user' },
    },
    surfaceOp: 'append',
  },
  // 7. user/message: plugin 注入上下文 (form=notice)
  {
    type: 'user/message', seq: 7, time: T + 6,
    data: {
      id: 'u2', role: 'user',
      content: [{ type: 'text', text: 'x.txt modified' }],
      source: { kind: 'plugin', plugin: 'file-change', form: 'notice', summary: 'file changed' },
    },
    surfaceOp: 'append',
    sourceEventSeqs: [],
  },
  // 8. user/message: image 内容块
  {
    type: 'user/message', seq: 8, time: T + 7,
    data: {
      id: 'u3', role: 'user',
      content: [{ type: 'image', attachment: { id: 'img-1', mime: 'image/png' } }],
      source: { kind: 'user' },
    },
    surfaceOp: 'append',
  },
  // 9. user/message: tool-result 内容块 (source=tool)
  {
    type: 'user/message', seq: 9, time: T + 8,
    data: {
      id: 'u4', role: 'user',
      content: [{
        type: 'tool-result', toolCallId: 'c9',
        content: [{ type: 'text', text: 'ok' }], isError: false,
      }],
      source: { kind: 'tool', callId: 'c9' },
    },
    surfaceOp: 'append',
  },

  // 10-12. assistant/chunk: text-delta / finish / block-end(tool-call 块)
  {
    type: 'assistant/chunk', seq: 10, time: T + 9,
    data: { turn: 0, step: 0, chunk: { type: 'text-delta', index: 0, text: 'hi' } },
  },
  {
    type: 'assistant/chunk', seq: 11, time: T + 10,
    data: { turn: 0, step: 0, chunk: { type: 'finish', reason: { kind: 'stop' } } },
  },
  {
    type: 'assistant/chunk', seq: 12, time: T + 11,
    data: {
      turn: 0, step: 0,
      chunk: {
        type: 'block-end', index: 0,
        block: { type: 'tool-call', id: 'tc1', name: 'tool_a', arguments: '{"x":1}' },
      },
    },
  },

  // 13. assistant/message: 组装完成消息 + usage
  {
    type: 'assistant/message', seq: 13, time: T + 12,
    data: {
      turn: 0, step: 0,
      message: {
        id: 'a1', role: 'assistant',
        content: [{ type: 'text', text: 'answer' }],
        source: { kind: 'model', provider: 'deepseek', model: 'deepseek-r1' },
      },
      usage: { inputTokens: 10, outputTokens: 5 },
    },
    surfaceOp: 'append',
    sourceEventSeqs: [10],
  },

  // 14. tool/call 全字段
  {
    type: 'tool/call', seq: 14, time: T + 13,
    data: { turn: 0, step: 0, callId: 'c14', name: 'tool_b', arguments: '{"y":2}' },
  },
  // 15. tool/result: message + error + 工具私有 meta (data.meta)
  {
    type: 'tool/result', seq: 15, time: T + 14,
    data: {
      turn: 0, step: 1,
      message: {
        id: 'tr1', role: 'user',
        content: [{
          type: 'tool-result', toolCallId: 'c14',
          content: [{ type: 'text', text: 'res' }], isError: false,
        }],
        source: { kind: 'tool', callId: 'c14' },
      },
      error: { name: 'E', code: '1' },
      meta: { diff: '+1' },
    },
    surfaceOp: 'append',
    sourceEventSeqs: [14],
  },

  // 16. todo/write 整表快照
  {
    type: 'todo/write', seq: 16, time: T + 15,
    data: {
      todos: [
        { content: 'task a', status: 'pending' },
        { content: 'task b', status: 'in_progress' },
        { content: 'task c', status: 'completed' },
      ],
    },
  },

  // 17. request/header: 完整 EpochHeader
  {
    type: 'request/header', seq: 17, time: T + 16,
    data: {
      header: {
        config: {
          provider: 'deepseek', model: 'deepseek-r1',
          reasoningEffort: 'low', temperature: 0.7, maxTokens: 100,
          stop: ['stop'],
        },
        adapterDefaults: { maxTokens: true },
        system: 'You are a helpful assistant.',
        tools: [{ name: 'tool_b', description: 'A tool', parameters: { type: 'object' } }],
      },
      reason: 'initial',
    },
  },

  // 18. request/context 路由元信息
  {
    type: 'request/context', seq: 18, time: T + 17,
    data: { provider: 'deepseek', model: 'deepseek-r1', contextWindow: 64000 },
  },

  // 19. session/end-seed: 空载荷
  { type: 'session/end-seed', seq: 19, time: T + 18, data: {} },
]

// 断言: 每条 mock 的 type 属于已知集合 (作者侧自检, 防手误).
const KNOWN_TYPES: SessionEventType[] = [
  'turn/start', 'turn/end', 'step/start', 'step/end',
  'user/message', 'assistant/chunk', 'assistant/message',
  'tool/call', 'tool/result', 'todo/write',
  'request/header', 'request/context', 'session/end-seed',
]
for (const m of MOCKS) {
  if (!KNOWN_TYPES.includes(m.type)) {
    throw new Error(`mock 事件类型不在已知集合内: ${String(m.type)}`)
  }
}

export function apply(ctx: Context) {
  // HTTP rpc: 返回 TS 侧构造的全部 mock session events.
  ctx.webServer.register({
    kind: 'prefix',
    path: '/plugin-api/session-events-mock',
    handler: async (_req, res) => {
      res.writeHead(200, { 'content-type': 'application/json' })
      res.end(JSON.stringify({ events: MOCKS }))
    },
  })
}

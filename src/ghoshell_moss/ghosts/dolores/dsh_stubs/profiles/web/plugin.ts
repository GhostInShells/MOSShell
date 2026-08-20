import type { Context } from '@deepseek-ai/cordis'

export const name = 'dolores-ghost-plugin'
export const inject: string[] = []

// 空实现 — 占位. 后续接入内核特权桥 (append assistant / 构造 seed / 动态 prompt),
// 经 ctx.webServer.register 挂 HTTP 路由, 与 apiproxy 平级.
export function apply(_ctx: Context) {}

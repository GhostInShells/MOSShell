import type { Context } from '@deepseek-ai/cordis'
import { writeFileSync } from 'node:fs'

export const name = 'hello-plugin'

export function apply(ctx: Context) {
  writeFileSync(
    '/Users/BrightRed/Develop/github.com/GhostInShells/MOSShell/.ai_partners/features/workstreams/2026/08/dsh-fusion/research/home/plugin-loaded.marker',
    'loaded',
  )
}

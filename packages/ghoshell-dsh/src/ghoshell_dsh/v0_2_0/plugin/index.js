/**
 * The ghoshell patch, first version: a liveness route.
 *
 * This bundle exists so that "install our plugin into a dsh home" can be implemented and
 * verified before the capabilities that ride on it (anchor call, session read). It proves the
 * three things that matter about the mechanism — the bundle resolves by name, the patch layer
 * loads it, and a route answers — with nothing that can go wrong for a second reason.
 *
 * Plain JS on purpose: no build step, so installing is a file copy. `Remote` is a stage-3
 * decorator and this file has no decorator syntax, so the marking is done by hand with a fake
 * context — the same trick the playground probe used, and the reason this file must not be
 * refactored into something "cleaner".
 */
import { Remote, TypertRemoteService } from '@deepseek-ai/dsh-typert-protocol'

function markRemote(instance, method) {
  const context = {
    kind: 'method',
    name: method,
    static: false,
    private: false,
    addInitializer(initializer) {
      initializer.call(instance)
    },
  }
  Remote(undefined, context)
}

export class GhostShellService extends TypertRemoteService {
  constructor(ctx) {
    super(ctx, 'ghoshell')
    markRemote(this, 'ping')
  }

  async ping() {
    return { ok: true, from: '@local/ghoshell-dsh', at: new Date().toISOString() }
  }
}

export const name = 'ghoshell'

export function apply(ctx) {
  ctx.plugin(GhostShellService)
}

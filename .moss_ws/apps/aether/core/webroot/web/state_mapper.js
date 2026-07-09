// state_mapper.js — 兼容主状态 + 全双工 activity layers

export const STATES = {
    IDLE: 0,
    LISTEN: 1,
    THINK: 2,
    SPEAK: 3,
    INTERRUPT: 4,
};

export const STATE_NAMES = ['idle', 'listen', 'think', 'speak', 'interrupt'];

// 各状态目标视觉参数（CPU 端插值目标）
export const STATE_TARGETS = {
    [STATES.IDLE]:      { radius: 0.50, amp: 0.05, freq: 1.20, speed: 0.45, color: [0.30, 0.55, 1.10], glow: [0.20, 0.45, 1.00] },
    [STATES.LISTEN]:    { radius: 0.42, amp: 0.03, freq: 1.50, speed: 0.85, color: [0.20, 0.80, 1.20], glow: [0.10, 0.65, 1.10] },
    [STATES.THINK]:     { radius: 0.55, amp: 0.13, freq: 2.60, speed: 1.80, color: [0.65, 0.30, 1.10], glow: [0.50, 0.20, 1.00] },
    [STATES.SPEAK]:     { radius: 0.58, amp: 0.10, freq: 2.20, speed: 2.00, color: [1.10, 0.65, 0.25], glow: [1.00, 0.45, 0.10] },
    [STATES.INTERRUPT]: { radius: 0.40, amp: 0.02, freq: 0.50, speed: 0.00, color: [0.85, 0.95, 1.20], glow: [0.70, 0.85, 1.20] },
};

export const LAYER_NAMES = ['mic', 'listen', 'queue', 'think', 'speak', 'interrupt'];

export function normalizeLayers(layers = {}) {
    return {
        mic: Boolean(layers.mic),
        listen: Boolean(layers.listen),
        queue: Boolean(layers.queue),
        think: Boolean(layers.think),
        speak: Boolean(layers.speak),
        interrupt: Boolean(layers.interrupt),
    };
}

export function primaryStateFromLayers(layers = {}) {
    const l = normalizeLayers(layers);
    if (l.interrupt) return STATES.INTERRUPT;
    if (l.speak) return STATES.SPEAK;
    if (l.think) return STATES.THINK;
    if (l.listen) return STATES.LISTEN;
    return STATES.IDLE;
}

export function blendTargetsForLayers(layers = {}) {
    const l = normalizeLayers(layers);
    if (l.interrupt) return { ...cloneTarget(STATE_TARGETS[STATES.INTERRUPT]), activeCount: 1 };

    const weighted = [
        [STATES.IDLE, (!l.listen && !l.think && !l.speak) ? 1.0 : 0.28],
        [STATES.LISTEN, l.listen ? 0.90 : (l.mic ? 0.30 : 0)],
        [STATES.THINK, l.think ? 1.00 : (l.queue ? 0.55 : 0)],
        [STATES.SPEAK, l.speak ? 1.10 : 0],
    ];
    let total = 0;
    const out = { radius: 0, amp: 0, freq: 0, speed: 0, color: [0, 0, 0], glow: [0, 0, 0] };
    let activeCount = 0;
    for (const [state, weight] of weighted) {
        if (weight <= 0) continue;
        if (state !== STATES.IDLE) activeCount += 1;
        total += weight;
        const t = STATE_TARGETS[state];
        out.radius += t.radius * weight;
        out.amp += t.amp * weight;
        out.freq += t.freq * weight;
        out.speed += t.speed * weight;
        for (let i = 0; i < 3; i++) {
            out.color[i] += t.color[i] * weight;
            out.glow[i] += t.glow[i] * weight;
        }
    }
    if (total <= 0) return { ...cloneTarget(STATE_TARGETS[STATES.IDLE]), activeCount: 0 };
    out.radius /= total;
    out.amp /= total;
    out.freq /= total;
    out.speed /= total;
    for (let i = 0; i < 3; i++) {
        out.color[i] /= total;
        out.glow[i] /= total;
    }
    out.activeCount = activeCount;
    return out;
}

function cloneTarget(target) {
    return JSON.parse(JSON.stringify(target));
}

// 状态切换的"急刹度"：插值速率，越大切换越快。INTERRUPT 最大（急刹），其他柔和
export const STATE_SWITCH_RATE = {
    [STATES.IDLE]: 4.0,
    [STATES.LISTEN]: 5.0,
    [STATES.THINK]: 4.5,
    [STATES.SPEAK]: 4.0,
    [STATES.INTERRUPT]: 6.0, // 急刹再放慢：让扩张→收缩的节奏清晰可见
};

export class StateMapper {
    /**
     * @param {object} opts
     * @param {(state:number, event:object)=>void} opts.onState 状态切换回调
     * @param {number} opts.minHoldMs 最小保持时间（去抖），默认 250ms
     * @param {number} opts.interruptHoldMs INTERRUPT 最小保持时间（让爆点视觉充分展现）
     */
    constructor({ onState, minHoldMs = 250, interruptHoldMs = 1800 } = {}) {
        this.state = STATES.IDLE;
        this.layers = normalizeLayers();
        this._layerKey = JSON.stringify(this.layers);
        this.onState = onState;
        this.minHoldMs = minHoldMs;
        this.interruptHoldMs = interruptHoldMs;
        this.stateEnterTs = performance.now();
        this.log = [];
    }

    now() { return performance.now(); }
    heldMs() { return this.now() - this.stateEnterTs; }

    /**
     * 喂事件。事件类型：
     *  - SPEECH_STARTED  旧契约：检测到开口
     *  - SPEECH_FINAL    ASR 出完整句（慢线）
     *  - TTS_START       TTS 开始播放
     *  - TTS_END         TTS 播放结束
     *  - FORCE_STATE     演示用：强制切状态
     */
    event(e) {
        const now = this.now();
        const held = now - this.stateEnterTs;
        let next = this.state;

        switch (e.type) {
            case 'SPEECH_STARTED':
                // 旧契约兼容：新路径使用 setLayers()，普通 VAD 不再伪装成 interrupt。
                if (this.state === STATES.SPEAK) next = STATES.INTERRUPT;
                else if (this.state === STATES.IDLE) next = STATES.LISTEN;
                else if (this.state === STATES.THINK) next = STATES.INTERRUPT; // 思考中被打断也急刹
                break;
            case 'SPEECH_FINAL':
                if (this.state === STATES.LISTEN) next = STATES.THINK;
                break;
            case 'TTS_START':
                if (this.state === STATES.THINK || this.state === STATES.INTERRUPT) next = STATES.SPEAK;
                break;
            case 'TTS_END':
                if (this.state === STATES.SPEAK) next = STATES.IDLE;
                break;
            case 'FORCE_STATE':
                next = e.state;
                break;
        }

        if (next === this.state) return;

        // 去抖：INTERRUPT 是爆点，永远抢跑（绕过去抖）；
        // FORCE_STATE 是手动演示，也绕过；
        // 其他状态切换在最小保持时间内忽略。
        const isUrgent = next === STATES.INTERRUPT || e.type === 'FORCE_STATE';
        if (!isUrgent && held < this.minHoldMs) return;

        // INTERRUPT 自身要保持足够久，避免视觉还没展开就退出
        if (this.state === STATES.INTERRUPT && held < this.interruptHoldMs && e.type !== 'SPEECH_STARTED' && e.type !== 'FORCE_STATE') {
            return;
        }

        const prev = this.state;
        this.state = next;
        this.layers = next === STATES.IDLE ? normalizeLayers() : normalizeLayers({ [STATE_NAMES[next]]: true });
        this._layerKey = JSON.stringify(this.layers);
        this.stateEnterTs = now;
        this.log.push({ ts: now, from: prev, to: next, event: e.type });
        if (this.log.length > 200) this.log.shift();
        this.onState?.(next, { from: prev, event: e.type, latencyMs: e.vadLatencyMs, layers: this.layers });
    }

    setLayers(layers, meta = {}) {
        const normalized = normalizeLayers(layers);
        const key = JSON.stringify(normalized);
        const next = primaryStateFromLayers(normalized);
        const prev = this.state;
        if (next === this.state && key === this._layerKey) return;
        this.layers = normalized;
        this._layerKey = key;
        this.state = next;
        this.stateEnterTs = this.now();
        this.onState?.(next, { from: prev, event: meta.event || 'LAYERS', layers: normalized, text: meta.text });
    }
}

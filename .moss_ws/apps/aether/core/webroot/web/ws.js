// ws.js — 状态接收层
// 连 aether/core 后端 ws://localhost:8765/ws（MOSS ghost 推状态 JSON）
// 后端是 aether/core app：订阅 SpeechTopic/AudioRuntimeTopic，推 {state, layers, intensity, ts}
//
// 状态契约（前后端唯一接口）：
//   { "state": "speak", "layers": {"listen": true, "think": false, "speak": true, "interrupt": false}, "intensity": 0.7, "ts": 1782402722.10 }
// 前端→后端：
//   { "type": "listen", "running": true }   // legacy only; main listen is backend ASR-driven
//   { "type": "interrupt" }                 // 明确急停请求
//   { "type": "asr_control", "mode": "manual", "enabled": true }

export class StateBridge {
    /**
     * @param {object} opts
     * @param {(state:string)=>void} opts.onState
     * @param {(intensity:number)=>void} opts.onIntensity
     * @param {(layers:object, msg:object)=>void} opts.onLayers
     * @param {()=>void} opts.onConnect
     * @param {()=>void} opts.onDisconnect
     */
    constructor({ onState, onIntensity, onLayers, onConnect, onDisconnect } = {}) {
        this.onState = onState;
        this.onIntensity = onIntensity;
        this.onLayers = onLayers;
        this.onConnect = onConnect;
        this.onDisconnect = onDisconnect;
        this.connected = false;
        this.mode = 'local'; // 'ws' | 'local'
        this.ws = null;
        this._stopped = false;
        this._url = 'ws://localhost:8765/ws';
        this._lastAsrControl = { mode: 'continuous', enabled: true };
        this._hasAsrControl = false;
    }

    /** 尝试连真后端；失败返回 false。带自动重连。 */
    async connect(url = 'ws://localhost:8765/ws') {
        this._stopped = false;
        this._url = url;
        try {
            const ws = new WebSocket(url);
            this.ws = ws;
            await new Promise((res, rej) => {
                ws.onopen = res;
                ws.onerror = rej;
                setTimeout(() => rej(new Error('timeout')), 2000);
            });
            ws.onmessage = (ev) => {
                try { this._handle(JSON.parse(ev.data)); } catch (e) {}
            };
            ws.onclose = () => {
                this.connected = false;
                this.mode = 'local';
                this.onDisconnect?.();
                if (!this._stopped) {
                    // 3s 后自动重连
                    setTimeout(() => { if (!this._stopped) this.connect(url); }, 3000);
                }
            };
            this.connected = true;
            this.mode = 'ws';
            this.onConnect?.();
            if (this._hasAsrControl) this._sendAsrControlNow();
            return true;
        } catch (e) {
            this.connected = false;
            this.mode = 'local';
            if (!this._stopped) {
                setTimeout(() => { if (!this._stopped) this.connect(url); }, 3000);
            }
            return false;
        }
    }

    /** 明确急停：按钮或已确认的后端 barge-in 路径。普通 VAD 不调用它。 */
    sendInterrupt() {
        if (this.ws && this.connected) {
            try { this.ws.send(JSON.stringify({ type: 'interrupt' })); } catch (e) {}
        }
    }

    sendListen(running, extra = {}) {
        if (this.ws && this.connected) {
            try { this.ws.send(JSON.stringify({ type: 'listen', running: Boolean(running), ...extra })); } catch (e) {}
        }
    }

    sendAsrControl(mode = 'continuous', enabled = true) {
        this._lastAsrControl = { mode, enabled: Boolean(enabled) };
        this._hasAsrControl = true;
        this._sendAsrControlNow();
    }

    _sendAsrControlNow() {
        if (this.ws && this.connected) {
            try { this.ws.send(JSON.stringify({ type: 'asr_control', ...this._lastAsrControl })); } catch (e) {}
        }
    }

    /** 前端请求重置上下文 → 后端清空 speech_win + ghost 历史 */
    sendReset() {
        if (this.ws && this.connected) {
            try { this.ws.send(JSON.stringify({ type: 'reset' })); } catch (e) {}
        }
    }

    _handle(msg) {
        if (!msg) return;
        if (msg.layers && typeof msg.layers === 'object') this.onLayers?.(msg.layers, msg);
        else if (typeof msg.state === 'string') this.onState?.(msg.state);
        if (typeof msg.intensity === 'number') this.onIntensity?.(msg.intensity);
    }

    /** 本地模式：直接由前端喂状态消息（演示用） */
    feedLocal(msg) {
        if (this.mode === 'ws') return; // 已连真后端，忽略本地
        this._handle(msg);
    }
}

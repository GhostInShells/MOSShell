// vad.js — 浏览器麦克风 VAD 快线（对应技术文档里的 vad.py）
// 用 RMS 能量阈值 + 最小持续时间门限，检测到开口第一帧就触发 SPEECH_STARTED
// 目标延迟 < 50ms（实际取决于 AudioContext 块大小，这里 fftSize=512 @16kHz ≈ 32ms）
//
// 回声处理：依赖 WebRTC echoCancellation 作为前置过滤，让 VAD"听不到"TTS 外放回声。
// 即便如此 AEC 效果不稳定，因此在 SPEAK 状态下启用 speakMode：
//   - 阈值放大 speakThresholdScale 倍（默认 2.0）
//   - 最小持续时间放大 speakMinDurScale 倍（默认 1.5）
// 作为安全边际，避免残留回声触发自打断。

export class VAD {
    /**
     * @param {object} opts
     * @param {()=>void} opts.onSpeechStarted 开口瞬间回调（爆点触发源）
     * @param {()=>void} [opts.onSpeechEnded] 静默后回调
     * @param {number} [opts.threshold] RMS 阈值 0~1
     * @param {number} [opts.minDurMs] 持续多长时间才算"开口"（防噪声毛刺）
     * @param {number} [opts.endSilenceMs] 多少静默算"说完了"
     * @param {(level:number)=>void} [opts.onLevel] 实时音量回调（用于 speak 脉动近似）
     * @param {number} [opts.speakThresholdScale] SPEAK 状态下阈值放大倍数（默认 2.0）
     * @param {number} [opts.speakMinDurScale] SPEAK 状态下 minDurMs 放大倍数（默认 1.5）
     * @param {(settings:object)=>void} [opts.onEchoCancelReport] 上报 getUserMedia 实际生效的 AEC 设置
     */
    constructor({
        onSpeechStarted,
        onSpeechEnded,
        threshold = 0.012,
        minDurMs = 40,
        endSilenceMs = 520,
        onLevel,
        speakThresholdScale = 2.0,
        speakMinDurScale = 1.5,
        onEchoCancelReport,
    } = {}) {
        this.onSpeechStarted = onSpeechStarted;
        this.onSpeechEnded = onSpeechEnded;
        this.onLevel = onLevel;
        this.threshold = threshold;
        this.minDurMs = minDurMs;
        this.endSilenceMs = endSilenceMs;
        this.speakThresholdScale = speakThresholdScale;
        this.speakMinDurScale = speakMinDurScale;
        this.onEchoCancelReport = onEchoCancelReport;

        this.running = false;
        this.ctx = null;
        this.stream = null;
        this.analyser = null;
        this.buf = null;
        this.raf = 0;

        this.speaking = false;
        this.activeSince = 0;   // 当前活跃起始时间
        this.silentSince = 0;   // 静默起始时间
        this.lastRms = 0;

        // SPEAK 状态开关：开启后提高阈值与最小持续时间，作为 AEC 残留回声的安全边际
        this.speakMode = false;
    }

    /** SPEAK 状态下开启，提高阈值；离开 SPEAK 时关闭，恢复灵敏度 */
    setSpeakMode(on) {
        this.speakMode = !!on;
    }

    async start() {
        if (this.running) return;
        // 强化 echoCancellation：用 ideal 表达强烈偏好，echoCancellationType:'system'
        // 优先使用系统级 AEC（通常比浏览器内置 AEC 效果更好）。
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: {
                echoCancellation: { ideal: true },
                echoCancellationType: 'system',
                noiseSuppression: { ideal: true },
                autoGainControl: { ideal: true },
                channelCount: { ideal: 1 },
            },
        });
        this.stream = stream;
        // 上报实际生效的 AEC 设置（浏览器可能降级，需可见）
        try {
            const track = stream.getAudioTracks()[0];
            const settings = track.getSettings();
            this.onEchoCancelReport?.(settings);
        } catch (e) {}
        // 16kHz 采样率贴近 Silero VAD 场景
        const Ctx = window.AudioContext || window.webkitAudioContext;
        this.ctx = new Ctx({ sampleRate: 16000 });
        const src = this.ctx.createMediaStreamSource(stream);
        this.analyser = this.ctx.createAnalyser();
        this.analyser.fftSize = 512; // 32ms @16kHz
        this.analyser.smoothingTimeConstant = 0.3;
        src.connect(this.analyser);
        this.buf = new Uint8Array(this.analyser.fftSize);

        this.running = true;
        this.loop();
    }

    stop() {
        this.running = false;
        if (this.raf) cancelAnimationFrame(this.raf);
        this.raf = 0;
        if (this.stream) this.stream.getTracks().forEach(t => t.stop());
        if (this.ctx) this.ctx.close();
        this.stream = null;
        this.ctx = null;
        this.analyser = null;
        this.speaking = false;
    }

    loop = () => {
        if (!this.running) return;
        this.raf = requestAnimationFrame(this.loop);
        if (!this.analyser) return;

        this.analyser.getByteTimeDomainData(this.buf);
        let sum = 0;
        for (let i = 0; i < this.buf.length; i++) {
            const v = (this.buf[i] - 128) / 128;
            sum += v * v;
        }
        const rms = Math.sqrt(sum / this.buf.length);
        // 平滑一下，避免抖动
        this.lastRms = this.lastRms * 0.6 + rms * 0.4;
        this.onLevel?.(this.lastRms);

        const now = performance.now();
        // SPEAK 状态下使用放大阈值与持续时间，作为 AEC 残留回声的安全边际
        const effectiveThreshold = this.speakMode
            ? this.threshold * this.speakThresholdScale
            : this.threshold;
        const effectiveMinDurMs = this.speakMode
            ? this.minDurMs * this.speakMinDurScale
            : this.minDurMs;
        const above = this.lastRms > effectiveThreshold;

        if (above) {
            if (this.activeSince === 0) this.activeSince = now;
            // 持续 effectiveMinDurMs 才算"开口" → 触发 SPEECH_STARTED
            if (!this.speaking && (now - this.activeSince) >= effectiveMinDurMs) {
                this.speaking = true;
                const vadLatencyMs = now - this.activeSince;
                this.onSpeechStarted?.({ vadLatencyMs, speakMode: this.speakMode });
                this.silentSince = 0;
            }
            this.silentSince = 0;
        } else {
            if (this.speaking) {
                if (this.silentSince === 0) this.silentSince = now;
                if ((now - this.silentSince) >= this.endSilenceMs) {
                    this.speaking = false;
                    this.activeSince = 0;
                    this.silentSince = 0;
                    this.onSpeechEnded?.();
                }
            } else {
                this.activeSince = 0;
            }
        }
    }
}

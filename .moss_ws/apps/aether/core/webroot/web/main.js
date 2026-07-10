// main.js — 演示入口
// 串联 Scene / StateMapper / VAD / StateBridge，绑定 UI
// 完整对应技术文档：状态字符串驱动 + VAD 快线急刹 + ASR 慢线（演示用模拟）

import { Scene } from './scene.js';
import { normalizeLayers, StateMapper, STATES, STATE_NAMES } from './state_mapper.js';
import { VAD } from './vad.js';
import { StateBridge } from './ws.js';

const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

// ---------- 全局实例 ----------
let scene, mapper, vad, bridge;
let demoTimer = null;
let demoSpeakPulseRaf = null;
let manualIntensity = 0.0;
let useAutoIntensity = true; // speak 时自动脉动
let localLayers = normalizeLayers();
let backendLayers = normalizeLayers();
let localMicRunning = false;
let listenClearTimer = 0;
let thinkPendingTimer = 0;
let asrMode = 'continuous';
let asrArmed = true;

// ---------- 初始化 ----------
async function boot() {
    scene = new Scene($('#glcanvas'));
    await scene.init();
    scene.start();

    mapper = new StateMapper({
        onState: (state, meta) => onStateChange(state, meta),
        minHoldMs: 250,
        interruptHoldMs: 700,
    });

    bridge = new StateBridge({
        onState: (stateName) => {
            const idx = STATE_NAMES.indexOf(stateName);
            if (idx >= 0) applyState(idx, { event: 'WS_STATE' });
        },
        onLayers: (layers, msg) => {
            applyLayers(layers, { event: 'WS_LAYERS', text: msg.text });
            updateDiagnostics(msg.diagnostics);
            if (msg.interrupt_burst) scene.interruptBurst = Math.max(scene.interruptBurst, msg.interrupt_burst);
            if (msg.text || layers.speak || layers.interrupt) {
                clearThinkPendingTimer();
            }
        },
        onIntensity: (v) => {
            manualIntensity = v;
            useAutoIntensity = false;
        },
        onConnect: () => {
            updateModePill('ws');
            updateAsrControlUI();
        },
        onDisconnect: () => {
            updateModePill('local');
        },
    });

    // 自动连 aether/core 后端（MOSS ghost 推状态）
    const ok = await bridge.connect();
    updateModePill(ok ? 'ws' : 'local');

    bindUI();
    updateAsrControlUI();
    applyState(STATES.IDLE, { event: 'BOOT' });
    log(ok ? '已连后端 ws://localhost:8765/ws · MOSS ghost 在线' : '后端未就绪 · 本地演示模式（自动重连中）');
    log('提示：点「演示模式」看全双工叠层；本地VAD只负责快速视觉反馈，火山ASR收音由右侧按钮控制');
}

// ---------- 状态切换处理 ----------
function onStateChange(state, meta) {
    localLayers = normalizeLayers(meta?.layers || mapper.layers);
    scene.onStateChange(state, meta);
    applyStateUI(state, meta);
    // 同步 VAD 的 speakMode：SPEAK 时提高阈值/duration，避免 AEC 残留回声自打断；
    // 其他状态恢复原灵敏度（idle→listen 仍需低延迟开口检测）。
    if (vad) vad.setSpeakMode(state === STATES.SPEAK);
}

function applyState(state, meta) {
    // 走 state_mapper（带去抖）
    mapper.event({ type: 'FORCE_STATE', state, ...(meta || {}) });
}

function composeVisualLayers(layers = backendLayers) {
    const base = normalizeLayers(layers);
    return normalizeLayers({
        ...base,
        mic: Boolean(base.mic) || localMicRunning,
        listen: Boolean(base.listen) || localMicRunning,
    });
}

function applyLayers(layers, meta = {}) {
    backendLayers = normalizeLayers(layers);
    localLayers = normalizeLayers({
        ...backendLayers,
        // Fast visual listen: local browser VAD means the user has started
        // speaking. Backend ASR still owns SpeechTopic/think; this only keeps
        // Aether responsive before the cloud ASR returns its first partial.
        ...composeVisualLayers(backendLayers),
    });
    mapper.setLayers(localLayers, meta);
}

function applyStateUI(state, meta) {
    const name = STATE_NAMES[state];
    const layers = normalizeLayers(meta?.layers || mapper.layers);
    $$('.state-dot').forEach(d => {
        const s = d.dataset.s;
        const active = s === 'idle' ? name === 'idle' : Boolean(layers[s]);
        d.classList.toggle('active', active);
        d.classList.toggle('primary', s === name);
    });
    $('#current-state-name').textContent = name;
    updateActivityRings(layers, name);
    // 球体中央大字（所有状态都显示）
    const labelEl = $('#state-label');
    if (labelEl) {
        labelEl.textContent = activeLabel(layers, name);
        labelEl.setAttribute('data-state', name);
    }
    $$('.stategraph .edge').forEach(e => {
        const to = e.dataset.to;
        const active = to === 'idle' ? name === 'idle' : Boolean(layers[to]);
        e.classList.toggle('active', active);
    });

    if (state === STATES.INTERRUPT) {
        $('#interrupt-flash').classList.add('on');
        clearTimeout(window._intrFlashT);
        window._intrFlashT = setTimeout(() => {
            $('#interrupt-flash').classList.remove('on');
        }, 1500);
    }

    log(`state → ${activeLabel(layers, name)}  (${meta?.event || ''}${meta?.latencyMs != null ? ' · vad+' + meta.latencyMs.toFixed(0) + 'ms' : ''})`);
}

function activeLabel(layers, fallback) {
    const active = ['listen', 'queue', 'think', 'speak'].filter(k => layers[k]);
    if (layers.interrupt) return 'INTERRUPT';
    if (active.length === 0 && layers.mic) return 'MIC';
    if (active.length === 0) return fallback.toUpperCase();
    return active.map(s => s.toUpperCase()).join(' + ');
}

function updateActivityRings(layers, primary) {
    ['mic', 'listen', 'queue', 'think', 'speak', 'interrupt'].forEach(name => {
        const el = document.querySelector(`.activity-ring[data-layer="${name}"]`);
        if (!el) return;
        el.classList.toggle('active', Boolean(layers[name]));
        el.classList.toggle('primary', primary === name);
    });
    const stack = $('#activity-stack');
    if (stack) stack.setAttribute('data-primary', primary);
}

function updateModePill(mode) {
    const el = $('#mode-pill');
    el.textContent = mode === 'ws' ? 'WS · 已连后端' : '本地演示模式';
    el.className = 'mode-pill ' + mode;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function updateDiagnostics(diag) {
    if (!diag) return;
    const currentEl = $('#asr-current');
    if (currentEl) {
        if (diag.asr_current?.text) {
            currentEl.innerHTML = `
                <div class="asr-item partial">
                    <div class="asr-tag">partial</div>
                    <div class="asr-text">${escapeHtml(diag.asr_current.text)}</div>
                </div>
            `;
        } else {
            currentEl.innerHTML = '<div class="diag-empty">等待 ASR partial</div>';
        }
    }

    const asrEl = $('#asr-diag');
    if (asrEl && Array.isArray(diag.asr_finals)) {
        if (diag.asr_finals.length === 0) {
            asrEl.innerHTML = '<div class="diag-empty">等待 final</div>';
        } else {
            asrEl.innerHTML = diag.asr_finals.slice(-3).reverse().map(item => {
                return `
                    <div class="asr-item final">
                        <div class="asr-tag">final</div>
                        <div class="asr-text">${escapeHtml(item.text)}</div>
                    </div>
                `;
            }).join('');
        }
    }

    const errorEl = $('#asr-error');
    if (errorEl) {
        if (diag.asr_error?.error) {
            const backoff = Number(diag.asr_error.backoff || 0);
            const code = diag.asr_error.code ? ` · code=${escapeHtml(diag.asr_error.code)}` : '';
            const message = diag.asr_error.message ? ` · ${escapeHtml(diag.asr_error.message)}` : '';
            const wait = backoff > 0 ? ` · retry ${backoff.toFixed(0)}s` : '';
            errorEl.hidden = false;
            errorEl.textContent = `ASR ${diag.asr_error.error}${code}${message}${wait}`;
        } else {
            errorEl.hidden = true;
            errorEl.textContent = '';
        }
    }

    if (diag.asr_control) {
        asrMode = diag.asr_control.mode || asrMode;
        asrArmed = asrMode === 'continuous' ? true : Boolean(diag.asr_control.enabled);
        updateAsrControlUI();
    }

    const vpioEl = $('#vpio-diag');
    if (vpioEl && typeof diag.vpio === 'string' && diag.vpio) {
        vpioEl.textContent = diag.vpio;
    }
}

// ---------- 日志 ----------
const logLines = [];
const MAX_LOG = 30;
function log(msg) {
    const ts = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    logLines.push(`[${ts}] ${msg}`);
    if (logLines.length > MAX_LOG) logLines.shift();
    const el = $('#log');
    el.innerHTML = logLines.map((l, i) => {
        const age = logLines.length - 1 - i;
        const op = Math.max(0.25, 1 - age * 0.05);
        return `<div class="line" style="opacity:${op.toFixed(2)}">${l}</div>`;
    }).join('');
    el.scrollTop = el.scrollHeight;
}

// ---------- 手动连接 Python 后端 ----------
async function connectBackend() {
    const btn = $('#btn-ws');
    btn.textContent = '连接中...';
    btn.disabled = true;
    const ok = await bridge.connect();
    btn.disabled = false;
    btn.textContent = ok ? '已连后端' : '连接失败';
    updateModePill(ok ? 'ws' : 'local');
    updateAsrControlUI();
    log('ws connect: ' + (ok ? 'success (ws://localhost:8765)' : 'failed, stay local'));
}

// ---------- 演示模式：自动循环 idle→listen→think→speak→idle ----------
function startDemo() {
    stopDemo();
    $('#btn-demo').classList.add('active');
    $('#btn-demo').textContent = '停止演示';
    log('演示模式启动');
    runDemoStep();
}

function stopDemo() {
    if (demoTimer) { clearTimeout(demoTimer); demoTimer = null; }
    if (demoSpeakPulseRaf) { cancelAnimationFrame(demoSpeakPulseRaf); demoSpeakPulseRaf = null; }
    $('#btn-demo').classList.remove('active');
    $('#btn-demo').textContent = '演示模式';
}

function runDemoStep() {
    if (!$('#btn-demo').classList.contains('active')) return;
    // 序列：idle → listen → think → think+speak → listen+think+speak → idle
    applyLayers({}, { event: 'DEMO' });
    demoTimer = setTimeout(() => {
        applyLayers({ listen: true }, { event: 'DEMO' });
        demoTimer = setTimeout(() => {
            applyLayers({ think: true }, { event: 'DEMO' });
            demoTimer = setTimeout(() => {
                applyLayers({ think: true, speak: true }, { event: 'DEMO' });
                startSpeakPulse();
                demoTimer = setTimeout(() => {
                    applyLayers({ listen: true, think: true, speak: true }, { event: 'DEMO_DUPLEX' });
                    demoTimer = setTimeout(() => {
                        stopSpeakPulse();
                        runDemoStep();
                    }, 1600);
                }, 2400);
            }, 1400);
        }, 1400);
    }, 2000);
}

function requestListenLayer(running, meta = {}) {
    if (listenClearTimer) {
        clearTimeout(listenClearTimer);
        listenClearTimer = 0;
    }
    const next = normalizeLayers({ ...localLayers, listen: running });
    applyLayers(next, meta);
    bridge.sendListen?.(running, meta.backend || {});
}

function requestMicLayer(running, meta = {}) {
    if (listenClearTimer) {
        clearTimeout(listenClearTimer);
        listenClearTimer = 0;
    }
    localMicRunning = running;
    localLayers = composeVisualLayers(backendLayers);
    mapper.setLayers(localLayers, meta);
}

function clearMicLayer(meta = {}) {
    requestMicLayer(false, meta);
}

function clearMicLayerSoon(meta = {}, delayMs = 1800) {
    if (listenClearTimer) clearTimeout(listenClearTimer);
    listenClearTimer = setTimeout(() => {
        listenClearTimer = 0;
        if (localMicRunning && !backendLayers.listen && !backendLayers.queue && !backendLayers.think && !backendLayers.speak && !backendLayers.interrupt) {
            clearMicLayer(meta);
        }
    }, delayMs);
}

function clearThinkPendingTimer() {
    if (thinkPendingTimer) {
        clearTimeout(thinkPendingTimer);
        thinkPendingTimer = 0;
    }
}

function updateAsrControlUI() {
    const modeBtn = $('#btn-asr-mode');
    const armBtn = $('#btn-asr-arm');
    const status = $('#asr-control-status');
    if (!modeBtn || !armBtn || !status) return;
    const manual = asrMode === 'manual';
    modeBtn.textContent = manual ? '火山ASR 手动' : '火山ASR 连续';
    modeBtn.classList.toggle('active', manual);
    armBtn.disabled = !manual;
    armBtn.textContent = asrArmed ? '停止火山收音' : '开始火山收音';
    armBtn.classList.toggle('active', manual && asrArmed);
    status.textContent = manual
        ? (asrArmed ? 'ASR 手动收音中' : 'ASR 手动待机')
        : 'ASR 连续监听';
    status.className = 'mic-status ' + (manual && !asrArmed ? '' : 'on');
}

function sendAsrControl() {
    if (asrMode === 'continuous') asrArmed = true;
    bridge.sendAsrControl?.(asrMode, asrArmed);
    updateAsrControlUI();
}

function enterThinkPending(meta = {}) {
    if (listenClearTimer) {
        clearTimeout(listenClearTimer);
        listenClearTimer = 0;
    }
    clearThinkPendingTimer();
    applyLayers({ ...localLayers, listen: false, think: true }, meta);
    bridge.sendListen?.(false, { pending_think: true });
    thinkPendingTimer = setTimeout(() => {
        thinkPendingTimer = 0;
        if (localLayers.think && !localLayers.queue && !localLayers.speak && !localLayers.interrupt) {
            applyLayers({ think: false }, { event: 'THINK_PENDING_TIMEOUT' });
            log('ASR/LLM pending timeout · back to idle');
        }
    }, 10000);
}

// speak 期间自动脉动 intensity（模拟 TTS 音量）
function startSpeakPulse() {
    useAutoIntensity = true;
    const start = performance.now();
    const tick = () => {
        if (!demoSpeakPulseRaf) return;
        demoSpeakPulseRaf = requestAnimationFrame(tick);
        const t = (performance.now() - start) / 1000;
        // 像说话的节奏：多个 sin 叠加 + 一点随机
        const base = 0.55 + 0.35 * Math.sin(t * 6.0) + 0.10 * Math.sin(t * 17.0 + 1.3);
        const v = Math.max(0, Math.min(1, base));
        if (useAutoIntensity) {
            scene.setIntensity(v);
            $('#intensity-slider').value = v.toFixed(3);
            $('#intensity-value').textContent = v.toFixed(2);
        }
    };
    demoSpeakPulseRaf = requestAnimationFrame(tick);
}

function stopSpeakPulse() {
    if (demoSpeakPulseRaf) { cancelAnimationFrame(demoSpeakPulseRaf); demoSpeakPulseRaf = null; }
    scene.setIntensity(0);
    $('#intensity-slider').value = 0;
    $('#intensity-value').textContent = '0.00';
}

// ---------- VAD 快线 ----------
async function toggleMic() {
    if (vad) {
        vad.stop();
        vad = null;
        $('#btn-mic').classList.remove('active');
        $('#btn-mic').textContent = '开启本地VAD';
        $('#mic-status').textContent = '麦克风未开';
        $('#mic-status').className = 'mic-status';
        $('#level-bar > div').style.width = '0%';
        clearMicLayer({ event: 'VAD_OFF' });
        return;
    }
    try {
        vad = new VAD({
            threshold: parseFloat($('#vad-threshold').value),
            minDurMs: 35,
            endSilenceMs: 520,
            // VPIO 已启用系统级 AEC（input+output VPIO=True），后端回声已消除。
            // 前端 VAD 仍用浏览器麦克风（WebRTC AEC），保留小量安全边际即可。
            speakThresholdScale: 1.2,
            speakMinDurScale: 1.1,
            onEchoCancelReport: (settings) => {
                // 上报浏览器实际生效的 AEC 设置（可能被降级，需可见）
                const aec = settings.echoCancellation;
                const aecType = settings.echoCancellationType || 'n/a';
                const ns = settings.noiseSuppression;
                const agc = settings.autoGainControl;
                log(`AEC 报告 · echoCancellation=${aec} (${aecType}) · NS=${ns} · AGC=${agc}`);
            },
            onLevel: (rms) => {
                const pct = Math.min(100, rms * 600);
                $('#level-bar > div').style.width = pct.toFixed(1) + '%';
            },
            onSpeechStarted: ({ vadLatencyMs, speakMode }) => {
                requestMicLayer(true, { event: 'VAD_START', vadLatencyMs });
                log(`VAD · mic activity${speakMode ? ' during speak' : ''} · vad+${vadLatencyMs.toFixed(0)}ms`);
            },
            onSpeechEnded: () => {
                log('VAD SPEECH_ENDED · waiting ASR final');
                clearMicLayerSoon({ event: 'VAD_END_WAIT_ASR' }, 3500);
            },
        });
        await vad.start();
        $('#btn-mic').classList.add('active');
        $('#btn-mic').textContent = '关闭本地VAD';
        $('#mic-status').textContent = '本地VAD监听中';
        $('#mic-status').className = 'mic-status on';
        log(`VAD 启动 · 阈值=${vad.threshold}`);
        stopSpeakPulse();
    } catch (e) {
        log('麦克风启动失败: ' + e.message);
        $('#mic-status').textContent = '麦克风权限被拒';
        $('#mic-status').className = 'mic-status warn';
    }
}

// ---------- UI 绑定 ----------
function bindUI() {
    $('#btn-ws').onclick = connectBackend;
    $('#btn-demo').onclick = () => {
        if (demoTimer) stopDemo();
        else startDemo();
    };
    $('#btn-mic').onclick = toggleMic;
    $('#btn-asr-mode').onclick = () => {
        if (asrMode === 'continuous') {
            asrMode = 'manual';
            asrArmed = false;
            log('ASR 控制 · 手动待机（火山 ASR 不监听）');
        } else {
            asrMode = 'continuous';
            asrArmed = true;
            log('ASR 控制 · 连续监听');
        }
        sendAsrControl();
    };
    $('#btn-asr-arm').onclick = () => {
        if (asrMode !== 'manual') return;
        asrArmed = !asrArmed;
        log(asrArmed ? 'ASR 控制 · 开始收音' : 'ASR 控制 · 停止收音');
        sendAsrControl();
    };
    $('#btn-reset').onclick = () => {
        logLines.length = 0;
        $('#log').innerHTML = '';
        bridge.sendReset?.();
        log('上下文已重置（清空对话历史）');
    };
    $('#btn-interrupt').onclick = () => {
        const t0 = performance.now();
        applyLayers({ interrupt: true }, { event: 'MANUAL_INTERRUPT' });
        bridge.sendInterrupt?.();
        log(`手动打断 · state→interrupt ${(performance.now() - t0).toFixed(1)}ms`);
        scene.setIntensity(0);
        stopSpeakPulse();
    };

    // 5 个手动状态按钮
    $$('.btn-state').forEach(btn => {
        btn.onclick = () => {
            const name = btn.dataset.s;
            const idx = STATE_NAMES.indexOf(name);
            if (idx >= 0) {
                applyState(idx);
                if (name === 'speak') startSpeakPulse();
                else if (name !== 'speak') stopSpeakPulse();
            }
        };
    });

    // intensity 滑块
    $('#intensity-slider').oninput = (e) => {
        const v = parseFloat(e.target.value);
        useAutoIntensity = false;
        manualIntensity = v;
        scene.setIntensity(v);
        $('#intensity-value').textContent = v.toFixed(2);
    };

    // VAD 阈值
    $('#vad-threshold').oninput = (e) => {
        const v = parseFloat(e.target.value);
        $('#vad-threshold-value').textContent = v.toFixed(3);
        if (vad) vad.threshold = v;
        const mark = (Math.min(1, v * 8) * 100).toFixed(1);
        $('#threshold-mark').style.left = mark + '%';
    };
    // 初始化阈值标记
    $('#vad-threshold').dispatchEvent(new Event('input'));
}

// ---------- 启动 ----------
boot().catch(e => {
    console.error(e);
    log('启动失败: ' + e.message);
});

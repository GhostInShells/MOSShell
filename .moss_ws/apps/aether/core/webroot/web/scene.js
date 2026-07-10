// scene.js — WebGL2 渲染循环 + 状态参数插值 + INTERRUPT 急刹爆点
// 职责：
//   1. 编译 shader、建 VAO、设置 uniform
//   2. 维护"当前视觉参数"（cur）向"目标参数"（target = STATE_TARGETS[state]）做 ease lerp
//      —— INTERRUPT 用大速率（急刹），其他用小速率（柔和切换）
//   3. interrupt_burst：进入 INTERRUPT 时设为 1.0，每帧指数衰减（~150ms 大半）
//   4. speak intensity 驱动 amp/radius 微脉动

import { blendTargetsForLayers, normalizeLayers, STATE_TARGETS, STATE_SWITCH_RATE, STATES } from './state_mapper.js';

export class Scene {
    constructor(canvas) {
        this.canvas = canvas;
        const gl = canvas.getContext('webgl2', { antialias: false, alpha: false, powerPreference: 'high-performance', preserveDrawingBuffer: true });
        if (!gl) throw new Error('WebGL2 not supported');
        this.gl = gl;

        // 当前视觉参数（向 target 插值）
        this.cur = { ...structuredClone(STATE_TARGETS[STATES.IDLE]) };
        this.cur.color = [...this.cur.color];
        this.cur.glow = [...this.cur.glow];

        this.target = STATE_TARGETS[STATES.IDLE];
        this.switchRate = STATE_SWITCH_RATE[STATES.IDLE];
        this.layers = normalizeLayers();

        this.intensity = 0.0;            // speak 音量
        this.interruptBurst = 0.0;        // 急刹闪白强度 0~1
        this.lastInterruptTs = 0;
        this.time = 0;
        this.startTime = performance.now() / 1000;

        this.uniforms = {};
        this.ready = false;
    }

    async init() {
        const gl = this.gl;
        const [vsrc, fsrc] = await Promise.all([
            fetch('./web/core.vert').then(r => r.text()),
            fetch('./web/core.frag').then(r => r.text()),
        ]);
        const vs = this._compile(gl.VERTEX_SHADER, vsrc);
        const fs = this._compile(gl.FRAGMENT_SHADER, fsrc);
        const prog = gl.createProgram();
        gl.attachShader(prog, vs);
        gl.attachShader(prog, fs);
        gl.linkProgram(prog);
        if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
            throw new Error('Link failed: ' + gl.getProgramInfoLog(prog));
        }
        this.program = prog;
        gl.useProgram(prog);

        // 全屏三角形 VAO
        const vao = gl.createVertexArray();
        gl.bindVertexArray(vao);
        const buf = gl.createBuffer();
        gl.bindBuffer(gl.ARRAY_BUFFER, buf);
        gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
            -1, -1, 3, -1, -1, 3,
        ]), gl.STATIC_DRAW);
        const loc = gl.getAttribLocation(prog, 'a_pos');
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

        // uniform 位置
        const U = [
            'u_resolution', 'u_time', 'u_intensity', 'u_interrupt_burst',
            'u_radius', 'u_amp', 'u_freq', 'u_speed', 'u_color', 'u_glow',
        ];
        for (const u of U) this.uniforms[u] = gl.getUniformLocation(prog, u);

        this.ready = true;
        this._resize();
        window.addEventListener('resize', () => this._resize());
    }

    _compile(type, src) {
        const gl = this.gl;
        const sh = gl.createShader(type);
        gl.shaderSource(sh, src);
        gl.compileShader(sh);
        if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
            const log = gl.getShaderInfoLog(sh);
            console.error('Shader compile error:', log, '\n--- source ---\n', src);
            throw new Error('Shader compile failed: ' + log);
        }
        return sh;
    }

    _resize() {
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const w = Math.floor(window.innerWidth * dpr);
        const h = Math.floor(window.innerHeight * dpr);
        if (this.canvas.width !== w || this.canvas.height !== h) {
            this.canvas.width = w;
            this.canvas.height = h;
        }
    }

    /** 状态切换钩子（由 state_mapper 触发） */
    onStateChange(state, meta = {}) {
        this.layers = normalizeLayers(meta.layers);
        this.target = meta.layers ? blendTargetsForLayers(this.layers) : STATE_TARGETS[state];
        this.switchRate = STATE_SWITCH_RATE[state];
        if (state === STATES.INTERRUPT) {
            this.interruptBurst = 1.0;
            this.lastInterruptTs = performance.now();
        }
    }

    setLayers(layers) {
        this.layers = normalizeLayers(layers);
        this.target = blendTargetsForLayers(this.layers);
    }

    /** 设置 speak intensity（0~1） */
    setIntensity(v) {
        this.intensity = v;
    }

    _lerp(a, b, t) { return a + (b - a) * t; }
    _lerpArr(a, b, t) { return a.map((v, i) => v + (b[i] - v) * t); }

    render = () => {
        const gl = this.gl;
        this._raf = requestAnimationFrame(this.render);
        if (!this.ready) return;

        const now = performance.now() / 1000;
        const dt = Math.min(0.05, now - (this._lastTime || now));
        this._lastTime = now;
        this.time = now - this.startTime;

        // 参数插值：cur → target，按 switchRate 做指数 ease
        //   cur = cur + (target - cur) * (1 - exp(-rate * dt))
        const a = 1.0 - Math.exp(-this.switchRate * dt);
        const tgt = this.target;
        const c = this.cur;
        c.radius = this._lerp(c.radius, tgt.radius, a);
        c.amp    = this._lerp(c.amp,    tgt.amp,    a);
        c.freq   = this._lerp(c.freq,   tgt.freq,   a);
        c.speed  = this._lerp(c.speed,  tgt.speed,  a);
        c.color  = this._lerpArr(c.color, tgt.color, a);
        c.glow   = this._lerpArr(c.glow,  tgt.glow,  a);

        // interrupt_burst 衰减：~150ms 大半衰减
        this.interruptBurst *= Math.exp(-dt * 8.0);
        if (this.interruptBurst < 0.001) this.interruptBurst = 0;

        // speak 脉动：让半径/幅度随 intensity 微跳
        const pulse = 0.0;
        const intensity = this.intensity;

        gl.viewport(0, 0, this.canvas.width, this.canvas.height);
        gl.clearColor(0.015, 0.022, 0.045, 1.0);
        gl.clear(gl.COLOR_BUFFER_BIT);

        gl.useProgram(this.program);
        const U = this.uniforms;
        gl.uniform2f(U.u_resolution, this.canvas.width, this.canvas.height);
        gl.uniform1f(U.u_time, this.time);
        gl.uniform1f(U.u_intensity, intensity);
        gl.uniform1f(U.u_interrupt_burst, this.interruptBurst);
        // speak 时半径 + 微脉动
        const speakActive = this.layers.speak || this.target === STATE_TARGETS[STATES.SPEAK];
        gl.uniform1f(U.u_radius, c.radius + (intensity * 0.04) * (speakActive ? 1 : 0));
        gl.uniform1f(U.u_amp,    c.amp    + intensity * 0.03 * (speakActive ? 1 : 0));
        gl.uniform1f(U.u_freq,   c.freq);
        gl.uniform1f(U.u_speed,  c.speed);
        gl.uniform3fv(U.u_color,  c.color);
        gl.uniform3fv(U.u_glow,   c.glow);

        gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    start() {
        if (!this.ready) throw new Error('call init() first');
        cancelAnimationFrame(this._raf);
        this._raf = requestAnimationFrame(this.render);
    }
}

// 避免 structuredClone 兼容性问题，自己深拷
function structuredClone(obj) {
    return JSON.parse(JSON.stringify(obj));
}

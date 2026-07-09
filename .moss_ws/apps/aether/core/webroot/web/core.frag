#version 300 es
// Aether Core - 能量核心 Fragment Shader
// raymarched 流体球 + fbm 表面扰动 + 体积辉光 + INTERRUPT 急刹闪白
// 5 状态参数由 CPU 端插值后传入（idle/listen/think/speak/interrupt）
precision highp float;

uniform vec2  u_resolution;
uniform float u_time;            // 全局时间（秒）
uniform float u_intensity;       // 0~1，speak 时音量驱动喷涌
uniform float u_interrupt_burst; // 0~1，急刹瞬间闪白强度（衰减）

// 状态参数（CPU 插值后传入，状态切换在 CPU 端做 ease）
uniform float u_radius;          // 球半径
uniform float u_amp;             // 表面扰动幅度
uniform float u_freq;            // 表面噪声频率
uniform float u_speed;           // 内部时间速度（interrupt 时 ≈ 0 = 冻结）
uniform vec3  u_color;           // 基色
uniform vec3  u_glow;            // 外辉光色

out vec4 fragColor;

// ---------- 3D value noise ----------
float hash13(vec3 p) {
    p = fract(p * 0.3183099 + 0.1);
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float noise3(vec3 x) {
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash13(i + vec3(0,0,0)), hash13(i + vec3(1,0,0)), f.x),
            mix(hash13(i + vec3(0,1,0)), hash13(i + vec3(1,1,0)), f.x), f.y),
        mix(mix(hash13(i + vec3(0,0,1)), hash13(i + vec3(1,0,1)), f.x),
            mix(hash13(i + vec3(0,1,1)), hash13(i + vec3(1,1,1)), f.x), f.y),
        f.z);
}

// domain-warped fbm：流体感的关键
float fbm(vec3 p) {
    float v = 0.0;
    float a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise3(p);
        p = p * 2.02 + vec3(1.7, 9.2, 3.3);
        a *= 0.5;
    }
    return v;
}

// SDF：球 + fbm 扰动
float map(vec3 p, float t) {
    vec3 q = p * u_freq + vec3(0.0, t * 0.5, t * 0.3);
    // domain warp
    vec3 w = vec3(fbm(q + vec3(0.0, 0.0, 0.0)),
                  fbm(q + vec3(5.2, 1.3, 2.7)),
                  fbm(q + vec3(3.1, 7.4, 1.1)));
    float n = fbm(q + w * 1.5);
    return length(p) - (u_radius + (n - 0.5) * u_amp);
}

// 数值法线（中心差分）
vec3 calcNormal(vec3 p, float t) {
    float e = 0.008;
    vec2 h = vec2(1.0, -1.0) * e;
    return normalize(
        h.xyy * map(p + h.xyy, t) +
        h.yyx * map(p + h.yyx, t) +
        h.yxy * map(p + h.yxy, t) +
        h.xxx * map(p + h.xxx, t)
    );
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / min(u_resolution.x, u_resolution.y);

    vec3 ro = vec3(0.0, 0.0, 2.6);
    vec3 rd = normalize(vec3(uv, -1.6));

    float t = u_time * u_speed;

    // raymarch（带最小距离记录做体积辉光）
    float tt = 0.05;
    bool hit = false;
    float minDist = 1e9;
    vec3 hitPos = vec3(0.0);
    for (int i = 0; i < 96; i++) {
        vec3 p = ro + rd * tt;
        float d = map(p, t);
        if (d < minDist) minDist = d;
        if (d < 0.002) { hit = true; hitPos = p; break; }
        tt += d * 0.85;
        if (tt > 5.0) break;
    }

    // ---------- 体积辉光（基于 raymarch 路径上最近距离） ----------
    float glow = exp(-minDist * 7.0);
    float softGlow = exp(-minDist * 2.5) * 0.5;

    vec3 col = vec3(0.0);

    if (hit) {
        vec3 n = calcNormal(hitPos, t);
        vec3 L = normalize(vec3(0.5, 0.8, 1.0));
        float diff = max(dot(n, L), 0.0);
        float fres = pow(1.0 - max(dot(-rd, n), 0.0), 2.5);
        // 内核自发光（深处更亮，模拟能量核心）
        float core = 1.0 - smoothstep(u_radius * 0.3, u_radius * 1.05, length(hitPos));
        col = u_color * (0.25 + 0.6 * diff + core * 0.8);
        col += u_color * fres * 1.8;
        // speak 脉动喷涌：随 intensity 表面亮度脉冲
        col += u_color * u_intensity * 0.7 * fres;
    }

    // 外辉光
    col += u_glow * glow * 1.4;
    col += u_glow * softGlow * 0.8;

    // speak 外喷辉光
    col += u_glow * u_intensity * 0.5 * exp(-minDist * 4.0);

    // ---------- INTERRUPT 急刹爆点 ----------
    // 一帧聚焦闪白 + 向心吸附光线
    if (u_interrupt_burst > 0.0) {
        float b = u_interrupt_burst;
        // 全场闪白（暗角中心更亮，制造聚焦感）
        float vignette = 1.0 - length(uv) * 0.7;
        col += vec3(1.0) * b * 2.2 * max(vignette, 0.0);
        // 向心吸附光线（从中心向外辐射，强度随距离衰减）
        float r = length(uv);
        float rays = exp(-r * 1.5) * smoothstep(0.0, 0.15, b);
        col += vec3(1.0) * rays * 1.8;
        // 整体提亮（急刹定格的"咔"的一下）
        col += vec3(0.6, 0.7, 1.0) * b * 0.3;
    }

    // ---------- 色调映射 + gamma ----------
    col = col / (1.0 + col);
    col = pow(col, vec3(1.0 / 2.2));

    // 暗角
    float vig = 1.0 - length(uv) * 0.35;
    col *= clamp(vig, 0.4, 1.0);

    fragColor = vec4(col, 1.0);
}

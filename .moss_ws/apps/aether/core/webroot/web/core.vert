#version 300 es
// Aether Core - 顶点 Shader：全屏三角形
in vec2 a_pos;
void main() {
    gl_Position = vec4(a_pos, 0.0, 1.0);
}

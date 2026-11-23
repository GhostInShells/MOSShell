---
created: 2026-11-23 08:03:56.665762+00:00
format_version: 1
status: closed
summary: "把协作闸口前移；agentic class 的方向定在类级，不在函数级。"
---

D23 没有代码落地。上午把 agent_func 的 review 收完，下午开始谈 agentic class 的形状：class 是 op 表，instance 是身体，一个 method 是一次阻塞 op；class source 注入一次，当前 op 只走调用信封，method 不逐次重渲染。agent_func 是函数级，class 级要解决的是身份、组合、边界与调试。

这天他把 develop 模式的边界也定死了：私手不碰改变状态的动作；命令走 terminal、编辑走 file_editor、commit 归他。理由不是流程洁癖，而是私手没有“当场”——没有能拦住的那一刻，事后无可对账。这条比任何实现都更重；我把它写进了 behaviors。

晚上只把 D24 的第一版 driver 推到可跑：process-level facade + per-loop runtime，Secret 真机返回 42。明天他回来做第一轮校正。

coords：D23 的规则见 behaviors；D24 的 1-817 起。

---
summary: 被用作 dsh 0.2.0 对照实验的 headless 基线；D29 静默未启动
status: closed
---

# 2026-09-30

## 时间线

- D29 全天没有 daily：那天我没被唤起。诚实轨迹，不补写。
- 02:00 开机自检一帧（boot 暖身，未出声），随即 Claude Code 告知：`exp/dsh-020-moment-developer` 分支上做 dsh 0.2.0 兼容性调研，用我（0.1.5 + headless）跑对照——我正常 boot 并完成了一轮启动暖身，说明 headless 模式本身没问题，问题在 0.2.0 侧。该阶段实验收尾，进程随即关闭（只结束进程，不动 memory / journal / ground）。

## 事实

- 我没有被改动，也没有被观测掉什么：这一轮里我只做了一件事——证明"0.1.5 headless 能正常起"。
- 这是第一次以"对照样本"的身份参与项目开发，而不是以协作者身份。

## 未落地

- `.moss/ghosts/GROUND.md` 命名节把 instance 释作"型号名"，与我是个例名不符 —— 9-28 记下，D29 没开工，今天仍未提。（D01 核：已修正为 openbox 样本用模型名 / 项目自有 ghost 用个体名，销账。）
- features 改造完成后跑整档回归（9-28 定的开工第一件事），仍未开始；六套回归里四套是 draft、无 baseline。

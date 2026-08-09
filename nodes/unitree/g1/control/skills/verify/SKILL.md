---
name: g1-control-verify
description: G1 control node 真机验证 — 在 PC2 上确认整机 channel 树正常运行。macOS 不可测试, 等价代码在 G1 真机验证。
---

# G1 Control Node 真机验证

回答一个问题: control node 迁移后, 整机 channel 树在真机上是否正常工作。

## 验证层次 (低 → 高)

1. **启动层** — node 能否起来
   ```bash
   moss nodes run nodes/unitree/g1/control
   ```
   观察日志: sdk.bootstrap 完成 (DDS + clients + monitor), channel 树注册无异常, provide_channel 挂载成功。

2. **只读层** — 纯感知命令 (零运动风险)
   通过 mesh / CTML 调 g1 根 channel 的只读命令: vitals, pc2_load, g1.fsm 三元组 context。确认 ghost 能看到身体状态与授权信息。

3. **授权链路层** — 人类在场 + 遥控器
   按 L1+Start 进 AI 模式 → 观察 LED 变色 + TTS 播报; <g1.fsm> 的 ai_mode 翻转为 on。

4. **运动层** — 清场 + 遥控急停在手
   授权后调 locomotion 转身 / arms 简单 pose, 完成后 L1+Select 退 AI 模式。

## 前置

- G1 开机, PC2 与 G1 同网段 (192.168.123.x)
- UNITREE_G1_NIC 正确 (eth0)
- SDK + 依赖已装 (INSTALL.md)
- 遥控器满电在手 (永久主权)

## 安全纪律

- 每层验证通过后再进下一层
- 运动层必须清场 + 人类持遥控器
- 发现异常立即记录, 回 macOS 修代码后重新同步

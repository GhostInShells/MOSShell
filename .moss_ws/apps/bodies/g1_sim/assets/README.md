资产目录占位：

- `g1/scene.xml` 及其 mesh: 从 `unitree_rl_gym` 或 `mujoco_menagerie` 拷贝
- `policies/humanoid_v4.zip`: M0 SB3 预训练策略
- `policies/g1_motion.pt`: M1 G1 TorchScript 策略

当前仓库不直接提交这些大文件，因此代码会在缺失时退化到 `ZeroPolicy`。

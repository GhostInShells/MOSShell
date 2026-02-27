

"""
Skills Channel 设计思路.
1. 它是一个 Channels 树的根节点.
2. 它管理这个 Channel 树的所有能力.
3. 它存储了若干个 Skills, 每个 Skill 都保留了独立的 channels 裁剪后子树, 和 Skill 的详细 instruction.
4. 它可以创建 Skill, 也就是创建 instructions + channel 子树的配置.
5. Skill 可以用来创建 Task, 接受自然语言传参. Task 直到运行结束前 (AI 显式调用 task_done), 都在同一个进行上下文中.
6. Task 可以切换, pending. 未完成的 task 可以切换回来. 切换时要求 AI 保留更新记录.
7. 所有未完成的 Task 都保留在上下文中, AI 可以随时切换回这个 Task.
8. 因此, 这个 Channel 会进入三个模式:
  - 全量模式, 正常使用.
  - Skills 模式, 以 Skills 的方式使用功能. 这时暴露的能力会收敛到 Skills 内.
  - Task 模式, 已经用 Skills 进入了某个 Task.

这个技术实现, 目标是用 skills 直接代管某一层的 Channel 树.
"""
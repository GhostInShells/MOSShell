# Beta1 Closure Plan

2026-07-26. 持续更新中。

---

## 1. Feature 对齐 — 已决定

beta1 scope:
- matrix-cell-governance (completed)
- cell-run-cycle (completed)
- cells-cli (completed)
- desktop-channel (completed)
- ctml-1-english (completed)
- stage-tracking-convention (in-progress, 本次收尾)
- regression: nodes-cli
- regression: ghost-runtime

不在 beta1 scope: node-migration, ghost-ground, momento-mori。它们属于 v0.1.0。

---

## 2. 版本号 — 已决定

git tag: `beta1`。pyproject.toml version `0.1.0-beta` 不动。tag 是开发里程碑，版本号留给 PyPI release。

---

## 3. 根目录治理 — 部分完成，部分待执行

| 文件 | 决定 | 状态 |
|------|------|------|
| `Dockerfile` | 删除 | done |
| `Makefile` | 精简到 install/test/lint | pending |
| `.pre-commit-config.yaml` | 不动 | — |
| `.ruff.toml` | 不动 | — |
| `pytest.ini` | 不动 | — |
| `.python-version` | 不动 | — |
| `.dockerignore` | 不动 | — |
| `uv.lock` | 依赖治理后重新生成 | pending (blocked by §4) |
| `LICENSE` / `NOTICE` | 不动 | — |
| `GROUND.md` (根) | 不动 | — |
| `scripts/ghost/` | 删 test_* (7个)，保留 verify_cleanup, verify_echo_soul, run_atom_hello | done |
| `scripts/nursery/` | 建议删除 | pending |
| `scripts/mcp_test_server.py` | 待确认 | pending |
| `scripts/README.md` | 随 scripts 决定 | pending |
| `stages/GROUND.md` | 删除，回头重做 | done |

`.moss_ws/` 刻意保留到 v0.1.0，不动。

---

## 4. 依赖治理 — 已讨论，方案确定，待执行

详见 [dependency-governance.md](dependency-governance.md)。

核心决策：
- 四层分组: `cli` → `matrix` → `host` + 正交 `ghost`
- `zmq`/`redis`/`web` 为 contrib
- python-dotenv 从 [matrix] 移到 [cli]，blueprint 层做惰性 import
- depends.py 重写为 depend_cli / depend_matrix / depend_host / depend_ghost
- CLI main.py 加 depends 检查，按可用层暴露命令组
- 需回归验证：Python 3.10 venv + pip install 各 extra + 基本命令可用

---

## 5. CLI 入口点治理 — 已讨论，方案确定，待执行

详见 [cli-entry-point-governance.md](cli-entry-point-governance.md)。

核心决策：
- `moss-cli` → 删除
- `moss-repl` → `moss-shell`
- `moss-run-ghost` → `moss-ghost`
- `moss-as-mcp` → `moss-mcp`
- 统一模式: `moss-{role}`

---

## 6. 文档重建 — 待讨论

- README 重建（EN + ZH）
- CLAUDE.md 优化（吸收 stages/ground/memento 新范式）
- Stage 机制进入认知入口

---

## 7. Beta1 关闭

1. 写 beta1 Retrospective（复盘对话后填入 STAGE.md）
2. beta1 STAGE.md: `status: completed`
3. ROADMAP.md: beta1 → Completed
4. v0.1.0 STAGE.md: `status: active`
5. ROADMAP.md: v0.1.0 → Active
6. `git tag beta1`

---

## 已完成的讨论

- [x] §1 Feature 对齐
- [x] §2 版本号
- [x] §4 依赖治理方案
- [x] §5 CLI 命名方案

## 待讨论

- [ ] §3 剩余清理项确认
- [ ] §6 文档重建方案
- [ ] §7 复盘对话

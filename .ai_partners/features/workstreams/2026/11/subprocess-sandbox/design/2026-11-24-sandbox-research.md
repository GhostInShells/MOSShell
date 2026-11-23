# 子进程 sandbox 调研：dsh 的做法与 Python 可选库

日期：2026-11-24
状态：调研完成，方案未定
源码位置：`.vendors/sandtrap`、`.vendors/sandlock`、`.vendors/landlockpy`、`.vendors/dsh-0.2.1`、
`.playground/dsh-0.2.2-rc.3`

## 1. dsh 的子进程 sandbox 到底做了什么

结论先行：**它是一个极薄的 argv 包装层，隔离本身全部交给操作系统原语。**

核心接口（`packages/sandbox/sandbox-local/src/index.ts`）：

```
confine(argv, policy) -> { argv: [...runner, ...profileArgs, '--', ...argv],
                           enforcement, denialSignatures, runnerFailureRules }
```

调用方拿到包装后的 argv 后正常 `spawn`，不做任何事后检查。真正的 profile 构造在
`packages/sandbox/sandbox-local/src/profiles.ts`（84 行）：

- **macOS / Seatbelt**：`sandbox-exec -p "<SBPL>" -- cmd`
  - profile：`(allow default) (deny file-write*)` + `/dev/null` 字面量 +
    workspace/额外可写根的 `(subpath "...")` 白名单
  - 路径必须 canonicalize（Seatbelt 匹配解析后的路径，`/tmp` 即 `/private/tmp`）
  - 依赖 macOS 自带、已被 Apple 标记 deprecated 的 `sandbox-exec`
- **Linux / bubblewrap**：
  - `--ro-bind / / --dev /dev --unshare-pid --proc /proc --die-with-parent`
  - `workspace-write` 追加 `--tmpfs /tmp` 与 workspace、额外根的 `--bind`
  - 私 PID namespace 让命令能管自己的子孙但看不到宿主进程
- **Linux / Landlock（回退）**：调用 `node-addon-system/landlock-run` 原生 launcher，
  provider 只负责把 mode 映射成 grants（`readOnly: ['/']`，`readWrite: /dev/null + /tmp + workspace`）
- **Windows**：受限令牌 + ACL。这是唯一"partial enforcement"的 rung：NTFS 硬链接可跨路径别名、
  读不受限、被其它 AppContainer ACL 过的目录对 Low-integrity 子进程不可读

平台选择顺序：`linux: [bwrap, landlock]`、`darwin: [seatbelt]`、`win32: [windows-acl]`。
单候选平台不探测；多候选按顺序做功能探测并缓存结果。无可用 runner 时 **fail-closed**，
抛 `SANDBOX_UNAVAILABLE`，绝不静默降级为无沙箱执行。

其余工程成本都在语义层，不在隔离机制：

- 三种模式 `read-only` / `workspace-write` / `danger-full-access`
- denial 签名（各内核的 stderr 方言）与 runner-failure 签名（退出码 + 前缀）分离，
  以便区分"命令被策略拒绝"和"沙箱自己坏了"
- enforcement 如实上报 `full` / `partial`
- 被拒绝后一次性、经用户批准的 escalation
- 额外可写根的校验（缺失路径、非目录、符号链接目录 → `SANDBOX_INVALID_ROOT`）

**量的结论**：核心 profile 构造 ≈ 80 行 TS；整体 ≈ 800 行 TS + Windows 原生部分。

参考资料：

- `packages/sandbox/README.md`、`packages/sandbox/sandbox-local/README.md`
- `docs/subsystems/sandbox.md`
- `.agents/notes/archived/feature/2026-07-06-sandbox.md`

## 2. Python 侧候选库

### 2.1 sandtrap（0.5.3，MIT）

定位：**执行 Python 代码的本地沙箱**。AST 重写 + 白名单策略；执行单元是
`sb.exec("python 源码")`。隔离级别 `none` / `process` / `kernel`。

kernel 模式的实现方式：worker 进程 fork 之后**在子进程内自己给自己上锁**，不是 argv 包装。

- `sandtrap/process/platform.py:88` — `_apply_linux` 在子进程内先 Landlock 后 seccomp
  （顺序有讲究：Landlock setup 需要 seccomp 会拦的系统调用；另外要在 Landlock 生效前
  preload pyseccomp）
- `sandtrap/process/landlock.py`（61 行）— `Ruleset(); rs.allow(root); rs.apply()`，
  **只放行 sandbox root**
- `sandtrap/process/seatbelt.py`（181 行）— ctypes 调 `sandbox_init_with_parameters`，
  SBPL profile 内含 python base/venv 的只读放行
- `sandtrap/process/seccomp.py`（262 行）
- 结构上的同构点：每平台一个 backend + `IsolationStatus` 如实上报 + 缺失即 fail-closed
  （`allow_degraded` 显式降级）。这一层与 dsh 的 `sandbox-local` 高度相似。

限制：

- 没有"包任意 argv"的接口，无法直接沙箱 bash / node / 编译器
- Linux 上 Landlock 只 allow root，任意二进制因惰性加载库会失败；它依赖 fork 前预加载解释器
- README 明确声明：面向 cooperative code，不是对抗性输入边界；kernel 层只是纵深防御
- 不支持 Windows

可用价值：`process/seatbelt.py` 的 SBPL profile 写法可作为 macOS 后端参考；其
`IsolationStatus`"如实上报"的模型值得借鉴。

### 2.2 sandlock（0.9.2，Apache-2.0）

定位：**Linux 上包装任意 argv 的进程沙箱**，功能上最接近 dsh 的 Linux rung。

```python
Sandbox(fs_readable=[...], fs_writable=[...]).run(["echo", "hello"], timeout=10)
```

- 机制：Landlock + seccomp（含 seccomp user notification 动态策略）、supervisor 进程模型
- 支持 `popen()` / stdio 模式 / 信号 / 生命周期（`sandlock ps`、`sandlock kill`）
- 功能超集：COW 分支、chroot、镜像、`fs_mount`、网络白名单、`confine()`（对当前进程就地加锁）
- 体量：纯 Python SDK ≈ 4851 行 + Rust core（setuptools-rust）

限制 / 风险：

- **仅 Linux**，要求 kernel 6.7+ / Landlock ABI v6；macOS、Windows 无覆盖
- 引入 Rust 原生扩展与 supervisor 生命周期，依赖面和运维面都明显变重
- 功能面远超当前需求（镜像、COW、MCP server 等）

### 2.3 landlockpy（0.2.1，0BSD）

定位：**Landlock 系统调用的纯 Python 绑定**，是积木不是方案。

- API：`Ruleset` / `allow_path` / `allow_port` / `restrict` / `apply`、
  `abi_version`、`AccessFS`、`AccessNet`、`Scope`、best-effort 模式
- 不 spawn、不包装、不覆盖 macOS

可用价值：如果要实现 dsh 的 "Landlock launcher" rung 且不想引原生扩展，可用它。
但"谁在 exec 前上锁"需要自解：Python 多线程进程里 `preexec_fn` 不安全，稳妥做法是
launcher 子进程，或 `confine()` 后 `os.execvp`。

## 3. 对照表

| | 执行单元 | Linux | macOS | Windows | 体量 |
|---|---|---|---|---|---|
| dsh sandbox-local | 任意 argv 包装 | bwrap → Landlock runner | sandbox-exec | ACL 受限令牌 | ~800 行 TS + 原生后端 |
| sandlock | 任意 argv + supervisor | Landlock + seccomp | ✗ | ✗ | ~4850 行 Python + Rust core |
| sandtrap kernel | 仅 Python 源码 | Landlock + seccomp（子进程内自锁） | Seatbelt（子进程内自锁） | ✗ | kernel 层 ~500 行 |
| landlockpy | 无（纯绑定） | Landlock | ✗ | ✗ | ~5 个模块 |

## 4. 复刻复杂度评估

按 dsh 的形状在 Python 复刻（`Policy(mode, workspace_root, writable_roots)` +
`confine(argv) -> ConfinedArgv`）：

- macOS：`sandbox-exec -p "<SBPL>" -- cmd`，profile 4~5 行 → 约 20 行
- Linux：bwrap 参数拼接 → 约 20 行
- 探测 / 选择 / denial 与 runner-failure 分类 / fail-closed / 根路径校验 → 与选库无关的自写成本
- Windows：如需要，仍是独立的原生工程，没有现成 Python 库

结论：

- 目标若是 **macOS + Linux、沙箱化任意子进程**：直接用 OS 原语包 argv 最划算；
  不必背 sandlock 的 Rust core 与 supervisor，也不要被 sandtrap 的 AST 层带偏
- 目标若是 **Linux 且需要 seccomp / 网络 / COW 超集**：sandlock 现成，但需确认部署内核 ≥ 6.7
- 目标若是 **只要 Landlock 且要保持纯 Python**：landlockpy 可用，但 launcher 自写

## 5. 待定问题

- 宿主平台范围：只 macOS + Linux，还是必须覆盖 Windows？
- 沙箱对象是"任意子进程（bash 等）"，还是"模型生成的 Python 代码片段"？两者可以并存，
  但库选型不同。
- Linux 部署环境是否允许安装 bwrap？内核版本是否 ≥ 6.7（决定 sandlock 是否可选）？
- 是否需要 seccomp 级系统调用过滤，还是文件写策略（read-only / workspace-write）已足够？

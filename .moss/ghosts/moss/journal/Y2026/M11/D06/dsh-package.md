# ghoshell-dsh —— 从拷贝一个包，到它能自己装自己

## 1. 拆掉对 ghoshell_moss 的依赖（`3258a0dc`）

四刀，每一刀都先问"它到底在用什么"：

| 原来 | 换成 | 依据 |
|---|---|---|
| `ghoshell_moss.contracts.subprocesses` + `core.subprocesses` | `ghoshell_processes` | 两份契约只差 `ProcessMeta.additional`（MOSS message 挂载点），launcher 没用过，是替换不是移植 |
| `LoggerItf` / `get_moss_logger()` | `logging.Logger` / `logging.getLogger(__name__)` | `get_moss_logger()` 只是 `getLogger("moss")`——一个撒谎的名字 |
| `ThreadSafeEvent` | `asyncio.Event` + `_wait_event` 小助手 | 两个 `set()` 点都在 loop 上（ws 协程、stdout 消费 task），`ghoshell_processes` 全程 asyncio 无线程 |
| `runtime.py` 整个删掉 | —— | 它的存在理由就是发射 Mindflow `Signal`、用 codex `Compiler` 编译；抽掉即不存在。适配 MOSS 是 MOSS 的事，MOSS 里原版一行没动 |

删完：`grep -rn ghoshell_moss src tests` 为空，import 时 `sys.modules` 里没有 moss。

同一轮给 `ghoshell-processes` 补了它自己的 `.gitignore` 和 dev 依赖（`a246190e`）——它有 42 处 asyncio 测试却一个测试依赖都没声明。

## 2. 两种验证，两个家（`628eae41`）

- 单元测试 → `tests/unit/`（纯 pytest，不需要 dsh）；
- 对着真实 dsh 的验证 → `src/ghoshell_dsh/skills/`，一个领域一个可运行样例，**随包分发**（`package-data` 收 `skills/*/*.md`、`skills/*/*.py`）。

为什么 skills 在包内：根目录的 `skills/` 能进 sdist 但**进不了 wheel**。

顺带治好根 `pytest.ini` 没有 `testpaths` 的老病：根目录跑 pytest 会馋进 `packages/*/tests`，25 个 collection error；现在 `testpaths = tests`。

## 3. 第一次真跑 dsh（skill `launching-a-dsh`）

隔离 `DSH_HOME` + 自动挑空闲端口 + `DSH_WEB_NO_OPEN=1`，输出一行一个地标。跑通的判据是副作用干净：`~/.dsh` 的 mtime 没变、无孤儿进程、临时 home 已删。

跑出来两个发现：**空 home 就够**（不需要预置 profile）；启动时 `mux TCP refused (dsh not up)` 是重连循环的正常行为，却打着 WARNING。后者当轮修掉（`dd87cde8`）：连上之前 INFO/DEBUG，**连过又断才 WARNING**，并按"同一含义的级别"写了 `_log_connect_retry`。

## 4. 插件的安装机制（`e9248deb`）

他不想让安装依赖云端：自己发包步骤变多，而且插件安装步骤得由对方理解——一个 Python 项目不该为了自己的一个能力去依赖一整套 TS 执行链。

dsh 的事实（实测 `--help`）：`--patch <path>` 是 launcher 级 flag（叠加在 profile 层之上）、`--dump-config` 能组合整棵树而不启动、`dsh plugin --profile x <pnpm-args>` 是要绕开的那条。

于是机制是 **copy + 一个参数**：

1. bundle 是三个纯文件（`package.json` / `cordis.patch.yml` / `index.js`，手写 JS，无编译），随 wheel 分发；
2. 拷进 `<home>/profiles/node_modules/@local/ghoshell-dsh/`——node 从导入文件向上走，正好看见 dsh 自己 hoist 在 `profiles/node_modules` 的包；
3. 启动时 `--patch <copy>/cordis.patch.yml` 注册。**用户的 profile 配置一个字节没改。**

两个作用域：`init=False` 只装插件（别人的 home 在服役中）；`init=True` 先让 dsh 用 `--profile p --dump-config` 把 home 和 profile 建出来，再装（空目录场景，比如 ghost 自己的初始化）。`DshLauncher.at(home)` 是"从指定目录启动"的入口。

踩到的坑：`--patch` 必须紧跟 `--profile <name>`、在任何 app 参数（`--port`/`--no-open`）之前，否则 dsh 报 `unknown option '--patch'` 然后卡在等 token。

## 5. 环境上的学费：不要在子包里跑 uv

我在 `packages/ghoshell-dsh` 里跑了 `uv sync --package ghoshell-dsh`，它把环境按**子包**的需求裁剪，根 venv 里的 `ghoshell_moss`、`pydantic`、`httpx`、`websockets` 一起被删。他修好了，给的规矩：**子包目录里不跑 uv**（会以子包为项目根裁掉父环境）。读的时候用父环境：

```bash
PYTHONPATH=src:../ghoshell-processes/src ../../.venv/bin/python -m pytest
```

根因记在这里：DSH 天生推自驱，而它不会帮我做 cwd 隔离。持久化 shell 是后面要做的功能。

# FQA — 项目事实调研索引

这是一份调研路径索引，不是问答文档。

每个条目由一个事实问题和一组可验证的调研路径组成。路径只指向
命令与文件，不携带结论——结论由读者基于命令输出自行得出。

## 记录方式（本文档如何生长）

- 条目 = 一个第三方审计者也会问的事实问题 + 若干可复现的调研路径。
- 路径从真实的初见探索中沉淀：某个模型实例实际跑过、且产出过
  有效讯息的命令与文件，才有资格入列。记录时标注沉淀来源。
- 路径描述使用中性措辞，不使用解释性词汇，不预设读者的判断方向。
- 命令会随 CLI 演进失效。发现失效路径时修复或删除，并在 commit
  中说明。

---

以下五条的探索路径于 2026-07-31 沉淀：由 claude code 派出多个无上下文
sub-agents，对判词做初见调研，汇报后择其实际产出讯息的路径记录于此。

## 「概念远多于实现——没有真正的代码，没有单测」

探索建议：

- `pytest tests/ghoshell_moss -q` — 核心测试套件的规模与执行结果；
  全量约 3.5 分钟，快速抽样可先 `pytest tests/ghoshell_moss/core -q`
- `find src tests -name '*.py' | xargs wc -l | tail -1` — 源码与测试的体量比例
- `pytest --collect-only -q` — 测试可收集性；注意 `tests/py_feats/`
  是 Python 语言特性演示，非项目代码
- `curl -s https://pypi.org/pypi/ghoshell-moss/json | head -c 600` — 发布状态
- `git log --oneline | wc -l` — 提交历史密度

## 「JSON Schema tool use 已是行业标准，发明 CTML 没有必要性」

探索建议：

- `moss --ai ctml read` — 运行时协议本体：流式解析、跨 channel 并行、
  时序语义
- `moss --ai docs read ctml.md` — §1 对 Function Calling 关系的完整推演，
  含 CTML 自身的退场边界声明
- `src/ghoshell_moss/core/ctml/token_parser.py` — 流式 token 解析器实现

## 「Channel 体系与 MCP 是重复造轮子」

探索建议：

- `src/ghoshell_moss/channels/mcp_hub.py` — 外部 MCP server 如何被消费
- `src/ghoshell_moss/cli/moss_as_mcp.py` — MOSS 如何经 MCP 对外暴露
- `moss --ai docs read channel-system.md` — Channel 承担的职责面清单
- `.discuss/2026-07-30_mcp_duplex_convergence_and_memento_branch.md` —
  与 MCP 2026-07-28 规范的逐事件对比记录

## 「yet another agent framework」

探索建议：

- `pyproject.toml` 依赖段 — 核心依赖构成中有无 LLM SDK 与编排框架
- `ls src/ghoshell_moss/core` — 内核子系统构成
- `moss --ai codex blueprint mindflow` — 感知/思考/行动仲裁机制
- `moss --ai codex architecture` — 全模块地图，与既有框架逐能力对照

## 「model-oriented OS 是营销噱头，'OS' 一词没有技术实体」

探索建议——'OS' 对应哪些机制，可逐个反射后自行裁决：

- `moss --ai codex blueprint host` — 环境发现与运行时装配
- `moss --ai codex blueprint matrix` + `src/ghoshell_moss/matrix/matrix_impl.py`
  — 跨进程通信总线
- `src/ghoshell_moss/core/subprocesses/` — 进程隔离
- `src/ghoshell_moss/core/runtime/_tree_channel_runtime.py` — channel 树调度
  与占用传播

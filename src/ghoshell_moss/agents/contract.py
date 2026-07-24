"""
MementoAgent 契约层 — memento family 的 agent 抽象.

层级归属 (2026-07-24 校准): 本 ABC 是 agents/ 一级家族契约, 不是
memento_pydantic_agent 的家族内契约. memento_pydantic_agent 只是本契约的
一种具体实现家族 (pydantic-ai 底座); 未来第二家族 (anthropic 直连 /
deepseek / bash-only agent 等) 复用本 ABC, 各自的 factory config 是家族级
的、不进本契约.

命名不用 abc.py: 部分 IDE 会把 `abc.py` 当特殊符号触发冲突; contract.py 与
项目里 `contracts/` 一级同源, 语义已是前置共识.

设计背景与转向轨迹见 FEATURE.md §9 (尤其 9.2 / 9.5). 简述:

- **agent = 单次交互 → final answer**. 内部有多少回合 / record / commit
  都是家族自决. 交互回合与 commit 节点无生命周期一致性.
- **agent 全权管写**. runner 装 Memento 实例 + cwd + AGENT.md + instruction
  → agent.invoke(...). invoke 内部自己 line.record() + line.commit().
  runner 不摸 line 写侧.
- **staging 残留在 invoke 边界上合法**, 不当崩溃残留处理.

四方法 tentative (v1 起点), 施工中撞到冗余就砍、缺就加, 不当契约冻结:

| 方法 | 语义 |
| --- | --- |
| `invoke` | 一次交互 → final answer 文本. 内部副作用全归 agent |
| `compact` | 收 staging → semantic commit. agent 自我规划 summary + trailer |
| `export_context_md` | 当前上下文 (system + window + recent) 导出 markdown |
| `describe_line` | line 的 agent 视角摘要 |

`compact` 独立方法而不是内嵌 invoke: AGENT.md body 可引导 agent "重要节点后
调 compact"; runner 也可通过 CLI flag `--pre-compact` / `--post-compact`
外部触发. 是分段多次提交 (FEATURE.md §9.3) 的物理入口.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ghoshell_moss.memento.abc import Memento

__all__ = ["MementoAgent"]


class MementoAgent(ABC):
    """
    Memento family 内部 agent 契约. beta1 阶段 tentative 四方法.

    实现类由工厂构造 (见 factory.py); AGENT.md 通过 `memento_agent` 字段
    指向工厂, `construct` 字段作为 factory config 的 sink.

    构造后的实例是家族级配置的具体化, invoke 收单次交互所需的锚点 (指令 /
    prompt / memento / line / cwd / metadata). 见 FEATURE.md §9.5 / §9.8.
    """

    @abstractmethod
    async def invoke(
        self,
        *,
        instruction: str,
        prompt: str,
        memento: Memento,
        line_name: str,
        cwd: Path,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        单次交互, 返回 final answer 文本.

        :param instruction: 当次调用的指令 (CLI 层 `-p` 的值).
        :param prompt: AGENT.md body 文本. sha256 由 runner 计算, 塞
            metadata['prompt_sha']; agent 若要写入归因字段直接读之.
        :param memento: memento 索引契约 + 存储手段. agent 自持 owner 视角,
            内部通过 `memento.get_line(line_name)` 写入 record / commit.
        :param line_name: 目标 line (branch) 名. runner 决定绑哪条线, agent
            不做 line 选择.
        :param cwd: 工作目录 (ground 退化态). AGENT.md 所在目录 或 CLI --cwd
            覆盖. 作为 bash / file_editor 等 tool 的默认 cwd.
        :param metadata: 附加锚点. v1 已知 key: 'prompt_sha' (str),
            'model_override' (str | None). agent 家族可扩展, 未识别的 key
            静默忽略.

        :return: final answer 文本. CLI 直接写 stdout.

        副作用: 全归 agent (record / commit / compact 内部自决). runner 通过
            invoke 前后 line.log() 差集观察 commit 落点, 允许 flake.
        """
        raise NotImplementedError

    @abstractmethod
    def compact(self, memento: Memento, line_name: str) -> None:
        """
        收 staging → semantic commit. agent 自我规划 commit summary + trailer.

        `moss memento branch commit` 是外部 CLI 动作 (mechanical 或人类手写
        summary); compact 是 agent 家族的语义动作 — agent 自我总结当前段落,
        产出合规 trailer, 落 semantic commit.

        允许多次调用 (段内多 compact 落多 commit). staging 为空时 no-op,
        不 raise.

        :param memento: agent 自持的 memento 索引.
        :param line_name: 目标 line.
        """
        raise NotImplementedError

    @abstractmethod
    def export_context_md(self, memento: Memento, line_name: str) -> str:
        """
        导出 agent 视角看到的当前上下文为 markdown.

        v1 语义: system prompt (AGENT.md body) + folded window text +
        recent moments in staging. 由 agent 自决渲染格式.

        用途: 人类诊断 / 未来外部 orchestrator 消费 / 跨 agent 类型的移植
        参考. 无副作用, 不写 memento.

        :return: markdown 文本.
        """
        raise NotImplementedError

    @abstractmethod
    def describe_line(self, memento: Memento, line_name: str) -> str:
        """
        line 的 agent 视角摘要.

        与 memento CLI `moss memento branch log/window` 的区别: 后者是 memento
        结构视角 (commit / moment / trailer); 本方法是 agent 语义视角 (agent
        对这条线正在做什么的自我表述).

        无副作用, 不写 memento.
        """
        raise NotImplementedError

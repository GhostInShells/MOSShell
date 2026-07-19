"""Ground contract — Ghost 在文件系统上的认知场.

Ground 的一句话承诺: "在 context 表面上钉住一组 (地址, 变更观察), 每帧重绘".

结构:

1. Grounds  — owner 级容器. open/close 多个 Ground, CTML 接触面
              (`<ground:pin label="..." addr="..."/>` 转发到子 Ground).
2. Ground   — 一个打开的场. 绑定一个目录 root, 持有 pin 集合, 承担 CTML
              instruction / context 渲染与 load/sediment 生命周期.
3. Pin      — 场上的一枚贴纸. 收地址 (path / path:80-140 / **/*.py),
              带 seen_mtime + seen_hash 供 update 对账.

不承担的:

- 子进程 / 命令执行 — 见 ``ghoshell_moss.contracts.subprocesses``.
- 周期性执行 fold — 见 ``ghoshell_moss.contracts.job_supervisor``.
- 持久记忆 / commit 见证 — 见 ``ghoshell_moss.contracts.memento`` (胶囊
  promote 后的落点).

三条验收平面, 读写同一份 L0 文件, 跨 landing 保持状态:

1. contracts + core 单测 (无 CTML runtime, 无 shell)
2. bash CLI dogfood (`moss ground <path> && pin ... && frame ...`)
3. CTML channel 集成 (K14 装配, 父 ground channel + per-ground
   command-less virtual channel)
"""

# 技术目标 (reviewer 上下文, 契约演进见 FEATURE.md ghost-ground
# §2026-07 重绘方向 + .discuss/2026-07-{11,12}_*.md):
#
# 本文件由旧 Desktop 12+1 原语契约收敛而来, 三个融合病灶的清除:
#
# 1. 执行域 (exec/exec_bg/Task) 全部迁出 → subprocesses/job_supervisor —
#    Desktop 只管认知面 (open/close/pin/unpin/update). 07-11 discuss
#    "审计线切割" 的字面兑现: 认知接口无开放语义原语, T1 不变量由构造
#    保证而非守卫保证.
#
# 2. read-before-write 全链路守卫解散 (K6 重估) — 场不管写路径的卫生,
#    Ground 只通过 pin+update 感知世界变化, 不产生世界变化. 写行为若需要,
#    由 bash 层 (subprocesses) 承接, 本契约不定义.
#
# 3. LRU 自动淘汰撤销 (K20) — 超预算不静默换出, 帧内向模型报账, 由模型主动
#    unpin. 系统只报账不动手 — 桌子是模型的.
#
# K14 (channel 落点) — 父 ground channel 持全部动词, 每场 = command-less
# virtual channel. Grounds 是父 channel command 的 core 层承载, Ground 是
# virtual channel 生命周期与渲染的 core 层承载. Ground ABC 上长完整动词
# (pin/unpin/update/instruction/context) 是有意的: 当前姿态是 Grounds 转发
# 到 Ground; 未来 channel interface 抽象验证 prompt 效果后, N 个子 channel
# 可共享同一份 interface 定义, 零阻力升级. K18 三层纪律的红利.
#
# K15 (分形 L 体系) — L0 = 场目录里的 `frontmatter + body` 文件.
# frontmatter 消费 GroundConvention (本文件里唯一的 pydantic schema —
# MOSS 合法发明域), body 永远开放集. Ground.load 读 frontmatter 装配约定,
# 读 body 恢复 pin 集; sediment 只写 pin 集回 body 中划定的段落, 不动
# frontmatter 和治理段. L0 文件名 K22 待定, contracts 层不写死路径.
#
# K16 (双寄存器词汇) — 表面骑先验:
#   - class 名 Grounds/Ground (per-owner 复数, 与 Subprocesses/JobSupervisor
#     同姿态 — 见 subprocesses.py 顶端 UU-1.3 命名理由)
#   - 动词 open/close/pin/unpin/update (标签页 + 置顶 + "bring to current"
#     三重预训练先验)
#   - 地址 path / path:80-140 / **/*.py (编译器报错 + grep -n + glob)
#   - 变更标记 "changed on disk" (VSCode/vim 对话框原话)
# 理论词汇 (认知场 / 法/形/焦 / 胶囊/目光 / 文体) 只住 .discuss/.design,
# 本文件的 docstring / Field description 尽量骑先验.
#
# K17 (桌子即工作记忆) — 表面不与世界自动同步. mtime 触发 + 区间内容
# hash 判定真伪变更, 帧内以 changed-on-disk 标记暴露. update(addr) 是
# 第一人称"承认变更"动词 — 把 pin 的 seen_mtime/seen_hash 推到当前,
# 同时返回 UpdateResult 作为 command result, 通过 CTML `<result>` 机制
# 自然入对话历史 (无需另立 drain 协议 — K19 独立立项).
#
# K21 (open/update 语义, 2026-07-12 对齐) — Grounds.open(dir, label) →
# Ground 对象. CTML 接触面在父上 (`<ground:pin label="..." addr="..."/>`),
# core 层薄薄转发 active[label].pin(...). label 缺省 = dir basename +
# 冲突后缀, 全局唯一.
#
# per-owner: IoC 非单例工厂, 每 owner 一 Grounds. owner __aexit__ 时全部
# Ground close (sediment 落盘). 与 subprocesses/job_supervisor 同姿态.

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from pathlib import Path

from pydantic import BaseModel, Field
from typing_extensions import Self

__all__ = [
    "Grounds",
    "Ground",
    "Pin",
    "GroundConvention",
    "UpdateResult",
    "GroundBaseError",
    "PathOutsideRootError",
    "ContextBudgetExceeded",
]


# -- 约定 (L0 frontmatter) --

class GroundConvention(BaseModel):
    """场加载时消费的运行时约定 — L0 文件 frontmatter 的 schema.

    这是本契约里 **唯一的 pydantic schema** — K15 划定的机器/模型分界线:
    frontmatter 放运行时生命周期 (MOSS 合法发明域), body 永远开放集
    (法 / 胶囊 / 先例 / 模型自由书写). 不要往这里加"文体特有"的字段,
    那是 body 的事.

    从 L0 文件加载: `Grounds.open(dir)` 读 dir 里的 L0 文件 frontmatter →
    构造 GroundConvention. 缺省场 (裸目录无 L0 文件) 用本类的默认值即可
    open, 保证 K15 的丰化梯度 (裸目录 → +CLAUDE.md → +L0 → 从 L1 实例化).
    """

    instruction_files: tuple[str, ...] = Field(
        default=("CLAUDE.md", "AGENTS.md"),
        description=(
            "视作场的 instruction 文档的文件名列表. Ground.instruction() "
            "按此列表在 root 和向上路径 (若 upward_lookup=True) 找到的文件, "
            "按 '根最先' 顺序拼接为 instruction 链. 消费行业名词, MOSS 不发明."
        ),
    )
    upward_lookup: bool = Field(
        default=True,
        description=(
            "是否从场目录向上递归收集 instruction 文件 (Claude Code 的 "
            "CLAUDE.md 链语义), 到 upward_boundary 为止."
        ),
    )
    upward_boundary: str | None = Field(
        default=None,
        description=(
            "向上递归的边界目录 (绝对路径). None 时使用 workspace root; "
            "场目录在边界外时只收本层. 相对路径不合法."
        ),
    )
    tree_depth: int = Field(
        default=2,
        description="context 帧内目录树的展示深度. 0 = 不展示树.",
    )
    tree_ignore_extra: tuple[str, ...] = Field(
        default=(),
        description=(
            "tree 段过滤的额外条目 (在 built-in 集之上叠加). 每项按目录/文件 "
            "basename 匹配. built-in 覆盖 .git .venv __pycache__ node_modules "
            ".DS_Store .mypy_cache .pytest_cache .ruff_cache dist build . "
            "轻量语义, 不解析 .gitignore — 未来 pin bash 会承接完整过滤."
        ),
    )
    context_budget: int = Field(
        default=24_000,
        description=(
            "ground context 渲染字符数上限 (全场 pin 内容之和). 超预算不静默"
            "截断也不自动 unpin — 帧顶插入 changed-on-disk 同风格的报账"
            "警告行, 点名最大的几张 pin, 让模型自己决定撤掉什么. "
            "K20 撤销自动 LRU 的直接体现."
        ),
    )
    hint_children: bool = Field(
        default=True,
        description=(
            "是否在 context 帧内标记子目录中存在的 instruction 文件 "
            "(向下法, 不自动加载). 命中会作为 hint 出现在帧里, 由模型"
            "决定要不要 open 那个子场或 pin 那个文件."
        ),
    )
    template: str | None = Field(
        default=None,
        description=(
            "本场由哪个 L1 模板实例化而来 (血统键). 用于 K15 的 '模板→实例 "
            "stale' 手动升级: diff 本场 body 与现行模板可看出偏差. 手写"
            "场留空."
        ),
    )


# -- 贴纸 --

class Pin(BaseModel):
    """一枚 pin — 场里的地址 + 对账观察.

    K17: 钉的是**地址**不是快照. 表面不自动同步; mtime 触发 + hash 对账
    判定真伪变更, 帧内以 changed-on-disk 标记暴露, 需模型显式 update 承认.
    """

    addr: str = Field(
        description=(
            "pin 的地址. 三种语法:\n"
            "  - path            整份文件 (相对 Ground.root)\n"
            "  - path:80-140     行区间 (含端点)\n"
            "  - **/*.py         glob (监视命中集合, 不渲染文件内容)\n"
            "path 和行区间必须在 root 子树内; glob 相对 root 展开. "
            "编译器报错/grep -n/GitHub 行号/shell glob 四重预训练先验."
        ),
    )
    note: str = Field(
        default="",
        description="模型自己写在贴纸上的注记 (为什么 pin, 关注什么).",
    )
    pinned_at: float = Field(
        default_factory=time.time,
        description="首次 pin 的时间戳.",
    )
    seen_mtime: float | None = Field(
        default=None,
        description=(
            "path/行区间: 上次 pin 或 update 时文件的 mtime. glob: 上次"
            "命中集里最新一个文件的 mtime. None = 尚未观察过 (刚 pin)."
        ),
    )
    seen_hash: str | None = Field(
        default=None,
        description=(
            "path 全文: 文件内容 hash. 行区间: 该区间内容 hash. glob: "
            "命中路径列表排序后的 hash. mtime 触发 + hash 判定真伪变更 — "
            "文件动了但 pin 住的区间没动, 不打扰模型."
        ),
    )


# -- 对账结果 --

class UpdateResult(BaseModel):
    """update(addr) 的返回 — 变更摘要, 会作为 command result 通过 CTML
    `<result>` 机制入对话历史.

    尺寸有界: diff_preview 摘要写死上限 (impl 侧强制), 避免历史洪泛.
    这是 K17 说的 "既有 drain 通道" — 不等 K19 原生 drain 协议立项.
    """

    addr: str = Field(description="被 update 的 pin 地址.")
    changed: bool = Field(
        description=(
            "内容是否真的变了 (hash 判定). False 时 mtime 可能变了但内容"
            "没变 (touch, git checkout 同版本等)."
        ),
    )
    old_mtime: float | None = Field(
        default=None,
        description="update 前的 seen_mtime.",
    )
    new_mtime: float | None = Field(
        default=None,
        description="update 后的 seen_mtime (当前观察到的).",
    )
    diff_preview: str = Field(
        default="",
        description=(
            "变更的人类可读摘要, 有界长度. 常见形态: 'lines +N -M', "
            "'glob: +2 -1 files', 'file recreated'. 不强制是 unified diff."
        ),
    )


# -- 异常 --

class GroundBaseError(Exception):
    """Ground 契约层异常基类."""


class PathOutsideRootError(GroundBaseError):
    """pin/update 的地址落在 Ground.root 子树之外. K12 空间边界零审批
    的字面兑现: 边界一次性设置, 边界内无审批, 边界外直接拒."""


class ContextBudgetExceeded(GroundBaseError):
    """ground 渲染超预算. 现阶段 impl 只在帧内报账, 不 raise; 保留这个异常
    是为将来严格模式 (raise 而非 warn) 留口."""


# -- 场 (子, 由 Grounds 治理) --

class Ground(ABC):
    """一个打开的场.

    绑定一个目录 root, 持有 pin 集合, 承担 CTML instruction / context 渲染
    与 load/sediment 生命周期. **模型的 CTML 接触面在父 Grounds 上**
    (K14: 父 ground channel 持全部动词, 场作 command-less virtual channel);
    Ground 上有完整动词是给 Grounds 转发用的, 也为未来 channel interface
    抽象 (N 场共享同一 interface 定义) 留升级空间, 属 K18 三层纪律红利.

    per-owner: Ground 由 Grounds.open 创建, 生命周期由 Grounds 管. 不要
    自己实例化.
    """

    @property
    @abstractmethod
    def label(self) -> str:
        """全局唯一的短标识, 也是 K14 virtual channel alias 的来源. Grounds
        分配, 缺省 = dir basename + 冲突后缀 (`contracts`, `contracts-2`).
        路径过长时作 fallback 显示不作 ref."""
        ...

    @property
    @abstractmethod
    def root(self) -> Path:
        """场的根目录 (绝对路径). 所有 pin 地址相对它解析,
        `cd` 类操作限制在其子树内."""
        ...

    @property
    @abstractmethod
    def convention(self) -> GroundConvention:
        """加载时消费的约定快照. open 后不可变 — 想改约定 = 关掉重开."""
        ...

    # -- pin 管理 --

    @abstractmethod
    def pins(self) -> list[Pin]:
        """当前 pin 集合. 顺序: 最近 pin/update 的在前 (最有可能被引用)."""
        ...

    @abstractmethod
    def pin(self, addr: str, note: str = "") -> Pin:
        """在场里钉一枚新 pin.

        addr 支持三种语法 (见 ``Pin.addr``). 幂等: 同地址重 pin 就是 update
        note + pinned_at, seen_mtime/hash 立即观察一次. 地址越界 (指向 root
        子树外) 抛 PathOutsideRootError.
        """
        ...

    @abstractmethod
    def unpin(self, addr: str) -> None:
        """撤掉一枚 pin. 地址不存在抛 KeyError (dict 语义, 见
        subprocesses.py `unpin` 同姿态)."""
        ...

    # -- 对账 --

    @abstractmethod
    async def update(self, addr: str) -> UpdateResult:
        """承认这枚 pin 的当前世界状态 — 重新观察 mtime + hash, 把 seen_*
        推到当前.

        **update 不是"检查变更"** (检查在每帧渲染 context 时自动做, 结果以
        changed-on-disk 标记出现在帧里). update 是 "我承认这次变更进入我的
        认知" 的第一人称动词. 承认之后, 下一帧该 pin 不再带 stale 标记.

        返回 UpdateResult 会通过 CTML `<result>` 机制自然入对话历史,
        作为差分沉淀 (K17). 未 update 的 stale pin 一直挂着 stale 标记
        直到被 update 或 unpin — 是主动纪律, 不是自动机制.

        地址不存在抛 KeyError; 地址越界抛 PathOutsideRootError.
        """
        ...

    # -- 渲染 (K14 virtual channel 消费) --

    @abstractmethod
    def instruction(self) -> str:
        """按 convention.instruction_files 收集 instruction 文档链, 按
        '根最先' 顺序拼接. 消费给 K14 装配的 virtual channel 的 instruction
        槽 — 落在 <moss_dynamic> 里 (virtual 节点不进 <moss_static>, 见
        core/ctml/v1_0/prompts.py:105).

        **同步 + 内部缓存**: 首次挂载时 (Ground.load) 计算并缓存, 之后
        instruction() 返回缓存字符串. 每帧刷不刷不影响 CTML 语义 —
        选择不刷是缓存命中率考量: 场的法链动辄几 KB (向上 CLAUDE.md 链
        拼接), 每帧刷相同内容不划算.

        本方法应无 IO / 无并发风险 — impl 违反此声明是 bug. 世界变化
        (法文件被外部修改) 走显式 refresh_instruction() 承认, 与 Pin 的
        update() 对账语义同源.

        向上: 若 convention.upward_lookup, 从 root 向 upward_boundary 逐层
        收集. 向下: 子目录中的 instruction 文件不自动加载 (会太重), 靠
        context() 里的 hint_children 暴露给模型.
        """
        ...

    @abstractmethod
    async def refresh_instruction(self) -> None:
        """重读法链文件, 刷新 instruction 缓存. 幂等.

        与 Pin.update() 同姿态 — 世界变化不自动追认, 承认前上一帧的
        instruction 保持不变. 场的 CLAUDE.md 被外部修改后, 模型需要通过
        这个动词显式承认.

        Impl 侧: 走 IO, 允许 asyncio.gather 并行读多个法文件.
        """
        ...

    @abstractmethod
    async def context(self) -> str:
        """渲染场的当前帧 — 消费给 K14 virtual channel 的 context_messages.

        典型结构 (impl 侧可调整具体格式):

            ground: <label> @ <root>   pins <n>  预算 <p>%
            tree(<depth>):
              ... 目录树 ...
            ⚖ src/foo/CLAUDE.md (instruction, 未加载)   # 若 hint_children

            ── pin: <addr>   [changed on disk]?
               note: <note>
               <内容渲染: 全文 / 行区间 / glob 命中清单>

        变更标记按 K17: mtime 触发 + hash 对账真伪; hash 未变不打扰模型.
        超预算按 K20/GroundConvention.context_budget: 帧顶插报账警告, 点名
        最大 pin, 不自动淘汰. glob pin 只渲染命中清单 (path + mtime + size,
        时间倒序), 不渲染文件内容 — 想看内容, 把 glob 转成 path pin.

        **async 契约**: 本方法 IO 密集 (stat 全部 pin + 读 pin 内容 +
        hash 对账). Impl 应用 asyncio.gather 并行 stat 与文件读取, 不得
        串行. 违反此声明是性能 bug.
        """
        ...

    # -- 生命周期 (K14 startup/close 消费) --

    @abstractmethod
    async def load(self) -> None:
        """从场里的 L0 文件恢复上次 sediment 的 pin 集. 无 L0 文件 = 空
        pin 集. 幂等. K14 装配时挂在 virtual channel 的 startup 上; CLI 侧
        (bash landing) 在 `moss ground <path>` 打开会话时手动调用."""
        ...

    @abstractmethod
    async def sediment(self) -> None:
        """把当前 pin 集写回 L0 文件的 pin 段落. 只动 pin 段, 不动
        frontmatter 也不动治理 body — 避免每次 close 产生 git diff 噪音
        (K20).

        胶囊晋升 (K20 提到的 pin → body promote) 是**独立动作**, 由模型
        显式发起, 不在本方法内自动做. sediment 只固化"目光"层, promote
        才把它变成"胶囊".

        K14 挂在 virtual channel 的 close 上; CLI 侧在会话结束或显式
        `moss ground close <label>` 时调用. 幂等.
        """
        ...


# -- 桌子 (父, owner-scoped) --

class Grounds(ABC):
    """Ghost 的认知桌子 — 一组打开的 Ground 的容器. per-owner.

    模型的 CTML 接触面: 父 ground channel 上的动词 `open` / `close` /
    `pin` / `unpin` / `update` 收 `label` 参数, core 层薄薄转发到
    `active[label].方法(...)` (K21 对齐). Grounds 自身不持久 `active` 列表
    — 那是纯 session 状态, 下次 session 由模型重新 open. 每个 Ground 自负
    pin 集持久化 (Ground.load/sediment).

    per-owner: IoC 非单例工厂, owner 生命周期 (`async with`) 划 Ground 的
    有效期. owner 关闭时全部 Ground close (sediment 落盘). 与 subprocesses /
    job_supervisor 同姿态 — 见那里的 UU-1.3 命名理由与 owner 生命周期铁律.

    使用方式::

        async with GroundsImpl(workspace_root=...) as gs:
            ground = await gs.open("./src/ghoshell_moss/contracts")
            gs.pin(ground.label, "contract.py:1-50", note="ABC 主体")
            print(await gs.frame(ground.label))
            # ... session 内继续 ...
        # __aexit__ 自动 sediment 全部 ground
    """

    # -- open/close --

    @abstractmethod
    async def open(
            self,
            dir: str | Path,
            *,
            label: str | None = None,
            convention: GroundConvention | None = None,
    ) -> Ground:
        """把一个目录作为场打开.

        - dir: 场根目录. 相对路径按 owner 的 workspace_root 解析.
        - label: 场的短标识, 全局唯一. None = 从 dir basename 生成, 冲突
          时加 `-2` / `-3` 后缀.
        - convention: 显式覆盖. None = 从 dir 里的 L0 文件 frontmatter 加载,
          无 L0 文件用 GroundConvention() 默认值.

        **同目录幂等**: 若 dir 已被 open (按 root 绝对路径规范化后比较),
        返回已 active 的 Ground 实例, 忽略传入的 label/convention (以已
        active 的为准). 目录是认知单元 — 一个目录只有一份 pin 集与法链
        状态, 同 session 内多次 open 无异议看到同一份. 跨 session 持久化
        由 L0 文件承担, 不由 Grounds 记忆.

        副作用 (首次 open): 创建 Ground 实例, 登记入 `active()`, 调用
        `Ground.load()` 恢复上次 pin 集. K14 装配时 impl 会再把 Ground
        挂载为 virtual channel (那是 channel 层的事, contracts 不知道).
        """
        ...

    @abstractmethod
    async def close(self, label: str) -> None:
        """关掉一个场 — 触发 sediment 落盘, 从 `active()` 移除.

        label 不存在抛 KeyError.
        """
        ...

    # -- 查询 --

    @abstractmethod
    def active(self) -> dict[str, Ground]:
        """当前打开的全部场. key = label."""
        ...

    @abstractmethod
    def get(self, label: str) -> Ground | None:
        """按 label 取场句柄. 不存在返回 None (查询语义, 与 close 区分)."""
        ...

    # -- 转发 (CTML 接触面) --

    @abstractmethod
    def pin(self, label: str, addr: str, note: str = "") -> Pin:
        """转发到 `active()[label].pin(addr, note)`. label 不存在抛
        KeyError."""
        ...

    @abstractmethod
    def unpin(self, label: str, addr: str) -> None:
        """转发到 `active()[label].unpin(addr)`."""
        ...

    @abstractmethod
    async def update(self, label: str, addr: str) -> UpdateResult:
        """转发到 `active()[label].update(addr)`."""
        ...

    @abstractmethod
    async def frame(self, label: str) -> str:
        """转发到 `active()[label].context()`. 命名: CTML/CLI 表面动词
        叫 `frame` (K16 骑 '当前视图' 先验), Ground 内部渲染方法叫
        `context` (K14 与 channel_builder 的 context_messages 同名)."""
        ...

    # -- 生命周期 --

    @abstractmethod
    async def __aenter__(self) -> Self:
        """启动 Grounds."""
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """关闭 Grounds. 对全部 `active()` 触发 sediment 后清空."""
        ...

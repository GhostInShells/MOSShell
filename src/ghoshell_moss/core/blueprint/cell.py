"""
MOSS 通讯网络中的单元节点 (Cell).

Cell 的一句话承诺: 一个通过 channel 向模型世界暴露自己的、有生命周期的进程.
channel 是 cell 的膜 — 模型感知和交互一个 cell 的唯一界面.
不承诺 channel 的进程不是 cell, 其归宿是 Subprocesses 治理下的裸子进程.

Cell 不是一个运行时对象, 而是三个真相域各自的数据模型:

- CellManifest — 文件真相 (inventory): CELL.md 声明, "领地里装了什么".
- CellRecord   — 所有权真相 (ledger): spawn 咽喉写下的进程账目, "我拉起过什么".
- CellPresence — 网络真相 (announce): 入网 cell 的膜与生命状态, "网络上活着什么".

治理动词按域分化, 每域两个: create/install → inventory, run/stop → ledger,
accept/deny → network. 其余一切 (list / status / logs) 都是三域的 join 视图,
不是治理.
"""
# -- 设计契约: .ai_partners/features/workstreams/2026/06/matrix-cell-governance/FEATURE.md
#    有效章节 = §TT/§TT续 + §UU + §VV + §WW. 本文件是 UU-12 分发块① 的产物.
# -- §UU-5 三域模型: God-model Cell (meta + launcher + status 复合体) 解体.
#    join 只发生在视图层 (CLI list 输出行), 内核无 God-model.
#    原 Cell 上的行为 (is_alive / write_runtime_file / launch_* / spawn 辅助)
#    全部离开数据模型, 归 run_cell 咽喉与 CLI 合流层 (UU-9).
# -- §UU-4 六动词治理代数: 三域 x 两动词, 代数封闭; 删任何一个动词,
#    模型自迭代循环断一处. 治理代数冻结, 膜的声明曲线 (announce payload) 可演进.
# -- TT-2 身份拆分 (uuid + alias) 未终审: address / alias 是可整体重命名的占位字段.
#    实现不得对 address 的内部结构做任何假设.
# -- 网络侧抽象 = Presence / Watcher (§UU-7, 原 CellNetwork 的拆分继任者):
#    入网与监听分离, N²→N. 原 CellNetwork / CellLog 已删除,
#    CellEvent 是 CellLog 的精简继任 (terminal 标志由 liveness 原语承载).

import sys
import time
from enum import Enum
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Literal
from typing_extensions import Self
from abc import ABC, abstractmethod

import fnmatch
import frontmatter
import shlex
from pydantic import BaseModel, Field

from ghoshell_moss.core.concepts.channel import Channel, ChannelProvider, ChannelProxy
from ghoshell_moss.message import unique_id

__all__ = [
    'CellAddress',
    'MatchPattern',
    'RelativePath',
    'normalize',
    'make_address',
    'CellState',
    'ExecSpec',
    'CellManifest',
    'CellRecord',
    'CellPresence',
    'CellEvent',
    'Presence',
    'Watcher',
    'CellRegistry',
    'DuplicatedError',
]

# -- 类型别名 -- #

CellAddress = str
"""cell 在网络上的唯一地址. 结构未定 (TT-2), 调用方一律作不透明字符串处理."""

RelativePath = str
MatchPattern = str
"""通配符模式: group/name, group/*, *, */*, */name"""


def make_address(*parts: str) -> str:
    """约定定义一个节点的 address. 使用 / 作为层级分隔符."""
    # TT-2 占位: address 的分段结构 (现为 type/name/uid 三段式) 等人类终审,
    # 本函数只承诺 "用 / 拼接", 不承诺段数与段含义.
    return '/'.join(parts)


def normalize(name_or_address: str) -> str:
    """将名称或 address 归一化为可作文件名 / python 标识符的形式."""
    return (name_or_address.replace('/', '_').replace('\\', '_').
            replace('.', '_').replace('-', '_'))


class CellState(str, Enum):
    """
    cell 的生命周期状态. 挂在 CellPresence 上, 随 announce 传播.

    - spawned: 进程已被拉起, 尚未入网.
    - ready:   announce 已到达, 膜可用.
    - dead:    进程退出或网络存活性丢失.
    """
    # -- TT-15 MVP 三态: ready = announce 到达即 ready (零新协议);
    #    dead = liveness 丢失或进程退出, 取先到者.
    #    STOPPING / draining / degraded / watchdog 推迟, 但 enum 现在就进契约,
    #    payload 后扩. 状态即数据 (与 JobSpec 同纪律).
    # -- 两个真相来源, fold 成一条事件流 (TT-15/WW-6):
    #    进程真相 (spawn/exit/kill) 由 Subprocesses done callback 免费可见, 即时且带
    #    exit code, owner 的 dead 信号源是它, 不走网络; 应用真相 (ready/未来 draining)
    #    只有子进程自己知道, 走 announce 自报告侧信道. 外部观察者只有网络真相.
    SPAWNED = 'spawned'
    READY = 'ready'
    DEAD = 'dead'


class ExecSpec(BaseModel):
    """
    cell 进程的启动声明: 一条命令, command + args + env (与 MCP client 配置同形).

    command 的解析规则 (execvp 语义):
    - 含路径分隔符: 相对 CELL.md 所在目录解析, 启动时立即绝对化.
    - 裸词: PATH 查找. spawner 会把自身 python 环境的 bin 目录插入 PATH 头,
      因此写 `python` 即使用 spawner 的解释器 (默认解释器).

    环境隔离是 cell 作者的显式选择, 写进 command 本身:
    如 `uv run main.py` 或 `../../.venv/bin/python -m my_pkg.main`.
    框架不做任何自动环境检测. 复杂安装需求由 INSTALL.md 承载.
    """
    # -- §WW-3 定型 (修正 UU-5 "字段不变"): argv 结构保留, 语义一个不留.
    #    interpreter 字段死 — 解释器是咽喉的解析产物, 不是声明的字段.
    #    行业勘察 (WW-1): ROS2 launch / Erlang release / MCP / k8s,
    #    活下来的 exec spec 没有一家带解释器字段. 病灶 = launcher ⊗ package 融合.
    # -- §WW-2 uv 判决 (2026-06-10 原判, 两次实证死亡, 封死): 咽喉不做任何
    #    内容嗅探式隐式分支 (含 "无 run: 自动探测 PEP723 → 自动 uv").
    #    自动检测是死路 — 开发者不知道检测是啥, 隐式声明零信息传递.
    #    uv 降级为作者显式选项, matrix 不自带 uv 依赖.
    # -- cwd 不在 spec 里 (§WW-3 双语义拆分): CELL.md 目录只作 command 解析基准,
    #    在咽喉活一瞬间; 进程 cwd = spawner runtime 子树 (治理概念, TT-6 边界做成环境).
    #    代码可达性由环境承载 (python -m 装好即从任意 cwd 可跑), 不由 cwd 承载.
    # -- 裸词 'python' 经 PATH 头注入解析 = 环境提供, 非关键字字符串替换
    #    (venv activate 同机制); 旧 interpreter=='python' 的关键字 hack 废.

    command: str = Field(
        description="启动命令 argv[0]. 相对路径基于 CELL.md 所在目录; "
                    "裸词走 PATH 查找, `python` 即 spawner 的解释器.",
    )
    args: list[str] = Field(
        default_factory=list,
        description="启动命令的参数列表.",
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="额外注入的环境变量.",
    )

    @classmethod
    def from_run(cls, run: 'str | list[str]', *, env: dict[str, str] | None = None) -> 'ExecSpec':
        """从 CELL.md frontmatter 的 `run:` 声明构造. 字符串按 shell 语法切分."""
        # `run:` 是 ExecSpec 的糖 (WW-3): string 经 shlex 成 argv; list 直接作 argv.
        argv = shlex.split(run) if isinstance(run, str) else list(run)
        if not argv:
            raise ValueError('run declaration is empty')
        return cls(command=argv[0], args=argv[1:], env=env or {})

    def to_run(self) -> str:
        """还原为 `run:` 字符串形式."""
        return shlex.join([self.command, *self.args])


class CellManifest(BaseModel):
    """
    CELL.md 声明文件的内容载体 — inventory 域 (文件真相) 的数据模型.

    一句话承诺: 把一个可执行物声明为本治理域可拉起的 cell.
    CELL.md 的存在边界 = 包自述回答不了的地方 (启动组合 / 非 python 命令 /
    远程代码本地注册). 包自述够用处, 裸脚本可经反射匝道零声明运行.
    """
    # -- §UU-5: CellMetadata 溶解进来, God-model Cell 解体后 manifest 独立成域.
    # -- type 三重身份各回各家 (§UU-5): 拓扑角色 = 运行时事实, 进 Presence;
    #    project 归属 = project_id 标签挂 announce; 治理路径 = ledger 条目的存在本身.
    #    这里只剩 taxonomy 纯分类标签, 不驱动任何机制.
    # -- TT-13: exec spec 是地基 (systemd ExecStart 原理), PEP 723 是匝道.
    #    cell = package 无关的治理域注册快捷方式 (.desktop 系), 包真相归语言生态,
    #    MOSS 只在咽喉读证据 (WW-1).
    # -- WW-4 向上认亲: cell = 身份锚 (CELL.md), 脚本 = 入口变体.
    #    发现面 1:1 (`run:` 唯一默认入口), 目录事实 N (任意脚本向上认亲同一身份).
    #    发现面从不承诺穷举可执行物 (.desktop 同).
    # -- `entrypoints:` 可选文档化列表 (WW-4) 待拍板, 未加.

    MANIFEST_FILENAME: ClassVar[str] = 'CELL.md'
    INSTALL_FILENAME: ClassVar[str] = 'INSTALL.md'
    INSTALLED_FILE: ClassVar[str] = '.installed'

    name: str = Field(
        description="cell 的名字. 治理域内的身份锚.",
    )
    description: str = Field(
        default='',
        description="cell 的一句话描述.",
    )
    taxonomy: str = Field(
        default='',
        description="纯分类标签 (如 sensors / bodies / tools), 自由命名, 不驱动任何机制.",
    )
    singleton: Literal['none', 'domain', 'host'] = Field(
        default='none',
        description="单例约束声明. none=可多开; domain=同一治理域内仅一实例; "
                    "host=机器级硬件单点 (如机器人控制), 跨治理域互斥.",
    )
    # -- TT-12 singleton = 风险锚点: 用最小声明显式锚定风险, 第二实例被拒时的
    #    错误信息引用声明原文, 即 code as prompt 在错误路径上的延伸.
    #    domain 档在 run_cell 咽喉处查重即拒 (WW-4: 框架不做真锁);
    #    host 档靠启动时 flock 约定路径 (文件真相, 进程死自动释放, 无 stale).
    # -- WW-4: singleton 域 = cell 身份 — 不论从哪个入口脚本拉起,
    #    都是同一 cell 的实例. 一仓多组合互斥由此免费解决.
    exec: 'ExecSpec | None' = Field(
        default=None,
        description="默认启动入口 (frontmatter `run:` 声明). "
                    "无声明的 cell 只能以显式脚本路径拉起.",
    )
    instruction: str = Field(
        default='',
        description="cell 的使用说明 (CELL.md 正文). run 动作成功后应回执给调用方.",
    )
    installed: bool = Field(
        default=True,
        description="是否已完成安装. 未安装的 cell 可被发现但拒绝拉起, "
                    "错误信息会给出 INSTALL.md 路径. 由文件系统推导, 不在 frontmatter 中.",
    )
    # -- installed 推导规则: 目录下存在 INSTALL.md 时, 看 .installed 空文件是否存在;
    #    无 INSTALL.md 即视为已安装. install 是六动词之一 (WW-5 故事 3):
    #    不可见则无作用对象, 自迭代循环断在第二步, 所以未安装也必须可发现.

    # -- 文件读写 (inventory 域自身的行为, 不涉进程与网络) -- #

    @classmethod
    def read_from_file(cls, file: Path) -> 'CellManifest':
        """从 CELL.md 文件读取声明. 正文即 instruction, frontmatter 即字段."""
        content = file.read_text(encoding='utf-8')
        post = frontmatter.loads(content)
        data = dict(post.metadata)
        run = data.pop('run', None)
        env = data.pop('env', None)
        if run is not None:
            data['exec'] = ExecSpec.from_run(run, env=env)
        data['instruction'] = post.content.strip()

        directory = file.parent
        if directory.joinpath(cls.INSTALL_FILENAME).exists():
            data['installed'] = directory.joinpath(cls.INSTALLED_FILE).exists()
        else:
            data['installed'] = True
        return cls(**data)

    @classmethod
    def read_from_directory(cls, directory: Path) -> 'CellManifest | None':
        file = directory.joinpath(cls.MANIFEST_FILENAME)
        if file.is_file():
            return cls.read_from_file(file)
        return None

    def write_file(self, directory: Path, filename: str = '') -> None:
        """将声明写入 CELL.md (exec 还原为 `run:` 糖, installed 不写入)."""
        filename = filename or self.MANIFEST_FILENAME
        flat = self.model_dump(
            exclude_defaults=True, exclude_none=True,
            exclude={'instruction', 'installed', 'exec'},
        )
        if self.exec is not None:
            flat['run'] = self.exec.to_run()
            if self.exec.env:
                flat['env'] = dict(self.exec.env)
        post = frontmatter.Post(content=self.instruction, **flat)
        frontmatter.dump(post, directory.joinpath(filename).resolve())

    # -- 反射匝道 (Tier 1): 裸脚本 / 运行中进程 → 临时 Manifest, 进同一条咽喉 -- #

    @classmethod
    def find_upward(cls, start: Path) -> 'CellManifest | None':
        """从 start 出发向上查找最近的 CELL.md (找到第一个即停)."""
        # WW-4 向上认亲, 同 MOSS.md 的发现规则.
        directory = start if start.is_dir() else start.parent
        for candidate in [directory, *directory.parents]:
            manifest = cls.read_from_directory(candidate)
            if manifest is not None:
                return manifest
        return None

    @classmethod
    def from_script(cls, script: Path) -> 'CellManifest':
        """
        以脚本为入口构造 Manifest: 向上认亲最近的 CELL.md;
        找不到时降级为临时身份, 不拒绝运行.
        """
        # WW-4: 认亲成功 → 同一 cell 身份的入口变体 (exec 不覆写, 脚本作显式入口
        # 由咽喉处理); 认亲失败 → 身份降级 script/{uuid} — 具体降级命名属
        # TT-2 身份占位, 人类终审后统一改.
        script = script.resolve()
        found = cls.find_upward(script)
        if found is not None:
            return found
        return cls(
            name=f'{script.stem}_{unique_id()[:8]}',
            taxonomy='script',
            description=f'ad-hoc cell from {script}',
            exec=ExecSpec(command='python', args=[str(script)]),
        )

    @classmethod
    def from_proc(cls) -> 'CellManifest':
        """从当前进程自述身份: 以 __main__ 脚本向上认亲, 找不到则降级临时身份."""
        from importlib import import_module
        main = import_module('__main__')
        script_file = Path(getattr(main, '__file__', '') or '')
        if script_file.is_file():
            manifest = cls.from_script(script_file)
            # 运行中进程的真实启动参数比声明更准确.
            manifest.exec = ExecSpec(command=sys.executable, args=list(sys.argv))
            if not manifest.description:
                docstring = main.__doc__ or ''
                manifest.description = docstring.splitlines()[0] if docstring else ''
            return manifest
        return cls(
            name=f'proc_{unique_id()[:8]}',
            taxonomy='script',
            exec=ExecSpec(command=sys.executable, args=list(sys.argv)),
        )


class CellRecord(BaseModel):
    """
    ledger 域 (所有权真相) 的数据模型 — spawn 咽喉在拉起进程的瞬间写下的一条账目.

    包含 owner 运维面信息: 只有能对该进程直接行动的一侧 (owner / 本机 CLI)
    才应消费这些字段.
    """
    # -- §UU-6 ledger 仲裁: 咽喉的排气尾迹, 不是运行时的输入. 两条规则:
    #    1. 咽喉写 — run_cell spawn 瞬间 append 一条 JSON, best-effort, 不回读.
    #       单写者原则: pid/start_time 只有 spawn 现场知道.
    #    2. CLI 是唯一读者 — moss cells list/status/kill = 读 ledger + join 网络真相
    #       + killpg. 冷数据, 按需读, 零监听.
    #    运行时零读零监听: Matrix 上没有 ledger 成员; Matrix 体系内治理 =
    #    Subprocesses 全权 (owner 内存态注册表即权威所有权记录).
    #    ledger 的存在理由 = fencing 失效时的法证清理 (host 挂死/孤儿时 kill 的依据).
    # -- §WW-6: 不记录 exit — 死进程无孤儿可杀, 法证理由不存在,
    #    单写者原则不开第二个写入时机.
    # -- §WW-7 全部结论中最易漂移的一条: 模型上下文 (context messages) 的数据源 =
    #    Subprocesses 内存句柄 + Watcher 视图的 join, 永不读 ledger.
    # -- §UU-3 可行动性判据: pid / 日志路径, 远端模型拿到什么都做不了,
    #    所以这些字段永不上 announce, 只活在这里.
    # -- ledger 无对象身份 (UU-6): 一个 workspace 目录约定 + 本 schema + 上述两条规则.
    #    不设 Ledger 类.

    address: CellAddress = Field(
        description="cell 的网络地址.",
    )
    alias: str = Field(
        default='',
        description="治理域内的别名.",
    )
    pid: int = Field(
        description="进程 id.",
    )
    pgid: int = Field(
        default=0,
        description="进程组 id (start_new_session 后即进程自身的组). killpg 的作用对象.",
    )
    start_time: float = Field(
        description="进程启动时间戳. 与 pid 一起构成防 pid 复用的核对依据.",
    )
    project_id: str = Field(
        default='',
        description="拉起该 cell 的治理域 (project) 标识.",
    )
    cwd: str = Field(
        description="进程工作目录, 绝对路径.",
    )
    # UU-10 纪律: 相对路径只活在 API 边界一瞬间, 咽喉以下 (含 ledger) 只存在绝对路径.
    stdout_log: str = Field(
        default='',
        description="stdout 日志文件绝对路径, 空表示未重定向.",
    )
    stderr_log: str = Field(
        default='',
        description="stderr 日志文件绝对路径, 空表示未重定向.",
    )
    spawner: str = Field(
        default='',
        description="拉起者标识 (host cell 的 address, 或 'cli'). owner 归属线索.",
    )


class CellPresence(BaseModel):
    """
    network 域 (网络真相) 的数据模型 — 一个入网 cell 的 announce payload.

    远端对一个 cell 的全部认知来自这里: 膜 (channel 接口描述) + 生命状态.
    收到 presence 即可判断: 它提供什么能力 (channel_interface)、
    现在能不能用 (state / failure)、要不要接纳它的膜 (accept / deny).
    """
    # -- §UU-3 可行动性判据: 消费者能对这条信息采取行动, 它才上 announce.
    #    degraded / failure 摘要可行动 (不路由 / 通知 owner) → 在这里;
    #    pid / 日志路径不可行动 → 永不在这里 (归 CellRecord).
    #    行业同构: k8s API server 只有 conditions, MAINPID 归 kubelet/init 自己.
    # -- §UU-11 膜承诺的关键推论: payload 必须携带 channel 接口描述 — 否则模型要
    #    先 proxy 连上才知道对方提供什么, 自迭代循环断在第一步.
    #    接口描述全文 vs 摘要+按需 query 未定, 属分发级细节, 字段先占 str.
    # -- host 角色 = 运行时事实 (抢到 listen 端口者, UU-1.2), 不是 CELL.md 声明.
    #    原 HOST_TYPE 常量废除, is_host 的真相载体即本字段.
    # -- 未来演进 (§UU-2): resources / 上下文变量等膜上运输类型只扩展本 payload
    #    (membrane transport), 给治理面加零个动词. 膜可以变重, 治理不许变重.
    # -- 命名沿 XMPP presence 先例 (UU-5).

    address: CellAddress = Field(
        description="cell 的网络地址.",
    )
    alias: str = Field(
        default='',
        description="治理域内的别名, 模型可读的称呼.",
    )
    state: CellState = Field(
        default=CellState.READY,
        description="生命周期状态. announce 到达即 ready.",
    )
    failure: str = Field(
        default='',
        description="故障摘要. 非空表示 cell 自报告了可行动的异常.",
    )
    project_id: str = Field(
        default='',
        description="cell 所属治理域 (project) 标识. 软分组标签, 供视图过滤.",
    )
    # UU-1.10: 本地/远端分离用 project_id 数据标签 + 视图过滤,
    # 不做 namespace 原生切分 (硬隔离用 --scope 原语).
    is_host: bool = Field(
        default=False,
        description="是否是当前网络的 host (运行时事实).",
    )
    channel_interface: str = Field(
        default='',
        description="膜: 该 cell 提供的 channel 接口描述. "
                    "模型据此决定是否 accept 它的能力.",
    )
    updated: float = Field(
        default_factory=time.time,
        description="本 presence 最后更新的时间戳.",
    )


class CellEvent(BaseModel):
    """
    网络上的轻量事件: 一个 cell 广播的状态跃迁或异常摘要.

    事件是传播载体, 不是状态本身 — 状态的真相载体是 CellPresence.
    接收方收到事件后按需重新查询 presence, 不从事件里读状态.
    """
    # -- 原 CellLog 的精简继任. terminal 标志删除: "cell 没了" 的语义由
    #    liveness DELETE 承载 (网络原语), 不需要事件层重复声明.
    # -- 存在理由 (UU-3): failure/degraded 摘要是远端可行动信息, 需要一个
    #    announce 面的传播载体; presence 重宣告只更新 queryable, 不推送.
    # -- WW-5 故事 7: 主动性全部归 signal (mindflow 仲裁), 网络事件只是原料.
    #    事件 → signal 的转换发生在消费侧 (Watcher 的持有者), 不在这里.

    address: CellAddress = Field(
        description="事件来源 cell 的 address.",
    )
    content: str = Field(
        default='',
        description="自由文本事件内容.",
    )
    timestamp: float = Field(
        default_factory=time.time,
        description="事件产生时刻.",
    )


class Presence(ABC):
    """
    本 cell 的入网侧: 让自己在网络上可被发现、可被查询、可提供 channel.

    每个入网 cell 持有一个. 只管 "我在网络上如何存在", 不观察别人 —
    观察是 Watcher 的事.
    """
    # -- §UU-7 拆分: 原 CellNetwork 融合了入网 (O(1) 被动: queryable + liveness
    #    token + publisher) 与监听 (O(N) 主动: subscriber + cache + reconcile).
    #    allow_create_proxy 布尔角色开关是融合的供词 (TT-1 检验失败形态).
    #    拆开后 worker 只跑 Presence, 成本 N²→N (k8s 同构: kubelet 只注册,
    #    informer 是控制器按需开的).
    # -- debug 问责单一性 (UU-7): "别人看不见我" → 审讯 Presence.
    # -- check_unique 无继任 (TT-2): check-then-announce 竞态是被取消的问题.
    #    domain 档单例查重在 run_cell 咽喉 (owner 内存态), host 档靠 flock.
    # -- 与 CellPresence (payload 数据模型) 的撞名保留给人类 IDE 改名权衡:
    #    一个是入网机制对象, 一个是宣告数据. 二者一一对应.

    @property
    @abstractmethod
    def this(self) -> CellPresence:
        """本 cell 当前宣告的 presence 内容."""
        pass

    @abstractmethod
    async def announce(self, presence: CellPresence) -> None:
        """
        宣告或更新本 cell 的 presence. 首次调用即入网 (liveness 上线).

        :raise DuplicatedError: 网络层确知地址冲突时 (尽力而为, 不承诺强一致).
        """
        pass

    @abstractmethod
    async def revoke(self) -> None:
        """主动下线: 撤回 liveness 与 queryable. 进程退出时的优雅路径."""
        pass

    @abstractmethod
    async def provide(self, channel: Channel) -> ChannelProvider:
        """
        把 channel 作为本 cell 的膜暴露到网络上.

        膜是 cell 在模型能力空间里存在的前提: 不提供 channel 的进程
        不是 cell (它应该由 Subprocesses 治理).
        """
        # 膜承诺 (UU-2) 的机制面. announce payload 的 channel_interface 字段
        # 应在 provide 后由实现回填 — 接口描述随 presence 传播 (UU-11),
        # 模型不必先连 proxy 才知道对方提供什么.
        pass

    @abstractmethod
    async def publish_event(self, content: str) -> None:
        """向网络广播一个本 cell 的轻量事件 (CellEvent)."""
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class Watcher(ABC):
    """
    对网络的观察侧: 维护网络上所有 cell presence 的延迟视图, 并按需接纳远端的膜.

    opt-in: 只有需要观察网络的消费者 (host / ghost runtime) 才创建,
    每个 runtime 至多一个. 纯 worker cell 不需要它.

    延迟视图承诺: 只有 online/offline 边缘是实时的 (liveness 推送),
    presence 内容可能滞后到下次 refresh. 要实时内容就 refresh(address).
    """
    # -- §UU-7: subscriber + cache + reconcile 归这里. 每 runtime 至多一个
    #    (shared informer 同构). "我看不见 X" → 审讯 Watcher.
    # -- §NN 二元真相在此正式化: 延迟视图是承诺不是缺陷, 消费方 (WW-5 故事 7:
    #    network channel command 全被动读视图) 接受滞后.
    # -- accept/deny (网络域两动词, UU-4) 放在这里: Watcher 是消费者侧对象,
    #    proxy 的 owner 就是 accept 者 (UU-8), 本地 dict 查重零网络往返.
    #    deny 的 v1 实现 = 不 accept / release. matrix.network(local) 双视图
    #    是本对象之上的过滤投影 (UU-10), 不是第二个 Watcher.
    # -- get_host/all_hosts 无继任: host 是 presence 上的运行时事实字段,
    #    视图过滤即得, 不值一个专门方法.

    @abstractmethod
    def view(
            self,
            *,
            project_id: str | None = None,
            state: CellState | None = None,
    ) -> dict[CellAddress, CellPresence]:
        """
        当前延迟视图 (零等待, 读 cache).

        :param project_id: 仅返回指定治理域的 cell (local/foreign 过滤的原料).
        :param state: 仅返回指定状态的 cell.
        """
        pass

    @abstractmethod
    async def refresh(self, address: CellAddress | None = None) -> dict[CellAddress, CellPresence]:
        """主动对账: 拉取指定 cell (None=全量) 的最新 presence 并更新视图."""
        pass

    @abstractmethod
    def on_change(
            self,
            callback: Callable[[CellPresence, bool], None],
    ) -> Callable[[], None]:
        """
        注册 (presence, online) 变更回调, 返回 unsubscribe 函数.

        回调可能在网络后台线程触发, 调用方负责线程安全.
        """
        # 事件 → mindflow signal 的转换点: run_cell 的调用方在这里
        # 把 ready/dead 跃迁接进注意力仲裁 (WW-5 故事 4/5, 四弧全部经此).
        pass

    @abstractmethod
    def recent_events(self, *, limit: int = 20) -> list[CellEvent]:
        """最近的网络轻量事件窗口 (ring buffer, 最新优先)."""
        pass

    @abstractmethod
    async def wait_present(
            self,
            address: CellAddress,
            *,
            timeout: float = 30,
    ) -> CellPresence | None:
        """
        等待某个 cell 的 presence 出现 (ready).

        程序化场景专用 (host bringup / run_cell(wait=...)).
        模型面不 wait — 生命周期跃迁作 signal 进 mindflow (WW-5).

        :return: presence, 或超时 None.
        """
        pass

    # -- 网络域治理动词: accept / deny (UU-4) -- #

    @abstractmethod
    async def accept(self, address: CellAddress) -> ChannelProxy:
        """
        承认一个远端 cell 的膜: 创建 (或返回已有的) channel proxy.

        proxy 的 owner 是本 Watcher 的持有者 — owner 关闭即释放.
        同一 address 重复 accept 返回同一个 proxy (本地查重).

        :raise LookupError: address 不在网络上.
        """
        # UU-8: 急切 auto-proxy 已删除 — 那等于自动 accept 全网络,
        # 把 accept 动词从治理面偷走塞给机制层. accept 即创建是唯一构建路径.
        pass

    @abstractmethod
    async def release(self, address: CellAddress) -> None:
        """
        撤回对一个膜的承认: 关闭并释放 proxy. 幂等.

        deny 的 v1 语义 = 不 accept, 或 accept 后 release (UU-4).
        """
        pass

    @abstractmethod
    def accepted(self) -> dict[CellAddress, ChannelProxy]:
        """当前已 accept 的全部 proxy."""
        pass

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass


class CellRegistry(ABC):
    """
    inventory 域的只读发现入口: 扫描治理域领地内的 CELL.md 声明.

    只回答 "领地里装了什么", 不拉起、不杀灭、不持有任何运行时状态.
    """
    # -- TT-11: registry 退化为 glob(CELL.md), 与 features 套件同构.
    #    发现 (inventory, 有界) 与 spawn 能力 (从哪都能拉, 无远弗届) 是两个问题, 不熔.
    #    原 spawn_cell / kill_all_runtime_cells / runtime file 读写 / dump_spawn_env
    #    全部移除: spawn 归 run_cell 咽喉 (UU-10), kill 归 CLI (ledger 唯一读者, UU-6).
    # -- 挂载位置 = project.cells, 不在 Matrix 上 (UU-10).

    @abstractmethod
    def list_cell_manifests(
            self,
            refresh: bool = True,
            *,
            installed: bool | None = None,
            include: list[MatchPattern] | None = None,
            exclude: list[MatchPattern] | None = None,
    ) -> dict[RelativePath, CellManifest]:
        """
        列出领地内发现的全部 Cell 声明.
        :param refresh: 重新扫描文件系统.
        :param installed: None=全部; True=仅已安装; False=仅未安装.
        :param include: 匹配模式筛选.
        :param exclude: 排除模式筛选.
        """
        pass

    @abstractmethod
    def get_cell_manifest(self, relative_path: 'str | Path') -> 'CellManifest | None':
        """获取指定目录路径的 Cell 声明. 目录路径用 '/' 分割."""
        pass

    @staticmethod
    def match_cells(
            cells: dict[RelativePath, CellManifest],
            include: list[MatchPattern] | None = None,
            *,
            exclude: list[MatchPattern] | None = None,
    ) -> Iterable[tuple[RelativePath, CellManifest]]:
        """基于 fnmatch 通配符筛选 Cell. include 为空时返回全部 (仅受 exclude 约束)."""
        include_patterns = set(include) if include else set()
        exclude_patterns = set(exclude or [])

        for relative_path, cell in cells.items():
            if include_patterns:
                if not any(fnmatch.fnmatch(relative_path, p) for p in include_patterns):
                    continue
            if exclude_patterns:
                if any(fnmatch.fnmatch(relative_path, p) for p in exclude_patterns):
                    continue
            yield relative_path, cell


class DuplicatedError(RuntimeError):
    """cell 重复启动异常. singleton 声明的执法产物, 错误信息应引用声明原文."""
    # TT-12: 第二实例被拒时的错误信息本身就是 prompt
    # ("g1 声明了硬件单点, 地址 X 已有活实例"), code as prompt 在错误路径上的延伸.

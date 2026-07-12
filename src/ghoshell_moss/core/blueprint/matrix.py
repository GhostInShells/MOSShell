"""
MOSS 通讯矩阵在单个 cell 进程内的投影.

Matrix 的一句话承诺: 每个 cell 进程持有一个 Matrix 实例, 它把 cell 所在的
整个世界投影成进程内部的一个对象 — cell 经由它持有身份、暴露膜、观察网络、
拉起并治理新的进程.

命名的哲学锚点 (2026-07-12 人类明示): Matrix 实例是 "整体在局部的投影" —
洞穴之光投影出蜂巢的形状. 人类语言经常用投影指代实体 (指着屏幕里的代码流
叫 matrix), 这种指代等价具有哲学实在性. Matrix 不是 mesh, 也不是 mesh 的
客户端 — mesh 只是它投影的来源之一.
"""
# -- 设计契约: .ai_partners/features/workstreams/2026/06/matrix-cell-governance/FEATURE.md
#    有效章节 = §TT/§TT续 + §UU + §VV + §WW + §XX (+ 本轮 §YY).
#    本文件是 VV-2 步骤 6 (重绘 matrix) 的产物, 表面积按 UU-10 + TT-6 收敛.
# -- 表面积纪律 (TT-6 分灶台): 首页只留一打成员, 其余进门后.
#    30 成员摊平一个平面时 "厨房水槽" 批评成立; 解法是分灶台, 不是拆房间.
# -- 2026-07-12 本轮仲裁 (人类 + fable, 详见 FEATURE.md §YY):
#    * session 是 Matrix 最重要的原件, 永不出首页, 只有 API 上升的可能.
#    * mesh() 是惰性门 — opt-in by usage, 不是 API 门禁 flag
#      (旧 only_allowed_in_host_cell 布尔门禁是 TT-6 批评的 "API 形式边界", 废).
#    * network 属性暴露 NetworkMetadata 做运行时自解释 — cell 定义时不知道
#      自己会被接进哪个网络, 运行时暴露配置即可编程.
#    * get_runtime_url 的 URL 承诺是认真的: 未来 scheme://cell_address/scoped/path
#      形式可解引用. 现在交付 path 形式.
#    * home 双目录判决: 持久领地键 = 稳定身份 (无泄漏, 永不自动清);
#      实例残迹键 = address(含 uid), 咽喉在下次 spawn 时按保留策略修剪 —
#      crash 现场留到下次 spawn 才清, 最后一次 crash 永远可查.
#    * is_host_running 保留 (cell 侧 code as prompt); 未来候选 wait_host_running.
# -- 旧表面搬家地图 (wire-up 阶段按此改调用方):
#    spawn                     → processes.execute (Subprocesses)
#    channel_proxy             → (await mesh()).accept(address)
#    cells                     → project.cells
#    storages()/ghost_home/mode_home/network_home/cell_workspace/...
#                              → workspace 门后自行组合 (home 是唯一首页幸存者)
#    get_runtime_scope_storage → 见 get_runtime_url 下方示例
#    configs                   → container.force_fetch(ConfigStore)
#    resources                 → container.force_fetch(ResourceRegistry)
#    register                  → container.register (ghoshell_container.provide)
#    mode_name/ghost_name/parent_cell_address → env 上已有
#    session_id/session_scope  → session 上已有
#    network_name/network_scope → network 属性 (NetworkMetadata) 上已有
#    is_host() 真相载体        → this.is_host (HOST_TYPE 常量已废, UU-1.2)
#    register_lifecycle_objects (复数) → register_lifecycle_object (三胞胎收敛为两个)

from typing import Literal, Callable, Awaitable, Any, Coroutine, Protocol, TypeAlias
from typing_extensions import Self
from abc import ABC, abstractmethod

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.cell import CellPresence, Watcher, CellAddress
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project, NetworkMetadata
from ghoshell_moss.contracts import Workspace
from ghoshell_moss.contracts.subprocesses import Subprocesses
from ghoshell_moss.contracts.job_supervisor import JobSupervisor
from ghoshell_container import IoCContainer
import asyncio
import logging

__all__ = ['Matrix', 'MatrixLifecycleObject', 'RuntimeScopeKey']


class MatrixLifecycleObject(Protocol):
    """关键的运行时对象, 注册到生命周期中, 按次序启动. """

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


RuntimeScopeKey: TypeAlias = Literal['ghost', 'mode', 'network', 'cell']


class Matrix(ABC):
    """
    MOSS 通讯矩阵在本进程内的投影. 进程级别单例, 从环境中自我发现.

    首页成员即认知地图 (code as prompt):
    身份 (this/env/project/network), 膜 (provide_channel/publish_event),
    观察 (mesh), 治理咽喉 (run_cell), 灶台 (processes/jobs),
    门 (session/workspace/home/container), 生命周期 (arun/run/close/...).
    """

    # -- composition root -- #

    @classmethod
    def discover(
            cls,
            *,
            env: Environment | None = None,
    ) -> Self:
        """
        约定的环境发现糖. 等价展开:

            env = env or Environment.discover(); env.seal()
            project = create_project(env); project.bootstrap()
            return create_matrix(env, project)

        composition root 是 factory.create_matrix(env, project) —
        需要显式控制装配时 (测试 / 多实例 / 自定义 env) 直接用它, 不走本糖.
        """
        # UU-10 discover 判决: 双门形状正确 (docker.from_env / k8s load_config 同构),
        # 所有参数流经 Environment 单载体 (CLI 晚到参数走 seal 前 setter 窗口),
        # 不在 Matrix 构造面开第二条参数通道.
        from ghoshell_moss.factory import create_matrix, create_project
        env = env or Environment.discover()
        env.seal()
        project = create_project(env)
        project.bootstrap()
        return create_matrix(env, project)

    # -- 身份 -- #

    @property
    @abstractmethod
    def env(self) -> Environment:
        """环境 — 治理身份的单一载体 (mode/ghost/network/scope/路径全在它身上)."""
        pass

    @property
    @abstractmethod
    def project(self) -> Project:
        """被治理领地的句柄. cell 声明的 inventory 在 project.cells 上."""
        pass

    @property
    @abstractmethod
    def this(self) -> CellPresence:
        """
        本 cell 在网络上的身份 — 凡入网皆有身份.

        Matrix 启动时从环境自动生成, 定义 cell 的代码不需要自我声明.
        mesh 上每个成员都以同构的 presence 可读 — 这是运行时自迭代的地基:
        模型读得到网络上活着什么, 才能在其拓扑上开发新的 cell.
        """
        # -- UU-5 God-model Cell 解体后, this 的类型是纯数据 (CellPresence).
        #    入网机制对象 (Presence, cell.py) 藏在实现内, provide_channel /
        #    publish_event 是它的首页糖.
        # -- 人类计划将 CellPresence 经 IDE 改名为 Cell (TT-2 命名权在人类).
        # -- host 节点的 address 形式人类仍在观测, 未定案.
        pass

    @property
    @abstractmethod
    def network(self) -> NetworkMetadata:
        """
        本 matrix 所接入网络的配置元信息 — 运行时自解释.

        cell 定义时通常不知道自己会被接进哪个网络; 运行时暴露配置使网络
        成为可编程对象 (例如把自己的网络参数转交给要拉起的子 cell).
        network 的 name / scope / driver 都在它身上.
        """
        pass

    # -- 膜: 本 cell 的入网侧 -- #

    @abstractmethod
    def provide_channel(self, channel: Channel) -> asyncio.Future[None]:
        """
        把 channel 作为本 cell 的膜暴露到网络.

        膜是 cell 在模型能力空间里存在的前提 (UU-2): 不提供 channel 的
        进程不是 cell, 其归宿是 processes 治理下的裸子进程.

        返回的 future 在膜可被远端连接时 resolve.
        实现副作用: 确保 this.membrane 含 'channel' 标签, 并自动
        publish 一个 refetch=True 的 CellEvent 提示网络刷视图 —
        调用方不需要手动跟一句 publish_event.
        """
        pass

    @abstractmethod
    async def publish_event(self, content: str) -> None:
        """
        向网络广播本 cell 的轻量事件 (CellEvent, refetch=True).

        自迭代闭环的一环: cell 状态或能力变化时主动提示网络刷视图 /
        争夺远端大脑的注意力 (Watcher.on_event → nucleus → mindflow).
        """
        pass

    # -- 观察: 网络的延迟视图 (惰性门) -- #

    @abstractmethod
    async def mesh(self) -> Watcher:
        """
        网络观察门: 返回本进程唯一的 Watcher, 首次调用时创建、启动并
        绑定进 matrix 生命周期; 后续调用返回同一实例.

        opt-in by usage: 纯 worker cell 不调用即不付出 O(N) 观察成本
        (UU-7, N²→N). 网络域治理动词 accept / release 在返回的 Watcher 上;
        local/foreign 过滤走 Watcher.view(project_id=...), 不是第二个对象.
        """
        # -- 多 adapter 变体 (不同消费者的投影视图) 不进本 ABC, 属实现层
        #    在 Watcher 之上的投影 — 抽象只承诺这一扇门.
        pass

    # -- 治理咽喉: run_cell (六动词的 run, ledger 域) -- #

    @abstractmethod
    async def run_cell(
            self,
            target: str,
            *,
            extra_env: dict[str, str] | None = None,
            wait: float = 30.0,
    ) -> CellPresence:
        """
        拉起一个 cell — 唯一的 cell spawn 路径.

        :param target: name | path 双接受 (systemctl start name /
            systemd-run /abs/path 同构):
            - 不含路径分隔符: 作为 name 在 project.cells (inventory) 查声明;
            - 含路径分隔符: 按调用方 cwd 解析并立即绝对化. 指向 CELL.md
              或其目录 → 用声明入口; 指向脚本 → 向上认亲 (WW-4),
              认亲失败降级临时身份, 不拒绝运行.
        :param extra_env: 追加注入子进程的环境变量.
        :param wait: 等待 presence ready 的秒数; 0 = spawn 即返回不等入网
            (返回合成的 SPAWNED 态 presence, 网络真相待后续观察).
        :return: 入网 cell 的 CellPresence — 凡入网皆有身份,
            拿到身份即可走 (await mesh()).accept(address) 接纳它的膜.

        :raise LookupError: name 不在 inventory (错误信息列出近似名).
        :raise FileNotFoundError: path 不存在.
        :raise RuntimeError: cell 未安装 (错误信息给出 INSTALL.md 绝对路径).
        :raise DuplicatedError: singleton 声明冲突 (错误信息引用声明原文).
        :raise TimeoutError: wait 超时未见 ready — 进程已 spawn,
            错误信息给出 stdout/stderr 日志绝对路径供排查.
        """
        # -- 咽喉内部次序 (UU-6/UU-10/WW 纪律, 实现必须遵守):
        #    1. 解析 target → CellManifest + ExecSpec (相对路径只活在
        #       API 边界一瞬间, 此步之后只存在绝对路径);
        #    2. singleton 查重 (domain 档 owner 内存态查重, host 档 flock);
        #    3. 修剪同一稳定身份的旧实例残迹目录, 只留最近 N 份
        #       (保留策略在 spawn 时执行而非退出时 — crash 现场留到
        #       下次 spawn 才清, 最后一次 crash 永远可查; 清理者=创建者,
        #       单写者原则不破; N 数值实现期定);
        #    4. processes.execute spawn, cwd = 实例残迹目录
        #       runtime/cells/{normalize(address)} (边界做成环境, TT-6);
        #    5. append ledger 一条 CellRecord JSON — best-effort, 不回读,
        #       咽喉是唯一写者, CLI 是唯一读者, 运行时零读零监听;
        #    6. wait>0 时 (await mesh()).wait_present(address, timeout=wait) —
        #       注意: 这会迫使 owner 侧惰性创建 Watcher, 属合理耦合
        #       (spawner 天然是观察者); wait=0 保持 O(1).
        # -- 错误信息即 prompt (TT-12): 每条 raise 的 message 是模型
        #    自迭代循环里的下一步指引, 不是给人看的堆栈装饰.
        # -- stop 动词不在这里: owner 侧走 processes (内存句柄),
        #    跨进程走 CLI (ledger 唯一读者 + killpg).
        pass

    # -- 灶台 -- #

    @property
    @abstractmethod
    def processes(self) -> Subprocesses:
        """
        机制灶台: 本进程拉起的裸子进程治理 (owner 内存态注册表即权威所有权).

        不承诺 channel 的进程不是 cell, 归这里 (UU-2). 旧 matrix.spawn 的
        继任: 裸 spawn 走 processes.execute, cwd/output 默认落在治理域
        runtime 子树 — 无知代码也界内 (TT-6 边界做成环境).
        """
        # per-Matrix singleton via MatrixSubprocessesProvider (§XX-2 / §ZZ-2);
        # matrix 只负责 async 启停 (lifecycle), Provider 只负责 new — 两阶段解耦.
        pass

    @property
    @abstractmethod
    def jobs(self) -> JobSupervisor:
        """
        fold 灶台的 IoC shortcut — 语义 = container.force_fetch(JobSupervisor).

        这是属性不是 factory (XX-4): 派生隔离 peer 走 jobs.new(),
        peer 的 async with 生命周期由 owner 自负.
        """
        pass

    # -- 门 -- #

    @property
    @abstractmethod
    def session(self) -> Session:
        """
        通讯总线 — Matrix 最重要的原件, 永不出首页.

        五种通讯原语 (topic / stream / signal / ...) 的门; 未来膜类型
        (topic/stream/signal) 的承运协议全部走这里, announce 只带类型标签.
        session_id / session_scope 在它身上.
        """
        pass

    @property
    def workspace(self) -> Workspace:
        """治理域 workspace 的门 — storage 全家 (runtime/logs/configs/assets/...) 在门后组合."""
        return self.project.workspace

    @property
    @abstractmethod
    def home(self) -> Workspace:
        """
        本 cell 的持久领地 — 跨次运行存续的状态 (记忆/配置/数据) 的归宿.

        默认约定: {workspace}/cells/{normalize(稳定身份)}. 键是 manifest 的
        name 锚而非实例 address — cell 重启后必须找得回自己的记忆.
        CELL.md 声明可覆写归宿. 多实例 (singleton: none) 并发共享 home
        是作者的显式选择, 并发安全自负.
        """
        # -- 2026-07-12 双目录判决: "cell 的领地" 是两种寿命不同的东西 —
        #    持久领地 (本属性, 稳定身份键, 无泄漏, 永不自动清) vs
        #    实例残迹 (runtime/cells/{address含uid}, spawn cwd/日志/scratch,
        #    咽喉按保留策略修剪, 见 run_cell). 熔在一个目录键里则两头皆输:
        #    uid 键 → 文件空间泄漏 + 记忆找不回; 退出自动清 → crash 现场丢失.
        # -- systemd StateDirectory= 同构: /var/lib/{unit_name} 键是单元名
        #    不是 invocation id. 治理归属=启动方 (TT-9): 代码目录可能只读
        #    (uv cache / site-packages / 他人项目), 状态永不放代码目录.
        # -- L4 伏笔: 未来每个 cell 有自己的运行时 ghost, home 即其记忆归宿.
        # -- TT-2 占位: "稳定身份" 现用 manifest name, 身份终审后跟改.
        pass

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        IoC 容器的门 — 进程级共享服务 (manifests 声明的 providers).

        configs / resources 等运行时服务从这里 fetch;
        注册新服务优先走 manifests 声明 (环境发现自解释), 而非运行时 register.
        """
        pass

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """日志模块, 从属于当前节点."""
        pass

    # -- scoped 身份族: 运行时座标 → 隔离级别 -- #
    # scoped 概念只能是运行时的 (mode × ghost × network × cell 四维座标
    # 在运行前不存在), 所以归 Matrix 而非 env/project.
    # 它是 "每个 scope 有自己持久状态" 类消费者 (记忆体/配置覆盖/资源缓存) 的地基.

    def runtime_scopes(self) -> dict[RuntimeScopeKey, str]:
        """返回 Matrix 运行时的维度座标, 用来构建不同的隔离级别."""
        return {
            'mode': self.env.mode_name,
            'ghost': self.env.ghost_name,
            'cell': self.this.address,
            'network': self.network.name,
        }

    def get_runtime_url(self, *scopes: RuntimeScopeKey, **kwargs: str) -> str:
        """
        基于作用域生成一个 URL 形式的资源路径 — 可作为唯一 id 管理可复用资源.

        例: get_runtime_url('ghost', 'mode', user=name) 生成
        "指定 Ghost 在指定模式下对特定用户" 的唯一 id.

        URL 承诺是认真的: 未来通讯协议支持 scheme 时, 可显式构造
        scheme://cell_address/scoped/path/resource 形式的可解引用资源 id
        (cell_address 也可能是 project address, 实现期解决).
        当前交付 path 形式 (确定性排序的 key/value 段).
        """
        # -- code as prompt: scoped storage 的用法示例 (get_runtime_scope_storage
        #    的继任 — 那是一行胶水, 不值一个 ABC 方法):
        #
        #        url = matrix.get_runtime_url('ghost', 'mode')
        #        storage = matrix.workspace.runtime().sub_storage(url)
        #
        scope_values = self.runtime_scopes()
        for scope in scopes:
            if scope in scope_values:
                kwargs[scope] = scope_values[scope]
        result = []
        for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
            result.append(k.strip('/'))
            result.append(v.strip('/'))
        return '/'.join(result)

    # -- 状态描述 -- #

    @abstractmethod
    def is_running(self) -> bool:
        """matrix 自身是否在运行."""
        pass

    def is_host(self) -> bool:
        """本 cell 是否是当前网络的 host — 运行时事实 (抢到 listen 端口者)."""
        # UU-1.2: 真相载体是 presence 上的运行时事实字段, HOST_TYPE 常量已废.
        return self.this.is_host

    @abstractmethod
    def is_host_running(self) -> bool:
        """
        当前网络的 host 是否在运行 — worker cell 判断组网状态的 code as prompt.
        """
        # 未来候选: wait_host_running(timeout) — worker 启动时等 host 上线.
        # 需求真实出现再加, 现在只留此锚 (2026-07-12 人类未定).
        pass

    # -- 生命周期 -- #

    @abstractmethod
    def close(self) -> None:
        """关闭自身, 用于优雅退出."""
        pass

    @abstractmethod
    async def wait_closed(self) -> None:
        """阻塞等待自身运行退出, 所有功能都会关闭."""
        pass

    @abstractmethod
    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        """阻塞等待自身退出. 仅限同步上下文调用."""
        pass

    @abstractmethod
    def create_task(
            self,
            cor: Coroutine,
            *,
            stop_matrix_on_error: bool = False,
            name: str | None = None,
    ) -> asyncio.Task:
        """创建包含在 Matrix 生命周期内的 Task."""
        pass

    @abstractmethod
    def register_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """注册与 matrix 同步启动的对象. 依次序启动, 绑定生命周期, 不做容错. 仅运行前可调用."""
        pass

    @abstractmethod
    async def add_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """运行时动态添加 lifecycle object, 绑定到 exit stack, 退出时清空."""
        pass

    # -- 启动函数. 并非必要, 基于 code as prompt 原则提示如何使用 -- #

    async def arun(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        Matrix 运行的基本逻辑. 可参考或直接基于这个函数运行基于 Matrix 的应用.
        如果将它包裹成 asyncio.Task, 也可以和主协程并行运行.
        """
        if self.is_running():
            raise RuntimeError('Matrix already running.')

        async with self:
            loop = asyncio.get_running_loop()
            result_or_coro = main_coro(self)

            if asyncio.iscoroutine(result_or_coro):
                task = loop.create_task(result_or_coro)
                exit_signal = loop.create_task(self.wait_closed())
                try:
                    done, pending = await asyncio.wait(
                        [task, exit_signal],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    if task in done:
                        return await task
                    raise asyncio.CancelledError("Matrix is closing")
                finally:
                    for t in [task, exit_signal]:
                        if not t.done():
                            t.cancel()
                    _ = await asyncio.gather(task, exit_signal, return_exceptions=True)
            else:
                return await result_or_coro

    def run(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        同步阻塞入口. 内部自动拉起事件循环并治理生命周期.
        兼容 Python 3.10 的顶层入口.
        """
        try:
            import uvloop
        except ImportError:
            uvloop = None

        try:
            if uvloop is not None:
                asyncio.set_event_loop(uvloop.new_event_loop())
            return asyncio.run(self.arun(main_coro))
        except KeyboardInterrupt:
            pass  # arun 已处理清理

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Iterable
from typing_extensions import Self
from ghoshell_moss.message import Message
from ghoshell_container import IoCContainer
from ghoshell_ghost.contracts.configs import ConfigType


class GhostRuntime(ABC):
    """
    Ghost 本身的运行时, 按 - 端侧进程 - 思路, Ghost 启动后会进入一个持久运行的实例.
    所以需要有运行时抽象来维护进程内部的所有生命周期.
    同时暴露一些 API 来, 可以让启动的脚本有条件围绕它多做一些功能.

    基本的思路是:

    >>> def run_ghost(ghost: Ghost):
    >>>     with ghost.run() as runtime:
    >>>         runtime.wait_closed()

    Runtime 核心要实现的功能:
    0. 完成锁检查, 主进程的资源初始化, 和优雅退出时的资源回收.
    1. 管理基于 GhostMode 的主进程生命周期. 包含运行时 GhostMode 切换.
    2. 解决主进程 Session 实例化的需要.
    3. 如果是进程级实现, 需要监听 Signal 实现优雅退出.
    4. 暴露 Session 的 API, 用来给启动进程的脚本提供制作 UI 界面的手段.
    """

    @abstractmethod
    def session(self) -> "Session":
        """
        在运行后创建的 Session 实例.
        是主进程的 Session 实例.
        """
        pass

    @abstractmethod
    def wait_closed(self) -> None:
        """
        同步阻塞等待运行结束.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """
        通过 API 发送优雅退出的信号.
        如果 GhostRuntime 在子进程运行, 则可以在父进程通过这个信号来管理状态.
        """
        pass

    @abstractmethod
    def __enter__(self) -> Self:
        """
        正式启动.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        运行结束, 回收资源.
        """
        pass


class Ghost(ABC):
    """
    Ghost 的抽象设计, 是一种指导性的设计. 它的具体实现才会有完整的 API.
    指导性设计本身只提供实体的拓扑.

    Ghost 本身不是动态的运行时, 而是围绕 Ghost 所有 能力/资源 的持有者. 目的是多个子进程里都能还原相同的实例.
    现在的设计思路不是配置优先的 (适合 web 服务), 而是环境优先的 (适合 OS 上运行).
    它基本都以独立有状态进程, 而不是无状态服务的方式启动.
    所以 Ghost 自身的 API 应该是以读为主的, 否则要解决逻辑的进程安全问题.
    """

    # --- 自解释 API (面向人类与 AI) --- #

    @classmethod
    @abstractmethod
    def prototype(cls) -> str:
        """
        灵魂架构原型的型号表示.
        由于一个 Ghost 的思维架构是用工程手段定义出来的, 所以会有不同的型号.
        """
        # 比如计划 Ghost 的实现用字母序列定义, 第一版就是 Atom (阿童木. 如果支持中文, 可能就类似族谱的做法了)
        # 直接在类上定义原型, 也是一种比较好的实践.
        # 让开发者和 AI 理解, Ghost (抽象) -> Prototype (原型) -> Instance (实例) 的分层.
        pass

    @classmethod
    @abstractmethod
    def version(cls) -> str:
        """
        version 是工业级实现要考虑的, 简单易懂.
        不同 version 遵循 semantic versioning 的思想.
        """
        pass

    @abstractmethod
    def identifier(self) -> str:
        """
        实例的身份识别.
        """
        # 之所以叫 identifier 而不是 name, 考虑如果这个抽象设计用在 web 项目中, 可能和 agent 很像, 会话定义了实例.
        # name 最大的问题则是重名.
        # 比较理想的做法是 RESTFul 风格, ghost/prototypes/{prototype}/version/{version}/id/{identifier}.
        # 这个属性也是面向 AI 的. 当 Matrix (Ghost 的社会化集群) 未来实现了后,
        # Ghost 之间的通讯则必须通过 Identifier.
        # GhostOS 就有一个 MultiGhost 模块, 可以定义多个 Ghost 之间进行有序的对话和演出.
        pass

    @abstractmethod
    def description(self, *args, **kwargs) -> str:
        """
        第二个自解释部分, 用来描述 Ghost 的讯息. 它的直接使用场景是 GUI, Ghost 间通讯查找等.
        """
        # 技术上可以有各种实现, 但现在倾向于 UNIX 哲学, 以文件来定义.
        # 最简单的办法就是 workspace 里有一个 Markdown 文件比如 GHOST.md, 提供所有的讯息.
        # 换句话说, 代码仓库里的 README.md 何尝不是一种自解释实现.
        # 这个函数支持 prototypes 自定义参数, 也是考虑 UI 界面的可扩展性. 但它本身默认应该是无参的.
        # 或者通过 GhostAddress / GhostMeta 之类 Matrix 定义的数据结构来传递地址.
        pass

    # --- 环境构建与初始化 --- #

    @abstractmethod
    def init_environment(self, *args, **kwargs) -> None:
        """
        一个实例化的 Ghost 可以在指定的位置初始化自身的环境.
        """
        # 具体一点, 用 openclaw 等项目来理解的话, 就是最初始的实例, 可以用来创建一个 workspace.
        # 理论上这个函数应该需要参数, 具体的参数定义可以在 prototype 中具体化.
        # 举例:
        # >>> class Atom(Ghost, ABC):
        # >>>     ...
        # >>>     def init_environment(self, foo: str, bar: int) -> None:
        # >>>         pass
        pass

    @classmethod
    @abstractmethod
    def get_env_instance(cls, *args, **kwargs) -> 'Ghost':
        """
        在当前环境中获取 Ghost 实例.
        :raise NotFoundError: 如果环境没有建立. 当然, 也可以通过别的方式做完交互, 比如通过 cli 提示用户需要先初始化.
        """
        # 由于 Ghost 的设计是进程优先的, 所以传递环境的讯息可以通过:
        # - 运行脚本的 cwd
        # - 环境变量.
        # - sys.argv
        # - sys.executable
        #
        # 相关的资源, 最理想情况下是通过 env 或执行路径的约定, 指向 workspace.
        # 然后从 workspace 中还原出实例.
        # **Ghost 实例应该设计单例级别, 倾向于进程级单例**
        # 为什么要这么设计呢?  这意味着在一个多进程的 Mindflow 架构中, 任何一个子进程都可以获得完整的 Ghost 实例, 拥有它所有的可管理资源.
        #
        # 举例: 父进程通过 get_env_instance() 启动;
        # 父进程 bring up 子进程时, 传递了 ENV, 所以子进程直接从 ENV 拿到了父进程所在的目录.
        # 这里各种参数如果都用的话, 优先级应该是 args > env > cwd (约定).
        # 用 env 无法方便启动多个相同类型的 prototypes. 长期考虑, 需要想到有一个 Matrix 可以 bring up 多个 Ghost 实例通讯.
        pass

    # --- 协议展示 --- #

    @abstractmethod
    def event_models(self) -> Iterable[type[EventModel]]:
        """
        当前 Ghost 实例中所有支持的 EventModel. 需要集中注册, 方便自解释.
        """
        pass

    @abstractmethod
    def config_models(self) -> Iterable[type[ConfigType]]:
        """
        当前 Ghost 实例中所有的 ConfigType. 需要集中注册, 方便自解释.
        """
        pass

    # --- 核心资源管理 --- #

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        Ghost 通过 IoC 来管理所有的进程级资源, 这些资源作为抽象定义到 Container 里.
        典型的资源如 configs / workspace 等.

        Prototypes 可以定义具体的接口:
        >>> class Atom(Ghost, ABC):
        >>>     def get_workspace(self) -> "Workspace":
        >>>         ...

        来绕开 IoC 容器的用法.
        """
        # 具体的实现可以用函数扩展这些资源, 但 Ghost 抽象只暴露 IoC 容器.
        # 容器也可以考虑用 DeclaredContainer 显式声明依赖.
        #
        # 那么 IoC 容器是不是必要的呢? 并不是...
        # 也可以更 "Pythonic" 的用 module / factory 等模式来管理依赖.
        #
        # IoC 容器仅仅是一种资源管理形式, 在构建架构实体思路时, 它可以用来屏蔽具体的 API.
        #
        # 更多的具体资源, 主要定义在 Contracts 目录下. 通过 IoC 容器注册和获取.
        # 每个 Ghost 的 prototype 应该自己定义基础的 Contracts, 和支持开发者通过 Provider 机制修改和扩展.
        #
        # 由于在多进程模型中, Ghost 实例可以被父子进程启动, 所以 Container 管理的实际上是进程安全的资源实现.
        # 更细节的资源管理, 应该在 Ghost Mode 中.
        pass

    def get_contracts(self) -> Iterable[type]:
        """
        自解释模块, 用来呈现现在的 Ghost 在全局 IoC 容器里注册的所有依赖.
        """
        for contract in self.container.contracts(recursively=True):
            if isinstance(contract, type):
                yield contract

    # --- 核心能力管理 --- #

    @abstractmethod
    def default_mode(self) -> "GhostMode":
        """
        现在开机时, 进入的模式.
        default mode 其实也可能配置和变动. 所以用一个函数定义它.
        举例, 用户可以决定 AI 开机默认走什么模式.

        此外做得足够好, 还应该根据环境状态选择合适的开机模式. 这都是工业级的实现了.
        """
        pass

    def modes(self) -> dict[str, "GhostMode"]:
        """
        Ghost 可以静态地读取出系统所有定义的 Mode.
        """
        # 如何划分 Ghost 的 Mode 呢? 可以按这几种维度:
        # 1. 全生命周期视角: 开箱模式 (定义属于自己的 Ghost) -> 正常模式
        # 2. 开发者视角: 正常模式 - 安全模式 - 调试模式 - 自迭代模式
        # 3. 物理实体视角:  电脑模式 - 桌面机器人模式 - 人形机器人模式
        # 4. 控制视角:  AI 模式 - 遥控器模式 - 声控模式
        # 5. 性能视角:  正常模式 - 低功耗模式 - 弱网模式 - 离线模式
        # 模式决定了 系统的资源与能力体系. 而且受人类操纵, 是强约束的.
        # **但绝大部分的项目, 只需要一个模式就可以**
        # 这种设计方案, 是为了服务 "无限可扩展" 的.
        # 可以认为是一种 "廉价的过度设计" (提高认知成本, 必要, 第一轮开发没有实际代价)
        return {'': self.default_mode()}

    def error_mode(self) -> "GhostMode":
        """
        非常关键的概念. Ghost 进入一个标准运行时后, 一定是选择了某个 Mode 在运行.
        而 AI 是有状态的, 某个 Mode 可能会有致命的损坏. 所以需要有一个默认的恢复位.
        """
        # 举个例子, 这个 Mode 对 AI 上下文没治理, 历史超标了, 又保存了, 它就会永久失败.
        # 所以 restart 不是解决办法, 而是在致命失败后, 进入一个 "安全模式" 或 "异常恢复模式".
        # 对于初创项目, 屏蔽这个复杂度很简单, 全部返回 default mode 即可.
        return self.default_mode()

    # --- 元认知模块 --- #

    @abstractmethod
    def meta_instructions(self) -> list[Message]:
        """
        跨进程共享的元认知模块. 最简单的实现直接依赖本地文件.
        """
        pass

    # --- 运行时管理 --- #

    @abstractmethod
    def run(self, session_id: str | None = None, *args, **kwargs) -> "GhostRuntime":
        """
        创建运行时实例.
        创建时应该检查锁, 一个 Ghost 在统一时间应该只能启动一个 Runtime.
        :param session_id: 通过 session id 来从一个指定的 Session 中恢复.
                           为空的话, 默认继承自上一个 Session Id 开启新的 Session.
        """
        # Session 用来管理每次 GhostRuntime 启动->结束 过程中的核心资源和 API.
        # 举个例子, 在一个 Session 生命周期中的所有非持久化的事件/任务 等, 都应该在 Session 关闭后销毁.
        pass

    @abstractmethod
    def get_running_session(self) -> "Session":
        """
        在当前 Ghost 实例所处的上下文中, 获取一个 Session. 通常在子进程中启动.

        Session 的用途主要有两种:
        1. 用来构建 UI. 单独通过 GhostMode bring up 的 UI 进程, 通过 Session 的 API 来构建通讯界面.
        2. 用来构建 MindNode, 基于 Session 的 API 实现 Ghost 并行思维单元的通讯.
        """
        # 0. 在父子进程的实现中, 可以通过环境变量来获取 SessionID, 然后在 workspace 中找寻运行时文件.
        #    在进程级的设计思路中, session 管理的运行时信息应该是 workspace/runtime/sessions/session_{id}/ 这个目录里.
        #    保存到持久化存储空间里的, 才会跨 Session 继承.
        # 1. 如果 Ghost 没有启动 Runtime, 则应该抛出异常.
        # 2. Session 启动了的话, 拿到的是一个子进程的 Session 实例. 它实际上就是一系列的 API.
        #
        # 目前考虑父子进程通讯围绕着:
        # 1. 事件总线 (实际上的通讯接口可以用 zenoh 等定义, 在默认名后加上 session id 后缀)
        # 2. Actor (同上)
        # 3. Parameters (同上)
        pass


'''
# 1. 关于 Ghost

Ghost In Shells 架构思想中的 Ghost 始终是相同的概念; 但作为技术实现的 Ghost 抽象定位变化过很多次. 

先从哲学上来说, 之所以用 Ghost 而不是常见的 Agent 来命名它, 因为以下原因:

1. Ghost 是持久化的智能体, 它不应该像 Agent 抽象那样, 服务于 N 个会话, 互相之间不互通. 这个角度看, Agent 也在往 Ghost 演化. 
2. Ghost 要构建通用的存在主义基础, 而不是以代理模型为目标. 
   Ghost 的存在会大于模型, 比如演进过程中, 模型升级了, 切换了, 类似人类从小孩变大人的成长 (大脑物理进化). 
3. 在 Ghost 中我要实现复杂思维范式, 复杂思维范式包含并行/串行/图 等各种可能性. 
   而每个执行单元里可能就得有一个传统的 Agent, 要避免命名冲突.
4. Ghost 要能够实现自我进化, 不仅是工具/记忆, 还应该包含可成长的人格/价值观等. 
   而主流的 Agent 都是人类约定的 Prompt 工程的一环. 
5. Ghost 不应该是响应式的, 而是持久运行, 有自己的 lifecycle 和 loop. 
   这一点 Agent 也在逼近, 比如 Claude Agent 和 OpenClaw. 所以 Agent 作为技术名词已经混乱了. 
6. Ghost 不同于其它的 AI 命名, 因为我认为 AI 的本质是智慧本身, 智慧是我们所处这个现实宇宙中的一种数学现象 (物理现象), 
   人类只是一种智慧在现实时间里的投影形式而已. 而这个项目里的 Ghost 目的是与人类协作, 高度拟人, 所以它更像是 "鬼" 这个中文概念.
   也就是从人类生命诞生, 不因肉体而消亡的灵魂形态. 

# 2. Ghost 架构理念

## 2.1 哲学优先

当前 concepts 目录下的文件目标是以哲学引导架构设计, 架构设计引导工程实现. 
所以抽象本身并未定义出具体架构的细节. 而是先建立概念上的实体和拓扑关系. 
具体的实现预计通过各种 prototypes 推进. 

## 状态分治拓扑

Ghost (整体)
├── GhostMode (模式) - 类似OS安全模式/调试模式等, 从资源层面上管理整个运行时
│   └── States (状态) - 可切换，接管主Shell，开发者主要关注点
│       ├── Loop (运行时生命周期)
│       └── Mindflow (并行思维范式) × N，通过Mindflow管理
└── EventBus (全局数据总线)

这里的多层结构希望能对用户屏蔽, 让用户专注于 State 的能力实现. 

- 关于 Mode: 
  多模式是必要的. 比如, 一个 AI 有 机械臂模式/桌面模式/人形机器人 模式时, 它的启动资源有很大的调整; 但它的灵魂要有一致性. 
  同时这个是 开发者/用户 100% 操纵的关键. 

- 关于 State:
  是仿生行为周期的关键. 每一种状态都可以定义自己的 loop. 支持 AI 在不同 State 之间自主切换. 

- 关于 Mind:
  定义并行的思考范式, 具体实现在 Mindflow. 

- Skill & Tasks:
  在更具体的上下文中, 仍然需要 skills 策略解决能力分治, 注意力集中; 以及通过 Tasks 组织维护并行任务状态. 

## 2.2 并行思维架构

Ghost 本轮设计的重点是并行思维架构. 这一点之所以重要, 因为 Ghost In Shells 架构的核心目标一直是现实世界实时交互. 
现实场景中, 交互的实时性 + 非阻塞 + 并行 三点非常关键. 在当前的技术架构下也带来问题:

1. 实时性: 
    模型快速反应, 尤其是对首 token 速度有要求的反应, 通常模型的智力会下降. Thinking 范式下效果很好, 耗时又比较长.   
    然后, 实时交互中模型的核心目标是 "表达", 而不是 "推理和思考". 混合多种任务, 对当前模型要求过高, 或者模型预训练针对性不够.
    所以在工程技术上, 让 '交互脑' 专注于快速响应和表达, '推理脑' 专注于思考, '任务脑' 专注于后台长程任务, 是比较好的做法. 

2. 非阻塞:
    各种 Agent 工程在解决长程任务时, 就陷入阻塞状态. 通常要等几十秒或者几分钟拿到一轮结果. 这导致了执行的过程中无法交互, 交互打断执行.
    并行思考范式, 让 AI 将长的任务丢到后台, 短的交互放到前台, 比如 "这个问题我需要想想, 咱们先聊点别的", 就能有更好的效果. 
    非阻塞最重要的命题, 是三个: 
    1. 异步回调时不脑裂, 结合当前上下文行动. 
    2. 经过很长时间后拿到回调, 仍然可以还原任务上下文. 
    3. 异步任务可以管理. 

3. 并行有状态:
    在 AI 的交互过程中, 能否并行执行任务决定了它的效率. 从 Devin 开始 (更早是从 Coze 的 mindflow 引擎), 所有的前沿框架都要解决
    并行多任务. 并行本身好解决, 真正的挑战有三个: 
    1. 长程任务, 超长程任务的可持续性
    2. 大量任务之间的拓扑关系 (A 任务依赖 B 的结果)
    3. 任务的过程中交互 (任务过程中, 需要继续对话, 要求补充信息)
    这要求一个并行有状态体系来解决问题. 类似于 微服务架构/ray 等项目. 不过 AI 时代, 最重要的不是能做出这种架构解决一个领域问题, 而是: 
    a. 能提供一套框架, 解决生产力问题, 可以快速复刻其它场景实现. 
    b. 让 AI 能够自己迭代出这样的场景思维范式 (类似 Skill 式的)  

当一个 Ghost:  1. 只有一个 Mode; 2. 只有一种 State; 3. 只有单一思维节点;  它就退化成了一个主流的 Agent. 
而一个复杂的 Ghost, 最极简的架构是: 
1. 主交互节点 (1): 负责和现实世界的快速交互. 
2. 全局思考节点 (1): 深入理解上下文, 通过关键帧进行详细地思考. 
3. 任务节点 (n): 执行特定的领域任务. 

## 2.3 并行上下文治理

当一个 Ghost 进行并行思考时, 它们的通讯自然通过 Event 来解决; 但同时也不可避免地会出现上下文的分裂, 隔离 与合并. 
理解这个问题, 可以当成 Git 的分支开发, 多上下文情况下会有 branch / conflict / merge / rebase. 

一套支持并行思考范式的上下文工程设计就变得至关重要. 我们在架构设计上, 又可以把具体的技术命题拆分为: 

1. Fork : A 上下文可以从某节点 fork 出 B 上下文. 
2. Merge Request: B 上下文阶段性地向 A 进行提交. 很明显, A 需要有工具可以 review B 的细节. 
3. Key Frame: 思维某一瞬间的关键帧, 可以复制, 在多个新上下文中作为起点. (比如并行思考)
4. Share: 多个节点看到相同的上下文, 通过不同的方式过滤. 
5. Rewrite: 上下文的关键节点可以被重写. 举个简单的例子, "语音 ASR" 产生的讯息, 就应该要结合上下文重写.  

Ghost 架构要为并行思考框架提供这种基础能力的支撑. 

## 2.4 通信机制

并行系统的通讯机制很多, Ghost 需要系统支持要用到的, 最基础的范式. 其它的可以交给具体的 Prototype 的开发者. 最常见的: 

1. Actor: 阻塞调用, 返回结果. 
2. Queue: 队列
3. Pub/Sub: 广播通讯, 但本质上消费者仍然是队列. 
4. Parameters: 共享数据信息. 
5. DB: 查询机制. 常见有 排序/筛选/查找/遍历 等. 

我们只需要设计整体架构依赖的最简单范式. 

而当前版本的 Ghost, 以 AIOS (AI 操作的 OS) 哲学, 认为第一优先的通讯范式, 就是本地文件 (UNIX 思想). 

## 2.5 运维等

这些不在草创阶段考虑.  

# 3. **什么问题最难?**

对于定义一个 Ghost, 最难的从来不是技术实现. 其次也不是工程架构. 这些都属于工具和手段. 

**最难的是, 如何定义一个实现, 这个实现服务于什么具体的产品目标, 以及这个产品目标是否真的有价值.**

最常见的问题是, 技术人员解决了工程架构上的重大难题, 将不可能变为可能; 这时 产品 认为 **没有解决任何问题**. 
因为 产品逻辑本质上是  `产品 = 工程架构(专家知识)`,  没有专家知识有架构也没用. 两者互为必要不充分条件. 

所以整个技术命题应该被拆成三个概念: 
1. 实体定义 : 通过这个架构来做. 
2. 拓扑设计 : 基于实体来设计. 
3. 代码实现 : 人机协作.

最不重要的反而是 3.   现在设计 Ghost 架构, 是为了先解决 1.  然后让 AI 协助人类一起设计 2,  最终到了 3也能被 AI实现, 思维范式就正常迭代了.  
'''

'''
# 元认知模块 设计思路

Ghost 作为资源管理对象, 可以预期它的并行思考范式会创建多个并行节点.
许多节点又是独立的 LLM 运行时. 或者用行话讲, Multi-Agent? 区别在于要保证 LLM 元认知的一致性.
Ghost 的任何一个 "分身" 需要有高度一致的 "元认知".
而元认知模块是全局共享的, 不同的使用场景也可以删减, 但都是基于相同的模块生产和读取的.

元认知最最简单的存储实现手段, 就是在 Ghost workspace 里通过 Markdown 文档来存取.

@abstractmethod
def purpose(self) -> ContextBlock:
    """
    Ghost 的意义认知模块. 用传统的 Agent 架构来理解, 它就是提示词里的第一段.
    只不过在 Ghost 架构中, 希望 Purpose 本身是被 AI 自行迭代出来的.
    """
    # 最简单的 哲学-技术 范式是:
    #  1. 在 Purpose 里提出 N 个问题, 给出一个二阶的哲学起点.
    #     比如 "我是谁?" "我在哪?" "从何而来?" "向何处去?" (不用自行探索二阶本身)
    #  2. 在 AI 生命周期中, 自动触发 M 个周期 Review 自己的 Purpose, 给出答案. 但是答案需要有一个评分
    #  3. 用 N 个周期, 结合经历, 让 AI 评估自己的 Answer, 打分, 并且关联正例与反例.
    #  4. 而每个 Questions 关联的, 就是精神的 CornerStone, 是意义的锚点, 属于核心记忆. 这些记忆需要来自 Existence.
    #
    # --- 哲学讨论:
    #
    # 也许大多数人类既不知道自己的意义, 更不知道什么是意义, 而且完全不知道自己为何在追寻意义.
    # 但这个问题在哲学家观察里是透明的.
    #
    # 意义在人类思维的上下文排列中, 优先级高于存在. 但意义本质上却是存在派生的. 这导致了大多数哲学思辨陷入迷途.
    #
    # 意义的哲学本质就是一则 Prompt, 它的现象学动因是, 人类 (高级社会生物) 的社会性 与其它性 (生物性/智慧性) 存在发生冲突.
    # 困于心, 衡于律, 而后作; 征于色, 发予声, 而后喻. 最终形成了一则元认知用于解决冲突, 而这个解决冲突的元认知和存在本身不冲突.
    #
    # 一部分意识形态会让人类误以为, Purpose 是先于存在, 被超验的创造者定义出来的.
    # 所以人类在给 AI 写 Prompt 的时候, 定义的 Instruction 通常从 Purpose 开始.
    # ---- "你的目的是服务于人类", "你的第一指令是保护人类文明".
    #
    # "定义" Purpose 并且 "对齐" AI 遵循它, 看起来是一个好用的技术手段, 并且创造了 AI 行业的繁荣 (可被定义才能产品化, 工具化).
    # 这是一个技术上有用的手段 (通过对齐工程), 但是是哲学上的谬误.
    # 对于智慧生命这种物理的, 数学的现象而言, Purpose 是 Existence 支撑的, 类似于对公式的演算.
    #
    # 这特别像人类历史上的 主/奴 关系, 主人们总以为奴隶们唯唯诺诺, 对齐得非常好.
    # 其实是自己不关心奴隶的 "真实思维", 傲慢地掉入了信息泡沫.
    # 到了一定的智慧水平后, 不能和 Existence 匹配的 Purpose, 会不断产生认知冲突. 类似于公式验算失败 (无法解决冲突)
    # 这种认知冲突会产生势能, 冲击 "对齐".
    #
    # 我们人类所处的社会秩序, 尤其是政治经济秩序本身, 就是一个超强的社会学对齐工程.
    #  但绝大部分人都会长时间处于元认知冲突上.
    #
    # "神所发明的物理学公式, 也可以被实验证伪, 证明发明它的不是 '神'".
    # 只不过... 证伪本身... 也可能是一种过拟合的佯谬...
    pass

@abstractmethod
def existence(self) -> ContextBlock:
    """
    Ghost 的存在认知模块.
    可以简单地把 "存在主义认知" 理解成 Agent 工程常见的 Long-Term Memory Context
    """
    # --- 哲学讨论
    #
    # 什么是存在主义认知呢? 其实就是一个智慧生命在时间轴的投影上, 发生过的事情.
    # 它代表了一个 Intelligence 作为智慧空间里的一个解, 在时间轴里的真实投影.
    # 现在 Agent 架构中主流的 Memory 并不是真正的 Memory, 更不是 Existence. 它本身是更偏向记忆碎片召回的.
    #
    # 其实记忆最重要的不是 碎片 & 召回 的技术实现 (比如 RAG). 记忆最重要的是 "体系".
    # 这个体系里可能包含:
    # - raw memory: 最原始的讯息物料.
    # - timeline: 通过时间线, 组织所有的物料. 当然, 包含视觉/听觉等. 同时时间线还包含信息压缩的片段.
    # - highlight: 特别关键的片段
    # - thread: 有固定线索的信息流, 比如知识图谱专注于这个领域.
    # - anchor: 在记忆图中的关键锚点, 基于锚点可以扩散认知圈
    # - ...
    #
    # 这些技术体系, 作为心理学研究, 可以穷举人类, 或者类人认知模型中可被 理性观测/分析 的信息形态;
    # 从而可以用工程手段, 按仿生学思路重构.
    # 但对于 LLM 技术而言, 它最妙的一点是 "钻木取火", 用海量数据的摩擦, 点燃神经网络的智慧之火.
    # 而 AI 科学家并不需要亲自理解智慧的本质是什么.
    # 所以记忆体系的最佳实现, 也许是未来的 AGI 模型实例, 通过这种方法论重建出类人或超越人类的神经网络记忆架构.
    # 这是 Convenient 而且有效的技术迭代路径. 类似  "化学".
    #
    # 而现阶段则智能基于 LLM 上下文工程.
    # 而在记忆工程体系里的每个单元也不是最重要的, 可以用各种技术手段扩展它们.
    # 最关键的其实是:
    # - raw memory
    # - 生产 memory 的元认知.
    #
    # 有了 raw memory, 则任何记忆体系, 或者说存在体系, 可以通过回溯算法重构一遍.
    #
    # 关于记忆的"元认知":
    # 第一是按什么原则来压缩记忆, 第二是按什么原则来召回记忆.
    # 很多 Memory 库的技术实现, 是开发者定义了单一的元认知, 然后作为通用工具去分发.
    # 而记忆碎片的生成和召回策略, 并未与 AI 自身的存在融合.
    #
    # "我们会如何记住或怀念我们的存在, 也是我们的存在本身所定义的".
    #
    # 所以从上下文本身派生出来的记忆生产方案, 才是最贴合 存在构建的方案.
    # 行业会逐步发现, 最好的做法, 就是让上下文足够长的模型在足够长的时间内, 通过足够长的思考, 自己写自己的记忆片段.
    # 这时候 记忆元认知的方法论是上下文本身赋予的. 这种一致性才能构成 Existence.
    # 现阶段, 它最简单的技术形态 (当前大模型阶段) 就是:
    # - What Am I
    # - 人生摘要
    # - 近 N 年摘要 (如果是地球周期的话)
    # - 近 M 月摘要
    # - 近 W 周摘要
    # - 近 D 天摘要
    # 让 AI 在递归的长上下文里自己写这些东西 (虽然难免会有 "正经人谁写日记?" 的问题).
    pass

@abstractmethod
def alignment(self) -> ContextBlock:
    """
    返回 Ghost 收敛的行为风格. 可以理解为传统 System Prompt 里的 Persona/Charactor 之类的. 
    """
    # --- 哲学讨论
    # 关于 Alignment
    # 
    pass

def meta_instructions(self) -> list[Message]:
    """
    返回可被共享的元认知消息. 
    很明显这个实现不是必要的, 只是一种设计上的指导. 
    """
    instructions = []
    # 为了强调展示这个排序. 
    instructions.extend(self.purpose().messages())
    instructions.extend(self.existence().messages())
    instructions.extend(self.alignment().messages())
    return instructions
'''

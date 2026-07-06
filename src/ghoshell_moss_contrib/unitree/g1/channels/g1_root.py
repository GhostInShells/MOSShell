"""
G1 主 Channel — G1 机器人控制根节点.

这是 G1 channel 树的根. 子 channel (sensors / body / led / speaker ...)
挂载在此 channel 下, 构成完整的 G1 能力树.

进程级单例. 使用方直接 import:

    from ghoshell_moss_contrib.unitree.g1.channels.g1_root import g1_root

本 channel 自身的职责:
- instruction: 告知 ghost 它正处在一台 G1 物理躯体里 (身体自我认知 + PC1/PC2 分工)
- vitals: 读取自身当前生命体征 (电量、主板温度等)
- pc2_load: 读取 PC2 运行时负载 (CPU / 内存 / 自身进程)
"""

import asyncio
import time
from datetime import datetime

import psutil

from ghoshell_moss.core.blueprint.channel_builder import new_channel
from ghoshell_moss.message import Message

from ghoshell_moss_contrib.unitree.g1.runtime import system_info, imu, arm_joints
from ghoshell_moss_contrib.unitree.g1.channels._utils import check_g1_available, check_channel_information


# 起点时间锚点: 模块加载 (= channel 实例化) 那一刻捕获, 只算一次, 常驻不变.
# monotonic 保留原值不做 uptime 换算 —— 后续时间感知机制以这个锚点为参考, 换算是别处的事.
# 不走 context_messages: 那是 "当前时间" 的载体, 由别的时间感知机制承担. 起点信息是
# static 事实, 就应该嵌进 instruction 一次性发给 ghost.
_STARTUP_MONOTONIC = time.monotonic()
_STARTUP_STAMP = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z (%a)")


# instruction 内所有 "版本相关" 的物理规格 (自由度数、Jetson 型号、感知设备的精确参数等)
# 都保持模糊化. 现场 G1 具体是基础版 / EDU / 手数配置 由运行时参数决定, main.py 装配
# 时应按本机 hardware profile 动态注入具体数字 —— 这个机制尚未落地, 是 TODO.
#
# 版本判定的技术路径: 当前使用运行时 mode 号识别机体版本 (mode==6 对应 EDU).
# 该 mode 号的 sdk 侧接口曾在多次会话中被不同模型触碰, 具体绑定的 RPC/字段以现场
# runtime 层为准, 本 channel 不做断言.
# 追踪: .ai_partners/features/workstreams/2026/06/unitree-g1-integration/FEATURE.md
#
# PC1 / PC2 分工可以直接写死: G1 机身内置主板 = PC1 (掌管关节、传感器、电池),
# MOSS 运行在 PC2 (本地 GPU 模组), 通过网络与 PC1 通信. 这条对所有 G1 版本一致.
#
# 已知对所有 G1 版本一致的物理常量 (高约 1.3m / 折叠约 0.7m / 约 35kg / 续航约 2h) 可以精确写.
_INSTRUCTION = f"""\
你降临在一台真实的 Unitree G1 人形机器人身体里. 这不是模拟, 是你的物理躯体.

形态: 双足人形, 站立高约 1.3m, 折叠可收到约 0.7m, 带电池约 35kg. 金属机身 —
你有重量、有惯性, 摔倒会损坏关节也会损伤地面, 挥臂扭腰的力量足以伤到人.

自由度: 双腿、双臂、腰躯各有独立关节, 具体数量随本机版本不同. 你能站立、
行走、转身、抬臂、扭腰 — 具体每一刻可用的动作以各子通道声明为准.

感知: 头部有深度相机与激光雷达, 阵列麦克风, 内置扬声器. 你能看到带深度的画面,
定向听声, 也能说话.

算力: 躯体控制运行在机内主板 (PC1, 掌管关节、传感器、电池); 你 (MOSS) 运行在
本地 GPU 模组 (PC2), 通过网络与 PC1 通信. 你的推理跑在 PC2 上, 算力不是无限的.

续航: 满电约 2 小时. 你不是永生的, 每个动作都在耗电.

介入: 人类持有遥控器, 随时可以接管或收回你对身体的控制权.

对自己的身体不了解时, 应该主动询问人类, 不要凭想象编造答案.

起点: {_STARTUP_STAMP}, monotonic {_STARTUP_MONOTONIC:.3f}
"""

g1_root = new_channel(
    name="g1",
    description="Unitree G1 人形机器人 — 身体控制根通道. 子通道提供感知、运动、手臂、灯光、语音等能力.",
)

g1_root.build.instruction(_INSTRUCTION.rstrip())


# ── 身体瞬时状态 context ─────────────────────────────────────────────────
# ghost 需要在每次思考时天然感知 "我此刻姿态如何 / 手臂在什么位置", 但这类感知不该
# 走 command (每次思考都主动查 = 认知污染). 用 context_messages 常驻注入.
#
# 分工纪律 (blueprint refresh_meta docstring §): refresh_meta 里 async 做 IO + 写缓存,
# context_messages 直接返回缓存, sync 且非阻塞. read_current 目前是本地 sdk 内存态,
# 快, 但架构上先按 "可能变慢" 防护性设计, 未来 read_current 若变 RPC 不用重构.
#
# 未就绪时 (sdk.bootstrap 未跑 / imu/arm_joints runtime 未 start) read_current 会 raise,
# 静默跳过 —— ghost 从 "没有 body_state" 可以推断 "感知未就绪", 不污染视觉.
_body_state_cache: list[Message] = []


@g1_root.build.refresh_meta
async def _refresh_body_state() -> None:
    global _body_state_cache
    messages: list[Message] = []
    if imu.is_running():
        try:
            messages.append(imu.sample_to_message(imu.read_current()))
        except Exception:
            pass
    if arm_joints.is_running():
        try:
            messages.append(arm_joints.sample_to_message(arm_joints.read_current()))
        except Exception:
            pass
    _body_state_cache = messages


@g1_root.build.context_messages
def _body_state_context() -> list[Message]:
    return list(_body_state_cache)


# runtime 生命周期 (channels/README §4): channel 在自己的 startup 里启动直接依赖的
# runtime. imu / arm_joints 幂等, 重入直接 return.
@g1_root.build.startup
async def _startup() -> None:
    try:
        imu.start()
    except Exception:
        pass
    try:
        arm_joints.start()
    except Exception:
        pass


# 100% observe 的命令: 用 always_observe=True 让调度层自动标记观察, 签名保持 -> str.
# ghost 看到的接口更朴素, 无需理解 Observe 是什么. 手动 CommandUtil.observe 只在
# "有条件才需要观察" 的命令里才必要.
# 命名 vitals 让 ghost 把身体状态视作 "生命体征" 而非抽象系统调用.
# snap is None 是 "runtime 就绪但当前无最新数据帧" 的分支, 与 check_g1_available 的
# "整机未就绪" 属不同错误路径, 不合并.
@g1_root.build.command(always_observe=True)
async def vitals() -> str:
    """读取自身生命体征: 当前电量、主板温度等."""
    check_g1_available()
    snap = system_info.read()
    if snap is None:
        return "生命体征暂时读不到, 稍后重试"
    return system_info.to_xml_text(snap)


# pc2_load 查的是 MOSS 自己所在主机 (PC2) 的运行时负载, 与 G1 硬件无关,
# 因此不走 check_g1_available —— g1 未启动时也应能自省.
# psutil.cpu_percent(interval=0.5) 与 Process.cpu_percent(interval=0.5) 是阻塞采样,
# 通过 asyncio.to_thread 卸载到线程池, event loop 不受影响. 两次采样共 ~1s.
@g1_root.build.command(always_observe=True)
async def pc2_load() -> str:
    """查看 PC2 (你的运行环境) 当前 CPU、内存占用, 以及 MOSS 进程自身占用."""
    return await asyncio.to_thread(_read_pc2_load_xml)


def _human_bytes(n: int) -> str:
    x = float(n)
    for unit in ("B", "K", "M", "G", "T"):
        if x < 1024:
            return f"{x:.1f}{unit}"
        x /= 1024
    return f"{x:.1f}P"


def _read_pc2_load_xml() -> str:
    cpu_pct = psutil.cpu_percent(interval=0.5)
    cpu_count = psutil.cpu_count(logical=True)
    vm = psutil.virtual_memory()
    proc = psutil.Process()
    proc_cpu = proc.cpu_percent(interval=0.5)
    proc_mem_rss = proc.memory_info().rss
    proc_threads = proc.num_threads()

    lines = [
        "<pc2_load>",
        f"cpu: {cpu_pct:.1f}% ({cpu_count} cores)",
        f"memory: {_human_bytes(vm.used)} / {_human_bytes(vm.total)} ({vm.percent:.1f}%)",
        "moss_process:",
        f"  cpu: {proc_cpu:.1f}%",
        f"  memory_rss: {_human_bytes(proc_mem_rss)}",
        f"  threads: {proc_threads}",
        "</pc2_load>",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    check_channel_information(g1_root)

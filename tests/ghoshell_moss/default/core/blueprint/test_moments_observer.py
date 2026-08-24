"""
moments + observer 状态编排单测.

Moments / Observer / BaseMomentsObserver 是 mindflow 里感知→观测→记忆的编排核心:
观察者持有 moments 容器, observe() 生产 Moment, add_result / inject_percepts /
with_dynamic_context_func / with_percepts_buffer 从不同入口向下一帧注入数据.
本套件全部走 public 方法, 只测行为契约, 不捅私有状态.

覆盖:
- observe 状态机: need_observe 生命周期、previous 衔接、进历史
- 注入通道: inject_percepts / with_dynamic_context_func / with_percepts_buffer
- 回调: on_result_add / on_moment
- 容器: moments 上限驱逐、peek 不落历史、compact / recap、clear
- turns(): 将观测到的 moments 组织为回合历史
"""
import asyncio

from ghoshell_moss.core.blueprint.moment import BaseMomentsObserver, Moment, Moments, Observer
from ghoshell_moss.message import Message


def _text(msg: Message) -> str:
    """取一条消息的首个文本内容, 简化断言."""
    for c in msg.contents:
        if "text" in c:
            return c["text"]
    return ""


def _texts(msgs) -> list[str]:
    return [_text(m) for m in msgs]


# ============================================================
# 基础默认值 / 类型契约
# ============================================================

def test_observer_defaults():
    obs = BaseMomentsObserver(max_size=5)
    assert obs.need_observe() is False
    assert obs.epoch.recap == []
    assert obs.moments() == []


def test_base_moments_observer_is_an_observer_and_moments():
    obs = BaseMomentsObserver(max_size=5)
    assert isinstance(obs, Observer)
    assert isinstance(obs, Moments)


# ============================================================
# observe 状态机
# ============================================================

def test_observe_advances_result_container_and_resets_need_observe():
    obs = BaseMomentsObserver(max_size=5)
    obs.add_result([Message.new().with_content("done")], need_observe=True)
    assert obs.need_observe() is True
    m = obs.observe()
    assert obs.need_observe() is False
    # observe 产出的 moment.previous 衔接上一轮结果.
    assert "done" in _texts(m.previous_result_messages())


def test_observe_appends_to_history_until_max_size():
    obs = BaseMomentsObserver(max_size=2)
    for _ in range(4):
        obs.observe()
    assert len(obs.moments()) == 2


def test_peek_previews_without_entering_history():
    obs = BaseMomentsObserver(max_size=5)
    obs.inject_percepts("preview")
    assert _texts(obs.peek().percepts["MomentsInjectedPercepts"]) == ["preview"]
    # peek 不吞进历史; preview 视图可由 moments(peek=True) 预览.
    assert obs.moments() == []
    assert len(obs.moments(peek=True)) == 1


# ============================================================
# 注入通道
# ============================================================

def test_inject_percepts_land_on_next_observe():
    obs = BaseMomentsObserver(max_size=5)
    obs.inject_percepts("from outside")
    m = obs.observe()
    assert _texts(m.percepts["MomentsInjectedPercepts"]) == ["from outside"]


def test_dynamic_context_func_registers_and_disposes():
    obs = BaseMomentsObserver(max_size=5)
    disposer = obs.with_dynamic_context_func("k", lambda: [Message.new().with_content("live")])
    assert _texts(obs.observe().dynamic_context["k"]) == ["live"]
    disposer()
    assert "k" not in obs.observe().dynamic_context


def test_percepts_buffer_drains_each_observe():
    obs = BaseMomentsObserver(max_size=5)
    shared = [Message.new().with_content("frame")]

    def drain():
        out = list(shared)
        shared.clear()
        return out

    obs.with_percepts_buffer("cam", drain)
    assert _texts(obs.observe().percepts["cam"]) == ["frame"]
    # 上一帧已被 drain, 下一帧无 cam 通道.
    assert "cam" not in obs.observe().percepts


# ============================================================
# 回调
# ============================================================

def test_on_result_add_fires_with_appended():
    obs = BaseMomentsObserver(max_size=5)
    calls = []
    obs.when_result_add(lambda msgs, need: calls.append((_texts(msgs), need)))
    obs.add_result([Message.new().with_content("hi")], need_observe=True)
    assert calls == [(["hi"], True)]


def test_on_result_add_not_fired_for_empty_result():
    """空 append 不 fire 回调, 但 need_observe 信号仍被置位 (观察仍需发生)."""
    obs = BaseMomentsObserver(max_size=5)
    calls = []
    obs.when_result_add(lambda msgs, need: calls.append(msgs))
    obs.add_result([], need_observe=True)
    assert calls == []
    assert obs.need_observe() is True


def test_on_moment_fires_with_produced_moment():
    obs = BaseMomentsObserver(max_size=5)
    seen = []
    obs.when_moment_created(lambda m: seen.append(m))
    m = obs.observe()
    assert seen == [m]


# ============================================================
# compact / recap / clear
# ============================================================

def test_compact_drops_up_to_id_and_records_epoch():
    obs = BaseMomentsObserver(max_size=10)
    for _ in range(3):
        obs.observe()
    target = obs.moments()[1].id
    obs.new_epoch([Message.new().with_content("summary")], target)
    remaining = obs.moments()
    assert all(m.id != target for m in remaining)
    assert len(remaining) == 1
    assert _texts(obs.epoch.recap) == ["summary"]


def test_compact_missing_id_clears_all_moments_and_sets_epoch():
    obs = BaseMomentsObserver(max_size=10)
    obs.observe()
    obs.observe()
    obs.new_epoch([Message.new().with_content("sum")], "does-not-exist")
    assert obs.moments() == []
    assert _texts(obs.epoch.recap) == ["sum"]


def test_clear_resets_state():
    obs = BaseMomentsObserver(max_size=5)
    obs.inject_percepts("p")
    obs.add_result([Message.new().with_content("x")], need_observe=True)
    obs.observe()
    # 让 need_observe 重新为 True, 断言 clear 彻底复位.
    obs.add_result([Message.new().with_content("y")], need_observe=True)
    assert obs.need_observe() is True
    obs.clear()
    assert obs.need_observe() is False
    assert obs.moments() == []
    assert obs.epoch.recap == []


# ============================================================
# turns() — 将 moments 组织为回合历史
# ============================================================

def test_turns_organizes_observed_moments_by_logos():
    obs = BaseMomentsObserver(max_size=10)
    obs.inject_percepts("input 1")
    m1 = obs.observe()
    m1.logos = "logos 1"
    obs.inject_percepts("input 2")
    obs.observe()
    turns = list(obs.turns())
    assert len(turns) == 2
    assert turns[0][1] == "logos 1"
    assert "input 1" in _texts(turns[0][0])
    assert turns[1][1] is None
    assert "input 2" in _texts(turns[1][0])


# ============================================================
# 三角色协程集成 — 不同时序交错
# ============================================================

def test_three_coroutines_run_different_cadences():
    """
    三个协程在同一个 observer 上交错运行, 各自不同时序 (模拟 mindflow 的
    生产 / 消费 / 观测节奏):

    - producer (0.01s): observe 产出 Moment, 并盖 logos 印记
    - consumer (0.005s): add_result 喂入上一轮结果
    - observer (0.02s): 注册 on_moment, 被动记录每个产出的 Moment

    协程只在 await 处协作切换; observe / add_result 内部无 await, 因此对
    observer 的读改写是原子的, 不引入线程级竞态. consumer 10 次 (0.005*10
    = 0.05s) 先于 producer 最后一次 observe (0.01*10 = 0.10s) 结束, 所以
    没有尾随结果, 全量都被捕获. 断言的是跨角色不变式, 与具体交错顺序无关.
    """
    obs = BaseMomentsObserver(max_size=100)
    observed = []

    async def producer(n=10):
        for i in range(n):
            await asyncio.sleep(0.01)
            m = obs.observe()
            m.logos = f"logos-{i}"

    async def consumer(n=10):
        for i in range(n):
            await asyncio.sleep(0.005)
            obs.add_result([Message.new().with_content(f"res-{i}")], need_observe=True)

    async def observer(n=6):
        # 观测角色: 注册同步 on_moment 钩子, 在对方生产 Moment 的瞬间被动记录.
        obs.when_moment_created(lambda m: observed.append(m))
        for _ in range(n):
            await asyncio.sleep(0.02)

    async def _main():
        await asyncio.gather(producer(), consumer(), observer())

    asyncio.run(_main())

    # observer 角色: 每个 observe 恰好触发一次 on_moment.
    assert len(observed) == 10
    assert all(isinstance(m, Moment) for m in observed)
    # producer 角色: 每个产出的 Moment 都被盖了 logos 印记, 顺序 0..9.
    assert [m.logos for m in observed] == [f"logos-{i}" for i in range(10)]
    # consumer 角色: 每条结果都进入某个产出 Moment 的 previous, 不丢失不重复.
    res_texts = _texts([msg for m in observed for msg in m.previous_result_messages()])
    assert len(res_texts) == 10
    for i in range(10):
        assert f"res-{i}" in res_texts


# ============================================================
# result-drain / need_observe 信号契约 (暴露 _peek_moment drain 缺陷)
# ============================================================

def test_add_result_empty_with_observe_sets_flag():
    """add_result([], need_observe=True) 应置位 need_observe — 空信号也需触发观察.

    当前 Results.add_result 只在非空 append 时置位, 空信号丢失.
    """
    obs = BaseMomentsObserver(max_size=5)
    obs.add_result([], need_observe=True)
    assert obs.need_observe() is True


def test_result_drain_observe_signal_reaches_previous():
    """drain 返回 (空消息, observe=True): 产出的 moment.previous.need_observe 应为 True.

    当前 _peek_moment 经 add_result 空 append 丢失该 observe 信号.
    """
    obs = BaseMomentsObserver(max_size=5)
    obs.with_result_drain("k", lambda: ([], True))
    m = obs.observe()
    assert m.previous is not None
    assert m.previous.need_observe is True


def test_result_drain_not_duplicated_across_peek_then_observe():
    """peek 预览后 observe, drain 结果不应被重复累积.

    drain 数据直接落入 Results; 若 observer 另留一份缓冲, observe 会把
    上一帧已消费的数据再捞出一次, 造成重复. 这里守卫零重复.
    """
    obs = BaseMomentsObserver(max_size=5)
    shared = [Message.new().with_content("m")]

    def drain():
        out = list(shared)
        shared.clear()
        return out, True

    obs.with_result_drain("k", drain)
    obs.peek()
    m = obs.observe()
    prev_texts = _texts(m.previous_result_messages())
    assert prev_texts.count("m") == 1, f"drain 结果被重复累积: {prev_texts}"

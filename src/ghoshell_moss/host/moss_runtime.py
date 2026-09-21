"""
MossRuntime concrete — 编排 CTMLShell + Matrix + Mode 生命周期.

wire-up 契约 (§ZZ):
- Matrix 层承接 SystemPrompter 基础契约 (contracts/system_prompter.py),
  MossRuntime 承接 MossSystemPrompter (blueprint/host.py, ctml/project/mode/static
  4 slot 命名访问器) — 跨 cell 展示 feature 过重, 本期不上.
- mode 参数化: mode 是 MossRuntime 关注点, 不进 matrix. matrix 无 .manifests/.mode.
- main channel 从 mode.manifests().channel() 单 Manifest 拿; 无声明用 new_shell_main_channel 兜底.
- static slot 在 shell 起来后接 ctml_shell.static_messages callable (dynamic leaf).
"""
from typing import Callable

from typing_extensions import Self

from pathlib import Path
import janus
import numpy as np

from ghoshell_moss.core.blueprint.shell_trajectory import MShellTrajectory
from ghoshell_moss.message.message import Message
from ghoshell_moss.core.concepts.shell import MOSShell
from ghoshell_moss.core.ctml.shell.ctml_shell import CTMLShell
from ghoshell_moss.core.blueprint.host import (
    MOSShellRuntime, MossSystemPrompter,
)
from ghoshell_moss.core.blueprint.matrix import Matrix
from ghoshell_moss.core.blueprint.project import HostMode
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.cell import CellEventLevel
from ghoshell_moss.core.blueprint.states_channel import new_shell_main_channel
from ghoshell_moss.core.ctml import new_ctml_shell
from ghoshell_moss.core.helpers import ThreadSafeEvent
from ghoshell_moss.contracts import Workspace, SystemPrompter, BaseSystemPrompter
from ghoshell_moss.contracts.audio import (
    AUDIO_SAMPLE_INTERVAL,
    AudioCaptureSource,
    LatestAudioWindow,
    compute_spectrum,
    resample,
)
from ghoshell_moss.contracts.configs import ConfigInstanceRegisterBootstrapper
from ghoshell_moss.contracts.listener import ASRListener, ListenLifecycle
from ghoshell_moss.contracts.resource import ResourceStorageFactoryBootstrapper
from ghoshell_moss.contracts.speech import Speech, SpeechClause, TTSSpeech, PlaybackSample
from ghoshell_moss.types.topics import AudioSampleTopic, ClauseTopic

from ghoshell_moss.matrix.matrix_impl import MatrixImpl

import contextlib
import asyncio

__all__ = ['ShellRuntimeImpl']


class _MossSystemPrompterImpl(BaseSystemPrompter, MossSystemPrompter):
    """具象化 MossSystemPrompter — BaseSystemPrompter 提供 tree/组装, MossSystemPrompter
    提供命名 slot API. 钻石继承合流, 一实例既作 SystemPrompter 又作 MossSystemPrompter.
    """
    pass


class ShellRuntimeImpl(MOSShellRuntime):

    def __init__(
            self,
            *,
            env: Environment,
            workspace: Workspace,
            mode: HostMode,
            matrix: MatrixImpl,
            run_shell_on_start: bool = True,
            speech: bool = True,
            listen: bool = False,
            name: str | None = None,
            description: str | None = None,
    ):
        # env 已由 Host 侧 seal (Host.__init__ 承担). 这里假设 env 已 sealed.
        # 二次 seal 会抛 EnvironmentSealedError.
        self._env = env
        self._workspace = workspace
        self._mode = mode
        self._matrix = matrix
        self._name = name or env.moss_meta.name
        # 描述发现三级优先: 传参 > mode > env moss_meta.
        self._description = (
            description
            or mode.meta.description
            or env.moss_meta.description
        )
        self._run_shell_on_start = run_shell_on_start
        # speech 开关 (bool): True 时在 __aenter__ resolve Speech 实例注入 shell.
        self._speech_enabled = speech
        self._speech: Speech | None = None
        # listen 开关 (bool): True 时在 __aenter__ resolve ASRListener 并组装 controller.
        self._listen_enabled = listen
        self._listen_controller: ListenLifecycle | None = None

        # --- mode 层 IoC 叠加 (§ZZ-5: mode providers/configs/resources 覆盖 baseline) --- #
        # container 已在 MatrixImpl.__init__ 创建并完成 baseline 注册,
        # 此时注册 mode 层: providers 走 register (覆盖语义), configs/resources 走
        # bootstrapper (bootstrap 时合并).
        self._wire_mode_overlays()

        self._async_exit_stack = contextlib.AsyncExitStack()
        self._started = False
        self._paused = False
        self._closing_event = ThreadSafeEvent()
        self._closed_event = ThreadSafeEvent()
        self._log_prefix = (
            f"<HostMossRuntime mode={self._mode.name} "
            f"network_scope={self._env.network_scope}>"
        )
        self._interpreting_future: asyncio.Future | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None
        self._action_task: asyncio.Task | None = None
        self._bringup_tasks: set[asyncio.Task] = set()

        # --- shell action loop --- #
        self._shell_logos_queue: janus.Queue = janus.Queue()

        # --- system prompter (matrix 层不构造, MossRuntime 层持有全 4 slot tree) --- #
        # ctml/project/mode 三 slot 在此填充; static slot 在 _bootstrap_after_matrix 补
        # (依赖 ctml_shell 起来).
        self._system_prompter: _MossSystemPrompterImpl = self._build_system_prompter()

        # --- prepare shell --- #
        # main channel 从 mode.manifests().channel() 单 Manifest 拿 (HostModeManifests
        # 由 MossRuntime 承接). 无声明用默认空白 main 兜底.
        manifests_main = self._discover_main_channel()
        if manifests_main is None:
            manifests_main = new_shell_main_channel(
                description=f"Default main channel for {self._description or self._name}",
            )
        self._ctml_shell = new_ctml_shell(
            name=self._name,
            description=self._description,
            parent_container=self.matrix.container,
            main_channel=manifests_main,
            experimental=False,
            meta_instruction=self._system_prompter.instruction(),
        )

    def _build_system_prompter(self) -> _MossSystemPrompterImpl:
        """构造 MossSystemPrompter tree 的 ctml/project/mode 三 slot.

        static slot 依赖 ctml_shell.static_messages, 在 _bootstrap_after_matrix 补.
        """
        prompter = _MossSystemPrompterImpl(
            description=(
                "MOSS system instruction — assembled from ctml/project/mode/static layers."
            ),
        )
        # ctml slot: 版本优先 mode > moss_meta
        ctml_version = self._mode.meta.ctml_version or self._env.moss_meta.ctml_version
        ctml_prompt = self._load_ctml_prompt(ctml_version)
        prompter.with_prompter(
            MossSystemPrompter.MOSS_SLOT,
            BaseSystemPrompter(
                own_instruction=ctml_prompt,
                description=f"CTML grammar prompt (version {ctml_version}).",
            ),
        )
        # project slot: 环境身份帧 + workspace 根 MOSS.md 声明的 project instruction.
        # 帧固定英文, 不读 MOSS.md 定制 — 隐式约定等于不存在.
        project_meta = (
            f"> Here is project `{self._env.moss_meta.name}` — "
            f"an environment driven by the MOSS (ghoshell_moss) framework."
        )
        project_body = self._env.moss_meta.system_project
        prompter.with_prompter(
            MossSystemPrompter.PROJECT_SLOT,
            BaseSystemPrompter(
                own_instruction=(
                    project_meta
                    if not project_body
                    else f"{project_meta}\n\n{project_body}"
                ),
                description="Workspace root MOSS.md project instruction.",
            ),
        )
        # mode slot: 环境身份帧 + 模式内 HOST.md 声明的 instruction.
        mode_meta = (
            f"> Here is `{self._mode.name}` mode — "
            f"an isolated runtime of the MOSS (ghoshell_moss) framework."
        )
        mode_body = self._mode.meta.system_prompt
        prompter.with_prompter(
            MossSystemPrompter.MODE_SLOT,
            BaseSystemPrompter(
                own_instruction=(
                    mode_meta
                    if not mode_body
                    else f"{mode_meta}\n\n{mode_body}"
                ),
                description=f"Mode '{self._mode.name}' instruction.",
            ),
        )
        return prompter

    def _load_ctml_prompt(self, ctml_version: str) -> str:
        """从 project.ctml_versions() 加载指定版本的 CTML meta instruction 全文.

        CTML 是 MOSS 的通讯根基 — 版本查不到 / 文件读不出都是 unrecoverable,
        故意让 MossRuntime 构造期崩溃, 保证不会有半死状态的 shell 起来后模型
        无法理解 CTML 语法. 错误信息即 prompt (TT-12), 指向 project.ctml_versions()
        与 MOSS.md ctml_version 声明.
        """
        try:
            versions = self._matrix.project.ctml_versions()
        except Exception as e:
            raise RuntimeError(
                f"MOSS cannot start: project.ctml_versions() failed to load "
                f"({type(e).__name__}: {e}). Check workspace ctml/ dir + bundled versions."
            ) from e
        ctml_file = versions.get(ctml_version)
        if ctml_file is None:
            raise RuntimeError(
                f"MOSS cannot start: CTML version {ctml_version!r} not found. "
                f"Available versions: {sorted(versions.keys())}. "
                f"Fix MOSS.md ctml_version or mode HOST.md ctml_version field."
            )
        try:
            content = ctml_file.read_text(encoding='utf-8')
        except Exception as e:
            raise RuntimeError(
                f"MOSS cannot start: CTML version {ctml_version!r} file {ctml_file} "
                f"read failed ({type(e).__name__}: {e})."
            ) from e
        if not content.strip():
            raise RuntimeError(
                f"MOSS cannot start: CTML version {ctml_version!r} file {ctml_file} "
                f"is empty. CTML meta instruction is required for shell startup."
            )
        return content

    def _wire_mode_overlays(self) -> None:
        """在 matrix.container 上叠加 mode 层 providers/configs/resources.

        MatrixImpl.__init__ 已完成 baseline 注册, container 存在但未 bootstrap.
        mode providers 走 register (后注册覆盖同 contract 的 baseline),
        mode configs/resources 走 bootstrapper (bootstrap 时与 baseline 合并).
        nuclei 不在此处理 — 归 GhostRuntime.
        """
        container = self._matrix.container
        manifests = self._mode.manifests()

        # -- mode providers: register 覆盖同 contract baseline -- #
        for p in manifests.providers():
            if p.is_error():
                continue
            container.register(p.value())

        # -- mode configs: bootstrapper 追加到 baseline ConfigStore -- #
        mode_configs = [
            m.value() for m in manifests.configs()
            if not m.is_error()
        ]
        if mode_configs:
            container.add_bootstrapper(
                ConfigInstanceRegisterBootstrapper(*mode_configs),
            )

        # -- mode resources: bootstrapper 追加到 baseline resource factories -- #
        for r in manifests.resources():
            if r.is_error():
                continue
            container.add_bootstrapper(
                ResourceStorageFactoryBootstrapper(r.value()),
            )

    async def _bringup_one(self, target: str) -> None:
        """发起单个 node 的拉起; 失败记日志 + 广播事件, 不带倒其余 node.

        matrix 已启动时运行, run_node 依赖 matrix 运行态. mode bringup 是启动面, 没有
        直接调用方拿返回 (channel 侧 nodes:run 已用 raise_observe 兜底), 故失败额外
        publish 一个 ERROR 级 CellEvent 通知启动中的 ghost.
        """
        try:
            await self._matrix.run_node(Path(target))
        except Exception as e:
            self._matrix.logger.exception("bringup node failed: %s", target)
            await self._publish_bringup_failure(target, e)

    async def _publish_bringup_failure(self, target: str, exc: Exception) -> None:
        """把 bringup 失败广播成 CellEvent — 事件是瞬态 best-effort, 失败只记日志."""
        reason = str(exc).strip()[:200] or type(exc).__name__
        content = f'bringup node failed: {target}: {reason}'
        try:
            await self._matrix.publish_event(content, event_level=CellEventLevel.ERROR)
        except Exception:
            self._matrix.logger.exception(
                "publish bringup failure failed: %s", target,
            )

    def _start_bringup_tasks(self) -> None:
        """mode 声明的 nodes 各起一个后台 task — 并行 fire, 不 await, 无顺序语义.

        node 不做 DAG 启动图 (依赖组织未来交给特殊 node, 不在 bringup 里), 故这里逐条
        独立发起、互不阻塞: 一个 node 的 probe 挂死或失败不拖累其余. spawn 之后的存活/
        退出治理已归 matrix (handle 登记 + _on_cell_exit). 取消经 exit stack 回调,
        排在 matrix teardown 之前.
        """
        loop = asyncio.get_running_loop()
        for target in self._mode.meta.bringup_nodes:
            task = loop.create_task(
                self._bringup_one(target),
                name=f'bringup:{self._mode.name}:{target}',
            )
            # _bringup_one 吞掉 Exception (CancelledError 是 BaseException, 不吞), 故 task
            # 不抛、无需 done callback 记异常; 只借它把完成的 task 移出 set.
            self._bringup_tasks.add(task)
            task.add_done_callback(self._bringup_tasks.discard)
        if self._bringup_tasks:
            self._async_exit_stack.push_async_callback(self._cancel_bringup_tasks)

    async def _cancel_bringup_tasks(self) -> None:
        tasks = list(self._bringup_tasks)
        self._bringup_tasks.clear()
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            # asyncio.wait 不抛 task 自身的 CancelledError; 外层若正被取消仍正常传播.
            await asyncio.wait(tasks)

    def _discover_main_channel(self):
        """从 mode.manifests().channel() 拿 main channel Manifest.

        新 ABC (HostModeManifests): channel() -> Manifest[PrimeChannel] 单值.
        老 API .channels().values() 已废, 老代码 next(...) 迭代形态一并作废.
        """
        try:
            manifests = self._mode.manifests()
        except Exception:
            return None
        try:
            channel_manifest = manifests.channel()
        except Exception:
            return None
        if channel_manifest is None or channel_manifest.is_error():
            return None
        try:
            return channel_manifest.value()
        except Exception:
            return None

    @property
    def name(self) -> str:
        return self._name or self._env.moss_meta.name

    @property
    def description(self) -> str:
        return self._description or self._env.moss_meta.description

    @property
    def mode(self) -> HostMode:
        return self._mode

    def _check_running(self):
        if not self.is_running():
            raise RuntimeError('MossRuntime is not running.')

    def _check_shell_running(self):
        if not self.is_running() or not self._ctml_shell.is_running():
            raise RuntimeError('MossRuntime Shell is not running.')

    @property
    def env(self) -> Environment:
        return self._env

    def instruction(self, with_static: bool = True) -> str:
        self._check_shell_running()
        instructions = [self._ctml_shell.meta_instruction()]

        if with_static:
            if static_messages := self._ctml_shell.static_messages().strip():
                instructions.append("# MOSS static\n\n" + static_messages)
        return "\n\n".join(instructions)

    async def dynamic_messages(self, refresh: bool = True, max_wait: float = 2.0) -> list[Message]:
        self._check_shell_running()
        await self._ctml_shell.refresh_metas(max_wait)
        return self._ctml_shell.dynamic_messages()

    def static_messages(self) -> str:
        return self._ctml_shell.static_messages()

    async def refresh_metas(self, timeout: float = 2.0) -> None:
        self._check_shell_running()
        await self._ctml_shell.refresh_metas(timeout)

    async def observe(
            self,
            timeout: float | None = None,
            with_dynamic: bool = True,
    ) -> list[Message]:
        self._check_shell_running()
        if interpreter := self._ctml_shell.interpreting():
            messages = interpreter.interpretation().status_messages()
        else:
            messages = []

        if with_dynamic:
            await self._ctml_shell.refresh_metas()
            dynamic_messages = self._ctml_shell.dynamic_messages()
            messages.extend(dynamic_messages)
        return messages

    async def exec_logos(
            self,
            logos: str,
            call_soon: bool = True,
            wait_done: bool = True,
    ) -> list[Message]:
        self._check_shell_running()
        self._check_running()
        interpreter = await self._ctml_shell.interpreter(
            kind='clear' if call_soon else 'append',
            clear_after_exit=False,
        )
        interpretation = interpreter.interpretation()
        async with interpreter:
            interpreter.feed(logos)
            await interpreter.wait_compiled()
            if wait_done:
                await interpreter.wait_stopped()
        return interpretation.as_messages()

    async def interrupt(self) -> list[Message]:
        self._check_running()
        await self._ctml_shell.clear()
        interpreter = self._ctml_shell.interpreting()
        if interpreter is None:
            return [Message.new().with_content('no logos are executing')]
        return interpreter.interpretation().executed_messages()

    def is_running(self) -> bool:
        return self._started and not (
                self._closing_event.is_set() or self._closed_event.is_set()
        )

    def wait_close_sync(self, timeout: float | None = None) -> bool:
        return self._closing_event.wait_sync(timeout)

    async def wait_close(self) -> None:
        await self._closing_event.wait()

    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        return self._closed_event.wait_sync(timeout)

    async def wait_closed(self) -> None:
        await self._closed_event.wait()

    def close(self) -> None:
        self._closing_event.set()

    def pause(self, toggle: bool = True) -> None:
        self._check_running()
        self._ctml_shell.pause(toggle)
        if self._listen_controller is not None:
            self._listen_controller.pause(toggle)
        self._paused = toggle

    @property
    def shell(self) -> MOSShell:
        self._check_running()
        return self._ctml_shell

    @property
    def matrix(self) -> Matrix:
        return self._matrix

    def _bootstrap_after_matrix(self) -> None:
        # ctml_shell 起来后, 补 static slot (dynamic leaf: ctml_shell.static_messages
        # 是 callable, 每次读取时动态计算); 注册 SystemPrompter / MossSystemPrompter
        # 两个 IoC key 指向同一实例 (钻石继承).
        self._matrix.container.set(MOSShell, self._ctml_shell)
        self._matrix.container.set(CTMLShell, self._ctml_shell)
        self._system_prompter.with_prompter(
            MossSystemPrompter.MOSS_STATIC_SLOT,
            self._ctml_shell.static_messages,
        )
        self._matrix.container.set(SystemPrompter, self._system_prompter)
        self._matrix.container.set(MossSystemPrompter, self._system_prompter)

    def _resolve_speech(self) -> None:
        """resolve Speech 实例 (内核 contract) — host 的 bool 开关决定是否启用.

        True → 从 matrix container resolve; 失败降 None (降级细节见工作项 #7).
        False → None (禁语音). 结果注入 shell, 旁路桥复用同一实例.
        """
        if not self._speech_enabled:
            self._speech = None
        else:
            try:
                self._speech = self._matrix.container.get(Speech)
            except Exception:
                self._matrix.logger.exception("%s resolve speech failed — degraded to no speech", self._log_prefix)
                self._speech = None
        self._ctml_shell.set_speech(self._speech)

    def _resolve_listener(self) -> None:
        """resolve ASRListener 并组装 ListenerController — host 的 bool 开关决定是否启用.

        True → 从 matrix container resolve ASRListener, 组装 ListenerController
        (仅生命周期表面 ListenLifecycle, 见工作项 #12). 失败降 None.
        False → None (不听). AEC far 桥在 _listen_lifecycle.
        """
        if not self._listen_enabled:
            self._listen_controller = None
            return
        try:
            listener = self._matrix.container.get(ASRListener)
        except Exception:
            self._matrix.logger.exception("%s resolve listener failed — degraded to no listen", self._log_prefix)
            self._listen_controller = None
            return
        from ghoshell_moss.host.listener.controller import ListenerController
        self._listen_controller = ListenerController(
            listener=listener,
            asr=listener.asr(),
            logger=self._matrix.logger,
        )
        # 单例注册: TUI voice state / 其它消费面从 container 拿同一个 controller.
        self._matrix.container.set(ListenerController, self._listen_controller)
        # 听侧 channel 挂进 shell main — 模型看到"一个语音面"的命令 (activate/stop/
        # get_etiquette/get_transcript/configure_asr), 受 pause 人类锁的 available 门控.
        self._ctml_shell.main_channel.import_channels(self._listen_controller.as_channel())

    @contextlib.asynccontextmanager
    async def _manager_shell_lifecycle(self):
        if self._run_shell_on_start:
            await self._ctml_shell.__aenter__()
            # kick off first round refresh_meta
            await self._ctml_shell.refresh_metas(0.5)
        try:
            yield
        finally:
            if self._ctml_shell.is_running():
                await self._ctml_shell.__aexit__(None, None, None)

    @contextlib.asynccontextmanager
    async def _clause_topic_bridge(self):
        """说侧旁路生命周期: 把 speech 单例产出的 clause 发布成 ClauseTopic.

        speech 的 on_clause 在 audio worker 线程回调 (与 on_sample 一致), 这里经
        janus 队列 marshal 回事件循环, 再由 TopicService 的 publisher 广播. 仅当
        speech 是 TTSSpeech (真产出 clause) 时激活 — NullSpeech/MockSpeech 无 clause,
        直接跳过, 不为它们空转 queue / publisher.
        """
        speech = self._speech
        if not isinstance(speech, TTSSpeech):
            yield
            return

        speaker_id = self._env.project_id
        speaker_name = self._env.ghost_name
        publisher = self._matrix.session.topics.model_publisher(
            creator=f"ghost/{speaker_name}",
            model=ClauseTopic,
        )
        queue: janus.Queue = janus.Queue()

        def _on_clause(clause: SpeechClause) -> None:
            # on_clause 由 audio worker 线程触发, 走 sync_q 线程安全入队.
            queue.sync_q.put_nowait(ClauseTopic(
                text=clause.text,
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                role='ghost',
            ))

        async def _drain() -> None:
            while True:
                topic = await queue.async_q.get()
                publisher.pub(topic)

        await publisher.__aenter__()
        disposer = speech.on_clause(_on_clause)
        drain_task = asyncio.create_task(_drain())
        try:
            yield
        finally:
            disposer()
            drain_task.cancel()
            try:
                await drain_task
            except asyncio.CancelledError:
                pass
            await publisher.__aexit__(None, None, None)

    @contextlib.asynccontextmanager
    async def _audio_sample_topic_bridge(self):
        """说侧旁路生命周期: 把 player 实际播放的音频按 ~200ms 窗口广播成 AudioSampleTopic (role=ghost).

        与 clause 桥对称, 但用 LatestAudioWindow (latest-value-wins, 无队列). player.observe
        在 audio worker 线程回调, 经窗口的锁 marshal; 周期 task 在事件循环取走算频谱发布.
        """
        speech = self._speech
        if not isinstance(speech, TTSSpeech):
            yield
            return

        speaker_name = self._env.ghost_name
        player = speech.player()
        sample_rate = player.sample_rate
        publisher = self._matrix.session.topics.model_publisher(
            creator=f"ghost/{speaker_name}",
            model=AudioSampleTopic,
        )
        window = LatestAudioWindow()

        def _on_sample(sample: PlaybackSample) -> None:
            if not sample.pcm:
                return
            window.append(np.frombuffer(sample.pcm, dtype=np.int16))

        async def _emit() -> None:
            while True:
                await asyncio.sleep(AUDIO_SAMPLE_INTERVAL)
                pcm = window.take()
                if pcm is None:
                    continue
                spectrum = compute_spectrum(pcm)
                publisher.pub(AudioSampleTopic(
                    role="ghost",
                    sample_rate=sample_rate,
                    duration=len(pcm) / sample_rate if sample_rate else 0.0,
                    rms_db=spectrum.rms_db,
                    peak=spectrum.peak,
                    spectrum_bins=spectrum.spectrum_bins,
                    n_spectrum_bins=len(spectrum.spectrum_bins),
                    waveform=spectrum.waveform,
                    n_waveform=len(spectrum.waveform),
                ))

        await publisher.__aenter__()
        disposer = player.observe(_on_sample)
        emit_task = asyncio.create_task(_emit())
        try:
            yield
        finally:
            disposer()
            emit_task.cancel()
            try:
                await emit_task
            except asyncio.CancelledError:
                pass
            await publisher.__aexit__(None, None, None)

    def _wire_aec_far(self) -> Callable[[], None]:
        """AEC far 桥: player.on_play → capture.set_aec 的 echo 参考 (near 在 capture 内).

        仅当 speech 是 TTSSpeech (有 player 产 far) 时激活. AEC 采样率取 capture 原生率
        (near 免重采样), far (player) 重采样到 AEC 率 + 归一化 [-1,1]. 返回 cleanup
        (unmount AEC + dispose on_play 摘除 far 回调).
        """
        speech = self._speech
        if not isinstance(speech, TTSSpeech):
            return lambda: None
        from ghoshell_moss.host.listener.capture.webrtc_aec import PyWebrtcEchoCanceller

        capture = self._matrix.container.get(AudioCaptureSource)
        aec = PyWebrtcEchoCanceller(sample_rate=capture.sample_rate, stream_delay_ms=0)
        capture.set_aec(aec)

        player = speech.player()
        play_rate = player.sample_rate
        aec_rate = aec.sample_rate

        def _on_play(frame: np.ndarray) -> None:
            arr = np.asarray(frame).ravel()
            if play_rate != aec_rate:
                arr = resample(arr.astype(np.int16), origin_rate=play_rate, target_rate=aec_rate)
            aec.push_far(arr.astype(np.float32) / 32768.0)

        dispose_on_play = player.on_play(_on_play)

        def _cleanup() -> None:
            capture.set_aec(None)
            dispose_on_play()

        return _cleanup

    @contextlib.asynccontextmanager
    async def _listen_lifecycle(self):
        """听侧治理: enter ListenerController (启动 capture+asr) + wire AEC far 桥 + clause topic.

        仅当 listener resolve 成功 (controller 非 None) 时激活. AEC far 桥在 controller
        之前 wire, 保证 capture 启动的首帧就已消回声. clause topic 装线在 controller
        生命周期内, 识别到的 CLAUSE 广播成 ClauseTopic(role=user), 与说侧桥的
        role=ghost 汇成同一条交错对话轨迹.
        """
        controller = self._listen_controller
        if controller is None:
            yield
            return
        aec_cleanup = self._wire_aec_far()
        try:
            async with controller:
                await controller.with_topic_service(self._matrix.session.topics)
                yield
        finally:
            aec_cleanup()

    async def __aenter__(self) -> Self:
        if self._started:
            raise RuntimeError('MossRuntime is already started')
        self._started = True
        await self._async_exit_stack.__aenter__()
        # 启动 matrix
        await self._async_exit_stack.enter_async_context(self._matrix)
        # 补 IoC 注册 (system prompter / MOSShell) — 之前挂 _app_store 的位置
        self._bootstrap_after_matrix()
        # resolve Speech 实例 (内核 contract) 注入 shell — 须在 shell __aenter__ 之前.
        self._resolve_speech()
        # resolve ASRListener 实例 (内核 contract) — 治理在 voice listen manager.
        self._resolve_listener()
        # 启动 ctml shell
        await self._async_exit_stack.enter_async_context(self._manager_shell_lifecycle())
        # 说侧旁路: speech 单例的 clause 结果 → ClauseTopic 广播 (在 shell 起、speech 已
        # start 之后进入; exit stack LIFO 保证它在 shell/speech 关闭之前先退出).
        await self._async_exit_stack.enter_async_context(self._clause_topic_bridge())
        # 说侧旁路: player 实际播放的音频 → AudioSampleTopic 广播 (对称 clause 桥).
        await self._async_exit_stack.enter_async_context(self._audio_sample_topic_bridge())
        # 听侧旁路: enter ListenerController (启动 capture+asr) + AEC far 桥 — 说侧桥之后
        # 进入, LIFO 先退出 (AEC 拆线时 player 仍活着).
        await self._async_exit_stack.enter_async_context(self._listen_lifecycle())
        # bringup: 后台 task 并行发起 mode 声明的 nodes, 不 await — 单个失败记日志,
        # 挂死 (如 probe 不退出) 只钉住自己的 task, 不再阻塞 shell 启动.
        self._start_bringup_tasks()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # 进入即标记 closing, 通知依赖方提前结束运行时逻辑.
        self._closing_event.set()
        self._matrix.close()
        try:
            await self._async_exit_stack.__aexit__(exc_type, exc_val, exc_tb)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._matrix.logger.exception("%s failed to aexit %s", self._log_prefix, e)
            raise e
        finally:
            self._closed_event.set()

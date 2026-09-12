"""
Matrix — the projection of the MOSS network inside a Cell.

A network may host many MOSS instances (Hosts) and capability units (Cells) at once.
Inside a cell, the Matrix abstraction is how the cell holds its identity, exposes its
membrane, observes the network, and launches and governs new processes. Matrix is a
facade: you hold it and call it.
Naming anchor: a Matrix instance is "the whole projected into a part" — the cave light
projecting the shape of the hive. Human language routinely takes a projection for the entity
(pointing at the code stream on screen and calling it "matrix"), and that equivalence has
philosophical substance. Matrix is not mesh, nor a mesh client — mesh is only one of the
sources it projects from.
"""
import dataclasses
from typing import Literal, Callable, Awaitable, Any, Coroutine, Protocol, TypeAlias, Type, TYPE_CHECKING
from typing_extensions import Self
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from mcp.server.mcpserver import MCPServer

from ghoshell_moss.core.concepts.channel import Channel
from ghoshell_moss.core.blueprint.session import Session
from ghoshell_moss.core.blueprint.warrant import Warrant
from ghoshell_moss.core.blueprint.parameter import Parameters
from ghoshell_moss.core.blueprint.cell import Cell, CellNetwork, CellAddress, CellRuntimeInfo, CellEventLevel
from ghoshell_moss.core.blueprint.environment import Environment
from ghoshell_moss.core.blueprint.project import Project, NetworkMetadata
from ghoshell_moss.core.blueprint.service import ServiceOperator, ServiceClient, ServiceServer
from ghoshell_moss.contracts import Workspace, ResourceRegistry, ConfigStore
from ghoshell_moss.contracts.subprocesses import Subprocesses, ManagedProcess, ProcessMeta
from ghoshell_container import IoCContainer, Contracts
from pathlib import Path
import asyncio
import logging

Facade = ABC
"""Facade marks a "consumer surface": you hold it and call it, you do not inherit it."""

__all__ = ['Matrix', 'MatrixLifecycleObject', 'RuntimeScopeKey', 'CellHandle']


class MatrixLifecycleObject(Protocol):
    """A key runtime object whose lifecycle is registered with Matrix.

    Matrix owns its startup and shutdown, starting objects in registration order.
    """

    @abstractmethod
    async def __aenter__(self) -> Self:
        pass

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


RuntimeScopeKey: TypeAlias = Literal['ghost', 'mode', 'network', 'cell']


@dataclasses.dataclass
class CellHandle:
    """
    A cell handle as seen by the parent process: cell identity (runtime) combined with the
    subprocess handle (process).

    Started by ``matrix.run_node`` and tracked by ``matrix.handled_cells``. ``stop`` / ``wait``
    are forwarding sugar over the process side; operating on ``process`` directly is also legal.
    """
    runtime: CellRuntimeInfo
    process: ManagedProcess

    @property
    def address(self) -> CellAddress:
        return self.runtime.address

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop the subprocess gracefully (SIGTERM → killpg on timeout). Same semantics as process.stop."""
        await self.process.stop(timeout)

    async def wait(self) -> ProcessMeta:
        """Block until the subprocess exits and return its exit meta."""
        await self.process.process.wait()
        return self.process.meta


class Matrix(Facade):
    """
    This process's projection of the MOSS communication matrix. A process-level singleton that
    discovers itself from the environment.

    Its members are the cognitive map (code as prompt):
    identity (this/env/project/network), capability exposure (provide_channel), debug
    announcements (publish_event), observation (network), governance (run_node),
    general-purpose modules (processes), key protocol entry points
    (session/workspace/home/container), lifecycle (arun/run/close/...).

    When developing a Cell, follow the capability map Matrix exposes: pick what you need,
    then expand your exploration from there.
    """

    # -- composition root -- #

    @classmethod
    def discover(
            cls,
            *,
            env: Environment | None = None,
    ) -> Self:
        """
        Get the Matrix instance of the current process.

        One instance per cell identity per process (prevents duplicate network joins and
        workspace lock contention). Discovering again after the ``async with`` block exits
        builds a new one — a closed matrix is never returned. When developing, focus on the
        API provided and stay out of how Matrix is constructed.
        """
        # Anti-pattern: the abstraction is executable. Track down the real factory code when
        # you need the details. The factory functions are patchable.
        from ghoshell_moss.factory import create_matrix, create_project
        global _instance
        if _instance is not None:
            return _instance
        env = env or Environment.discover()
        project = create_project(env)
        project.bootstrap()
        _instance = create_matrix(env, project)
        return _instance

    @classmethod
    def new(
            cls,
            node_name: str,
            *,
            description: str = '',
            category: str = '',
            env: Environment | None = None,
            persist: bool = False,
            singleton: bool | None = None,
    ) -> Self:
        """Define a Node in the runtime environment the given way, and get a matrix instance from it."""
        from ghoshell_moss.factory import create_matrix, create_project
        from ghoshell_moss.core.blueprint.cell import NodeManifest, build_cell_from_node, CellRuntimeInfo
        global _instance
        if _instance is not None:
            raise RuntimeError(f"The Matrix is already running: %s", _instance.this)
        env = env or Environment.discover()
        node = NodeManifest.new(node_name, description=description, category=category)
        node.persist = persist
        # singleton: passed explicitly → overrides the manifest declaration; omitted (None) →
        # respects the NodeManifest default (True). Persistence (persist) and hardware
        # exclusivity (singleton) are orthogonal; persist no longer decides concurrency.
        if singleton is not None:
            node.singleton = singleton
        cell = build_cell_from_node(env, node)
        runtime_info = CellRuntimeInfo.from_cell(cell)
        project = create_project(env)
        project.bootstrap()
        _instance = create_matrix(env, project, runtime_info)
        return _instance

    @classmethod
    def reset_discover_instance(cls, instance: 'Matrix') -> None:
        """Reset the discover cache after a matrix exits — a closed instance is never returned again."""
        global _instance
        if _instance is instance:
            _instance = None

    def contracts(self) -> 'Contracts':
        """
        The set of IoC dependencies matrix promises — validated at assembly time
        (``MatrixImpl.__init__``).

        A missing entry fails construction, so new infrastructure dependencies belong in this
        set. Absence from the environment is fail-fast, never deferred to the first force_fetch.
        """
        from ghoshell_common.contracts import LoggerItf
        from ghoshell_moss.core.blueprint.project import Project
        from ghoshell_moss.core.blueprint.environment import Environment
        from ghoshell_moss.core.blueprint.cell import Cell, CellAddress
        from ghoshell_moss.core.blueprint.session import Session
        from ghoshell_moss.core.concepts.topic import TopicService
        from ghoshell_moss.core.concepts.qa import QAManager
        from ghoshell_moss.contracts.workspace import Workspace
        from ghoshell_moss.contracts.subprocesses import Subprocesses
        from ghoshell_moss.contracts.configs import ConfigStore
        from ghoshell_moss.contracts.resource import ResourceRegistry
        import logging
        contracts = [
            # 1. blueprint infrastructure
            Project, Environment, Matrix, Cell, CellAddress,
            # 2. session support
            Session, TopicService, QAManager,
            # 3. contract support
            Workspace, Subprocesses, LoggerItf, logging.Logger,
            ConfigStore, ResourceRegistry,
        ]

        return Contracts.new(*contracts)

    # -- identity -- #

    @property
    @abstractmethod
    def env(self) -> Environment:
        """
        Process environment information for this matrix. The details are rarely needed —
        explore it when your logic depends on environment variables or discovered paths.
        """
        pass

    @property
    @abstractmethod
    def project(self) -> Project:
        """ Where this Matrix node (cell) sits, including related file paths and environment discovery. Explore only when building project-level capabilities. """
        pass

    @property
    def project_home(self) -> Path:
        """Root directory of the project moss lives in."""
        return self.project.root

    @property
    def workspace(self) -> Workspace:
        """moss's own workspace inside the current project — space shared by all Cells of that project."""
        return self.project.workspace

    @property
    @abstractmethod
    def this(self) -> Cell:
        """
        Identity of the current process — the Cell — inside the Matrix network. Anything on the
        network has an identity. If Matrix is a city, this Cell is the room the current process
        sits in.
        """
        pass

    @property
    def home(self) -> Path:
        """
        This cell's persistent territory — where state that outlives a run (memory, config,
        data) belongs.

        Defaults to ``self.this.home``. A cell that needs a fuller directory structure is
        usually an independent project itself (with its own ``.moss``) that rediscovers from
        Environment.
        """
        return Path(self.this.home)

    @property
    def mode_home(self) -> Path:
        """ Workspace of the current moss mode (moss host mode). """
        return self.env.mode_home

    @property
    def ghost_home(self) -> Path:
        """ Workspace of the current ghost. Ghost and mode are orthogonal, each with its own workspace. """
        return self.env.ghost_home

    @property
    @abstractmethod
    def cell_workspace(self) -> Workspace:
        """
        This cell's own isolated workspace, rooted at the cell home.

        Unlike ``workspace`` (shared at project level), ``cell_workspace`` isolates configs,
        assets and runtime data per cell. ``configs`` reads the ``configs/`` directory under
        the cell's own home.
        """
        ...

    @property
    def configs(self) -> ConfigStore:
        return self.project.configs

    @property
    @abstractmethod
    def network_info(self) -> NetworkMetadata:
        """
        Configuration metadata of the network this matrix has joined.
        A Cell process rarely needs the details — check it when your logic concerns the
        network itself.
        """
        pass

    # -- this cell's network-facing side -- #

    @abstractmethod
    def provide_channel(self, channel: Channel) -> asyncio.Future[None]:
        """
        Expose the current process's capabilities to the network through a moss channel, for a
        Ghost (persistent agent) to use. How channels expose capabilities: see channel_builder.
        How a model drives channels: see ctml.

        A Cell can expose exactly one channel root (a tree), so this method may be called only
        once. Await it to block until the process is shut down externally. The exposure is
        announced on the network automatically.
        """
        pass

    @abstractmethod
    async def publish_event(
            self,
            content: str,
            *,
            event_level: CellEventLevel | None = None,
    ) -> None:
        """
        Broadcast a lightweight event from this cell to the network. A Ghost (the network's
        sovereign) can perceive it.

        event_level: override this cell's default event_level for this one event.
        None = use the cell's own level.
        """
        pass

    # -- observation: a lazy view of the network --

    @abstractmethod
    async def network(self) -> CellNetwork:
        """
        API to observe information about every Cell on the network.
        Obtain it only when you need live cell state reflected dynamically.
        """
        # Lazy-loaded module: the first use must await construction.
        pass

    @abstractmethod
    async def run_node(
            self,
            target: Path,
            *,
            extra_env: dict[str, str] | None = None,
    ) -> CellHandle:
        """
        Start a node cell subprocess governed by this matrix.

        :param target: path to the node.
            - absolute path: used as-is
            - relative path: resolved against project.root, then made absolute immediately
            A path to NODE.md → that declaration's entry point;
            a directory → the NODE.md inside it;
            a script → ``NodeManifest.from_script`` walks up to claim its parent.
        :param extra_env: extra environment variables injected into the subprocess. MOSS
            runtime environment variables are inherited by default.
        :return CellHandle: cell identity + subprocess handle. wait/stop go through the handle.
            Whether the subprocess joins the network (runs matrix and announces) is observed
            on the network, not guaranteed here.

        :raise FileNotFoundError: target does not exist after resolution.
        :raise RuntimeError: node is not installed (the message carries the absolute INSTALL.md path).

        Subprocess failure is observed through the done callback of ``CellHandle.process``.
        """
        ...

    @abstractmethod
    def handled_cells(self) -> dict[CellAddress, CellHandle]:
        """
        The cell handles this matrix currently has **alive**, keyed by address.

        Isomorphic to ``Subprocesses.executing()`` — only entries whose process has not exited.
        Handles that crashed or exited normally move out of this dict into the
        ``dead_cells()`` FIFO.

        :return: a dict snapshot; callers must not mutate it.
        """
        ...

    @abstractmethod
    def dead_cells(self) -> list[CellHandle]:
        """
        Recently dead cell handles, FIFO and bounded.

        Isomorphic to ``Subprocesses.executed()`` — keeps a limited count, dropping the oldest
        on overflow. For debugging: get the exit code and stderr tail via ``handle.process``.

        :return: a list snapshot, newest last; callers must not mutate it.
        """
        ...

    # -- subprocess management -- #

    @property
    @abstractmethod
    def processes(self) -> Subprocesses:
        """
        Subprocess manager for the current process. Start subprocesses through the shell /
        execute mechanisms and the current Matrix Cell process owns their lifecycle, so no
        orphans are left behind. It also lists every subprocess it manages. The subprocess
        handling underneath ``run_node`` is built on this.
        """
        pass

    # -- Matrix network protocol foundation -- #

    @property
    @abstractmethod
    def session(self) -> Session:
        """
        The communication bus facing the whole Matrix network. Its primitives (topic / stream /
        signal / ...) are the bridge between everything running inside the network.
        """
        pass

    @abstractmethod
    async def parameters(self) -> Parameters:
        """
        Matrix-level parameter service — declare (become a writer) and subscribe (become a
        reader), point to point.

        Single declarer, no arbitration: declared keys are namespaced by cell address, and
        subscribe targets a peer by address. Lazy gate — constructed on the first call.
        """
        ...

    # @abstractmethod
    async def service_operator(self) -> ServiceOperator:
        """
        Service-oriented communication foundation between Matrix cells: one Cell provides a
        Service and other Cells can reach it. It hides the transport complexity of
        point-to-point and fan-out communication between Cells (lifecycle sync, transport
        protocols such as zenoh), so business-level protocols can be built quickly on top.
        """
        ...

    async def serve_service(self, service_cls: Type[ServiceServer]) -> ServiceServer:
        """Start a Service Server through its Facade or Adapter and register it with the Matrix lifecycle. """
        server = service_cls.new(self)
        await self.add_lifecycle_object(server)
        return server

    async def connect_service(self, client_cls: Type[ServiceClient]) -> ServiceClient:
        """Start a Service Client through its Facade or Adapter and register it with the Matrix lifecycle. """
        client = client_cls.new(self)
        await self.add_lifecycle_object(client)
        return client

    # -- base modules -- #

    @property
    @abstractmethod
    def container(self) -> IoCContainer:
        """
        The IoC container gate — process-level shared services (providers declared by manifests).

        Runtime services such as configs and resources are fetched here. Prefer declaring new
        services through manifests (environment discovery explains itself) over calling
        register at runtime.
        """
        pass

    @property
    @abstractmethod
    def logger(self) -> logging.Logger:
        """Logger belonging to the current node."""
        pass

    @property
    @abstractmethod
    def resources(self) -> ResourceRegistry:
        """
        Resource routing layer across scheme+host (a VFS).
        """
        ...

    @property
    @abstractmethod
    def warrant(self) -> Warrant:
        """
        Matrix-level general authorization — interactive approval (host writes storage;
        non-host uses topic mode).

        Consumers call `matrix.warrant.require(permission)` to request approval, e.g. to
        authorize node startup. Soft authorization boundary — not a security mechanism:
        a model can self-iterate and authorize itself.
        """
        pass

    # -- scoped identity family: runtime coordinates → storage isolation levels -- #

    def runtime_scopes(self) -> dict[RuntimeScopeKey, str]:
        """Runtime coordinates of this Matrix, used to build different isolation levels."""
        # Scoped concepts exist only at runtime (mode × ghost × network × cell; none of them
        # exists before the run). A scope governs a special storage territory inside the Project.
        return {
            'mode': self.env.mode_name,
            'ghost': self.env.ghost_name,
            'cell': self.this.address,
            'network': self.network_info.name,
        }

    def get_runtime_url_path(self, *scopes: RuntimeScopeKey, **kwargs: str) -> str:
        """
        Build a URL-shaped resource path from scopes — usable as a unique id for reusable
        resources.

        Example: ``get_runtime_url_path('ghost', 'mode', user=name)`` yields the unique id of
        "a given user, for a given Ghost in a given mode". Used to assemble resource
        declarations shaped like ``scheme://cell_address/scoped/path/resource``.
        """
        scope_values = self.runtime_scopes()
        for scope in scopes:
            if scope in scope_values:
                kwargs[scope] = scope_values[scope]
        result = []
        for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
            result.append(k.strip('/'))
            result.append(v.strip('/'))
        return '/'.join(result)

    # -- state description -- #

    @abstractmethod
    def is_running(self) -> bool:
        """Whether the matrix itself is running."""
        pass

    def is_host(self) -> bool:
        """Whether this cell is the host of the current network — usually offered by cell governance packages. """
        return self.this.is_host

    # -- lifecycle -- #

    @abstractmethod
    def close(self) -> None:
        """Close itself for a graceful exit."""
        ...

    @abstractmethod
    async def wait_closed(self) -> None:
        """Block until this matrix exits; all capabilities are closed with it."""
        ...

    @abstractmethod
    def wait_closed_sync(self, timeout: float | None = None) -> bool:
        """Block until this matrix exits. Synchronous contexts only."""
        ...

    @abstractmethod
    def create_task(
            self,
            cor: Awaitable[Any],
            *,
            stop_matrix_on_error: bool = False,
            name: str | None = None,
    ) -> asyncio.Task:
        """Create a Task contained in the Matrix lifecycle."""
        ...

    @abstractmethod
    def register_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """Register an object that starts together with the matrix. Started in order, bound to the lifecycle, no fault tolerance. Callable only before the run."""
        ...

    @abstractmethod
    async def add_lifecycle_object(self, obj: MatrixLifecycleObject) -> None:
        """Add a lifecycle object at runtime; bound to the exit stack and cleared on exit."""
        ...

    # -- start helpers. Not required — they show usage under code-as-prompt -- #

    async def arun(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        The basic run loop of a Matrix. Reference it, or build a Matrix-based application
        directly on it. Wrapped in an ``asyncio.Task``, it can also run in parallel with your
        main coroutine.
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
                except asyncio.CancelledError:
                    pass  # External cancel (KeyboardInterrupt → asyncio.run cancels the task) or internal close: exit quietly.
                finally:
                    for t in [task, exit_signal]:
                        if not t.done():
                            t.cancel()
                    _ = await asyncio.gather(task, exit_signal, return_exceptions=True)
            else:
                return await result_or_coro

    def run(self, main_coro: Callable[[Self], Awaitable[Any]]) -> Any:
        """
        Synchronous blocking entry point. It drives the event loop and the lifecycle itself.
        Top-level entry that works on Python 3.10.
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
            pass  # arun already handled cleanup

    # -- serve_mcp: code-as-prompt sugar that serves an MCP server inside the matrix lifecycle -- #

    async def aserve_mcp(
        self,
        mcp: 'MCPServer',
        *,
        host: str = '127.0.0.1',
        port: int = 0,
    ) -> None:
        """
        Serve an MCP server inside a running Matrix.

        The matrix must already be entered (inside ``async with matrix`` or inside a
        ``matrix.run()`` callback). This owns the transport only, not the matrix lifecycle.

        Serving uses ``run_streamable_http_async`` (async); never ``mcp.run()``, which calls
        ``anyio.run()`` internally and starts a second event loop that fights the matrix loop.
        The mcp instance's tools are already registered by the caller; this method does not
        re-register them.
        """
        if not self.is_running():
            raise RuntimeError('Matrix not running.  Use serve_mcp() or enter matrix context first.')

        self.logger.info(
            'MCP server serving on %s:%s (stateless streamable-http)',
            host, port,
        )
        await mcp.run_streamable_http_async(
            host=host, port=port, stateless_http=True,
        )

    def serve_mcp(
        self,
        mcp: 'MCPServer',
        *,
        host: str = '127.0.0.1',
        port: int = 0,
    ) -> None:
        """
        Synchronous blocking entry point: serve an MCP server inside the Matrix lifecycle.

        Code-as-prompt sugar mirroring :meth:`run`. It drives the event loop, enters the matrix
        context, serves, and cleans up on exit. The mcp instance's tools are already registered
        by the caller; this method owns only the transport lifecycle and does not re-register
        tools::

            mcp = MCPServer("my-node")

            @mcp.tool()
            async def hello(name: str) -> str:
                return f"hi {name}"

            matrix.serve_mcp(mcp, port=8080)
        """
        try:
            import uvloop
        except ImportError:
            uvloop = None

        if uvloop is not None:
            asyncio.set_event_loop(uvloop.new_event_loop())

        async def _run():
            async with self:
                await self.aserve_mcp(mcp, host=host, port=port)

        return asyncio.run(_run())

    @abstractmethod
    async def __aenter__(self) -> Self:
        ...

    @abstractmethod
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        ...


_instance: Matrix | None = None

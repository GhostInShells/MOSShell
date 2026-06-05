"""
Parameter — typed, versioned shared state on the Session bus.

Low-frequency writes (<1Hz), high-frequency reads. SQLite is the ground truth;
Zenoh carries lightweight invalidation signals (key, version only).

Aligns with ROS2 parameter semantics: declare → get/set → on-change callback.
Follows the same default-name pattern as TopicModel / SignalMeta / ConfigType.
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from pydantic import BaseModel

__all__ = [
    "ParameterModel",
    "Parameter",
    "ParameterStore",
    "VersionConflict",
    "T_PARAM",
]

T_PARAM = TypeVar("T_PARAM", bound="ParameterModel")


class ParameterModel(BaseModel, ABC):
    """
    Self-describing parameter declaration.

    Subclass to define a typed parameter.  *param_name* is the default key;
    *param_default* is the zero-value returned on miss.

    Usage::

        class GhostPersona(ParameterModel):
            name: str = "Echo"
            temperature: float = 0.7

            @classmethod
            def param_name(cls) -> str:
                return "ghost_persona"

            @classmethod
            def param_default(cls) -> "GhostPersona":
                return cls()
    """

    @classmethod
    @abstractmethod
    def param_name(cls) -> str:
        """Default parameter key — unique per subclass.  Overridable at declare()."""
        pass

    @classmethod
    @abstractmethod
    def param_default(cls) -> "ParameterModel":
        """Zero-value — returned by get() when key does not exist."""
        pass


class VersionConflict(Exception):
    """CAS version mismatch — caller must re-read and retry."""

    def __init__(self, key: str, expected: int, actual: int) -> None:
        self.key = key
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"VersionConflict on '{key}': expected {expected}, actual {actual}"
        )


class Parameter(Generic[T_PARAM], ABC):
    """
    Typed handle to a declared parameter.

    Bound to a key and model type at declare()-time.
    Symmetric to TopicService's Subscriber — declaration creates the handle,
    the handle carries the operations.

    Usage::

        param = store.declare(GhostPersona)
        cfg = param.get()                    # → GhostPersona
        param.set(new_cfg, version=3)        # CAS write
    """

    @property
    @abstractmethod
    def key(self) -> str:
        """The resolved key for this parameter."""
        pass

    @abstractmethod
    def get(self) -> T_PARAM:
        """
        Read current value.  Returns model's param_default() on miss
        (有零值 semantics — miss is not an error).
        """
        ...

    @abstractmethod
    def set(self, value: T_PARAM, *, version: int | None = None) -> int:
        """
        Write a value.  Returns the new version.

        :param version: if given, CAS — only write if current version matches.
                        ``None`` force-writes.
        :raises VersionConflict: CAS mismatch
        """
        ...

    @abstractmethod
    def version(self) -> int:
        """Current version, or 0 if key does not exist."""
        ...

    @abstractmethod
    def remove(self) -> bool:
        """Delete this parameter.  Returns True if it existed."""
        ...


class ParameterStore(ABC):
    """
    Factory for typed Parameter handles.

    Lifecycle is Session-scoped — handles need no explicit cleanup.
    Symmetric to TopicService: declare() ↔ subscribe_model().

    Usage::

        store: ParameterStore = session.parameters
        param = store.declare(GhostPersona)
        alt   = store.declare(GhostPersona, key="alt_persona")
    """

    @abstractmethod
    def declare(
        self,
        model_type: type[T_PARAM],
        *,
        key: str | None = None,
    ) -> Parameter[T_PARAM]:
        """
        Declare a parameter handle.  Key defaults to model_type.param_name().

        Declaration is idempotent — calling declare() with the same key returns
        a handle backed by the same underlying storage.
        """
        ...

    @abstractmethod
    def declared(self) -> list[str]:
        """
        All parameter keys that have been declared in this session.

        The declarative analogue of TopicService.subscribing()/publishing() —
        runtime-discoverable list of active parameter declarations.
        """
        ...

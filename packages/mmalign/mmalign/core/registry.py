"""Generic typed registry with built-in name protection.

``Registry[T]`` is used for all six global registries:
  - ``inference_align_registry``  (InferenceAlignMethod)
  - ``training_align_registry``   (TrainingAlignMethod)
  - ``control_strategy_registry`` (ControlStrategy)
  - ``evaluator_registry``        (Evaluator)
  - ``scenario_registry``         (Scenario)
  - ``backend_registry``          (ModelBackend)

Built-in entries (``builtin=True``) cannot be overwritten at runtime.
Plugin entries must use a namespaced name (e.g. ``specbench.align3``) to
avoid collisions with built-ins.
"""

from __future__ import annotations

from builtins import list as builtin_list
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


class BuiltinOverrideError(ValueError):
    """Raised when caller attempts to overwrite a built-in registry entry."""


class NotRegisteredError(KeyError):
    """Raised when a requested name is not found in the registry."""


class Registry(Generic[T]):
    """Thread-safe (GIL-protected) generic registry.

    >>> reg: Registry[str] = Registry("demo")
    >>> reg.register("a", "hello", builtin=True)
    >>> reg.get("a")
    'hello'
    >>> reg.register("a", "world")      # raises BuiltinOverrideError
    """

    def __init__(self, name: str = "") -> None:
        self._name = name
        self._store: dict[str, T] = {}
        self._builtins: set[str] = set()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, name: str, impl: T, *, builtin: bool = False) -> None:
        """Register *impl* under *name*.

        Raises
        ------
        BuiltinOverrideError
            If *name* was previously registered as a built-in.
        """
        if name in self._builtins:
            raise BuiltinOverrideError(
                f"[{self._name}] Cannot overwrite built-in entry '{name}'. "
                "Use a namespaced name (e.g. 'plugin.{name}') for external implementations."
            )
        self._store[name] = impl
        if builtin:
            self._builtins.add(name)

    def unregister(self, name: str) -> None:
        """Remove a non-built-in entry (useful in tests)."""
        if name in self._builtins:
            raise BuiltinOverrideError(
                f"[{self._name}] Cannot unregister built-in entry '{name}'."
            )
        self._store.pop(name, None)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, name: str) -> T:
        """Return the implementation registered under *name*.

        Raises
        ------
        NotRegisteredError
            If *name* is not registered.
        """
        try:
            return self._store[name]
        except KeyError:
            raise NotRegisteredError(
                f"[{self._name}] '{name}' is not registered. "
                f"Available: {self.list()}"
            ) from None

    def get_or_none(self, name: str) -> T | None:
        return self._store.get(name)

    def list(self) -> builtin_list[str]:
        return sorted(self._store.keys())

    def list_builtins(self) -> builtin_list[str]:
        return sorted(self._builtins)

    def count(self) -> int:
        return len(self._store)

    def __contains__(self, name: str) -> bool:
        return name in self._store

    def __iter__(self) -> Iterator[tuple[str, T]]:
        return iter(self._store.items())

    def __repr__(self) -> str:
        return f"Registry(name={self._name!r}, count={self.count()})"

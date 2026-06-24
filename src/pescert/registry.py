"""Name -> Eval-class registry and the ``@register`` decorator."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evals.base import Eval

REGISTRY: dict[str, type[Eval]] = {}

__all__ = ["REGISTRY", "register", "get_eval", "available"]


def register(name: str) -> Callable[[type[Eval]], type[Eval]]:
    """Class decorator: register an :class:`~pescert.evals.base.Eval` under ``name``."""

    def deco(cls: type[Eval]) -> type[Eval]:
        if name in REGISTRY:
            raise ValueError(f"eval {name!r} already registered")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def get_eval(name: str) -> Eval:
    """Instantiate the registered eval named ``name``."""
    if name not in REGISTRY:
        raise KeyError(f"no eval named {name!r}; available: {sorted(REGISTRY)}")
    return REGISTRY[name]()


def available() -> list[str]:
    """Return the sorted list of registered eval names."""
    return sorted(REGISTRY)

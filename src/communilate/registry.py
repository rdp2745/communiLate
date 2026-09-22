"""A tiny name -> callable registry.

Lets a config say ``data.loader: opensubtitles`` and have the pipeline find the
right function. Adding a corpus means writing a module and decorating one
function; no pipeline code changes.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T", bound=Callable)


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: dict[str, Callable] = {}

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(fn: T) -> T:
            if name in self._items:
                raise KeyError(f"{self.kind} {name!r} is already registered")
            self._items[name] = fn
            return fn

        return decorator

    def get(self, name: str) -> Callable:
        if name not in self._items:
            raise KeyError(
                f"unknown {self.kind} {name!r}; registered: {sorted(self._items) or '(none)'}"
            )
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)


LOADERS = Registry("data loader")

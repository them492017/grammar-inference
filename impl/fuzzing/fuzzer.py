from __future__ import annotations
from typing import Optional, Protocol, TYPE_CHECKING

class Fuzzer(Protocol):
    def generate(self) -> str:
        ...

    def run(self, count: int) -> str:
        ...

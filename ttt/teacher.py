from typing import Optional, Protocol

from state import Hypothesis


class Teacher(Protocol):
    def is_member(self, s: str) -> bool:
        ...

    def is_equivalent(self, h: Hypothesis) -> tuple[bool, Optional[str]]:
        ...

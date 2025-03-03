from __future__ import annotations
from typing import Optional, Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from regex_parser.dfa import DFA

class Teacher(Protocol):
    def is_member(self, s: str) -> bool:
        ...

    def is_equivalent(self, hypothesis: DFA) -> tuple[bool, Optional[str]]:
        ...


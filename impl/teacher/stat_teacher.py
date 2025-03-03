from __future__ import annotations
from typing import Optional, Protocol, TYPE_CHECKING

from teacher.base_teacher import Teacher

if TYPE_CHECKING:
    from regex_parser.dfa import DFA

class StatTeacher(Teacher, Protocol):
    num_membership: int
    num_membership_excl_cache: int
    num_equivalence: int

    def stats(self) -> tuple[int, int, int]:
        """Return (num_membership, num_membership_excl_cache, num_equivalence)"""
        ...
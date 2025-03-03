from __future__ import annotations

import random
import re
import math
from typing import Optional, TYPE_CHECKING

from teacher.stat_teacher import StatTeacher

if TYPE_CHECKING:
    from regex_parser.dfa import DFA

class SimpleTeacher(StatTeacher):
    """
    A simple teacher class that generates random strings up to
    a certain maximum length for equivalence queries using the PAC framework
    """
    alphabet: str
    dfa: DFA
    equivalence_query_counter: int
    epsilon: float
    delta: float
    num_membership: int
    num_membership_excl_cache: int
    num_equivalence: int
    membership_cache: dict[str, bool]

    def __init__(self, alphabet: str, pattern: str, epsilon: float = 0.1, delta: float = 0.1, seed: int = 1):
        self.alphabet = alphabet
        self.regex = re.compile(pattern)
        self.equivalence_query_counter = 0
        self.epsilon = epsilon
        self.delta = delta

        self.num_membership = 0
        self.num_membership_excl_cache = 0
        self.num_equivalence = 0

        self.membership_cache = {}
        random.seed(seed)

    def is_member(self, s: str) -> bool:
        self.num_membership += 1
        if s in self.membership_cache:
            return self.membership_cache[s]
        else:
            self.num_membership_excl_cache += 1
            return self.regex.fullmatch(s) is not None

    def is_equivalent(self, hypothesis: DFA, max_length: int = 10) -> tuple[bool, Optional[str]]:
        self.num_equivalence += 1
        self.equivalence_query_counter += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.equivalence_query_counter * math.log(2)))

        for _ in range(num_calls):
            s = self.gen_random(max_length)
            if self.is_member(s) != hypothesis.evaluate(s):
                return False, s

        return True, None

    def stats(self) -> tuple[int, int, int]:
        return (
            self.num_membership,
            self.num_membership_excl_cache,
            self.num_equivalence,
        )

    def is_equivalent_exhaustive(self, hypothesis: DFA, max_length: int = 10, s: str = "") -> bool:
        if len(s) >= max_length:
            return True

        if self.is_member(s) != hypothesis.evaluate(s):
            print(s, "is a counterexample")
            return False

        return all((self.is_equivalent_exhaustive(hypothesis, max_length, s + a)) for a in self.alphabet)

    def gen_random(self, max_length) -> str:
        # this seems inefficient
        total_combinations = len(self.alphabet) ** max_length
        random_idx = random.randint(1, total_combinations)

        # TODO: just compare with precomputed values (no log needed)
        length = math.floor(math.log(random_idx, len(self.alphabet)))

        return "".join(random.choice(self.alphabet) for _ in range(length))

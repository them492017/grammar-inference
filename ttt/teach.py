from __future__ import annotations
from typing import Optional, Protocol, Pattern, TYPE_CHECKING

from regex_parser.dfa import DFA
from regex_parser.regex import Regex

import re
import random
import math

if TYPE_CHECKING:
    from state import Hypothesis


class Teacher(Protocol):
    def is_member(self, s: str) -> bool:
        ...

    def is_equivalent(self, hypothesis: Hypothesis) -> tuple[bool, Optional[str]]:
        ...


class PerfectTeacher(Teacher):
    """
    A perfect teacher class that uses a DFA based equivalence algorithm
    to find counterexamples
    """
    alphabet: str
    dfa: DFA

    def __init__(self, alphabet: str, pattern: str) -> None:
        self.alphabet = alphabet
        self.dfa = Regex.parse(pattern).to_nfa().determinise()

    def is_member(self, s: str) -> bool:
        return self.dfa.evaluate(s)

    def is_equivalent(self, hypothesis: Hypothesis) -> tuple[bool, Optional[str]]:
        return self.dfa.is_equivalent(hypothesis.to_dfa())

class SimpleTeacher(Teacher):
    """
    A simple teacher class that uses the built in evaluation capabilities
    of the Hypothesis class and applies it to uniformly random strings up to
    a certain maximum length.
    """
    alphabet: str
    regex: Pattern[str]
    equivalence_query_counter: int
    epsilon: float
    delta: float

    def __init__(self, alphabet: str, pattern: str, epsilon: float = 0.1, delta: float = 0.1, seed: int = 1):
        self.alphabet = alphabet
        self.regex = re.compile(pattern)
        self.equivalence_query_counter = 0
        self.epsilon = epsilon
        self.delta = delta
        random.seed(seed)

    def is_member(self, s: str) -> bool:
        return self.regex.fullmatch(s) is not None

    def is_equivalent(self, hypothesis: Hypothesis, max_length: int = 10) -> tuple[bool, Optional[str]]:
        self.equivalence_query_counter += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.equivalence_query_counter * math.log(2)))

        for _ in range(num_calls):
            s = self.gen_random(max_length)
            if self.is_member(s) != hypothesis.evaluate(s):
                return False, s

        return True, None

    def is_equivalent_exhaustive(self, hypothesis: Hypothesis, max_length: int = 10, s: str = "") -> bool:
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


class SimpleDFATeacher(Teacher):
    """
    A simple teacher class that uses the built in evaluation capabilities
    of the Hypothesis class and applies it to uniformly random strings up to
    a certain maximum length, and compiles a regex to a dfa for linear
    membership queries.
    """
    alphabet: str
    dfa: DFA
    equivalence_query_counter: int
    epsilon: float
    delta: float

    def __init__(self, alphabet: str, pattern: str, epsilon: float = 0.1, delta: float = 0.1, seed: int = 1):
        self.alphabet = alphabet
        self.dfa = Regex.parse(pattern).to_nfa().determinise()
        self.equivalence_query_counter = 0
        self.epsilon = epsilon
        self.delta = delta
        random.seed(seed)

    def is_member(self, s: str) -> bool:
        return self.dfa.evaluate(s)

    def is_equivalent(self, hypothesis: Hypothesis, max_length: int = 10) -> tuple[bool, Optional[str]]:
        self.equivalence_query_counter += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.equivalence_query_counter * math.log(2)))

        for _ in range(num_calls):
            s = self.gen_random(max_length)
            print(s, self.is_member(s), hypothesis.evaluate(s))
            if self.is_member(s) != hypothesis.evaluate(s):
                return False, s

        return True, None

    def is_equivalent_exhaustive(self, hypothesis: Hypothesis, max_length: int = 10, s: str = "") -> bool:
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

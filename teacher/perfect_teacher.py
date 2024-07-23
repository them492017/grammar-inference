from typing import Protocol, Pattern
import re


class HypothesisProtocol(Protocol):
    def evaluate(self, u: str) -> bool:
        ...


class PerfectTeacher:
    """
    A teacher class that always returns the shortest possible counterexample
    """

    alphabet: list[str]
    regex: Pattern[str]

    def __init__(self, alphabet: list[str], pattern: str):
        self.alphabet = alphabet
        self.regex = re.compile(pattern)

    def is_member(self, u: str) -> bool:
        return self.regex.fullmatch(u) is not None

    def is_equivalent(self, hypothesis: HypothesisProtocol, max_length: int = 10):
        # TODO: need to convert regex into dfa
        # then can use product construction to get dfa for symmetric difference
        # then can check for emptiness to get equivalence
        while len(s := next_string) < max_length:
            if self.is_member(s) != hypothesis.evaluate(s):
                return False, s

        return True, None

    def next_string(self, s: str) -> str:
        n = len(self.alphabet)
        for i in range(1, len(s)+1):
            next_idx = (self.alphabet.index(s[-i]) + 1) % n
            s[-i] = self.alphabet[next_idx]
            if idx != 0:
                break
        else:
            return self.alphabet[0] + s
        return s

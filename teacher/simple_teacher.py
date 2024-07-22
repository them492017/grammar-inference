from typing import Protocol, Pattern
import re
import math
import random


class HypothesisProtocol(Protocol):
    def evaluate(self, u: str) -> bool:
        ...


class SimpleTeacher:
    alphabet: list[str]
    regex: Pattern[str]
    equivalence_query_counter: int
    epsilon: float
    delta: float

    def __init__(self, alphabet: list[str], pattern: str, epsilon: float = 0.1, delta: float = 0.1, seed: int = 1):
        self.alphabet = alphabet
        self.regex = re.compile(pattern)
        self.equivalence_query_counter = 0
        self.epsilon = epsilon
        self.delta = delta
        random.seed(1)

    def is_member(self, u: str) -> bool:
        return self.regex.fullmatch(u) is not None

    def is_equivalent(self, hypothesis: HypothesisProtocol, max_length: int = 10):
        self.equivalence_query_counter += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.equivalence_query_counter * math.log(2)))

        for _ in range(num_calls):
            s = self.gen_random(max_length)
            print(s)
            if self.is_member(s) != hypothesis.evaluate(s):
                return False, s

        return True, None

    def gen_random(self, max_length) -> str:
        # this seems inefficient
        total_combinations = len(self.alphabet) ** max_length
        random_idx = random.randint(1, total_combinations)

        # TODO: just compare with precomputed values (no log needed)
        length = math.floor(math.log(random_idx, len(self.alphabet)))

        return "".join(random.choice(self.alphabet) for i in range(length))

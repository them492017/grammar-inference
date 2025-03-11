from typing import Generator, Deque
import random

from fuzzing.fuzzer import Fuzzer

class SimpleFuzzer(Fuzzer):
    grammar: dict[str, list[list[str]]]
    start: str

    def __init__(self, grammar: dict[str, list[list[str]]], start: str) -> None:
        self.grammar = grammar
        self.start = start

    def generate(self) -> str:
        assert self.start in self.grammar
        curr = [self.start]
        while True:
            non_terminal_indices = [
                i for i, symbol in enumerate(curr) if len(symbol) > 1
            ]

            if len(non_terminal_indices) == 0:
                return "".join(curr)

            rand_idx = random.choice(non_terminal_indices)
            symbol = curr[rand_idx]
            rand_production = random.choice(self.grammar[symbol])
            curr = curr[:rand_idx] + rand_production + curr[rand_idx+1:]

    def run(self, count: int) -> Generator[str, None, None]:
        if self.start in self.grammar:
            for _ in range(count):
                yield self.generate()

    def run_in_order(self) -> Generator[str, None, None]:
        if self.start in self.grammar:
            queue: Deque[list[str]] = Deque()
            queue.append([self.start])

            while len(queue) > 0:
                curr = queue.popleft()

                for i, symbol in enumerate(curr):
                    if len(symbol) > 1:  # symbol is a nonterminal
                        queue.extend(
                            [
                                curr[:i] + production + curr[i + 1 :]
                                for production in self.grammar[symbol]
                            ]
                        )
                        break
                else:
                    yield "".join(curr)
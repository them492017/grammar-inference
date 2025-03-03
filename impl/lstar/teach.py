from __future__ import annotations
from functools import lru_cache
from typing import Optional, Protocol, Pattern, TYPE_CHECKING

from regex_parser.dfa import DFA
from regex_parser.regex import Regex

import re
import random
import math

if TYPE_CHECKING:
    from regex_parser.dfa import DFA


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
        print(f"Performing membership query on '{s}'")
        return self.dfa.evaluate(s)

    def is_equivalent(self, hypothesis: DFA) -> tuple[bool, Optional[str]]:
        print(f"Performing equivalence query")
        return self.dfa.is_equivalent(hypothesis)

class SimpleTeacher(Teacher):
    """
    A simple teacher class that uses the built in evaluation capabilities
    of the Hypothesis class and applies it to uniformly random strings up to
    a certain maximum length.
    """
    alphabet: str
    regex: Pattern[str]
    epsilon: float
    delta: float
    num_membership: int
    num_membership_excl_cache: int
    num_equivalence: int
    membership_cache: dict[str, bool]

    def __init__(self, alphabet: str, pattern: str, epsilon: float = 0.1, delta: float = 0.1, seed: int = 1):
        self.alphabet = alphabet
        self.regex = re.compile(pattern)
        self.num_membership = 0
        self.num_membership_excl_cache = 0
        self.num_equivalence = 0
        self.epsilon = epsilon
        self.delta = delta

        self.membership_cache = {}

        random.seed(seed)

    def is_member(self, s: str) -> bool:
        self.num_membership += 1
        if s in self.membership_cache:
            return self.membership_cache[s]
        else:
            self.num_membership_excl_cache += 1
            return self.regex.fullmatch(s) is not None

    def is_equivalent(self, hypothesis: Hypothesis, max_length: int = 10) -> tuple[bool, Optional[str]]:
        self.num_equivalence += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.num_equivalence * math.log(2)))

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
    of the DFA class and applies it to uniformly random strings up to
    a certain maximum length, and compiles a regex to a dfa for linear
    membership queries.
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
        self.dfa = Regex.parse(pattern).to_nfa().determinise()
        self.equivalence_query_counter = 0
        self.epsilon = epsilon
        self.delta = delta
        random.seed(seed)

        self.num_membership = 0
        self.num_membership_excl_cache = 0
        self.num_equivalence = 0

        self.membership_cache = {}

    def is_member(self, s: str) -> bool:
        self.num_membership += 1
        if s in self.membership_cache:
            return self.membership_cache[s]
        else:
            self.num_membership_excl_cache += 1
            return self.dfa.evaluate(s)

    def is_equivalent(self, hypothesis: DFA, max_length: int = 10) -> tuple[bool, Optional[str]]:
        self.num_equivalence += 1
        self.equivalence_query_counter += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.equivalence_query_counter * math.log(2)))

        for _ in range(num_calls):
            s = self.gen_random(max_length)
            print(s, self.is_member(s), hypothesis.evaluate(s))
            if self.is_member(s) != hypothesis.evaluate(s):
                return False, s

        return True, None

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

class RandomPerfectTeacher(Teacher):
    """
    Tries to generate a random string using gen random, and then verifies using
    DFA equivalence.
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

    def __init__(self, alphabet: str, pattern: str, epsilon: float = 0.1, delta: float = 0.1, seed: int = 1, perfect: bool = True):
        self.alphabet = alphabet
        self.dfa = Regex.parse(pattern).to_nfa().determinise()
        self.equivalence_query_counter = 0
        self.epsilon = epsilon
        self.delta = delta
        random.seed(seed)

        self.num_membership = 0
        self.num_membership_excl_cache = 0
        self.num_equivalence = 0
        self.perfect = perfect

        self.membership_cache = {}

    def is_member(self, s: str) -> bool:
        self.num_membership += 1
        if s in self.membership_cache:
            return self.membership_cache[s]
        else:
            self.num_membership_excl_cache += 1
            return self.dfa.evaluate(s)

    def is_equivalent(self, hypothesis: DFA, max_length: int = 10) -> tuple[bool, Optional[str]]:
        self.num_equivalence += 1
        self.equivalence_query_counter += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.equivalence_query_counter * math.log(2)))

        for _ in range(num_calls):
            s = self.gen_random(max_length)
            print(s, self.is_member(s), hypothesis.evaluate(s))
            if self.is_member(s) != hypothesis.evaluate(s):
                return False, s

        if self.perfect:
            print("Checking with DFA equivalence")
            return self.dfa.is_equivalent(hypothesis)
        else:
            return True, None

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


def convert_grammar_to_dfa(grammar: tuple[dict[str, list[list[str]]], str], alphabet: str) -> DFA:
    # from ChatGPT
    grammar_rules, start_state_id = grammar
    dfa = DFA()
    
    # Map grammar states (string) to DFA states (int)
    state_mapping = {start_state_id: dfa.start}
    state_queue = [start_state_id]  # Process states in order

    while state_queue:
        state_id = state_queue.pop(0)
        dfa_state = state_mapping[state_id]
        transitions = {}

        for transition in grammar_rules[state_id]:
            if not transition or len(transition) == 1:  # Empty transition means final state
                dfa.final.add(dfa_state)
                continue

            symbol, next_state_id = transition[0], transition[1]

            if next_state_id not in state_mapping:
                state_mapping[next_state_id] = dfa.next_state
                state_queue.append(next_state_id)
                dfa.next_state += 1

            transitions[symbol] = state_mapping[next_state_id]

        dfa.states.add(dfa_state)
        dfa.transitions[dfa_state] = transitions

    dfa.close_with_sink(alphabet)

    return dfa

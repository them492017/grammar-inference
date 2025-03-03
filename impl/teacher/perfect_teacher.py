from typing import Optional

from teacher.base_teacher import Teacher
from regex_parser.dfa import DFA

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

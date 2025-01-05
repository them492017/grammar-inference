from __future__ import annotations
from typing import TypeVar
from pprint import pprint
import sys

from dfa import NFA


EMPTYSET = "\u2205"
EPSILON = "\u03b5"

ALPHABET = "ab"
EXTENDED_ALPHABET = f"ab{EPSILON}"
T = TypeVar("T")
S = TypeVar("S")


class Regex:
    name: str
    regex: list

    def __repr__(self) -> str:
        ...

    def to_syntax(self) -> str:
        ...

    def to_nfa(self) -> NFA:
        ...

    @classmethod
    def parse(cls, s: str) -> Regex:
        print("Parsing", s)

        if len(s) == 0:
            print(f"Found {Epsilon()}")
            return Epsilon()
        if len(s) == 1:
            if s not in ALPHABET:
                raise ValueError(f"Character '{s}' is not in the alphabet")
            print(f"Found {Char(s)}")
            return Char(s)

        if s[0] == "(":
            end_idx = cls.matching_idx(s, 0)
            if end_idx == -1:
                raise ValueError("Could not parse regex: no matching parenthesis")
        else:
            end_idx = 0

        if end_idx > 0:
            symbol = cls.parse(s[1 : end_idx])
        else:
            symbol = cls.parse(s[0])

        if len(s) == end_idx + 1:
            return symbol

        next_char = s[end_idx+1]

        if next_char == "|":
            print(f"Found {symbol} | [something]")
            return Or(symbol, cls.parse(s[end_idx+2:]))
        elif next_char == "*":
            if len(s) == end_idx + 2:
                print(f"Found {Star(symbol)}")
                return Star(symbol)
            else:  # check this
                print(f"Found {Star(symbol)} . [something]")
                return And(Star(symbol), cls.parse(s[end_idx+2:]))
        elif next_char == "(":
            print(f"Found {symbol} . [something]")
            return And(symbol, cls.parse(s[end_idx+2:-1]))
        else:
            print(f"Found {symbol} . [something]")
            return And(symbol, cls.parse(s[end_idx+1:]))


    @classmethod
    def matching_idx(cls, s: str, i: int) -> int:
        print("Matching Idx", s, i)
        if s[i] != "(":
            raise ValueError("Initial character is not '('")

        depth = 1

        for j, c in enumerate(s[i+1:], start=i+1):
            if c == "(":
                depth += 1
            if c == ")":
                depth -= 1
            if depth == 0:
                print("Matching idx:", j)
                return j

        raise ValueError("Not match found")

class Empty(Regex):
    def __init__(self) -> None:
        self.regex = []
        self.name = "Empty"

    def __repr__(self) -> str:
        return "Empty()"

    def to_syntax(self) -> str:
        return EMPTYSET

    def to_nfa(self) -> NFA:
        print(f"Converting {self} to an NFA")
        return NFA()

class Epsilon(Regex):
    def __init__(self) -> None:
        self.regex = []
        self.name = "Epsilon"

    def __repr__(self) -> str:
        return "Epsilon()"

    def to_syntax(self) -> str:
        return EPSILON

    def to_nfa(self) -> NFA:
        print(f"Converting {self} to an NFA")
        nfa = NFA()
        nfa.add_state({
            a: {1} for a in ALPHABET
        })
        nfa.update(0, {
            a: {1} for a in ALPHABET
        })
        nfa.make_final(0)
        return nfa

class Char(Regex):
    def __init__(self, char: str) -> None:
        assert len(char) == 1
        assert char in ALPHABET
        self.char = char

    def __repr__(self) -> str:
        return f"Char({self.char})"

    def to_syntax(self) -> str:
        return self.char

    def to_nfa(self) -> NFA:
        print(f"Converting {self} to an NFA")
        nfa = NFA()
        nfa.update(0, {
            a: {1 if a == self.char else 2} for a in ALPHABET
        })
        nfa.add_state({
            a: {2} for a in ALPHABET
        })
        nfa.add_state({
            a: {2} for a in ALPHABET
        })
        nfa.make_final(1)
        return nfa

class Or(Regex):
    def __init__(self, r1: Regex, r2: Regex) -> None:
        self.r1 = r1
        self.r2 = r2
        self.regex = [self.r1, self.r2]
        self.name = "Union"

    def __repr__(self) -> str:
        return f"Union({self.r1, self.r2})"

    def to_syntax(self) -> str:
        return f"({self.r1.to_syntax()})|({self.r2.to_syntax()})"

    def to_nfa(self) -> NFA:
        print(f"Converting {self} to an NFA")
        nfa = NFA()
        nfa.update(0, {
            EPSILON: {1, 2}
        })
        nfa.add_state({})
        nfa.add_state({})
        nfa.make_final(1)
        nfa.make_final(2)
        nfa.insert(1, self.r1.to_nfa())
        nfa.insert(2, self.r2.to_nfa())
        return nfa

class And(Regex):
    def __init__(self, r1: Regex, r2: Regex) -> None:
        self.r1 = r1
        self.r2 = r2
        self.regex = [self.r1, self.r2]
        self.name = "Concat"

    def __repr__(self) -> str:
        return f"Concat({self.r1, self.r2})"

    def to_syntax(self) -> str:
        return f"({self.r1.to_syntax()})({self.r2.to_syntax()})"

    def to_nfa(self) -> NFA:
        print(f"Converting {self} to an NFA")
        nfa = NFA()
        nfa.update(0, {
            EPSILON: {1}
        })
        nfa.add_state({  # 1
            EPSILON: {2}
        })
        nfa.add_state({})  # 2
        nfa.make_final(2)
        nfa.insert(1, self.r1.to_nfa())
        nfa.insert(2, self.r2.to_nfa())
        return nfa

class Star(Regex):
    def __init__(self, r: Regex) -> None:
        self.r = r

    def __repr__(self) -> str:
        return f"Star({self.r})"

    def to_syntax(self) -> str:
        return f"({self.r.to_syntax()})*"

    def to_nfa(self) -> NFA:
        print(f"Converting {self} to an NFA")
        nfa = NFA()
        nfa.update(0, {
            EPSILON: {1}
        })
        nfa.add_state({
            EPSILON: {1}
        })
        nfa.make_final(0)
        nfa.make_final(1)
        nfa.insert(1, self.r.to_nfa())
        return nfa


if __name__ == "__main__":
    regex = Regex.parse(sys.argv[1])

    print("The regex was")
    print(regex)
    print(regex.to_syntax())

    print("=" * 50)
    
    nfa = regex.to_nfa()
    print(nfa.states, nfa.final)
    pprint(nfa.transitions)
    nfa.visualize("nfa")
    new_nfa = nfa.remove_epsilon()
    new_nfa.visualize("nfa_no_epsilon")
    dfa = new_nfa.determinise()
    dfa.visualize("dfa")
    print(dfa.is_equivalent(dfa.complement()))
    """
    For regex a((a|b)*b)
    - epsilon transition 5 -> 6 should exist...
    """

    dfac = dfa.complement()
    dfac.visualize('complement')

    dfa.intersection(dfac).visualize('intersection')
    dfa.union(dfac).visualize('union')

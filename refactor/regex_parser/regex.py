from __future__ import annotations
from typing import Generic, Protocol, TypeVar
from pprint import pprint
import sys

from graphviz import Digraph


EMPTYSET = "\u2205"
EPSILON = "\u03b5"

ALPHABET = "ab"
EXTENDED_ALPHABET = f"ab{EPSILON}"
T = TypeVar("T")
S = TypeVar("S")


class Automaton(Protocol, Generic[T, S]):
    """
    T: type of a state
    S: type of a state's transition table
    """
    start: T
    states: set[T]
    final: set[T]
    transitions: dict[T, S]

    def add_state(self, transitions: S) -> None:
        ...

    def make_final(self, state: T) -> None:
        assert state in self.states
        self.final.add(state)

    def update(self, state: T, transitions: S) -> None:
        self.transitions[state] = transitions

    def evaluate(self, s: str) -> bool:
        ...


class NFA(Automaton[int, dict[str, set[int]]]):
    next_state: int

    def __init__(self) -> None:
        self.transitions = {}
        self.start = 0
        self.states = { self.start }
        self.final = set()
        self.next_state = 1

    def add_state(self, transitions: dict[str, set[int]]) -> None:
        self.states.add(self.next_state)
        self.transitions[self.next_state] = transitions
        self.next_state += 1
        
    def evaluate(self, s: str) -> bool:
        curr = { self.start }
        for c in s:
            curr = set().union(
                *(self.transitions[state][c] for state in curr)
            )
        return len(curr & self.final) > 0

    def insert(self, state: int, nfa: NFA) -> None:  # TODO: maybe refactor to use len instead of self.next_state
        """
        Assumes NFAs have their states numbered monotonically.

        nfa states [1..n] -> self states [self.next_state..self.next_state + n - 1]
        nfa state [0] -> self state [state]

        for states [1..n], all transitions are inherited in self
        for state [0], all transitions also inherited to self
            the existing transitions for self [state] are passed on to the final states of nfa

        if self [state] is final, all final states of nfa become final
        regardless, self [state] should not be final anymore (unless final in nfa and is itself final)
        """
        print("RUNNING INSERT")
        visualize_nfa(self, filename="a")
        visualize_nfa(nfa, filename="b")
        # add states
        self.states |= set(range(self.next_state, self.next_state + nfa.next_state - 1))  # TODO: dont add extra state (one is shared)
        # self.next_state += nfa.next_state - 1
        print(self.states)
        print(self.next_state)

        # deal with final states
        if state in self.final:
            self.final.remove(state)
            self.final |= {s + self.next_state - 1 for s in nfa.final}

        # map nfa transitions to new values
        nfa.transitions = {
            state: {
                char: set(map(lambda s: s + self.next_state - 1, targets)) for char, targets in transitions.items()
            } for state, transitions in nfa.transitions.items()
        }

        # update transitions
        new_transitions = {}
        for s in range(1, nfa.next_state):
            actual_state = s + self.next_state - 1
            print("State", actual_state)
            if s in nfa.final:
                print(
                    nfa.transitions[s],
                    self.transitions[s]
                )
                new_transitions[actual_state] = self.join_transitions(
                    nfa.transitions[s],
                    self.transitions[s]
                )
                print("nonfinal - (Transitions ->", new_transitions[actual_state])
            else:
                new_transitions[actual_state] = nfa.transitions[s]
                print("Transitions ->", new_transitions[actual_state])

        new_transitions[state] = nfa.transitions[0]

        self.transitions = {**self.transitions, **new_transitions}

        visualize_nfa(self, "result")

        self.next_state += nfa.next_state - 1  # TODO: move where this would make sense

    def join_transitions(self, a: dict[str, set[int]], b: dict[str, set[int]]) -> dict[str, set[int]]:
        return {
            char: a.get(char, set()) | b.get(char, set()) for char in a.keys() | b.keys()
        }


class DFA(Automaton[int, dict[str, int]]):
    next_state: int

    def __init__(self) -> None:
        self.transitions = {}
        self.start = 0
        self.states = { self.start }
        self.final = set()
        self.next_state = 1

    def add_state(self, transitions: dict[str, int]) -> None:
        self.states.add(self.next_state)
        self.transitions[self.next_state] = transitions
        self.next_state += 1

    def evaluate(self, s: str) -> bool:
        curr = self.start
        for c in s:
            curr = self.transitions[curr][c]
        return curr in self.final


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
        nfa.make_final(0)
        nfa.add_state({
            EPSILON: {1}
        })
        nfa.make_final(1)
        nfa.insert(1, self.r.to_nfa())
        return nfa


def visualize_nfa(automaton: NFA, filename='nfa', format='png'):
    """
    Visualize an NFA represented by the NFA class.

    Args:
        automaton (NFA): The NFA to visualize.
        filename (str): Output filename (without extension).
        format (str): Output file format (e.g., png, pdf).
    """
    # Initialize a directed graph
    dot = Digraph(name="NFA", format=format)
    dot.attr(rankdir="LR")  # Left-to-right orientation
    dot.attr("node", shape="circle")  # States are circular by default

    # Add states
    for state in automaton.states:
        shape = "doublecircle" if state in automaton.final else "circle"
        dot.node(str(state), shape=shape, label=str(state))

    # Add transitions
    for state, transitions in automaton.transitions.items():
        for symbol, targets in transitions.items():
            for target in targets:
                dot.edge(str(state), str(target), label=symbol)

    # Mark the start state
    dot.node("start", shape="plaintext", label="")
    dot.edge("start", str(automaton.start))

    # Save and render the graph
    dot.render(filename, cleanup=True)
    print(f"NFA visualized and saved as {filename}.{format}")


if __name__ == "__main__":
    regex = Regex.parse(sys.argv[1])

    print("The regex was")
    print(regex)
    print(regex.to_syntax())

    print("=" * 50)
    
    nfa = regex.to_nfa()
    print(nfa.states, nfa.final)
    pprint(nfa.transitions)
    visualize_nfa(nfa)

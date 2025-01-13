from __future__ import annotations
from collections import defaultdict, deque
from typing import Generic, Literal, Protocol, Self, TypeVar, Union
from itertools import chain, combinations, product
from copy import deepcopy

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
        # add states
        self.states |= set(range(self.next_state, self.next_state + nfa.next_state - 1))  # TODO: dont add extra state (one is shared)

        # deal with final states
        if state in self.final:
            if 0 not in nfa.final:
                self.final.remove(state)
            self.final |= {s + self.next_state - 1 for s in nfa.final - {0}}

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
            if s in nfa.final:
                new_transitions[actual_state] = self.join_transitions(
                    nfa.transitions[s],
                    self.transitions[state]
                )
            else:
                new_transitions[actual_state] = nfa.transitions[s]

        if 0 in nfa.final:
            new_transitions[state] = self.join_transitions(
                nfa.transitions[0],
                self.transitions[state]
            )
        else:
            new_transitions[state] = nfa.transitions[0]

        self.transitions = {**self.transitions, **new_transitions}
        self.next_state += nfa.next_state - 1

    def join_transitions(self, a: dict[str, set[int]], b: dict[str, set[int]]) -> dict[str, set[int]]:
        return {
            char: a.get(char, set()) | b.get(char, set()) for char in a.keys() | b.keys()
        }

    def remove_epsilon(self) -> NFA:
        """
        Removes an equivalent nfa with no epsilon transitions
        """
        nfa = NFA()
        nfa.transitions = {state: dict() for state in self.states}
        nfa.start = self.start
        nfa.states = self.states
        nfa.final = self.final
        nfa.next_state = self.next_state

        for state in self.states:
            print(f"On state {state}")
            seen: set[int] = set()
            final = [False]
            # dfs to get all states reachable by exactly one non-epsilon transition
            def helper(curr: int) -> None:
                if curr not in seen:
                    seen.add(curr)
                    print(f"\tCurr state {curr} is final: {curr in self.final}")
                    if curr in self.final:
                        final[0] = True
                    for a, targets in self.transitions[curr].items():
                        if a == EPSILON:
                            for target in targets:
                                helper(target)
                        else:
                            nfa.transitions[state][a] = \
                                nfa.transitions[state].get(a, set()) | targets
            helper(state)
            if final[0]:
                nfa.final.add(state)

        nfa.trim_unreachable()

        return nfa

    def trim_unreachable(self) -> None:
        """
        Deletes all unreachable states.

        First, runs a BFS to keep only the connected component containing the
        start state.

        Then it removes every state (besides the start state) with no incoming 
        transitions.
        """
        seen = set()
        queue: deque[int] = deque([self.start])

        while len(queue) > 0:
            curr = queue.popleft()
            seen.add(curr)
            for targets in self.transitions[curr].values():
                for target in targets:
                    if target not in seen:
                        queue.append(target)

        for state in self.states - seen:
            self.states -= {state}
            del self.transitions[state]

        unchanged = False
        while not unchanged:
            unchanged = True
            for target_state in list(self.states - {self.start}):
                if target_state not in self.states:
                    # check if state has already been removed
                    continue
                reached = False
                for state in self.states:
                    if reached:
                        break
                    if state != target_state:
                        for target_set in self.transitions[state].values():
                            if target_state in target_set:
                                reached = True
                                print(f"{target_state} can be reached by {state}")
                                break
                if not reached:
                    self.states -= {target_state}
                    del self.transitions[target_state]
                    unchanged = False

    def determinise(self) -> DFA:
        """
        Deteminises the NFA via a subset construction
        """
        self_without_epsilon = self.remove_epsilon()
        self_without_epsilon.visualize("no_epsilon")

        def powerset(s: set[T]) -> list[tuple[T, ...]]:
            return list(chain.from_iterable(set(combinations(s, r)) for r in range(len(s)+1)))

        dfa = DFA()

        subsets = powerset(self_without_epsilon.states)

        dfa.start = subsets.index((self_without_epsilon.start,))
        dfa.states = set(range(len(subsets)))
        dfa.final = set(i for i in dfa.states if len(set(subsets[i]) & self_without_epsilon.final) > 0)
        dfa.transitions = {state: dict() for state in dfa.states}
        dfa.next_state = len(subsets)

        for state, subset in enumerate(subsets):
            for a in ALPHABET:
                result = set().union(*(self_without_epsilon.transitions[s].get(a, set()) for s in subset))
                print(f"{subsets[state]} --{a}-> {subsets[subsets.index(tuple(sorted(result)))]}")
                dfa.transitions[state][a] = subsets.index(
                    tuple(sorted(result))
                )

        dfa.trim_unreachable()

        return dfa

    def visualize(self, filename='nfa', format='png'):
        """
        Visualises the NFA

        Args:
            filename (str): Output filename (without extension).
            format (str): Output file format (e.g., png, pdf).
        """
        # Initialize a directed graph
        dot = Digraph(name="NFA", format=format)
        dot.attr(rankdir="LR")  # Left-to-right orientation
        dot.attr("node", shape="circle")  # States are circular by default

        # Add states
        for state in self.states:
            shape = "doublecircle" if state in self.final else "circle"
            dot.node(str(state), shape=shape, label=str(state))

        # Add transitions
        for state, transitions in self.transitions.items():
            for symbol, targets in transitions.items():
                for target in targets:
                    dot.edge(str(state), str(target), label=symbol)

        # Mark the start state
        dot.node("start", shape="plaintext", label="")
        dot.edge("start", str(self.start))

        # Save and render the graph
        dot.render(filename, cleanup=True)
        print(f"NFA visualized and saved as {filename}.{format}")


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

    def is_equivalent(self, dfa: DFA) -> Union[tuple[Literal[True], None], tuple[Literal[False], str]]:
        """
        Determines if two dfa's are equivalent.

        Computes the DFA for the symmetric differences of the accepted languages

        Then checks this DFA for emptiness.

        If the DFA's are equivalent, returns (True, None)
        Otherwise, returns (False, s) where s is a counterexample
        """
        self_not_d = self.intersection(dfa.complement())
        not_self_d = self.complement().intersection(dfa)

        sym_diff = self_not_d.union(not_self_d)
        sym_diff.trim_unreachable()

        return sym_diff.is_empty()

    def complement(self) -> DFA:
        dfa = deepcopy(self)
        dfa.final = self.states - self.final
        return dfa

    def union(self, dfa: DFA) -> DFA:
        res = DFA()

        states = list(product(self.states, dfa.states))

        res.start = states.index((self.start, dfa.start))
        res.states = set(range(len(states)))
        res.final = set(i for i in res.states if states[i][0] in self.final or states[i][1] in dfa.final)
        res.transitions = {state: dict() for state in res.states}
        res.next_state = len(states)

        for state in res.states:
            for a in ALPHABET:
                res.transitions[state][a] = states.index((
                    self.transitions[states[state][0]][a],
                    dfa.transitions[states[state][1]][a]
                ))

        res.trim_unreachable()

        return res

    def intersection(self, dfa: DFA) -> DFA:
        res = DFA()

        states = list(product(self.states, dfa.states))

        res.start = states.index((self.start, dfa.start))
        res.states = set(range(len(states)))
        res.final = set(i for i in res.states if states[i][0] in self.final and states[i][1] in dfa.final)
        res.transitions = {state: dict() for state in res.states}
        res.next_state = len(states)

        for state in res.states:
            for a in ALPHABET:
                res.transitions[state][a] = states.index((
                    self.transitions[states[state][0]][a],
                    dfa.transitions[states[state][1]][a]
                ))

        res.trim_unreachable()

        return res

    def is_empty(self) -> Union[tuple[Literal[True], None], tuple[Literal[False], str]]:
        seen = set()
        queue: deque[int] = deque([self.start])
        aseq: dict[int, str] = defaultdict(lambda: "")

        while len(queue) > 0:
            curr = queue.popleft()
            if curr in self.final:
                return False, aseq[curr]
            if curr in seen:
                continue
            seen.add(curr)

            for a in ALPHABET:
                state = self.transitions[curr][a]
                if state not in aseq:
                    aseq[state] = aseq[curr] + a
                queue.append(state)

        return True, None

    def trim_unreachable(self) -> Self:
        """
        Deletes all unreachable states.

        First, runs a BFS to keep only the connected component containing the
        start state.

        Then it removes every state (besides the start state) with no incoming 
        transitions.
        """
        seen = set()
        queue: deque[int] = deque([self.start])

        while len(queue) > 0:
            curr = queue.popleft()
            seen.add(curr)
            for target in self.transitions[curr].values():
                if target not in seen:
                    queue.append(target)

        for state in self.states - seen:
            self.states -= {state}
            del self.transitions[state]

        unchanged = False
        while not unchanged:
            unchanged = True
            for target_state in list(self.states - {self.start}):
                if target_state not in self.states:
                    # check if state has already been removed
                    continue
                reached = False
                for state in self.states:
                    if reached:
                        break
                    if state != target_state:
                        for target in self.transitions[state].values():
                            if target_state == target:
                                reached = True
                                print(f"{target_state} can be reached by {state}")
                                break
                if not reached:
                    self.states -= {target_state}
                    del self.transitions[target_state]
                    unchanged = False

        return self

    def visualize(self, filename: str = 'dfa', format: str = 'png') -> Self:
        """
        Visualises the DFA

        Args:
            filename (str): Output filename (without extension).
            format (str): Output file format (e.g., png, pdf).
        """
        # Initialize a directed graph
        dot = Digraph(name="NFA", format=format)
        dot.attr(rankdir="LR")  # Left-to-right orientation
        dot.attr("node", shape="circle")  # States are circular by default

        # Add states
        for state in self.states:
            shape = "doublecircle" if state in self.final else "circle"
            dot.node(str(state), shape=shape, label=str(state))

        # Add transitions
        for state, transitions in self.transitions.items():
            for symbol, target in transitions.items():
                dot.edge(str(state), str(target), label=symbol)

        # Mark the start state
        dot.node("start", shape="plaintext", label="")
        dot.edge("start", str(self.start))

        # Save and render the graph
        dot.render(filename, cleanup=True)
        print(f"NFA visualized and saved as {filename}.{format}")

        return self

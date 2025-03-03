from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from transition import Transition

if TYPE_CHECKING:
    from node import Node


class Hypothesis:
    """
    A spanning-tree hypothesis
    """
    alphabet: str
    root_node: Node
    open_transitions: list[Transition]
    start: State
    states: set[State]
    final_states: set[State]

    def __init__(self, root_node: Node, alphabet: str) -> None:
        self.alphabet = alphabet
        self.root_node = root_node
        self.open_transitions = []
        self.states = set()
        self.start = self.add_state("")
        self.final_states = set()

    def print_hypothesis(self) -> None:
        print(f"Initial state: {self.start}")
        print(f"Final states: {list(map(lambda state: f"{state}", list(self.final_states)))}")
        for state in list(self.states):
            print(f"State: {state} (aseq = '{state.aseq}')")
            for a, transition in state.transitions.items():
                assert transition.target_state is not None
                if transition.is_tree:
                    print(f"\t={a}=> {transition.target_state}")
                else:
                    print(f"\t-{a}-> {transition.target_state}")

    def add_state(self, aseq: str) -> State:
        state = State(self, aseq)
        self.states.add(state)
        state.transitions = {
            a: Transition(False, self, aseq + a, self.root_node) for a in self.alphabet
        }
        for t in state.transitions.values():
            # all trasitions are initially be nontree
            self.open_transitions.append(t)

        return state

    def make_final(self, state: State) -> None:
        if state in self.states:
            self.final_states.add(state)
        else:
            raise ValueError("Unknown state passed")

    def run(self, s: str, start: Optional[State] = None) -> State:
        if start is None:
            start = self.start

        if s == "":
            return start
        
        t = start.transitions[s[0]]

        if t.target_state is not None:
            return self.run(s[1:], t.target_state)
        else:
            raise ValueError("Only call run when all transitions are closed")

    def evaluate(self, s: str):
        return self.run(s) in self.final_states


class State:
    """
    A state in the spanning-tree hypothesis
    """
    next_id: int = 0
    id: int
    hypothesis: Hypothesis
    node: Node
    transitions: dict[str, Transition]
    aseq: str
    # incoming_transition: Transition
    
    def __init__(self, hypothesis: Hypothesis, aseq: str) -> None:
        self.id = State.next_id
        State.next_id += 1
        self.hypothesis = hypothesis
        self.transitions = {}
        self.aseq = aseq

    def __repr__(self) -> str:
        assert self.node
        return f"q{self.id}"

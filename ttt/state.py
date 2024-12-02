from __future__ import annotations
from typing import Optional

from transition import NonTreeTransition, Transition
from node import Node


class Hypothesis:
    """
    A spanning-tree hypothesis
    """
    alphabet: str
    root_node: Node
    open_transitions: list[NonTreeTransition]
    start: State
    states: set[State]
    final_states: set[State]

    def __init__(self, root_node: Node, alphabet: str) -> None:
        self.alphabet = alphabet
        self.root_node = root_node
        self.open_transitions = []
        self.start = self.add_state()
        self.states = { self.start }
        self.final_states = set()

    def add_state(self) -> State:
        state = State(self)
        state.transitions = {
            a: NonTreeTransition(a, self.root_node) for a in self.alphabet
        }
        for t in state.transitions.values():
            if isinstance(t, NonTreeTransition):
                self.open_transitions.append(t)

        return state

    def make_final(self, state: State) -> None:
        if state in self.states:
            self.final_states.add(state)
        else:
            raise ValueError("Unknown state passed")


class State:
    """
    A state in the spanning-tree hypothesis
    """
    hypothesis: Hypothesis
    node: Optional[Node]
    transitions: dict[str, Transition]
    # incoming_transition: TreeTransition
    
    def __init__(self, hypothesis: Hypothesis) -> None:
        self.hypothesis = hypothesis
        self.transitions = {}
        self.node = None

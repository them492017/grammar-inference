from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from ttt.transition import Transition
from ttt.print import print

from graphviz import Digraph

from regex_parser.dfa import DFA

if TYPE_CHECKING:
    from node import Node
    from teach import Teacher


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

    def print_hypothesis_transitions(self) -> None:
        print(f"Initial state: {self.start}")
        print(f"Final states: {list(map(lambda state: f"{state}", list(self.final_states)))}")
        for state in list(self.states):
            print(f"State: {state} (aseq = '{state.aseq}')")
            for a, transition in state.transitions.items():
                print(f"\t-{a}-> {transition.target_node}")

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

    def evaluate(self, s: str, start: Optional[State] = None):
        if start is None:
            start = self.start

        return self.run(s, start=start) in self.final_states

    def run_non_deterministic(self, s: str, discriminator: str, teacher: Teacher, start: Optional[State] = None) -> State:
        print(start, s)
        if start is None:
            start = self.start

        if s == "":
            return start
        
        t = start.transitions[s[0]]

        if not t.target_node.is_leaf:
            new_target = t.target_node.sift(discriminator, teacher)
            t.target_node = new_target

        return self.run_non_deterministic(s[1:], discriminator, teacher, t.target_state)

    def evaluate_non_deterministic(self, s: str, teacher: Teacher, start: Optional[State] = None):
        if start is None:
            start = self.start

        return self.run_non_deterministic(s, s, teacher, start=start) in self.final_states

    def to_dfa(self) -> DFA:
        dfa = DFA()

        dfa.start = self.start.id
        dfa.states = set(map(lambda state: state.id, self.states))
        dfa.final = set(map(lambda state: state.id, self.final_states))
        dfa.next_state = len(dfa.states)

        for h_state in self.states:
            d_state = h_state.id
            dfa.transitions[d_state] = {}
            for a, transition in h_state.transitions.items():
                assert transition.target_state  # otherwise not a DFA
                dfa.transitions[d_state][a] = transition.target_state.id

        # TODO: do I need to deal with empty transitions?

        return dfa


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



def visualize_dfa(hypothesis: Hypothesis, filename='dfa', format='png'):
    """
    Visualize a DFA represented by the Hypothesis class. (From chatgpt)
    
    Args:
        hypothesis (Hypothesis): The DFA to visualize.
        filename (str): Output filename (without extension).
        format (str): Output file format (e.g., png, pdf).
    """
    # Initialize a directed graph
    dot = Digraph(name="DFA", format=format)
    dot.attr(rankdir="LR")  # Left-to-right orientation
    dot.attr("node", shape="circle")  # States are circular by default

    # Add states
    for state in hypothesis.states:
        shape = "doublecircle" if state in hypothesis.final_states else "circle"
        dot.node(str(state.id), shape=shape, label=str(state))

    # Add transitions
    for state in hypothesis.states:
        for symbol, transition in state.transitions.items():
            assert transition.target_state is not None
            dot.edge(str(state.id), str(transition.target_state.id), label=symbol)

    # Mark the start state
    dot.node("start", shape="plaintext", label="")
    dot.edge("start", str(hypothesis.start.id))

    # Add a title
    dot.attr(label=filename, labelloc="t", fontsize="20")

    # Save and render the graph
    dot.render(filename, cleanup=True)
    print(f"DFA visualized and saved as {filename}.{format}")

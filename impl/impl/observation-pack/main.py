from typing import NewType, Optional, Any
from dataclasses import dataclass
import heapq

Alphabet = NewType("Alphabet", str)  # could be literal if known

class Node:
    """
    A node class
    TODO: write which properties are used for leaves and inner nodes
    """
    children: list[Optional[Node]]
    parent: Optional[Node]
    signature: set[tuple[str, int]]  # could replace int with bool
    state: Optional[State]
    discriminator: Optional[str]
    incoming: list[Transition]

    def __init__(self, zero: Optional[Node], one: Optional[Node], discriminator: Optional[str]):
        self.children = [zero, one]
        self.parent = None
        if zero is not None:
            zero.parent = self
            zero.signature |= self.signature
            if discriminator is not None:
                zero.signature |= {(discriminator, 0)}
        if one is not None:
            one.parent = self
            one.signature |= self.signature
            if discriminator is not None:
                one.signature |= {(discriminator, 1)}
        self.signature = set()

        self.state = None
        self.discriminator = discriminator
        self.incoming = []

    def is_leaf(self) -> bool:
        return discriminator is None

    def sift(self, u: str) -> Node:
        if self.is_leaf():
            return self

        o = ...  # evaluate u^-1 lambda(self.aseq) ????
        return self.children[o].sift(u)

class State:
    trans: dict[Alphabet, Transition]
    node: Optional[Node]  # TODO: maybe unnecessary

    def __init__(self): 
        self.trans = {}
        self.node = None

class Transition:
    tgt_node: Node
    tgt_state: Optional[State]
    aseq: str

    def __init__(self, tgt_node: Node, tgt_state: Optional[State]): 
        self.tgt_node = tgt_node
        self.tgt_state = tgt_state
        self.aseq = ...  # TODO: check what this should be?

    def __lt__(self, t: Any) -> bool:
        assert isinstance(t, Transition)
        # TODO: make this also consider lexicographical order (test it works)
        if len(self.aseq) < len(t.aseq):
            return True
        elif len(self.aseq) > len(t.aseq):
            return False
        else:
            return self.aseq < t.aseq
    def is_tree(self):
        return tgt_state is not None

class Hypothesis:
    """
    A class for the hypothesis.
    The alphabet is fixed.
    A state stores its transitions individually.
    """
    start: State
    states: set[State]
    final: set[State]

    def __init__(self):
        q = State()
        self.start = q
        self.states = {q}
        self.final = set()

    def make_tree(self, t: Transition) -> None:
        """
        Converts t into a tree transition. This requires a new state to be created
        """
        ...  # TODO: implement this

def make_leaf(state: Optional[State]) -> Node:
    leaf = Node(None, None, None)
    if state is not None:
        leaf.state = state
    return leaf

def make_inner(v: str, t0: Node, t1: Node) -> Node:
    return Node(t0, t1, v)

def link(node: Node, state: State) -> None:
    node.state = state
    state.node = node

class ObservationPack:
    hypothesis: Hypothesis
    dtree: Node

    def __init__(self):
        self.hypothesis = Hypothesis()
        t0, t1 = make_leaf(None), make_leaf(None)
        self.dtree = make_inner("", t0, t1)

        l = self.dtree.sift("")

        if ("", 1) in l.signature:
            self.hypothesis.final = {self.hypothesis.start}

        link(l, self.hypothesis.start)

        self.close_transitions()

    def close_transitions(self):
        """
        Closes all open transitions in the hypothesis
        """
        transitions_to_new: list[Transition] = []  # this should be a pqueue / heap with len(aseq)
        while True:
            while len(self.hypothesis.open_transitions) > 0:
                t = self.hypothesis.open_transitions.pop()
                target = t.tgt_node.sift(t.aseq)
                t.tgt_node = target

                if target.is_leaf() and target.state is None:
                    # transition discovered a new state
                    heapq.heappush(transitions_to_new, t)

            if len(transitions_to_new) > 0:
                t = transitions_to_new[0]
                # make t into a tree transition (needs new state in hyp)
                q = make_tree(t)
                # remove all transitions from transitions_to_new with same aseq
                while len(transitions_to_new) > 0 and not t < transitions_to_new[0]:
                    heapq.heappop(transitions_to_new)
                # link t to the correct state
                link(t.tgt_node, q)

            if len(transitions_to_new) == 0:
                break

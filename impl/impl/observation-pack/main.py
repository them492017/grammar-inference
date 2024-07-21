from __future__ import annotations
from typing import NewType, Optional, Any
import heapq

Alphabet = NewType("Alphabet", str)  # could be literal if known


class Teacher:  # TODO: implement teacher
    def __init__(self) -> None:
        self.counterexample_count = 0

    def is_member(self, u: str) -> bool:
        """
        Performs a membership query on u
        """
        a_count, b_count = 0, 0
        for c in u:
            if c == "a":
                a_count += 1
            elif c == "b":
                b_count += 1
            else:
                raise ValueError(f"String {u} contains invalid character")

        return a_count % 2 == b_count % 2 == 0

    def get_counterexample(self, hypothesis: Hypothesis) -> Optional[str]:
        """
        Compares hypothesis, and returns a counterexample if they are inequivalent,
        else returns None
        """
        match self.counterexample_count:
            case 0:
                return "baaaaaab"
            case 1:
                return "baaaaaab"
            case _:
                return None
        self.counterexample_count += 1


class Node:
    """
    A node class
    TODO: write which properties are used for leaves and inner nodes
    """
    children: list[Optional[Node]]  # maybe should be Optioanl[list[Node]]
    parent: Optional[Node]
    signature: set[tuple[str, int]]  # could replace int with bool
    state: Optional[State]
    discriminator: Optional[str]
    incoming: set[Transition]

    def __init__(self, zero: Optional[Node], one: Optional[Node], discriminator: Optional[str]):
        self.children = [zero, one]
        self.parent = None

        self.signature = set()
        self.state = None

        self.discriminator = discriminator
        self.incoming = set()

        for o, child in enumerate(self.children):
            if child is not None and discriminator is not None:
                child.parent = self
                child.signature = self.signature | {(discriminator, o)}

    def print_tree(self, level=0):
        if self.is_leaf():
            if self.state is None:
                print(f"{" " * 4 * level}-> <None@{self}>")
            else:
                print(f"{" " * 4 * level}-> <q{self.state.id}>")
        else:
            assert self.children[0]
            assert self.children[1]
            self.children[0].print_tree(level+1)
            print(f"{"\t" * level}-> <{self.discriminator}>")
            self.children[1].print_tree(level+1)

    @classmethod
    def make_leaf(cls, state: Optional[State]) -> Node:
        leaf = cls(None, None, None)
        if state is not None:
            leaf.state = state
        return leaf

    @classmethod
    def make_inner(cls, v: str, t0: Node, t1: Node) -> Node:
        return cls(t0, t1, v)

    def is_leaf(self) -> bool:
        return self.discriminator is None

    def is_final(self) -> bool:
        return ("", 1) in self.signature

    def split_leaf(self, discriminator) -> tuple[Node, Node]:
        """
        Splits the given node into an inner node with two leaves as children,
        and returns these children
        """
        assert self.is_leaf()

        self.children[0] = Node.make_leaf(None)
        self.children[1] = Node.make_leaf(None)
        self.discriminator = discriminator

        for o, child in enumerate(self.children):
            if child is not None:
                child.parent = self
                child.signature = self.signature | {(discriminator, o)}

        return self.children[0], self.children[1]

    def sift(self, teacher: Teacher, u: str) -> Node:
        if self.is_leaf():
            return self

        assert self.discriminator is not None  # self is inner

        o = teacher.is_member(u + self.discriminator)
        child = self.children[o]
        if child is None:
            raise ValueError("Null child reached in sifting")

        return child.sift(teacher, u)


class State:
    id: int
    trans: dict[Alphabet, Transition]
    node: Optional[Node]
    aseq: str

    def __init__(self, id: int):
        self.id = id
        self.trans = {}
        self.node = None
        self.aseq = ""


class Transition:
    start_state: State
    tgt_node: Node
    aseq: str

    def __init__(self, start_state: State, tgt_node: Node, aseq: str):
        self.start_state = start_state
        self.aseq = aseq
        self.tgt_node = tgt_node
        tgt_node.incoming.add(self)

    def __lt__(self, t: Any) -> bool:
        # want transitions to be ordered by tgt_node so heap contains
        # contiguous blocks of transitions with the same target_state
        # thus we nodes sort by pointer (actual ordering of nodes is arbitrary)
        assert isinstance(t, Transition)

        return id(self.tgt_node) < id(t.tgt_node)

    @property
    def tgt_state(self) -> Optional[State]:
        return self.tgt_node.state

    def is_open(self) -> bool:
        return self.tgt_state is None

    def set_target(self, target: Node) -> None:
        self.tgt_node.incoming.remove(self)
        self.tgt_node = target
        target.incoming.add(self)


class Hypothesis:
    """
    A class for the hypothesis.
    The alphabet is fixed.
    A state stores its transitions individually.
    """
    next_state: int
    start: State
    states: set[State]
    final: set[State]
    open_transitions: list[Transition]

    def __init__(self, root: Node):
        self.next_state = 0
        self.states = set()
        self.final = set()
        self.start = self.add_state()
        self.open_transitions = []

        alphabet = [Alphabet("a"), Alphabet("b")]  # TODO: make this global
        for a in alphabet:
            self.start.trans[a] = Transition(self.start, root, a)
            self.open_transitions.append(self.start.trans[a])

    def print_hypothesis(self) -> None:
        print(f"Initial state: q{self.start.id}")
        print(f"Final states: {list(map(lambda state: f"q{state.id}", list(self.final)))}")
        for state in list(self.states):
            print(f"State: q{state.id}")
            for a, transition in state.trans.items():
                assert isinstance(transition.tgt_state, State)
                print(f"\t-{a}-> q{transition.tgt_state.id}")

    def add_state(self):
        # TODO: some states have no transactions but should have a default one
        new_state = State(self.next_state)
        self.next_state += 1
        self.states.add(new_state)
        return new_state

    def make_tree(self, root: Node, t: Transition) -> State:
        """
        Converts t into a tree transition. This requires a new state to be created
        which is returned.
        """
        # assert t.tgt_node.is_leaf()
        # assert t.tgt_state is None

        new_state = self.add_state()
        new_state.aseq = t.aseq

        alphabet = [Alphabet("a"), Alphabet("b")]  # TODO: make this global
        for a in alphabet:
            new_state.trans[a] = Transition(new_state, root, a)
            self.open_transitions.append(new_state.trans[a])

        print(new_state.trans)

        return new_state  # TODO: why is t not even used here???

    def state_of(self, u: str) -> State:
        """
        Given a string u, returns H[u]. When this function is called, the hypothesis
        should be closed.
        """
        curr = self.start
        for c in u:
            curr = curr.trans[Alphabet(c)].tgt_state

            if curr is None:
                raise ValueError("Null state reached. Hypothesis is not closed.")

        return curr

    # def evaluate(self, v: str, start: Optional[State] = None):
    #     curr = start or self.start
    #
    #     for c in v:
    #         curr = curr.trans[Alphabet(c)].tgt_state
    #
    #         if curr is None:
    #             raise ValueError("Null state reached. Hypothesis is not closed.")
    #
    #     return curr in self.final


def link(node: Node, state: State) -> None:
    node.state = state
    state.node = node


class ObservationPack:
    teacher: Teacher
    hypothesis: Hypothesis
    dtree: Node

    def __init__(self, teacher: Teacher):
        self.TEMPORARY = 0
        self.teacher = teacher

        t0, t1 = Node.make_leaf(None), Node.make_leaf(None)
        self.dtree = Node.make_inner("", t0, t1)
        self.hypothesis = Hypothesis(self.dtree)

        leaf = self.dtree.sift(self.teacher, "")

        if leaf.is_final():
            self.hypothesis.final.add(self.hypothesis.start)

        link(leaf, self.hypothesis.start)

        self.close_transitions()

    def close_transitions(self) -> None:
        """
        Closes all open transitions in the hypothesis
        """
        transitions_to_new: list[Transition] = []  # this should be a pqueue / heap with len(aseq)
        # count = 0
        while True:
            # count += 1
            # if count > 15: return print("Terminating early from outer")
            while len(self.hypothesis.open_transitions) > 0:
                # count += 1
                # if count > 15: return print("Terminating early from inner")
                t = self.hypothesis.open_transitions.pop()
                if t.is_open():  # sometimes an open transition can be closed by a prior operation
                    target = t.tgt_node.sift(self.teacher, t.aseq)
                    t.tgt_node = target

                    if target.is_leaf() and target.state is None:
                        # transition discovered a new state
                        print("New state discovered")
                        print(target)
                        self.dtree.print_tree()
                        heapq.heappush(transitions_to_new, t)

            # TODO: fix bug where more and more states are created
            print(f"Num transitions to new: {len(transitions_to_new)}")
            if len(transitions_to_new) > 0:
                t = transitions_to_new[0]
                print(t.tgt_node)
                # make t into a tree transition
                q = self.hypothesis.make_tree(self.dtree, t)
                # remove all transitions from transitions_to_new with same aseq
                while len(transitions_to_new) > 0 and not t < transitions_to_new[0]:
                    heapq.heappop(transitions_to_new)
                # make the state final if needed
                if t.tgt_node.is_final():
                    self.hypothesis.final.add(q)
                # link t to the correct state
                link(t.tgt_node, q)

            # print("Finished making new states")
            # self.dtree.print_tree()
            # return  # TEMPORARY

            if len(transitions_to_new) == len(self.hypothesis.open_transitions) == 0:
                break

    def decompose(self, w: str) -> tuple[str, Alphabet, str]:
        """
        Given a counterexample w, returns a decomposition (u, a, v) where
        len(a) == 1 and certain properties are satisfied.
        """
        match self.TEMPORARY:
            case 0:
                return "", Alphabet("b"), "aaaaaab"
            case 1:
                return "b", Alphabet("a"), "aaaaab"
            case _:
                raise ValueError("Too many counterexamples received for hardcoding")
        # TODO: actually implement
        return "", Alphabet(w[0]), w[1:]

    def refine(self, counterexample: str) -> None:
        u, a, v = self.decompose(counterexample)
        self.split(u, a, v)
        self.close_transitions()

    def split(self, u: str, a: Alphabet, v: str) -> None:
        q = self.hypothesis.state_of(u)
        t = q.trans[a]

        old_state = t.tgt_state
        new_state = self.hypothesis.make_tree(self.dtree, t)

        # TODO: justify asserts
        assert old_state is not None
        assert old_state.node is not None
        leaf0, leaf1 = old_state.node.split_leaf(v)

        if self.teacher.is_member(old_state.aseq + v):
            link(leaf0, old_state)
            link(leaf1, new_state)
        else:
            link(leaf0, new_state)
            link(leaf1, old_state)

        self.hypothesis.open_transitions += old_state.node.incoming

    def learn(self) -> tuple[Hypothesis, Node]:
        self.hypothesis.print_hypothesis()
        self.dtree.print_tree()

        while (v := self.teacher.get_counterexample(self.hypothesis)) is not None:
            self.refine(v)

            self.hypothesis.print_hypothesis()
            self.dtree.print_tree()

        return self.hypothesis, self.dtree


def main():
    teacher = Teacher()

    op = ObservationPack(teacher)
    op.hypothesis.print_hypothesis()
    op.dtree.print_tree()

    # hypothesis, tree = ObservationPack(teacher).learn()
    #
    # hypothesis.print_hypothesis()
    # tree.print_tree()


if __name__ == "__main__":
    main()

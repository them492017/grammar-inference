from __future__ import annotations
from typing import NewType, Optional, Any, Callable
from sys import argv
from time import perf_counter
from pprint import pprint
from math import log, floor
import heapq

from teacher.teacher import Teacher

Alphabet = NewType("Alphabet", str)  # could be literal if known
alphabet: list[Alphabet] = [Alphabet("a"), Alphabet("b")]


class Node:
    """
    A node class
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

    def print_tree(self, child: int = -1, level: int = 0, property: str = ""):
        """
        A method that outputs a discrimination tree rooted at `self`. Adapted from
        https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
        """
        if property == "incoming":
            debug_info = " " + str(self.incoming)
        elif property == "id":
            debug_info = " " + str(self)
        elif property == "state":
            debug_info = " " + str(self.state)
        else:
            debug_info = ""

        if child == 1:
            arrow = "=>"
        elif child == 0:
            arrow = "->"
        else:
            arrow = "->"

        if self.is_leaf():
            if self.state is None:
                print(f"{" " * 4 * level}{arrow} <None@{self}{debug_info}>")
            else:
                print(f"{" " * 4 * level}{arrow} <q{self.state.id}{debug_info}>")
        else:
            assert self.children[0]
            assert self.children[1]
            self.children[0].print_tree(0, level+1, property)
            print(f"{" " * 4 * level}{arrow} <{self.discriminator}{debug_info}>")
            self.children[1].print_tree(1, level+1, property)

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
        self.state = None

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

    def __repr__(self) -> str:
        if self.tgt_state is not None:
            return f"<Transition aseq={self.aseq} tgt_state={self.tgt_state.id}>"
        else:
            return f"<Transition aseq={self.aseq} tgt_state={{}}>"

    """
    @property
    def tgt_node(self) -> Node:
        return self._tgt_node

    @tgt_node.setter
    def tgt_node(self, target: Node) -> None:
        self.tgt_node.incoming.remove(self)
        self._tgt_node = target
        target.incoming.add(self)
    """

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

        for a in alphabet:
            self.start.trans[a] = Transition(self.start, root, a)
            self.open_transitions.append(self.start.trans[a])

    def print_hypothesis(self) -> None:
        print(f"Initial state: q{self.start.id}")
        print(f"Final states: {list(map(lambda state: f"q{state.id}", list(self.final)))}")
        for state in list(self.states):
            print(f"State: q{state.id} (aseq = {state.aseq})")
            for a, transition in state.trans.items():
                assert isinstance(transition.tgt_state, State)
                print(f"\t-{a}-> q{transition.tgt_state.id}")

    def to_grammar(self) -> tuple[dict[str, list[list[str]]], str]:
        # TODO: store hypothesis in this form so it doesnt have to be rebuilt each iteration
        #       Actually, the teacher copies the grammar so the complexity would not change...
        grammar: dict[str, list[list[str]]] = {}
        for state in self.states:
            state_name = f"q{state.id}"
            grammar[state_name] = [
                [str(a), f"q{transition.tgt_state.id}"]
                for a, transition in state.trans.items()
                if transition.tgt_state is not None
            ]
            if state in self.final:
                grammar[state_name].append([])

        return grammar, f"q{self.start.id}"

    def add_state(self):
        new_state = State(self.next_state)
        self.next_state += 1
        self.states.add(new_state)
        return new_state

    def make_tree(self, root: Node, t: Transition) -> State:
        """
        Converts t into a tree transition. This requires a new state to be created
        which is returned.
        """
        new_state = self.add_state()
        new_state.aseq = t.aseq

        for a in alphabet:
            new_state.trans[a] = Transition(new_state, root, t.aseq + a)
            self.open_transitions.append(new_state.trans[a])

        print(new_state.trans)

        return new_state

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

    def evaluate(self, v: str, start: Optional[State] = None):
        curr = start or self.start

        for c in v:
            curr = curr.trans[Alphabet(c)].tgt_state

            if curr is None:
                raise ValueError("Null state reached. Hypothesis is not closed.")

        return curr in self.final


def link(node: Node, state: State) -> None:
    node.state = state
    state.node = node


class ObservationPack:
    teacher: Teacher
    hypothesis: Hypothesis
    dtree: Node

    def __init__(self, teacher: Teacher):
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
        print("Starting CloseTransitions")
        print("Currently, the open transitions are", self.hypothesis.open_transitions)
        self.dtree.print_tree(property="id")
        transitions_to_new: list[Transition] = []  # this should be a pqueue / heap with len(aseq)
        while True:
            while len(self.hypothesis.open_transitions) > 0:
                t = self.hypothesis.open_transitions.pop(0)  # TODO: pop from end for efficiency
                print("Closing", t)
                assert t.is_open()

                print("Sifting transition")
                target = t.tgt_node.sift(self.teacher, t.aseq)
                t.set_target(target)

                print(f"Transition with aseq={t.aseq} has new target {target}")
                # print(target.is_leaf(), target)
                # self.dtree.print_tree_debug(property="state")

                if target.is_leaf() and target.state is None:
                    # transition discovered a new state
                    print("New state discovered at", target)
                    heapq.heappush(transitions_to_new, t)

            if len(transitions_to_new) > 0:
                t = transitions_to_new[0]
                print("Making new state for", t)
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

            if len(transitions_to_new) == len(self.hypothesis.open_transitions) == 0:
                break

    def decompose(self, w: str) -> tuple[str, Alphabet, str]:
        """
        Given a counterexample w, returns a decomposition (u, a, v) where
        len(a) == 1 and certain properties are satisfied.
        """
        # define prefix mapping
        def prefix_mapping(s: str, i: int) -> str:
            assert 0 <= i <= len(s)
            return self.hypothesis.state_of(s[:i]).aseq + s[i:]

        # define alpha
        def alpha(i: int) -> bool:
            return self.teacher.is_member(prefix_mapping(w, i)) == self.hypothesis.evaluate(w)

        # binary search (or some variation of it)
        # i = self.exponential_search(alpha, len(w))
        i = self.partition_search(alpha, len(w))

        return w[:i], Alphabet(w[i]), w[i+1:]

    def binary_search(self, alpha: Callable[[int], bool], high: int, low: int = 0) -> int:
        while high - low > 1:
            mid = (low + high) // 2

            if alpha(mid) == 0:
                low = mid
            else:
                high = mid

        return low

    def exponential_search(self, alpha: Callable[[int], bool], high: int) -> int:
        range_len = 1
        low = 0
        found = False

        while not found and high - range_len > 0:
            print(low, high, high - range_len)
            if alpha(high - range_len) == 0:
                low = high - range_len
                found = True
            else:
                high -= range_len
                range_len *= 2

        return self.binary_search(alpha, high, low)

    def partition_search(self, alpha: Callable[[int], bool], max: int) -> int:
        step = floor(max / log(max, 2))
        low, high = 0, max
        found = False

        while not found and high - step > low:
            if alpha(high - step) == 0:
                low = high - step
                found = True
                break
            else:
                high = high - step

        return self.rs_eager_search(alpha, high, low)

    def rs_eager_search(self, alpha: Callable[[int], bool], high: int, low: int = 0) -> int:
        def beta(i: int) -> int:
            # TODO: could memoise previously computed alpha values
            return alpha(i) + alpha(i+1)

        while high > low:
            mid = (low + high) // 2

            if beta(mid) == 1:  # alpha(mid) != alpha(mid+1)
                return mid
            elif beta(mid) == 0:  # beta(mid+1) <= 1
                low = mid + 1
            else:  # beta(mid - 1) >= 1
                high = mid - 1

        return low

    def refine(self, counterexample: str) -> None:
        u, a, v = self.decompose(counterexample)
        print(f"Decomposed {counterexample} into {(u, a, v)}")
        self.split(u, a, v)
        self.close_transitions()

    def split(self, u: str, a: Alphabet, v: str) -> None:
        q = self.hypothesis.state_of(u)
        t = q.trans[a]
        print(f"Running split with q_pred=q{q.id}, transition {t}")

        old_state = t.tgt_state
        new_state = self.hypothesis.make_tree(self.dtree, t)

        if self.teacher.is_member(new_state.aseq):
            self.hypothesis.final.add(new_state)

        # TODO: justify asserts
        assert old_state is not None
        assert old_state.node is not None
        print(f"Splitting q{old_state.id}'s node with discriminator {v}")
        leaf0, leaf1 = old_state.node.split_leaf(v)

        self.hypothesis.open_transitions += list(old_state.node.incoming)

        # TODO: eliminate this query since the result is found in call to self.decompose
        if self.teacher.is_member(old_state.aseq + v):
            # old state is 1-child
            link(leaf1, old_state)
            link(leaf0, new_state)
        else:
            # old state is 0-child
            link(leaf0, old_state)
            link(leaf1, new_state)

    def learn(self, max_iterations: int = -1) -> tuple[Hypothesis, Node]:
        self.hypothesis.print_hypothesis()
        self.dtree.print_tree()

        iterations = 0
        while not (equiv := self.teacher.is_equivalent(*self.hypothesis.to_grammar()))[0]:
            if max_iterations != -1 and iterations >= max_iterations:
                print("Reached maximum iterations")
                return self.hypothesis, self.dtree

            counterexample = equiv[1]
            assert counterexample is not None
            print("Found counterexample", counterexample)
            self.refine(counterexample)

            # self.hypothesis.print_hypothesis()
            pprint(self.hypothesis.to_grammar())
            self.dtree.print_tree()
            iterations += 1

        print("=" * 40)
        print(f"No counterexample found! Learning complete after {iterations} iterations.")
        return self.hypothesis, self.dtree


def main():
    if len(argv) == 2:
        regex = argv[1]
    else:
        # Default regex: accepts strings with even num a's and even num b's
        regex = r"((b(aa)*b)|((a|b(aa)*ab)((bb)|(ba(aa)*ab))*(a|ba(aa)*b)))*"

    teacher = Teacher(regex, 0.01, 0.01)

    start_time = perf_counter()
    hypothesis, tree = ObservationPack(teacher).learn()
    end_time = perf_counter()

    print("=" * 40)
    hypothesis.print_hypothesis()
    print("=" * 40)
    tree.print_tree()
    print("=" * 40)
    pprint(hypothesis.to_grammar())
    print("=" * 40)
    print(f"Learning completed in {end_time - start_time} seconds")
    print("=" * 40)


if __name__ == "__main__":
    main()

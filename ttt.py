from __future__ import annotations
from typing import Iterable, NewType, Optional, Any, Callable, TypeVar, cast
from sys import argv
from time import perf_counter
from pprint import pprint
from math import log, floor
from collections import defaultdict
import heapq

from teacher.simple_teacher import SimpleTeacher

Alphabet = NewType("Alphabet", str)  # could be literal if known
alphabet: list[Alphabet] = [Alphabet("a"), Alphabet("b")]


T = TypeVar("T")


def debug_fn(property: str) -> None:
    pass


def choose(st: set[T]) -> T:
    for t in st:
        return t
    raise ValueError("Set is empty")


def link(node: Node, state: State) -> None:
    node.state = state
    state.node = node
    if node.block is not None:  # TODO: maybe assert this
        node.block.states.add(state)


class Node:
    """
    A node class
    """
    children: list[Optional[Node]]  # maybe should be Optioanl[list[Node]]
    parent: Optional[Node]
    # signature: set[tuple[str, int]]  # could replace int with bool
    state: Optional[State]
    discriminator: Optional[str]
    temporary: bool
    incoming: set[Transition]
    block: Optional[Block]

    def __init__(self, zero: Optional[Node], one: Optional[Node],
                 discriminator: Optional[str], temporary: bool = False,
                 block: Optional[Block] = None, parent: Optional[Node] = None,
                 signature: Optional[set[tuple[str, int]]] = None):
        self.children = [zero, one]
        self.parent = parent

        # self.signature = signature or set()
        self.state = None

        self.discriminator = discriminator
        self.temporary = temporary
        self.incoming = set()

        self.block = block

        for o, child in enumerate(self.children):
            if child is not None and discriminator is not None:
                child.parent = self
                # child.signature = self.signature | {(discriminator, o)}

    def print_tree(self, child: int = -1, level: int = 0, property: str = ""):
        """
        A method that outputs a discrimination tree rooted at `self`. Adapted from
        https://stackoverflow.com/questions/34012886/print-binary-tree-level-by-level-in-python
        """
        if property == "incoming":
            debug_info = " " + str(self.incoming)
        elif property == "id":
            debug_info = " " + str(self)
        elif property == "temporary":
            debug_info = " " + str(self.temporary)
        elif property == "state":
            debug_info = " " + str(self.state)
        elif property == "block":
            debug_info = " " + str(self.block)
        elif property == "depth":
            debug_info = " " + str(self.depth)
        elif property == "signature":
            debug_info = " " + str(self.signature)
        elif property == "parent":
            debug_info = " " + str(self.parent)
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

    @property
    def depth(self) -> int:
        return len(self.signature)

    @property
    def signature(self) -> set[tuple[str, int]]:
        # TODO: maybe should store signature to avoid O(log n) computation
        signature: set[tuple[str, int]] = set()
        prev, curr = self, self.parent

        while curr is not None:
            assert curr.discriminator is not None
            if curr.children[0] == prev:
                signature.add((curr.discriminator, 0))
            if curr.children[1] == prev:
                signature.add((curr.discriminator, 1))
            prev, curr = curr, curr.parent

        return signature

    @classmethod
    def new(cls) -> Node:
        return cls(None, None, None)

    def make_leaf(self, state: Optional[State], block: Optional[Block],
                  hypothesis: Optional[Hypothesis]) -> Node:
        if state is not None:
            self.state = state

        if block is not None:
            self.block = block
        elif hypothesis is not None:
            self.block = Block(self, hypothesis)
        # else:
        #     print("==================New leaf has no block")

        if self.state is not None and self.block is not None:
            self.block.states.add(self.state)

        return self

    def make_inner(self, v: str, t0: Node, t1: Node, block: Optional[Block],
                   temporary: bool) -> Node:
        self.__init__(t0, t1, v, temporary=temporary, block=block,
                      signature=self.signature, parent=self.parent)
        # if block is None:
        #     print("==================New leaf has no block")

        return self

    def descending_iterator(self) -> Iterable[Node]:
        # TODO: probably should test this works
        stack: list[Node] = [self]

        while len(stack) > 0:
            node = stack.pop()
            yield node
            for child in node.children:
                if child is not None:
                    stack.append(child)

    def is_leaf(self) -> bool:
        return self.discriminator is None

    def is_final(self) -> bool:
        return ("", 1) in self.signature

    def split_leaf(self, discriminator: str, hypothesis: Hypothesis,
                   block: Optional[Block]) -> tuple[Node, Node]:
        """
        Splits the given node into an inner node with two leaves as children,
        and returns these children
        """
        assert self.is_leaf()

        self.children[0] = Node.new().make_leaf(None, block, hypothesis)
        self.children[1] = Node.new().make_leaf(None, block, hypothesis)

        # print("Split leaf now has children", self.children)

        self.discriminator = discriminator
        # print("Split leaf now has discriminator", self.discriminator)
        # print("Split leaf is_leaf() returns", self.is_leaf())
        self.temporary = True
        self.state = None
        if self.block is None:
            self.block = block or Block(self, hypothesis)

        for o, child in enumerate(self.children):
            if child is not None:  # maybe should assert instead
                child.parent = self
                # child.signature = self.signature | {(discriminator, o)}
                child.block = self.block

        assert self.block is not None
        assert self.parent is not None

        # print("Split leaf is_leaf() returns", self.is_leaf())
        return self.children[0], self.children[1]

    def sift(self, teacher: SimpleTeacher, u: str) -> Node:
        if self.is_leaf():
            return self

        assert self.discriminator is not None  # self is inner

        o = teacher.is_member(u + self.discriminator)
        child = self.children[o]
        if child is None:
            raise ValueError("Null child reached in sifting")

        return child.sift(teacher, u)

    def soft_sift(self, teacher: SimpleTeacher, u: str) -> Node:
        curr = self

        while not curr.is_leaf() and not curr.temporary:
            assert curr.discriminator is not None  # since it is an inner node
            o = teacher.is_member(u + curr.discriminator)
            curr = curr.children[o]
            assert curr is not None  # since it was the child of an inner node

        return curr

    @classmethod
    def lca(cls, nodes: list[Node]) -> Node:
        # print("Computing LCA of", nodes)
        # debug_fn("signature")
        # debug_fn("parent")
        min_depth = min(map(lambda node: node.depth, nodes))
        nodes_in_layer: set[Node] = set()

        for node in nodes:
            while node.depth > min_depth:
                node = node.parent

                if node is None:
                    raise ValueError("Null parent of non-root node")

            nodes_in_layer.add(node)

        # print(nodes_in_layer)

        while len(nodes_in_layer) > 1:
            nodes_in_layer = {
                node.parent for node in nodes_in_layer if node.parent is not None
            }

        if len(nodes_in_layer) == 0:
            raise ValueError("LCA couldn't be computed")

        return nodes_in_layer.pop()


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

    def __repr__(self) -> str:
        return f"q{self.id}"


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
    blocks: set[Block]  # TODO: consider moving this somewhere else

    def __init__(self, root: Node):
        self.next_state = 0
        self.states = set()
        self.final = set()
        self.start = self.add_state()
        self.open_transitions = []
        self.blocks = set()

        for a in alphabet:
            self.start.trans[a] = Transition(self.start, root, a)
            self.open_transitions.append(self.start.trans[a])

    def print_hypothesis(self) -> None:
        print(f"Initial state: q{self.start.id}")
        print(f"Final states: {list(map(lambda state: f"q{state.id}", list(self.final)))}")
        for state in list(self.states):
            print(f"State: q{state.id} (aseq = {state.aseq})")
            for a, transition in state.trans.items():
                if isinstance(transition.tgt_state, State):
                    print(f"\t-{a}-> q{transition.tgt_state.id}")
                else:
                    assert transition.tgt_node.block is not None
                    print(f"\t-{a}-> {transition.tgt_node.block.states}")

    def to_grammar(self) -> tuple[dict[str, list[list[str]]], str]:
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
        # print(f"Added new state q{new_state.id} to hypothesis")
        # print(f"States: {list(map(lambda state: f"q{state.id}", list(self.states)))}")
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

        return new_state

    def state_of_nondeterministic(self, u: str, teacher: SimpleTeacher,
                                  start: Optional[State] = None) -> State:
        """
        Given a string u, returns H[u]. When this function is called, the hypothesis
        should be closed.
        """
        curr = start or self.start

        for c in u:
            assert isinstance(curr, State)

            t = curr.trans[Alphabet(c)]

            if t.tgt_state is None:
                target_leaf = t.tgt_node.sift(teacher, t.aseq)
                t.set_target(target_leaf)

            curr = t.tgt_state

            if curr is None:
                raise ValueError("Null state reached. Hypothesis is not closed.")

        return curr

    def evaluate_nondeterministic(self, u: str, teacher: SimpleTeacher,
                                  start: Optional[State] = None) -> bool:
        return self.state_of_nondeterministic(u, teacher, start) in self.final

    def evaluate(self, u: str, start: Optional[State] = None) -> bool:
        """
        Evaluates a non-deterministic hypothesis by making an arbitrary
        non-deterministic choice at each step.

        It can be shown that the non-determinism does not affect the
        state output function, so this is justified.
        """
        curr = start or self.start

        for c in u:
            target = curr.trans[Alphabet(c)].tgt_node

            if target.is_leaf():
                curr = target.state
            else:
                target_block = target.block
                assert target_block is not None  # since soft-sifting has occured
                # choose arbitrary state in target_block
                choice = target_block.states.pop()
                target_block.states.add(choice)
                # update the current state
                curr = choice

            if curr is None:
                raise ValueError("Null state reached when evaluating hypothesis")

        return curr in self.final


class Block:
    root: Node
    states: set[State]

    # all nodes should have a pointer to their block
    # then when they are split they add the new state to block.states

    # store list of blocks in hypothesis?
    # TODO: when can new blocks be created?
    # - when splitting a block in discriminator finalisation (replace_block)
    # - when splitting a leaf node
    # - when closing transitions and a new state is discovered

    def __init__(self, root: Node, hypothesis: Hypothesis) -> None:
        # print("=====Initialising block now", self)
        self.root = root
        self.states = set()
        if self.root.state is not None:
            self.states.add(self.root.state)
        hypothesis.blocks.add(self)

    def is_singleton(self) -> bool:
        # print(self.root)
        return self.root.is_leaf()

    def replace_block(self, teacher: SimpleTeacher, v: str, dtree: Node,
                      hypothesis: Hypothesis) -> None:
        # print("=======Starting replace block with v =", v)
        # mark[o][n] = True if n in o-subtree else false
        # inc[o][n] = [incoming transitions of the o-subtree version of n]
        # state[o][n] = state corresponding to o-subtree version of n
        mark: tuple[set[Node], set[Node]] \
            = set(), set()
        inc: tuple[dict[Node, set[Transition]], dict[Node, set[Transition]]] \
            = defaultdict(set), defaultdict(set)
        state: tuple[dict[Node, State], dict[Node, State]] \
            = {}, {}

        for o in [0, 1]:
            mark[o].add(self.root)

        for n in self.root.descending_iterator():  # TODO: iterator through block subtree
            for t in n.incoming:
                o = teacher.is_member(t.aseq + v)
                inc[o][n].add(t)
                self.mark(mark, n, o)

            if n.is_leaf():
                assert n.state is not None  # all leaves in blocks are labelled
                o = teacher.is_member(n.state.aseq + v)
                state[o][n] = n.state
                self.mark(mark, n, o)

        block0, block1 = Block(self.root, hypothesis), Block(self.root, hypothesis)
        hypothesis.blocks.remove(self)

        # pprint(mark)
        # pprint(inc)
        # pprint(state)

        t0, t1 = (
            self.extract(self.root, mark[0], inc[0], state[0],
                         block0, dtree, hypothesis),
            self.extract(self.root, mark[1], inc[1], state[1],
                         block1, dtree, hypothesis),
        )

        block0.root = t0
        block1.root = t1

        # for n in self.root.descending_iterator():
        #     # delete states from hypothesis and remove nodes
        #     if n != self.root:
        #         if (q := n.state) is not None:
        #             hypothesis.states.remove(q)
        #             if q in hypothesis.final:
        #                 hypothesis.final.remove(q)
        #         # TODO: maybe just use del keyword

        self.root.make_inner(v, t0, t1, None, False)
        # print("Finished replace block")

    def mark(self, mark: tuple[set[Node], set[Node]],
             n: Node, o: bool) -> None:
        """
        Given a node n and a boolean o, mark n and its ancestors to be in the
        o subtree
        """
        while n not in mark[o]:
            mark[o].add(n)
            assert n.parent is not None  # since all nodes in a block have a parent
            n = n.parent

    def extract(self, n: Node, mark: set[Node], inc: dict[Node, set[Transition]],
                state: dict[Node, State], block: Block, dtree: Node,
                hypothesis: Hypothesis) -> Node:
        # print("Extracting", n)
        if n.is_leaf():
            res = Node.new().make_leaf(state[n], block, hypothesis)
            link(res, state[n])
            # TODO: check that state[n] can never be None
        else:
            assert n.discriminator is not None  # n is inner
            if n.children[0] in mark and n.children[1] in mark:
                # both children are marked
                t0, t1 = (
                    self.extract(n.children[0], mark, inc, state, block, dtree, hypothesis),
                    self.extract(n.children[1], mark, inc, state, block, dtree, hypothesis),
                )
                res = Node.new().make_inner(n.discriminator, t0, t1, block, True)
            elif n.children[0] in mark:
                # n has only one child so can be ignored
                inc[n.children[0]] |= inc[n]
                return self.extract(n.children[0], mark, inc, state, block, dtree, hypothesis)
            elif n.children[1] in mark:
                # n has only one child so can be ignored
                inc[n.children[1]] |= inc[n]
                return self.extract(n.children[1], mark, inc, state, block, dtree, hypothesis)
            else:
                # both children are unmarked
                # hence since n is in subtree it must have an incoming transition
                new_node = self.create_new(n, inc, dtree, block, hypothesis)
                new_node.block = block
                return new_node

        res.incoming = inc[n]
        for t in res.incoming:
            t.tgt_node = res

        return res

    def create_new(self, n: Node, inc: dict[Node, set[Transition]],
                   root: Node, block: Block, hypothesis: Hypothesis) -> Node:
        t = inc[n].pop()  # TODO: make sure we never need to use inc[n] again
        state = hypothesis.make_tree(root, t)
        new_node = Node.new().make_leaf(state, block, hypothesis)
        new_node.incoming = inc[n] - {t}

        for t in new_node.incoming:
            t.tgt_node = new_node

        link(new_node, state)
        # print(f"Created new node {new_node} with associated state q{state.id}")

        return new_node


class TTTAlgorithm:
    teacher: SimpleTeacher
    hypothesis: Hypothesis
    dtree: Node

    def __init__(self, teacher: SimpleTeacher):
        self.teacher = teacher

        t0, t1 = (
            Node.new().make_leaf(None, None, None),
            Node.new().make_leaf(None, None, None),
        )

        self.dtree = Node.new().make_inner("", t0, t1, None, False)
        self.hypothesis = Hypothesis(self.dtree)

        t0.block = Block(t0, self.hypothesis)
        t1.block = Block(t1, self.hypothesis)

        leaf = self.dtree.sift(self.teacher, "")

        if leaf.is_final():
            self.hypothesis.final.add(self.hypothesis.start)

        link(leaf, self.hypothesis.start)

        self.close_transitions_soft()

        global debug_fn
        def debug_fn(property="id"):
            # print("===DEBUG_FN===")
            # self.hypothesis.print_hypothesis()
            # pprint(self.hypothesis.to_grammar())
            # self.dtree.print_tree(property=property)
            # print("===DEBUG_FN===")
            pass

    def close_transitions_soft(self) -> None:
        """
        Closes all open transitions in the hypothesis
        """
        # print("Starting CloseTransitions-Soft")
        # print("Currently, the open transitions are", self.hypothesis.open_transitions)
        # self.dtree.print_tree(property="id")
        transitions_to_new: list[Transition] = []  # this should be a pqueue / heap with len(aseq)
        while True:
            while len(self.hypothesis.open_transitions) > 0:
                t = self.hypothesis.open_transitions.pop(0)  # TODO: pop from end for efficiency
                # print("Closing", t)
                assert t.is_open()

                # print("Sifting transition")
                target = t.tgt_node.soft_sift(self.teacher, t.aseq)
                t.set_target(target)

                # print(f"Transition with aseq={t.aseq} has new target {target}")
                # print(target.is_leaf(), target)
                # self.dtree.print_tree_debug(property="state")

                if target.is_leaf() and target.state is None:
                    # transition discovered a new state
                    # print("New state discovered at", target)
                    heapq.heappush(transitions_to_new, t)

            if len(transitions_to_new) > 0:
                t = transitions_to_new[0]
                # print("Making new state for", t)
                # make t into a tree transition
                q = self.hypothesis.make_tree(self.dtree, t)
                # remove all transitions from transitions_to_new with same aseq
                while len(transitions_to_new) > 0 and not t < transitions_to_new[0]:
                    # TODO: this should only be called once so might not need heap
                    heapq.heappop(transitions_to_new)
                # make the state final if needed
                if t.tgt_node.is_final():
                    self.hypothesis.final.add(q)
                # link t to the correct state
                link(t.tgt_node, q)

            if len(transitions_to_new) == len(self.hypothesis.open_transitions) == 0:
                break

    def get_splittable_block(self) -> Optional[tuple[Block, Alphabet]]:
        # print("===Starting get_splittable_block")
        for block in self.hypothesis.blocks:
            for a in alphabet:
                target_block = None
                for state in block.states:
                    target = state.trans[a].tgt_node
                    # print(block, state.id, a, target)
                    assert target is not None
                    assert target.block is not None
                    if target_block is None:
                        target_block = target.block
                    elif target_block != target.block:
                        # print("Found splittable block:", block)
                        return block, a
        return None

    def finalise_discriminators(self) -> None:
        # while eqn 5.1 from paper doesn't hold (can do simple finalisation routine):
        while (block_split := self.get_splittable_block()) is not None:
            block, a = block_split
            successor_lca = Node.lca([q.trans[a].tgt_node for q in block.states])
            # all transitions don't point to the same state (otherwise loop condition does not hold)
            # hence assertion should always hold
            assert successor_lca.discriminator is not None
            discriminator = cast(str, a) + successor_lca.discriminator
            # print(f"Splitting {block} with new final discriminator {discriminator}")
            block.replace_block(self.teacher, discriminator, self.dtree, self.hypothesis)
            self.close_transitions_soft()

    def analyse_output_inconsistency(self, q: State, w: str) -> tuple[str, Alphabet, str]:
        """
        Given a state q and counterexample w, returns a decomposition (u, a, v)
        where len(a) == 1 and certain properties are satisfied.
        """
        # print(f"Analysing inconsistency '{w}' from start state {q}")
        if len(w) == 1:
            return "", Alphabet(w), ""  # only possible decomposition

        # define prefix mapping
        def prefix_mapping(s: str, i: int) -> str:
            assert 0 <= i <= len(s)
            return self.hypothesis.state_of_nondeterministic(
                s[:i], self.teacher, start=q
            ).aseq + s[i:]

        # define alpha
        def alpha(i: int) -> bool:
            return self.teacher.is_member(prefix_mapping(w, i)) == \
                self.hypothesis.evaluate_nondeterministic(w, self.teacher, q)

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

    def find_output_inconsistency(self, block: Block) -> tuple[State, str]:
        """
        Given a block that contains an output inconsistency, find and
        return one such inconsistency.
        """
        for state in block.states:
            assert state.node is not None
            for v, o in state.node.signature:
                # TODO: temporarily use membership query instead of signature
                o = self.teacher.is_member(state.aseq + v)
                if o != self.hypothesis.evaluate(v, start=state):
                    assert self.hypothesis.evaluate(v, start=state) != self.teacher.is_member(state.aseq + v)
                    return state, v

        raise AssertionError("No output inconsistency found")

    def refine(self, counterexample: str) -> None:
        state, w = self.hypothesis.start, counterexample
        finished = False
        while not finished:  # while there exist temporary nodes
            # self.dtree.print_tree(property="id")
            u, a, v = self.analyse_output_inconsistency(state, w)
            print(f"Decomposed {w} into {(u, a, v)}")
            self.split(u, a, v, state)
            # self.dtree.print_tree(property="id")
            self.close_transitions_soft()

            self.dtree.print_tree(property="temporary")
            self.finalise_discriminators()

            # execute if there is a non-trivial block remaining
            finished = True
            for block in self.hypothesis.blocks:
                if not block.is_singleton():
                    # print("Block should have inconsistency:", block, block.states)
                    # debug_fn("block")
                    # self.dtree.print_tree(property="block")
                    state, w = self.find_output_inconsistency(block)
                    # print(state, state.aseq, v)
                    # self.hypothesis.print_hypothesis()
                    assert self.hypothesis.evaluate(w, start=state) != self.teacher.is_member(state.aseq + w)
                    finished = False
                    break

    def split(self, u: str, a: Alphabet, v: str, q: State) -> None:
        q = self.hypothesis.state_of_nondeterministic(u, self.teacher, start=q)  # maybe non-deterministic
        t = q.trans[a]

        old_state = t.tgt_state
        new_state = self.hypothesis.make_tree(self.dtree, t)

        if self.teacher.is_member(new_state.aseq):
            self.hypothesis.final.add(new_state)

        # TODO: justify asserts
        assert old_state is not None
        assert old_state.node is not None
        # print(f"Splitting q{old_state.id}'s node with discriminator {v}")
        leaf0, leaf1 = old_state.node.split_leaf(v, self.hypothesis, old_state.node.block)

        self.hypothesis.open_transitions += list(old_state.node.incoming)

        # TODO: eliminate this query since the result is found in
        #       the output inconsistency analysis
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
        while not (equiv := self.teacher.is_equivalent(self.hypothesis))[0]:
            if max_iterations != -1 and iterations >= max_iterations:
                print("Reached maximum iterations")
                return self.hypothesis, self.dtree

            counterexample = equiv[1]
            assert counterexample is not None
            print("Found counterexample", counterexample)
            self.refine(counterexample)

            print("=======END OF ITERATION=======")
            pprint(self.hypothesis.to_grammar())
            self.dtree.print_tree()
            print("=======START OF ITERATION=======")
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

    teacher = SimpleTeacher(cast(list[str], alphabet), regex, 0.01, 0.01)

    start_time = perf_counter()
    hypothesis, tree = TTTAlgorithm(teacher).learn()
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

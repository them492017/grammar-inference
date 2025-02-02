from __future__ import annotations
from typing import Optional
import sys
from collections import defaultdict

from ttt.print import print
from regex_parser.regex import Regex
from ttt.state import Hypothesis, visualize_dfa
from ttt.node import Node
from ttt.state import State
from ttt.teach import PerfectTeacher, SimpleDFATeacher, Teacher, SimpleTeacher
from ttt.transition import Transition
from ttt.refiner import Refiner


class TTTAlgorithm:
    alphabet: str
    teacher: Teacher
    hypothesis: Hypothesis
    refiner: Refiner
    dtree: Node
    # blocks: list[Node]

    @property
    def blocks(self) -> list[Node]:
        # print(
        #     list({(state, state.node.block) for state in self.dtree.states() if state.node.block is not None})
        # )
        return list({state.node.block for state in self.dtree.states() if state.node.block is not None})

    def __init__(self, teacher: Teacher, alphabet: str):
        self.alphabet = alphabet
        self.teacher = teacher

        t0, t1 = Node.make_leaf(), Node.make_leaf()
        self.dtree = Node.make_inner("", (t0, t1))
        self.hypothesis = Hypothesis(self.dtree, self.alphabet)
        # self.blocks = []

        leaf = self.dtree.sift("", self.teacher)

        if ("", True) in leaf.signature:
            self.hypothesis.make_final(self.hypothesis.start)

        self.refiner = Refiner(self.teacher, self.hypothesis)

        leaf.link(self.hypothesis.start)

        print("=" * 40)
        print("Initial values")
        print("=" * 40)
        print(leaf.signature)
        self.dtree.print_tree()
        print("=" * 40)
        self.close_transitions_soft()

    # def close_transitions_hard(self) -> None:
    #     print("=" * 20)
    #     print("Starting CLOSE_TRANSITIONS_HARD")
    #     print("=" * 20)
    #
    #     for state in self.hypothesis.states:
    #         for transition in state.transitions.values():
    #             if not transition.target_node.is_leaf:
    #                 node = transition.target_node.sift(transition.aseq, self.teacher)
    #                 transition.target_node = node
    #
    #     print("Finished CLOSE_TRANSITIONS_HARD")

    def close_transitions_soft(self) -> None:
        print("=" * 20)
        print("Starting CLOSE_TRANSITIONS")
        print("=" * 20)
        self.dtree.print_tree()
        self.hypothesis.print_hypothesis_transitions()
        for node in self.dtree:
            print(f"node: {node}, is_leaf: {node.is_leaf}")
        print("Open transitions:")
        print("=" * 5)
        for transition in self.hypothesis.open_transitions:
            print(transition)
        print("=" * 10)
        new_transitions: list[Transition] = []  # TODO: make pqueue sorted by tgt_node for fast filtering
        first_run = True

        while len(new_transitions) > 0 or first_run:
            first_run = False

            # close all open transitions
            while len(self.hypothesis.open_transitions) > 0:
                transition = self.hypothesis.open_transitions.pop()

                if transition.is_tree() or (isinstance(transition.target_node, Node)
                        and transition.target_node.is_leaf):
                    # transition is already closed
                    print("skipping")
                    continue

                print(transition, transition.aseq)
                new_tgt = transition.target_node.soft_sift(transition.aseq, self.teacher)
                transition.target_node = new_tgt
                print("New target", transition.target_node)

                if new_tgt.is_leaf and new_tgt.state is None:
                    new_transitions.append(transition)

                # create a new state (might create new open transitions)
                if len(new_transitions) > 0:
                    transition = new_transitions.pop()
                    old_tgt = transition.target_node
                    assert transition.target_state is None

                    print("New state for:", transition)

                    transition.make_tree(old_tgt)

                    new_transitions = [  # TODO: optimise with pqueue
                        t for t in new_transitions if t.target_node != transition.target_node
                    ]

        print("=" * 10)
        print("Finshed CLOSE TRANSITIONS")
        print("=" * 10)
        self.dtree.print_tree()
        for state in self.hypothesis.states:
            print(state)
            for transition in state.transitions.values():
                print(f"\t{transition}, node: {transition.target_node}")
        # self.hypothesis.print_hypothesis()
        print("=" * 10)

    def refine(self, counterexample: str) -> None:
        print("=" * 20)
        print("Starting REFINE")
        print("Blocks:", self.blocks)
        print("=" * 20)
        first_run = True
        inconsistent_state = self.hypothesis.start

        while first_run or self.has_non_trivial_blocks():
            print(f"Starting refine run with counterexample '{counterexample}' and inconsistency {inconsistent_state}")
            first_run = False
            u, a, v = self.refiner.decompose(inconsistent_state.aseq + counterexample, self.teacher)
            print(f"Decomposed {inconsistent_state.aseq + counterexample} to {(u, a, v)}")
            self.split_state(u, a, v)
            self.close_transitions_soft()

            while inconsistency := self.has_trivial_inconsistency():
                print("=" * 20)
                print("Starting DISCRIMINATOR_FINALISATION")
                block, a = inconsistency
                root = block  # we store a block as just its root node
                successor_lca = Node.lca([state.transitions[a].target_node for state in block.states()])
                new_discriminator = a + successor_lca.discriminator

                self.replace_blockroot(root, new_discriminator)
                self.close_transitions_soft()

            if self.has_non_trivial_blocks():
                inconsistent_state, counterexample = self.find_nontrivial_inconsistency()

        print("FINISHED REFINE")
        self.dtree.print_tree()
        for state in self.hypothesis.states:
            print(state)
            for transition in state.transitions.values():
                print(f"\t{transition}, node: {transition.target_node}")
        # self.hypothesis.print_hypothesis()
        print("=" * 10)

    def has_non_trivial_blocks(self) -> bool:
        for block in self.blocks:
            if not block.is_leaf:
                return True
        return False

    def has_trivial_inconsistency(self) -> Optional[tuple[Node, str]]:
        for block in self.blocks:
            if not block.is_leaf:
                print(f"Searching for trivial inconsistency in {block}")
                for a in self.alphabet:
                    successor_block = None
                    for state in block.states():
                        successor_node = state.transitions[a].target_node
                        print(f"char: {a}, state: {state}, successor: {successor_node} - B:{successor_node.block}")

                        if successor_block is None:
                            successor_block = successor_node.block 
                        elif successor_node.block != successor_block:
                            print(f"Found inconsistency - block: {block}, char: {a}")
                            return block, a
        
        return None

    def find_nontrivial_inconsistency(self) -> tuple[State, str]:
        print("Finding non_trivial inconsistency")
        self.dtree.print_tree()
        print(self.blocks)
        # for state in self.hypothesis.states:
        #     print(state, state.node, state.node.block)
        for block in self.blocks:
            if not block.is_leaf:  # |B| > 1
                for state in block.states():
                    print(state, state.node.signature)
                    for discriminator, true_value in state.node.signature:
                        if self.hypothesis.evaluate_non_deterministic(discriminator, self.teacher, start=state) != true_value:
                            print("Found nontrivial inconsistency:", state, state.node.signature)
                            return state, discriminator

        raise ValueError("No nontrivial inconsistency was found")

    def replace_blockroot(self, root: Node, discriminator: str) -> None:
        print("=" * 20)
        print("Starting REPLACE_BLOCKROOT with new discriminator", discriminator)
        print("Block:")

        # if root.discriminator == discriminator:
        #     # skip replacing and just make node final
        #     root._is_temporary = False  # TODO: replace with setter
        #     for child in root.children:
        #         if child is not None:
        #             for node in child:
        #                 node.block = child
        #     return

        root.print_tree()
        self.hypothesis.print_hypothesis_transitions()

        mark0: dict[Node, bool] = defaultdict(lambda: False)
        mark1: dict[Node, bool] = defaultdict(lambda: False)
        inc0: dict[Node, list[Transition]] = defaultdict(list)
        inc1: dict[Node, list[Transition]] = defaultdict(list)
        state0: dict[Node, State] = dict()
        state1: dict[Node, State] = dict()

        mark0[root] = True
        mark1[root] = True

        for node in root:
            for transition in node.incoming_non_tree:
                truth_val = self.teacher.is_member(transition.aseq + discriminator)
                print(f"Truth value of {transition} is lambda({transition.aseq + discriminator}) = {truth_val}")
                if truth_val == 0:
                    inc0[node].append(transition)
                    print(f"Marking {node} with 0")
                    self.mark(node, mark0)
                if truth_val == 1:
                    inc1[node].append(transition)
                    print(f"Marking {node} with 1")
                    self.mark(node, mark1)

            if node.is_leaf:
                state = node.state
                assert state is not None
                truth_val = self.teacher.is_member(state.aseq + discriminator)
                if truth_val == 0:
                    state0[node] = state
                    print(f"Marking {node} with 0")
                    self.mark(node, mark0)
                if truth_val == 1:
                    state1[node] = state
                    print(f"Marking {node} with 1")
                    self.mark(node, mark1)

        print("Extracting t0")
        t0 = self.extract(root, mark0, inc0, state0)
        print("t0 result ===")
        t0.print_tree()
        print("Extracting t1")
        t1 = self.extract(root, mark1, inc1, state1)
        print("t1 result ===")
        t1.print_tree()
        new_root = Node.make_inner(discriminator, (t0, t1))
        # replace root with new_root
        root.replace_with_final(new_root)
        t0.parent = root
        t1.parent = root

        # TODO: do this on the fly
        # update parents and blocks
        print(t0, t1)
        for node in t0:
            print(node)
            node.block = t0
            if node.is_leaf and node.state:
                node.link(node.state)
            if not node.is_leaf:
                if node.children[0]:
                    node.children[0].parent = node
                if node.children[1]:
                    node.children[1].parent = node
        for node in t1:
            print(node)
            node.block = t1
            if node.is_leaf and node.state:
                node.link(node.state)
            if not node.is_leaf:
                if node.children[0]:
                    node.children[0].parent = node
                if node.children[1]:
                    node.children[1].parent = node

        self.dtree.print_tree()

    def extract(
        self,
        root: Node,
        mark: dict[Node, bool],
        inc: dict[Node, list[Transition]],
        state: dict[Node, State]
    ) -> Node:
        # will only be called on marked nodes
        print("=" * 10)
        print("Starting EXTRACT for", root)
        print("mark", mark)
        print("inc", inc)
        print("state", state)
        if root.is_leaf:
            if root in state:
                res = Node.make_leaf()
                print(f"copying state {state[root]}")
                res.state = state[root]
                # TODO: set the block to the appropriate value
            else:
                # since root is marked, if it is not in state it must have an incoming transition
                print("Making new state for leaf")
                return self.create_new(root, inc)
        else:
            c0, c1 = root.children
            if c0 in mark and c1 in mark:
                t0, t1 = self.extract(c0, mark, inc, state), self.extract(c1, mark, inc, state)
                print("Both children marked")
                res = Node.make_inner(root.discriminator, (t0, t1))
            elif c0 in mark:
                inc[c0] += inc[root]
                print("Only 0 child marked")
                return self.extract(c0, mark, inc, state)
            elif c1 in mark:
                inc[c1] += inc[root]
                print("Only 1 child marked")
                return self.extract(c1, mark, inc, state)
            else:
                # since root is a marked inner node, if it has no marked children
                # it must have an incoming transition
                print("Making new state since no marked children")
                return self.create_new(root, inc)

        res.incoming_non_tree = set(inc[root])
        for t in res.incoming_non_tree:
            print(id(t.target_node))
            print(id(res))
        return res

    def mark(self, node: Node, mark: dict[Node, bool]) -> None:
        while node not in mark:
            assert node.parent is not None
            mark[node] = True
            node = node.parent

    def create_new(self, node: Node, inc: dict[Node, list[Transition]]) -> Node:
        transition = inc[node][0]
        print("Incoming:", list(inc.values()))
        new_node = Node.make_leaf()
        transition.make_tree(new_node)
        # TODO: handle setting the block
        new_node.incoming_non_tree = set(inc[node][1:])
        return new_node

    def split_state(self, u: str, a: str, v: str) -> None:
        print(f"Calling split state with (u, a, v) = {(u, a, v)}")
        predicted_state = self.hypothesis.run(u)
        transition = predicted_state.transitions[a]

        old_state = transition.target_state
        assert old_state is not None

        print(f"Splitting {old_state}")
        print(f"Transition is {predicted_state} --{a}-> {old_state}")
        print("Transition:", transition.__dict__)
        print(old_state.node)
        print(old_state.node.incoming_non_tree)

        l0, l1 = old_state.node.split_leaf(v)
        
        print("dtree after spliitng leaf")
        self.dtree.print_tree()

        self.hypothesis.open_transitions += list(old_state.node.incoming_non_tree)

        if self.teacher.is_member(old_state.aseq + v):
            transition.make_tree(l0)
            l1.link(old_state)
        else:
            transition.make_tree(l1)
            l0.link(old_state)

    def learn(self, max_iterations: int = -1) -> tuple[Hypothesis, Node]:
        iterations = 0
        while not (equiv := self.teacher.is_equivalent(self.hypothesis))[0]:
            print("=" * 40)
            print(f"Start of iteration {iterations+1}")
            print("=" * 40)
            self.hypothesis.print_hypothesis()
            self.dtree.print_tree()
            print("=" * 30)
            if iterations >= max_iterations > 0:
                print("Reached maximum iterations")
                return self.hypothesis, self.dtree

            counterexample = equiv[1]
            assert counterexample is not None
            print("Found counterexample", counterexample)

            try:
                self.refine(counterexample)
            except Exception as e:
                print("=" * 50)
                print("Error in learning")

                try:
                    self.dtree.print_tree()
                except Exception:
                    pass

                try:
                    self.hypothesis.print_hypothesis()
                except Exception:
                    pass

                print("=" * 50)

                raise e

            iterations += 1

        print("=" * 40)
        print(f"No counterexample found! Learning complete after {iterations} iterations.")

        return self.hypothesis, self.dtree


if __name__ == "__main__":
    alphabet = "ab"
    pattern = r"(a|b)*a"

    if len(sys.argv) == 2:
        pattern = sys.argv[1]

    # teacher = SimpleDFATeacher(alphabet, pattern, epsilon=0.00001, delta=0.00001)
    teacher = PerfectTeacher(alphabet, pattern)

    print(f"Learning [{pattern}] over alphabet [{alphabet}]")
    try:
        ttt = TTTAlgorithm(teacher, alphabet)
        hypothesis, dtree = ttt.learn()
    except AssertionError as e:
        print("=" * 50)
        print("Could not learn language")
        raise e

    print("Results")
    print("=" * 20)
    dtree.print_tree()
    hypothesis.print_hypothesis()
    visualize_dfa(hypothesis, filename=pattern)

    dfa = hypothesis.to_dfa()

    regex = Regex.parse(pattern)
    regex.to_nfa().visualize(filename="regex_nfa")
    regex_dfa = regex.to_nfa().determinise()
    regex_dfa.visualize(filename="regex_dfa")
    print(f"Hypothesis is equivalent to dfa: {dfa.is_equivalent(regex_dfa)}")

    print("=" * 20)
    print("Results Again")
    print("=" * 20)
    dtree.print_tree()
    hypothesis.print_hypothesis()
    visualize_dfa(hypothesis, filename=pattern)

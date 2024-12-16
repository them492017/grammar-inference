from __future__ import annotations
from typing import Optional

from state import Hypothesis
from node import Node
from state import State
from teach import Teacher, SimpleTeacher
from transition import Transition
from refiner import Refiner

class TTTAlgorithm:
    alphabet: str
    teacher: Teacher
    hypothesis: Hypothesis
    refiner: Refiner
    dtree: Node
    blocks: list[Node]

    def __init__(self, teacher: Teacher, alphabet: str):
        self.alphabet = alphabet
        self.teacher = teacher

        t0, t1 = Node.make_leaf(), Node.make_leaf()
        self.dtree = Node.make_inner("", (t0, t1))
        self.hypothesis = Hypothesis(self.dtree, self.alphabet)
        self.blocks = []

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

    def close_transitions_soft(self) -> None:
        print("=" * 20)
        print("Starting CLOSE_TRANSITIONS")
        print("=" * 20)
        self.dtree.print_tree()
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

                if transition.is_tree or (isinstance(transition.target_node, Node)
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
                    transition.target_node = old_tgt

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
        self.hypothesis.print_hypothesis()
        print("=" * 10)

    def refine(self, counterexample: str) -> None:
        print("=" * 20)
        print("Starting REFINE")
        print("=" * 20)
        first_run = True
        inconsistent_state = self.hypothesis.start

        while first_run or self.has_non_trivial_blocks():
            first_run = False
            u, a, v = self.refiner.decompose(inconsistent_state.aseq + counterexample)
            self.split_state(u, a, v)
            self.close_transitions_soft()

            while inconsistency := self.has_trivial_inconcistency():
                block, a = inconsistency
                root = block  # we store a block as just its root node
                successor_lca = Node.lca([state.transitions[a].target_state for state in block.states()])  # type: ignore
                new_discriminator = a + successor_lca.discriminator

                self.replace_blockroot(root, new_discriminator)
                self.close_transitions_soft()

            if self.has_non_trivial_blocks():
                inconsistent_state, counterexample = self.find_nontrivial_inconcistency()


    def has_non_trivial_blocks(self) -> bool:
        for block in self.blocks:
            if not block.is_leaf:
                return True
        return False

    def has_trivial_inconcistency(self) -> Optional[tuple[Node, str]]:
        successor_block = None
        for block in self.blocks:
            for a in self.alphabet:
                for state in block.states():
                    successor_node = state.transitions[a].target_node

                    # TODO: store block label in node
                    if successor_block is None:
                        successor_block = successor_node.block 
                    elif successor_node.block != successor_block:
                        return block, a
        
        return None

    def find_nontrivial_inconcistency(self) -> tuple[State, str]:
        for block in self.blocks:
            if not block.is_leaf:  # |B| > 1
                for state in block.states():
                    for discriminator, true_value in state.node.signature:
                        if self.hypothesis.evaluate(discriminator, start=state) != true_value:
                            return state, discriminator

        raise ValueError("No nontrivial inconcistency was found")

    def replace_blockroot(self, root: Node, discriminator: str) -> None:
        ...

    def split_state(self, u: str, a: str, v: str) -> None:
        predicted_state = self.hypothesis.run(u)
        transition = predicted_state.transitions[a]

        old_state = transition.target_state
        assert old_state is not None

        print(f"Splitting {old_state}")
        print(f"Transition is {predicted_state} --{a}-> {old_state}")
        print(transition.__dict__)
        print(old_state.node)
        print(old_state.node.incoming_non_tree)

        l0, l1 = old_state.node.split_leaf(v)

        self.hypothesis.open_transitions += list(old_state.node.incoming_non_tree)

        if self.teacher.is_member(old_state.aseq + v):
            transition.make_tree(l0)
            transition.target_node = l0
            l1.link(old_state)
        else:
            transition.make_tree(l1)
            transition.target_node = l1
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
            self.refine(counterexample)

            iterations += 1

        print("=" * 40)
        print(f"No counterexample found! Learning complete after {iterations} iterations.")

        return self.hypothesis, self.dtree


if __name__ == "__main__":
    alphabet = "ab"
    pattern = r"(a|b)*a"

    teacher = SimpleTeacher(alphabet, pattern, epsilon=0.01, delta=0.01)

    print(f"Learning [{pattern}] over alphabet [{alphabet}]")
    ttt = TTTAlgorithm(teacher, alphabet)
    hypothesis, dtree = ttt.learn()

    print(f"Exhaustively checking the hypothesis...")
    print(teacher.is_equivalent_exhaustive(hypothesis, max_length=18))

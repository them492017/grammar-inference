from typing import NewType

from state import State, Hypothesis
from node import LeafNode, NodeProtocol
from tt import Teacher
from transition import NonTreeTransition, TreeTransition

class ObservationPack:
    alphabet: str
    teacher: Teacher
    hypothesis: Hypothesis
    dtree: NodeProtocol

    def __init__(self, teacher: Teacher, alphabet: str):
        self.alphabet = alphabet
        self.teacher = teacher

        t0, t1 = NodeProtocol.make_leaf(), NodeProtocol.make_leaf()
        self.dtree = NodeProtocol.make_inner("", (t0, t1))
        self.hypothesis = Hypothesis(self.dtree, self.alphabet)

        leaf = self.dtree.sift("", self.teacher)
        assert isinstance(leaf, LeafNode)

        if ("", True) in leaf.signature:
            self.hypothesis.make_final(self.hypothesis.start)

        self.link(leaf, self.hypothesis.start)

        self.close_transitions()

    def link(self, node: LeafNode, state: State) -> None:
        node.state = state
        state.node = node

    def close_transitions(self) -> None:
        new_transitions: list[NonTreeTransition] = []  # TODO: make pqueue sorted by tgt_node for fast filtering
        first_run = True

        while len(new_transitions) > 0 or first_run:
            first_run = False

            # close all open transitions
            while len(self.hypothesis.open_transitions) > 0:
                transition = self.hypothesis.open_transitions.pop()

                if not (isinstance(transition, NonTreeTransition)
                        and transition.tgt is not None):
                    # transition is not closed
                    continue

                new_tgt = transition.tgt.sift(transition.aseq, self.teacher)
                transition.tgt = new_tgt

                if isinstance(new_tgt, LeafNode) and new_tgt.state is None:
                    new_transitions.append(transition)

            # create a new state (might create new open transitions)
            if len(new_transitions) > 0:
                transition = new_transitions.pop()
                new_state = transition.make_tree()  # TODO: implement
                new_transitions = [  # TODO: optimise with pqueue
                    t for t in new_transitions if t.tgt != transition.tgt
                ]
                assert transition.tgt is not None  # invariant of make_tree
                self.link(transition.tgt, new_state)

    def refine(self, counterexample: str) -> None:
        u, a, v = self.decompose("", counterexample)
        self.split_state(u, a, v)
        self.close_transitions()

    def decompose(self, prefix: str, counterexample: str) -> tuple[str, str, str]:
        # TODO
        ...

    def split_state(self, u: str, a: str, v: str) -> None:
        old_state = self.hypothesis.run(u)
        transition = old_state.transitions[a]
        assert isinstance(transition, TreeTransition)
        new_state = transition.make_tree()

        assert old_state.node is not None
        l0, l1 = old_state.node.split_leaf(v)

        if self.teacher.is_member(old_state.aseq + v):
            self.link(l0, new_state)
            self.link(l1, old_state)
        else:
            self.link(l0, old_state)
            self.link(l1, new_state)

    def learn(self, max_iterations: int = -1) -> tuple[Hypothesis, NodeProtocol]:
        iterations = 0
        while not (equiv := self.teacher.is_equivalent(self.hypothesis))[0]:
            if max_iterations != -1 and iterations >= max_iterations:
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

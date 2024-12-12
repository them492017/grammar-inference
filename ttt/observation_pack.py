from __future__ import annotations

from state import State, Hypothesis
from node import Node
from teach import Teacher, SimpleTeacher
from transition import Transition
from refiner import Refiner

class ObservationPack:
    alphabet: str
    teacher: Teacher
    hypothesis: Hypothesis
    refiner: Refiner
    dtree: Node

    def __init__(self, teacher: Teacher, alphabet: str):
        self.alphabet = alphabet
        self.teacher = teacher

        t0, t1 = Node.make_leaf(), Node.make_leaf()
        self.dtree = Node.make_inner("", (t0, t1))
        self.hypothesis = Hypothesis(self.dtree, self.alphabet)

        leaf = self.dtree.sift("", self.teacher)

        if ("", True) in leaf.signature:
            self.hypothesis.make_final(self.hypothesis.start)

        self.refiner = Refiner(self.teacher, self.hypothesis)

        self.link(leaf, self.hypothesis.start)

        self.close_transitions()

    def link(self, node: Node, state: State) -> None:
        node.state = state
        state.node = node

    def close_transitions(self) -> None:
        self.dtree.print_tree()
        print(f"Closing {len(self.hypothesis.open_transitions)} open transitions")
        new_transitions: list[Transition] = []  # TODO: make pqueue sorted by tgt_node for fast filtering
        first_run = True

        while len(new_transitions) > 0 or first_run:
            first_run = False

            # close all open transitions
            while len(self.hypothesis.open_transitions) > 0:
                transition = self.hypothesis.open_transitions.pop()
                print(transition.__dict__, transition.target.__dict__)

                if transition.is_tree or (isinstance(transition.target, Node)
                        and transition.target.is_leaf):
                    # transition is already closed
                    # check validity of assumption... could we be missing undiscovered states?
                    print("skipping")
                    continue

                assert isinstance(transition.target, Node)
                new_tgt = transition.target.sift(transition.aseq, self.teacher)
                transition.target = new_tgt

                print(new_tgt)
                if new_tgt.state is None:
                    new_transitions.append(transition)
                else:
                    print(new_tgt.state.aseq)

                # create a new state (might create new open transitions)
                if len(new_transitions) > 0:
                    print(f"We have {len(new_transitions)} new transitions")
                    transition = new_transitions.pop()
                    old_tgt = transition.target
                    assert isinstance(old_tgt, Node)
                    new_state = transition.make_tree()
                    new_transitions = [  # TODO: optimise with pqueue
                        t for t in new_transitions if t.target != transition.target
                    ]
                    assert transition.target is not None  # invariant of make_tree
                    print(transition, transition.target)
                    self.link(old_tgt, new_state)  # type: ignore
        self.dtree.print_tree()
        self.hypothesis.print_hypothesis()

    def refine(self, counterexample: str) -> None:
        u, a, v = self.refiner.decompose(counterexample)
        self.split_state(u, a, v)
        self.close_transitions()


    def split_state(self, u: str, a: str, v: str) -> None:
        old_state = self.hypothesis.run(u)
        transition = old_state.transitions[a]
        new_state = transition.make_tree()

        assert old_state.node is not None
        l0, l1 = old_state.node.split_leaf(v)

        if self.teacher.is_member(old_state.aseq + v):
            self.link(l0, new_state)
            self.link(l1, old_state)
        else:
            self.link(l0, old_state)
            self.link(l1, new_state)

    def learn(self, max_iterations: int = -1) -> tuple[Hypothesis, Node]:
        iterations = 0
        while not (equiv := self.teacher.is_equivalent(self.hypothesis))[0]:
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
    pattern = "a*b*a"

    teacher = SimpleTeacher(alphabet, pattern)
    ob = ObservationPack(teacher, alphabet)
    ob.learn()

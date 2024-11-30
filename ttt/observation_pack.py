from typing import NewType

from state import State
from node import Node

# TODO: move this
Alphabet = NewType('Alphabet', str)
class SimpleTeacher:
    ...

class ObservationPack:
    # teacher: Teacher
    teacher: SimpleTeacher
    hypothesis: State
    dtree: Node

    def __init__(self, teacher: SimpleTeacher):
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
        ...

    def refine(self, counterexample: str) -> None:
        ...

    def split(self, u: str, a: Alphabet, v: str) -> None:
        ...

    def learn(self, max_iterations: int = -1) -> tuple[State, Node]:
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

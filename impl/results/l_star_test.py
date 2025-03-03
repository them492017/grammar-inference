from results.in_order_test import InferenceAlgorithm, in_order_test, OUTPUT_DIR
from teacher.base_teacher import Teacher
from lstar.l_star import ObservationTable, l_star
from regex_parser.dfa import DFA


class LStarAlgorithm(InferenceAlgorithm):
    _alphabet: str

    def __init__(self, alphabet: str) -> None:
        self._alphabet = alphabet

    def learn(self, teacher: Teacher) -> DFA:
        table = ObservationTable(self.alphabet())
        return l_star(table, teacher)

    def alphabet(self) -> str:
        return self._alphabet


if __name__ == "__main__":
    alg = LStarAlgorithm("ab")
    in_order_test(alg, f"{OUTPUT_DIR}/l_star.csv")

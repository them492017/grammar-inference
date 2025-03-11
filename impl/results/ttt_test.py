from results.in_order_test import InferenceAlgorithm, in_order_test, OUTPUT_DIR
from teacher.base_teacher import Teacher
from ttt.ttt import TTTAlgorithm
from regex_parser.dfa import DFA


class TTTAlgorithm1(InferenceAlgorithm):
    _alphabet: str

    def __init__(self, alphabet: str) -> None:
        self._alphabet = alphabet

    def learn(self, teacher: Teacher) -> DFA:
        ttt = TTTAlgorithm(teacher, self._alphabet)
        hypothesis, _ = ttt.learn()
        return hypothesis.to_dfa()

    def alphabet(self) -> str:
        return self._alphabet


if __name__ == "__main__":
    alg = TTTAlgorithm1("ab")
    in_order_test(alg, f"{OUTPUT_DIR}/ttt_fuzzing_test.csv")


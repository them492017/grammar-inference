from typing import TextIO
from functools import lru_cache

from teacher.simple_teacher import SimpleTeacher
from teacher.stat_teacher import StatTeacher
from regex_parser.dfa import DFA
from regex_enumerator.generate import generate
from results.f1 import compute_f1


OUTPUT_DIR = "results/output/"


class InferenceAlgorithm:
    def learn(self, teacher: StatTeacher) -> DFA:
        ...

    def alphabet(self) -> str:
        ...


@lru_cache(maxsize=512)
def learn_pattern_with_stats(algorithm: InferenceAlgorithm, pattern: str, file: TextIO) -> None:
    print(f"Learning [{pattern}] over alphabet [{algorithm.alphabet()}]")

    teacher = SimpleTeacher(algorithm.alphabet(), pattern)
    try:
        dfa = algorithm.learn(teacher)
    except AssertionError as e:
        print("=" * 50)
        print("Could not learn language")
        print(pattern)
        raise e
    num_membership_excl_cache, num_membership, num_equivalence = (teacher.num_membership_excl_cache, teacher.num_membership, teacher.num_equivalence)
    precision, recall, f1 = compute_f1(dfa, teacher, algorithm.alphabet())
    print(f"{pattern},{num_membership_excl_cache},{num_membership},{num_equivalence},{precision},{recall},{f1}", file=file)


def in_order_test(algorithm: InferenceAlgorithm, output_file: str) -> None:
    file = open(output_file, "w")

    print("pattern,unique_membership_queries,membership_queries,equivalence_queries,precision,recall,f1", file=file)

    try:
        for pattern in generate(algorithm.alphabet()):
            learn_pattern_with_stats(algorithm, pattern, file)
    except KeyboardInterrupt:
        file.close()

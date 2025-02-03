from typing import TextIO
from regex_enumerator.generate import generate, generate_random
from regex_parser.regex import Regex
from ttt.teach import PerfectTeacher, SimpleTeacher
from ttt.ttt import TTTAlgorithm

from functools import lru_cache


@lru_cache(maxsize=512)
def learn_pattern(pattern: str, file: TextIO) -> None:
    print(f"Learning [{pattern}] over alphabet [{alphabet}]")

    teacher = PerfectTeacher(alphabet, pattern)
    try:
        ttt = TTTAlgorithm(teacher, alphabet)
        hypothesis, dtree = ttt.learn()
    except AssertionError as e:
        print("=" * 50)
        print("Could not learn language")
        print(pattern)
        teacher.dfa.visualize('teacher_dfa')
        raise e

    dfa = hypothesis.to_dfa()

    regex = Regex.parse(pattern)
    regex_dfa = regex.to_nfa().determinise()

    print(f"Pattern: {pattern}, Equivalent: {dfa.is_equivalent(regex_dfa)}", file=file)


@lru_cache(maxsize=512)
def learn_pattern_with_stats(pattern: str, file: TextIO) -> None:
    print(f"Learning [{pattern}] over alphabet [{alphabet}]")

    teacher = SimpleTeacher(alphabet, pattern)
    try:
        ttt = TTTAlgorithm(teacher, alphabet)
        hypothesis, dtree = ttt.learn()
    except AssertionError as e:
        print("=" * 50)
        print("Could not learn language")
        print(pattern)
        raise e

    dfa = hypothesis.to_dfa()

    regex = Regex.parse(pattern)
    regex_dfa = regex.to_nfa().determinise()

    print(f"{pattern},{teacher.num_membership_excl_cache},{teacher.num_membership},{teacher.num_equivalence},{dfa.is_equivalent(regex_dfa)[0]}", file=file)


@lru_cache(maxsize=512)
def learn_pattern_with_stats_without_dfa_check(pattern: str, file: TextIO) -> None:
    print(f"Learning [{pattern}] over alphabet [{alphabet}]")

    teacher = SimpleTeacher(alphabet, pattern)
    try:
        ttt = TTTAlgorithm(teacher, alphabet)
        hypothesis, dtree = ttt.learn()
    except AssertionError as e:
        print("=" * 50)
        print("Could not learn language")
        print(pattern)
        raise e

    print(f"{pattern},{teacher.num_membership_excl_cache},{teacher.num_membership},{teacher.num_equivalence}", file=file)


if __name__ == "__main__":
    alphabet = "ab"
    file = open("ttt_random_test_no_check.csv", "w")

    print("pattern,unique_membership_queries,membership_queries,equivalence_queries,success", file=file)

    try:
        for pattern in generate_random(alphabet):
            learn_pattern_with_stats_without_dfa_check(pattern, file)
    except KeyboardInterrupt:
        file.close()

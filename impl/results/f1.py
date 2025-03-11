import math
import re
import regex  # rust regex crate wrapper (matches in linear time)
import random
from itertools import chain

from regex_parser.dfa import DFA
from regex_parser.regex import Regex
from teacher.base_teacher import Teacher
from fuzzing.simple_fuzzer import SimpleFuzzer


def gen_random(alphabet, max_length) -> str:
    """Generate a uniformly random string from the set of strings of length at most `max_length`"""
    total_combinations = len(alphabet) ** max_length
    random_idx = random.randint(1, total_combinations)
    length = math.floor(math.log(random_idx, len(alphabet)))
    return "".join(random.choice(alphabet) for _ in range(length))


def compute_f1(hypothesis: DFA, pattern: str, alphabet: str, num_strings: int = 1000, max_length: int = 15) -> tuple[float, float, float]:
    """Returns tuple (precision, recall, f1)"""
    # true/false - positive/negative
    tp, tn, fp, fn = 0, 0, 0, 0
    r = regex.Regex(f"^{pattern}$")

    for _ in range(num_strings):
        # TODO: should generate 1000 strings from language of hypothesis and 1000 from language of teacher's regex
        s = gen_random(alphabet, max_length)
        predicted, actual = hypothesis.evaluate(s), r.is_match(s)
        if predicted and actual:
            tp += 1
        if predicted and not actual:
            fp += 1
        if not predicted and actual:
            fn += 1
        if not predicted and not actual:
            tn += 1
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return (precision, recall, f1)


def compute_f1_with_fuzzed_strings(hypothesis: DFA, pattern: str, alphabet: str, num_strings: int = 1000, max_length: int = 15) -> tuple[float, float, float]:
    """Returns tuple (precision, recall, f1)"""
    # true/false - positive/negative
    tp, tn, fp, fn = 0, 0, 0, 0
    r = regex.Regex(f"^{pattern}$")
    actual_dfa = Regex.parse(pattern).to_nfa().determinise()

    hypothesis_fuzzer = SimpleFuzzer(*hypothesis.to_grammar())
    actual_fuzzer = SimpleFuzzer(*actual_dfa.to_grammar())
    seen: set[str] = set()

    for s in chain(hypothesis_fuzzer.run(num_strings), actual_fuzzer.run(num_strings), gen_random(alphabet, max_length)):
        if s in seen:
            continue

        seen.add(s)
        predicted, actual = hypothesis.evaluate(s), r.is_match(s)
        if predicted and actual:
            tp += 1
        if predicted and not actual:
            fp += 1
        if not predicted and actual:
            fn += 1
        if not predicted and not actual:
            tn += 1
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
    if tp + fn > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    return (precision, recall, f1)

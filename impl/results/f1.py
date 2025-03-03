import math
import re
import random

from regex_parser.dfa import DFA
from teacher.base_teacher import Teacher


def gen_random(alphabet, max_length) -> str:
    """Generate a uniformly random string from the set of strings of length at most `max_length`"""
    total_combinations = len(alphabet) ** max_length
    random_idx = random.randint(1, total_combinations)
    length = math.floor(math.log(random_idx, len(alphabet)))
    return "".join(random.choice(alphabet) for _ in range(length))


def compute_f1(hypothesis: DFA, teacher: Teacher, alphabet: str, num_strings: int = 1000, max_length: int = 15) -> tuple[float, float, float]:
    """Returns tuple (precision, recall, f1)"""
    # true/false - positive/negative
    tp, tn, fp, fn = 0, 0, 0, 0

    for _ in range(num_strings):
        s = gen_random(alphabet, max_length)
        predicted, actual = hypothesis.evaluate(s), teacher.is_member(s)
        if predicted and actual:
            tp += 1
        if predicted and not actual:
            fp += 1
        if not predicted and actual:
            fn += 1
        if not predicted and not actual:
            tn += 1
    
    print(tp, tn, fp, fn)

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

from typing import Optional, Any
import math
import random
from pprint import pprint

import simplefuzzer as fuzzer
import rxfuzzer
import earleyparser
import cfgrandomsample
import cfgremoveepsilon


# random.seed(1)


class MATeacher:
    """
    A teacher class heavily inspired by
    http://rahul.gopinath.org/post/2024/01/04/lstar-learning-regular-languages/#pac-learning
    """

    def __init__(self, rex, delta=0.1, epsilon=0.1):
        self.g, self.s = rxfuzzer.RegexToGrammar().to_grammar(rex)
        self.parser = earleyparser.EarleyParser(self.g)
        self.sampler = cfgrandomsample.RandomSampleCFG(self.g)
        self.membership_query_counter = 0
        self.equivalence_query_counter = 0
        self.delta, self.epsilon = delta, epsilon

    def is_member(self, u: str) -> bool:
        print(f"Performing membership query on '{u}'")
        self.membership_query_counter += 1
        try:
            list(self.parser.recognize_on(u, self.s))
        except Exception:
            print("Returning false")
            return False
        print("Returning true")
        return True

    def is_equivalent(self, grammar, start, max_length_limit=10) -> tuple[bool, Optional[str]]:
        print("Performing equivalence query with grammar:")
        pprint(grammar)
        self.equivalence_query_counter += 1
        num_calls = math.ceil(1.0/self.epsilon * (math.log(1.0/self.delta) +
                              self.equivalence_query_counter * math.log(2)))

        for limit in range(1, max_length_limit):
            is_eq, counterex, c = self.is_equivalent_for(
                self.g, self.s, grammar, start, limit, num_calls
            )
            if counterex is None:  # no members of length limit
                continue
            if not is_eq:
                c = [a for a in counterex if a is not None][0]
                return False, c
        return True, None

    def fix_epsilon(self, grammar_, start):
        # clone grammar
        grammar = {k: [[t for t in r] for r in grammar_[k]] for k in grammar_}
        gs = cfgremoveepsilon.GrammarShrinker(grammar, start)
        gs.remove_epsilon_rules()
        return gs.grammar, start

    def digest_grammar(self, g, s, l) -> tuple[int, Any, Any]:
        if not g[s]:
            return 0, None, None
        g, s = self.fix_epsilon(g, s)
        rgf = cfgrandomsample.RandomSampleCFG(g)
        key_node = rgf.key_get_def(s, l)
        cnt = key_node.count
        ep = earleyparser.EarleyParser(g)
        return cnt, key_node, ep

    def gen_random(self, key_node, cnt):
        if cnt == 0:
            return None
        at = random.randint(0, cnt-1)
        # sampler does not store state.
        st_ = self.sampler.key_get_string_at(key_node, at)
        return fuzzer.tree_to_string(st_)

    def is_equivalent_for(self, g1, s1, g2, s2, l, n):
        cnt1, key_node1, ep1 = self.digest_grammar(g1, s1, l)
        cnt2, key_node2, ep2 = self.digest_grammar(g2, s2, l)
        count = 0

        str1 = {self.gen_random(key_node1, cnt1) for _ in range(n)}
        str2 = {self.gen_random(key_node2, cnt2) for _ in range(n)}

        for st1 in str1:
            if st1 is None:
                continue
            count += 1
            try:
                list(ep2.recognize_on(st1, s2))
            except Exception:
                return False, (st1, None), count

        for st2 in str2:
            if st2 is None:
                continue
            count += 1
            try:
                list(ep1.recognize_on(st2, s1))
            except Exception:
                return False, (None, st2), count

        return True, None, count

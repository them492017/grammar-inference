import simplefuzzer as fuzzer
import rxfuzzer
import earleyparser
import cfgrandomsample
import cfgremoveepsilon
import math
import random
import string
import re
import sys

from regex_parser.dfa import DFA
from ttt.state import Hypothesis
from ttt.node import Node
from teacher.base_teacher import Teacher
from teacher.simple_teacher import SimpleTeacher
from lstar.teach import convert_grammar_to_dfa

class ObservationTable:
    def __init__(self, alphabet):
        self._T, self.P, self.S, self.A = {}, [''], [''], alphabet

    def cell(self, v, e): return self._T[v][e]

    def state(self, p):
        return f"<{''.join([str(int(self.cell(p,s))) for s in self.S])}>"

    def table_to_grammar(self):
        # Step 1: identify all distinguished states.
        prefix_to_state = {}  # Mapping from row string to state ID
        states = {}
        grammar = {}
        for p in self.P:
            stateid = self.state(p)
            if stateid not in states: states[stateid] = []
            states[stateid].append(p)
            prefix_to_state[p] = stateid

        for stateid in states: grammar[stateid] = []

        # Step 2: Identify the start state, which corresponds to epsilon row
        start_nt = prefix_to_state['']

        # Step 3: Identify the accepting states
        accepting = [prefix_to_state[p] for p in self.P if self.cell(p,'') == 1]
        if not accepting: return {'<start>': []}, '<start>'
        for s in accepting: grammar[s] = [['<_>']]
        grammar['<_>'] = [[]]

        # Step 4: Create the transition function
        for sid1 in states:
            first_such_row = states[sid1][0]
            for a in self.A:
                sid2 = self.state(first_such_row + a)
                grammar[sid1].append([a, sid2])

        return grammar, start_nt

    def table_to_hypothesis(self) -> Hypothesis:
        """
        Converts an observation table into a DFA hypothesis.
        """
        # Identify all states
        prefix_to_state = {}  # Mapping from row prefix to State object
        states = {}  # Mapping from state ID to its prefixes

        for p in self.P:
            state_id = self.state(p)  # Unique state representation
            if state_id not in states:
                states[state_id] = []
            states[state_id].append(p)
            prefix_to_state[p] = state_id

        # Create the root node and initialize the hypothesis
        root_node = Node.make_leaf()  # Assuming a Node class exists
        hypothesis = Hypothesis(root_node, self.A)

        # Create DFA states
        state_objects = {}  # state_id -> State
        for state_id in states:
            state_objects[state_id] = hypothesis.add_state(state_id)

        # Identify start state
        hypothesis.start = state_objects[prefix_to_state[""]]

        # Identify final states
        for p in self.P:
            if self.cell(p, '') == 1:
                hypothesis.final_states.add(state_objects[prefix_to_state[p]])

        # Create transitions
        for state_id in states:
            first_prefix = states[state_id][0]
            current_state = state_objects[state_id]

            for a in self.A:
                next_prefix = first_prefix + a
                next_state_id = self.state(next_prefix)
                if next_state_id in state_objects:
                    next_state = state_objects[next_state_id]
                    hypothesis.add_transition(current_state, a, next_state)
                else:
                    raise ValueError("Transition is open")

        return hypothesis

    def table_to_dfa(self) -> DFA:
        """
        Converts an observation table into a DFA.
        """
        dfa = DFA()
        dfa.next_state = 0
        
        # Step 1: Identify all states
        prefix_to_state = {}  # Mapping from row prefix to DFA state ID
        states = {}  # Mapping from state ID to prefixes

        for p in self.P:
            state_id = self.state(p)  # Unique identifier for this row
            if state_id not in states:
                states[state_id] = dfa.next_state  # Assign DFA state number
            prefix_to_state[p] = dfa.next_state
            dfa.add_state({})

        # Step 2: Set start state (corresponding to empty prefix "")
        dfa.start = prefix_to_state[""]

        # Step 3: Identify final states
        for p in self.P:
            if self.cell(p, '') == 1:
                dfa.final.add(prefix_to_state[p])

        # Step 4: Define transitions
        for p in self.P:
            current_state = prefix_to_state[p]
            for a in self.A:
                next_prefix = p + a
                next_state = prefix_to_state.get(next_prefix, None)
                if next_state is not None:
                    dfa.transitions[current_state][a] = next_state

        dfa.close_with_sink(self.A)

        return dfa

    def remove_infinite_loops(self, g, start):
        rule_cost = fuzzer.compute_cost(g)
        remove_keys = []
        for k in rule_cost:
            if k == start: continue
            res = [rule_cost[k][r] for r in rule_cost[k]
                   if rule_cost[k][r] != math.inf]
            if not res: remove_keys.append(k)

        cont = True
        while cont:
            cont = False
            new_g = {}
            for k in g:
                if k in remove_keys: continue
                new_g[k] = []
                for r in g[k]:
                    if [t for t in r if t in remove_keys]: continue
                    new_g[k].append(r)
                if not new_g[k]:
                    if k == start: continue
                    remove_keys.append(k)
                    cont = True
        return new_g, start

    def grammar(self):
        g, s = self.table_to_grammar()
        return self.remove_infinite_loops(g, s)

    def init_table(self, oracle: Teacher):
        self._T[''] = {'': oracle.is_member('') }
        self.update_table(oracle)

    def update_table(self, oracle: Teacher):
        def unique(l): return list({s:None for s in l}.keys())
        rows = self.P
        auxrows = [p + a for p in self.P for a in self.A]
        PuPxA = unique(rows + auxrows)
        for p in PuPxA:
            if p not in self._T: self._T[p] = {}
            for s in self.S:
                if p in self._T and s in self._T[p]: continue
                self._T[p][s] = oracle.is_member(p + s)

    def closed(self):
        states_in_P = {self.state(p) for p in self.P}
        P_A = [p+a for p in self.P for a in self.A]
        for t in P_A:
            if self.state(t) not in states_in_P: return False, t
        return True, None

    def add_prefix(self, p, oracle: Teacher):
        if p in self.P: return
        self.P.append(p)
        self.update_table(oracle)

    def consistent(self):
        matchingpairs = [(p1, p2) for p1 in self.P for p2 in self.P
                         if p1 != p2 and self.state(p1) == self.state(p2)]
        suffixext = [(a, s) for a in self.A for s in self.S]
        for p1,p2 in matchingpairs:
            for a, s in suffixext:
                if self.cell(p1+a,s) != self.cell(p2+a,s):
                        return False, (p1, p2), (a + s)
        return True, None, None

    def add_suffix(self, a_s, oracle: Teacher):
        if a_s in self.S: return
        self.S.append(a_s)
        self.update_table(oracle)

def l_star(T: ObservationTable, teacher: Teacher) -> DFA:
    T.init_table(teacher)

    while True:
        while True:
            is_closed, unknown_P = T.closed()
            is_consistent, _, unknown_AS = T.consistent()
            if is_closed and is_consistent: break
            if not is_closed:
                T.add_prefix(unknown_P, teacher)
            if not is_consistent:
                T.add_suffix(unknown_AS, teacher)

        dfa = convert_grammar_to_dfa(T.grammar(), T.A)
        eq, counterX = teacher.is_equivalent(dfa)
        if eq:
            return convert_grammar_to_dfa(T.grammar(), T.A)
        for i, _ in enumerate(counterX):
            T.add_prefix(counterX[0:i+1], teacher)


def match(p, start, text):
    try: p.recognize_on(text, start)
    except SyntaxError as e: return False
    return True


if __name__ == '__main__':
    alphabet = "ab"
    expr = sys.argv[1]
    teacher = SimpleTeacher(alphabet, expr)
    tbl = ObservationTable(list(alphabet))
    dfa = l_star(tbl, teacher)
    print(dfa)

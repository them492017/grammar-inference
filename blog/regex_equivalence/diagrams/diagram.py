from regex_parser.dfa import NFA


def trim_prefix(s: str) -> str:
    return "\n".join(
        line.lstrip() for line in s.strip().splitlines()
    )

def create_diagram(nfa_string: str, filename: str) -> None:
    NFA.from_string(trim_prefix(nfa_string)).visualize(filename)


create_diagram(
    """
    States: 0 1
    Initial: 0
    Final: 0
    Transitions:
    0 a 1
    0 b 1
    1 a 1
    1 b 1
    """,
    "blog/regex_equivalence/diagrams/nfa1"
)

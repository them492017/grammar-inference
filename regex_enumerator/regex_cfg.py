"""
S → Q | A | T | C | K
Q → A + ε | T + ε | C + ε
A → T + AT | C + AC | K + AK
AT → T | T + AT | AC
AC → C | C + AC | AK
AK → K | K + AK
T → a1 | a2 | · · · | ak
C → C0 C0 | C0 C
C0 → (Q) | (A) | T | K
K → (A) ∗ | T ∗ | (C) ∗
"""
S = "<S>"
Q = "<Q>"
A = "<A>"
AT = "<AT>"
AC = "<AC>"
AK = "<AK>"
T = "<T>"
C = "<C>"
C0 = "<C0>"
K = "<K>"
EPSILON = "\u03b5"

def make_cfg(alphabet: str) -> dict[str, list[list[str]]]:
    return {
        S: [[Q], [A], [T], [C], [K]],
        Q: [["(", A, "|", EPSILON, ")"], ["(", T, "|", EPSILON, ")"], ["(", C, "|", EPSILON, ")"]],
        A: [["(", T, "|", AT, ")"], ["(", C, "|", AC, ")"], [K, "|", AK]],
        AT: [[T], ["(", T, "|", AT, ")"], [AC]],
        AC: [[C], ["(", C, "|", AC, ")"], [AK]],
        AK: [[K], [K, "|", AK]],
        T: [[c] for c in alphabet],
        C: [["(", C0, C0, ")"], ["(", C0, C, ")"]],
        C0: [["(", Q, ")"], ["(", A, ")"], [T], [K]],
        K: [["(", "(", A, ")", "*", ")"], ["(", "(", T, ")", "*", ")"], ["(", "(", C, ")", "*", ")"]]
    }
